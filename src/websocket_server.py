"""
WebSocket server for keystroke streaming.

API Routes:
- WebSocket /ws/session - Main keystroke streaming connection
- GET /health - Health check
- GET /api/status - Server status

Message Protocol (JSON):
- Frontend -> Backend: {"type": "text_update", "full_text": "..."}
- Frontend -> Backend: {"type": "submit"}
- Backend -> Frontend: {"event": "text_update|submit_start|generation_complete", ...}
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import json
import time
import logging

from .stream_controller import (
    StreamController,
    KVCacheManager,
)

from .utils import load_model_and_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for API Documentation
# =============================================================================

class ServerStatus(BaseModel):
    status: str
    active_sessions: int
    uptime_seconds: float

# =============================================================================
# Global State (Model & Tokenizer)
# =============================================================================

ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load ML models on startup, unload on shutdown.
    """
    logger.info("Loading model and tokenizer... (This may take a moment)")
    
    # LOAD HERE - This runs before any request is accepted
    model, tokenizer, device = load_model_and_tokenizer("Qwen/Qwen2.5-1.5B-Instruct")
    
    ml_resources["model"] = model
    ml_resources["tokenizer"] = tokenizer
    ml_resources["device"] = device
    
    logger.info("Model loaded successfully!")
    
    yield # Server is running here
    
    # Cleanup (if needed)
    ml_resources.clear()
    logger.info("ML resources released.")

BASE_PROMPT = """
You are an expert SQL Assistant and Data Architect. Your goal is to generate accurate, syntactically correct, and efficient SQL queries based on the user's natural language question and the provided database schema.

### INSTRUCTIONS:

1.  **Context & Role**:
    * Act as a senior data analyst.
    * Do not explain the query. Output only the SQL query code block.
    * Use the dialect: **PostgreSQL**.

2.  **Schema Adherence**:
    * Use ONLY the tables and columns provided in the schema. Do not hallucinate columns or tables.
    * Pay attention to primary keys and foreign keys for JOIN conditions.
    * If a requested column is ambiguous, default to the most logical choice.

3.  **SQL Style & Constraints**:
    * Use standard capitalization: Keywords in UPPERCASE (SELECT, FROM, WHERE), identifiers in lowercase or snake_case matching the schema.
    * Always use table aliases (e.g., `u` for users, `o` for orders) for clarity.
    * Use explicit JOIN syntax (`JOIN ... ON`).
    * Limit results to 100 unless otherwise specified.
    * Handle NULLs appropriately (e.g., use `IS NULL` or `COALESCE`).

4.  **Reasoning Process**:
    * Identify relevant tables.
    * Identify necessary conditions (WHERE clause).
    * Determine aggregation level (GROUP BY) if counting or summarizing.

5.  **Safety**:
    * **Strictly READ-ONLY**: Never generate DDL (CREATE, DROP, ALTER) or DML (INSERT, UPDATE, DELETE) statements.
    * If the user asks for data outside the schema, respond with: "I cannot answer this question with the provided database schema."

### Schema:
Table: users (id, name, email, signup_date)
Table: orders (id, user_id, amount, status, created_at)

### EXAMPLE:
Question: 
Show me the top 5 users by total spending who signed up in 2023.

```sql
SELECT
    u.name,
    SUM(o.amount) as total_spent
FROM
    users u
JOIN
    orders o ON u.id = o.user_id
WHERE
    u.signup_date >= '2023-01-01' AND u.signup_date <= '2023-12-31'
GROUP BY
    u.id, u.name
ORDER BY
    total_spent DESC
LIMIT 5; 

### User Input
Question:
"""


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Keystroke Streaming API",
    description="WebSocket API for anticipatory prefill with keystroke streaming",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

server_start_time = time.time()
active_sessions: Dict[str, StreamController] = {}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/api/status", response_model=ServerStatus)
async def get_status():
    return ServerStatus(
        status="running",
        active_sessions=len(active_sessions),
        uptime_seconds=time.time() - server_start_time,
    )


@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket):
    """
    Main WebSocket endpoint for keystroke streaming.

    Message Protocol:

    Frontend -> Backend:
    - {"type": "start_session", "base_prompt": "..."}
    - {"type": "text_update", "full_text": "..."}
    - {"type": "submit"}

    Backend -> Frontend:
    - {"event": "connected", "session_id": "..."}
    - {"event": "session_started", "session_id": "..."}
    - {"event": "text_update", "current_text": "...", "confirmed_text": "...", "pending_text": "..."}
    - {"event": "submit_start", "final_text": "..."}
    - {"event": "generation_start", "time_to_first_token_ms": ...}
    - {"event": "generation_token", "token": "..."}
    - {"event": "generation_complete", "generated_text": "...", "total_time_ms": ...}
    - {"event": "error", "message": "..."}
    """
    await websocket.accept()
    session_id = f"session_{int(time.time() * 1000)}"

    if "model" not in ml_resources:
        logger.error("Model not loaded yet")
        await websocket.close(code=1011) # Internal Error
        return
    
    # Create controller with KV cache
    cache_manager = KVCacheManager(
        model=ml_resources["model"], 
        tokenizer=ml_resources["tokenizer"], 
        device=ml_resources["device"]
    )
    controller = StreamController(cache_manager, debounce_ms=300.0)

    async def on_flush(result: dict):
        try:
            await websocket.send_json(result)
        except Exception as e:
            logger.error(f"Error sending flush: {e}")

    controller.set_flush_callback(on_flush)
    active_sessions[session_id] = controller

    logger.info(f"New WebSocket connection: {session_id}")

    try:
        await websocket.send_json({
            "event": "connected",
            "session_id": session_id,
            "message": "WebSocket connected. Send 'start_session' to begin.",
        })

        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                msg_type = message.get("type", "")

                if msg_type == "start_session":
                    # Start new typing session
                    await controller.start_session(BASE_PROMPT)
                    await websocket.send_json({
                        "event": "session_started",
                        "session_id": session_id,
                        "base_prompt_length": len(BASE_PROMPT),
                    })
                
                elif msg_type == "keystroke":
                    # Handle keystroke
                    text = message.get("text", "")
                    if text:
                        print("Received keystroke...")
                        result = await controller.on_keystroke(text)
                        await websocket.send_json(result)
                
                elif msg_type == "submit":
                    # Handle submit - stream generation results
                    print("Received submit...")
                    async for result in controller.on_submit():
                        await websocket.send_json(result)

                else:
                    await websocket.send_json({
                        "event": "error",
                        "message": f"Unknown message type: {msg_type}",
                    })

            except json.JSONDecodeError:
                await websocket.send_json({
                    "event": "error",
                    "message": "Invalid JSON",
                })
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({
                    "event": "error",
                    "message": str(e),
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]


def run_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
