"""
WebSocket server for keystroke streaming.

API Routes:
- WebSocket /ws/session - Main keystroke streaming connection
- GET /health - Health check
- GET /api/status - Server status

Message Protocol (JSON):
- Frontend -> Backend: {"type": "text_update", "text": "...", "submit": false}
- Backend -> Frontend: {"event": "text_update|submit_start|generation_complete", ...}
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
import json
import time
import logging

from .stream_controller import (
    StreamController,
    MockKVCacheManager,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServerStatus(BaseModel):
    status: str
    active_sessions: int
    uptime_seconds: float


app = FastAPI(
    title="Keystroke Streaming API",
    description="WebSocket API for anticipatory prefill with keystroke streaming",
    version="2.0.0",
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
    - {"type": "text_update", "text": "...", "submit": false}
    - {"type": "text_update", "text": "...", "submit": true}

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

    cache_manager = MockKVCacheManager()
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
                    base_prompt = message.get("base_prompt", "")
                    await controller.start_session(base_prompt)
                    await websocket.send_json({
                        "event": "session_started",
                        "session_id": session_id,
                        "base_prompt_length": len(base_prompt),
                    })

                elif msg_type == "text_update":
                    text = message.get("text", "")
                    submit = message.get("submit", False)

                    if submit:
                        async for result in controller.on_text_submit(text):
                            await websocket.send_json(result)
                    else:
                        result = await controller.on_text_update(text)
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
