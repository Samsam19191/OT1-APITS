"""
WebSocket server for keystroke streaming.

This FastAPI server receives keystroke events from Riad's frontend
and calls StreamController methods to manage KV-cache prefill.

API Routes:
- WebSocket /ws/session - Main keystroke streaming connection
- GET /health - Health check
- GET /api/status - Server status

Message Protocol (JSON):
- Frontend -> Backend: {"type": "keystroke|delete|submit", "char": "a", ...}
- Backend -> Frontend: {"event": "keystroke|flush|generation_token", ...}
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import asyncio
import json
import time
import logging

from .stream_controller import (
    StreamController,
    MockKVCacheManager,
    GenerationStatus,
)
from .events import EventType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for API Documentation
# =============================================================================

class WebSocketMessage(BaseModel):
    """Message sent from frontend to backend."""
    type: str  # "keystroke", "delete", "submit", "start_session"
    char: Optional[str] = None  # For keystroke events
    base_prompt: Optional[str] = None  # For start_session


class ServerStatus(BaseModel):
    """Server status response."""
    status: str
    active_sessions: int
    uptime_seconds: float


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="Keystroke Streaming API",
    description="WebSocket API for anticipatory prefill with keystroke streaming",
    version="1.0.0",
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Server state
server_start_time = time.time()
active_sessions: Dict[str, StreamController] = {}


# =============================================================================
# REST Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/status", response_model=ServerStatus)
async def get_status():
    """Get server status."""
    return ServerStatus(
        status="running",
        active_sessions=len(active_sessions),
        uptime_seconds=time.time() - server_start_time,
    )


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws/session")
async def websocket_session(websocket: WebSocket):
    """
    Main WebSocket endpoint for keystroke streaming.
    
    Message Protocol:
    
    Frontend -> Backend:
    - {"type": "start_session", "base_prompt": "..."}  # Start new session
    - {"type": "keystroke", "char": "a"}               # User typed a character
    - {"type": "delete"}                                # User pressed backspace
    - {"type": "submit"}                                # User pressed Enter
    
    Backend -> Frontend:
    - {"event": "session_started", "session_id": "..."}
    - {"event": "keystroke", "current_text": "...", ...}
    - {"event": "flush", "confirmed_text": "...", "cache_extended": true}
    - {"event": "delete", "current_text": "...", "rollback": false}
    - {"event": "submit_start", "final_text": "..."}
    - {"event": "generation_token", "token": "SELECT"}
    - {"event": "generation_complete", "generated_text": "..."}
    - {"event": "error", "message": "..."}
    """
    await websocket.accept()
    session_id = f"session_{int(time.time() * 1000)}"
    
    # Create controller with mock cache (Rémi will provide real implementation)
    cache_manager = MockKVCacheManager()
    controller = StreamController(cache_manager, debounce_ms=300.0)
    
    # Set up flush callback to send to frontend
    async def on_flush(result: dict):
        try:
            await websocket.send_json(result)
        except Exception as e:
            logger.error(f"Error sending flush: {e}")
    
    controller.set_flush_callback(on_flush)
    active_sessions[session_id] = controller
    
    logger.info(f"New WebSocket connection: {session_id}")
    
    try:
        # Send session started message
        await websocket.send_json({
            "event": "connected",
            "session_id": session_id,
            "message": "WebSocket connected. Send 'start_session' to begin.",
        })
        
        while True:
            # Receive message from frontend
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                msg_type = message.get("type", "")
                
                if msg_type == "start_session":
                    # Start new typing session
                    base_prompt = message.get("base_prompt", "")
                    await controller.start_session(base_prompt)
                    await websocket.send_json({
                        "event": "session_started",
                        "session_id": session_id,
                        "base_prompt_length": len(base_prompt),
                    })
                
                elif msg_type == "keystroke":
                    # Handle keystroke
                    char = message.get("char", "")
                    if char:
                        result = await controller.on_keystroke(char)
                        await websocket.send_json(result)
                
                elif msg_type == "delete":
                    # Handle backspace
                    result = await controller.on_delete()
                    await websocket.send_json(result)
                
                elif msg_type == "submit":
                    # Handle submit - stream generation results
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
        # Cleanup
        if session_id in active_sessions:
            del active_sessions[session_id]


# =============================================================================
# Server Entry Point
# =============================================================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the WebSocket server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
