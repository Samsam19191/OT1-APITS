"""
Run the WebSocket server for keystroke streaming.

Usage:
    python run_server.py [--host HOST] [--port PORT]
    
Example:
    python run_server.py --port 8000
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Keystroke Streaming WebSocket Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║        Keystroke Streaming WebSocket Server                  ║
╠══════════════════════════════════════════════════════════════╣
║  WebSocket: ws://{args.host}:{args.port}/ws/session                    ║
║  Health:    http://{args.host}:{args.port}/health                      ║
║  Docs:      http://{args.host}:{args.port}/docs                        ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    import uvicorn
    uvicorn.run(
        "src.websocket_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
