"""
Test client for the WebSocket server.

Usage:
    python test_websocket_client.py [--url URL]
    
This simulates the frontend sending keystroke events.
"""

import asyncio
import json
import sys


async def test_websocket():
    """Test the WebSocket server with simulated keystrokes."""
    try:
        import websockets
    except ImportError:
        print("Installing websockets...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets"])
        import websockets
    
    url = "ws://localhost:8000/ws/session"
    print(f"Connecting to {url}...")
    
    async with websockets.connect(url) as ws:
        # Receive connection message
        response = await ws.recv()
        print(f"← {response}")
        
        # Start session
        print("\n→ Starting session...")
        await ws.send(json.dumps({
            "type": "start_session",
        }))
        response = await ws.recv()
        print(f"← {response}")
        
        # Simulate typing "Show patients"
        test_query = "Show users"
        print(f"\n→ Simulating typing: '{test_query}'")
        
        current_text = ""
        for char in test_query:
            current_text += char
            await ws.send(json.dumps({
                "type": "text_update",
                "full_text": current_text
            }))
            response = await ws.recv()
            data = json.loads(response)
            print(f"  ← text_update: '{data.get('current_text', '')}'")
            
            # Check for flush events (sent async by server)
            try:
                # Non-blocking check for additional messages
                flush = await asyncio.wait_for(ws.recv(), timeout=0.1)
                flush_data = json.loads(flush)
                if flush_data.get("event") == "flush":
                    print(f"  ← FLUSH: confirmed='{flush_data.get('confirmed_text')}'")
            except asyncio.TimeoutError:
                pass
            
            await asyncio.sleep(0.05)  # Small delay between keystrokes
        
        # Wait for final flush
        print("\n→ Waiting for debounce flush...")
        await asyncio.sleep(0.4)
        try:
            flush = await asyncio.wait_for(ws.recv(), timeout=0.5)
            print(f"← {flush}")
        except asyncio.TimeoutError:
            pass
        
        # Submit
        print("\n→ Submitting...")
        await ws.send(json.dumps({"type": "submit"}))
        
        # Receive generation results
        while True:
            try:
                # 1. Wait for the NEXT single message
                response = await asyncio.wait_for(ws.recv(), timeout=10000.0)
                
                # 2. Process it
                data = json.loads(response)
                print(f"← {data}")
                
                # 3. Check for exit condition
                if data.get("event") == "generation_complete":
                    break
                    
            except asyncio.TimeoutError:
                print("← (Timeout: Model took too long to generate next token)")
                break
            except Exception as e:
                print(f"← Error: {e}")
                break
        
        print("\n✓ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_websocket())
