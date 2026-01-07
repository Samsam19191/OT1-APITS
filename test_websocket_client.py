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
            "base_prompt": """
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
                "type": "keystroke",
                "text": current_text
            }))
            response = await ws.recv()
            data = json.loads(response)
            print(f"  ← keystroke: '{data.get('current_text', '')}'")
            
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
                response = await asyncio.wait_for(ws.recv(), timeout=2000.0)
                data = json.loads(response)
                print(f"← {data}")
                if data.get("event") == "generation_complete":
                    break
            except asyncio.TimeoutError:
                print("← (timeout waiting for generation)")
                break
        
        print("\n✓ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_websocket())
