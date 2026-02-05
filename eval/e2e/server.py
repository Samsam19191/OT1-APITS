"""
E2E Evaluation Server.

Supports client-driven evaluation where TTFT is measured in the browser.
Demonstrates:
1. Speed (TTFT)
2. Overhead (Resource Monitor)
3. Correctness (Execution against PG)
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from websockets.server import serve
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

from src.stream_controller import KVCacheManager, StreamController
from src.monitor import ResourceMonitor
from schema_loader import SchemaLoader
from eval import load_questions
from sql_executor import SQLExecutor, compare_results

import psutil
import gc


def cleanup_memory():
    """Clean up memory between evaluation modes."""
    gc.collect()
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except:
            pass


class E2EServer:
    """WebSocket server for true E2E evaluation."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        self.host = host
        self.port = port
        
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_name = None
        self._loaded = False
        
        # Per-connection state
        self.cache_manager = None
        self.controller = None
        self.questions = []
        
        # Tools
        self.monitor = ResourceMonitor(interval_ms=50) # 20Hz polling
        self.executor = None # SQLExecutor
        self.schema_loader = None
    
    def load_model(self, model_name: str):
        """Load LLM model."""
        if self._loaded and self.model_name == model_name:
            return
        
        print(f"Loading model: {model_name}...")
        
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device in ["mps", "cuda"] else torch.float32
        print(f"Using device: {self.device}, dtype: {dtype}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=True
        )
        
        self.model_name = model_name
        self._loaded = True
        print(f"✅ Model loaded on {self.device}")
        
        # Init Executor
        try:
             self.executor = SQLExecutor()
             test = self.executor.execute("SELECT 1")
             if not test.success:
                 print(f"⚠️  Executor DB Check Failed: {test.error}")
        except Exception as e:
             print(f"⚠️  Executor Init Failed: {e}")

    
    def _build_prompt_parts(self, question: str, schema: str):
        """Returns (system_part, user_part, full_prompt)"""
        system = f"""You are an expert SQL assistant. Generate a SQL query for the question.

### Schema:
{schema}

IMPORTANT: Output ONLY the SQL query in a code block.
"""
        user = f"""
### Question:
{question}

### Answer:
```sql
"""
        return system, user, system + user
    
    async def handle_connection(self, websocket):
        """Handle WebSocket connection."""
        print(f"[Server] New connection")
        
        await websocket.send(json.dumps({"event": "connected"}))
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(data, websocket)
        except Exception as e:
            print(f"[Server] Error: {e}")
            import traceback
            traceback.print_exc()
    
    async def broadcast_log(self, message: str):
        """Send a log message to all connected clients (or just the active one)."""
        # For simplicity, we just print to stdout, and the separate thread/connection handler 
        # needs to be aware of the WS. 
        # actually handle_connection holds the WS. 
        # We can't easily broadcast to *specific* WS from a global method without tracking them.
        # But we only have one user usually.
        print(message)
        # We can't rely on `self` having reference to the websocket if it's per-connection.
        # So we should pass WS to methods or stick to returning Logs.
        # Refactor: We will emit logs directly from within handle_message flow.
        pass

    async def _send_log(self, ws, msg: str):
        """Helper to send log event."""
        print(msg)
        try:
            await ws.send(json.dumps({
                "event": "server_log",
                "text": f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n"
            }))
        except:
            pass

    async def handle_message(self, data: dict, websocket):
        """Handle incoming message."""
        msg_type = data.get("type")
        
        if msg_type == "init":
            await self._send_log(websocket, "Received INIT command.")
            # Load model and questions
            model_name = data.get("model", "Qwen/Qwen2.5-Coder-3B-Instruct")
            db_filter = data.get("db", "car_1")
            limit = data.get("limit", 5)
            
            await self._send_log(websocket, f"Loading model: {model_name}...")
            self.load_model(model_name)
            await self._send_log(websocket, f"Model loaded on {self.device}.")

            # Initialize APITS Logic (once per connection)
            self.cache_manager = KVCacheManager(self.model, self.tokenizer, device=self.device, verbose=False)
            self.controller = StreamController(self.cache_manager, debounce_ms=30.0)
            
            # Load questions with prompts
            await self._send_log(websocket, f"Loading questions for DB: {db_filter}...")
            data_dir = PROJECT_ROOT / "eval" / "data"
            self.schema_loader = SchemaLoader(data_dir)
            
            raw_questions = load_questions(db_filter=db_filter)[:limit]
            
            # Build prompts
            self.questions = []
            for idx, q in enumerate(raw_questions):
                try:
                    db_id = q.get("db_id", "car_1")
                    schema = self.schema_loader.get_schema_prompt(db_id)
                    system_part, user_part, full_prompt = self._build_prompt_parts(q["question"], schema)
                    q_id = q.get("question_id") or f"{db_filter}_{idx+1}"
                    
                    self.questions.append({
                        "question_id": q_id,
                        "question": q["question"],
                        "gold_query": q["gold_query"],
                        "db_id": db_id,
                        "prompt": full_prompt,
                        "system_part": system_part,
                        "user_part": user_part,
                        "index": idx
                    })
                except Exception as e:
                    print(f"Skipping Q: {e}")
            
            await self._send_log(websocket, f"Ready with {len(self.questions)} questions.")
            await websocket.send(json.dumps({
                "event": "ready",
                "total": len(self.questions)
            }))
            
        elif msg_type == "prepare_query":
            # Prepare for a specific query index
            idx = data.get("index", 0)
            if idx < len(self.questions):
                q = self.questions[idx]
                
                # Pre-Initialize Monitor
                self.monitor.stop() # Ensure clean slate
                
                await self._send_log(websocket, f"Preparing Q{idx+1}: {q['question_id']}")

                # BACKGROUND PRE-COMPUTE: System Prompt (Schema pre-fill)
                if self.controller:
                    await self._send_log(websocket, f"[APITS] Pre-computing System Prompt in background...")
                    await self.controller.start_session(base_prompt=q['system_part'])
                    await self._send_log(websocket, f"[APITS] Background compute done.")

                await websocket.send(json.dumps({
                    "event": "question_ready",
                    "question": q
                }))
                
        elif msg_type == "run_baseline":
             idx = data.get("index", 0)
             q = self.questions[idx]
             
             await self._send_log(websocket, f"[Baseline] Starting generation for Q{idx+1}...")
             cleanup_memory()
             await asyncio.sleep(0.5)
             
             # Start Monitoring
             self.monitor.start()
             
             # Generation
             prompt = q["prompt"]
             inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
             streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
             
             gen_kwargs = dict(
                 inputs,
                 streamer=streamer,
                 max_new_tokens=256,
                 do_sample=False,
                 pad_token_id=self.tokenizer.eos_token_id,
             )
             
             try:
                 gen_kwargs["stop_strings"] = ["```"]
                 gen_kwargs["tokenizer"] = self.tokenizer
             except:
                 pass
             
             thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
             thread.start()
             
             accumulated_output = ""
             first_token_sent = False
             
             async for new_text in self._async_streamer(streamer):
                 accumulated_output += new_text
                 if not first_token_sent:
                     await self._send_log(websocket, "[Baseline] First token received!")
                     await websocket.send(json.dumps({"event": "baseline_first_token"}))
                     first_token_sent = True
                     
                 await websocket.send(json.dumps({
                     "event": "baseline_token",
                     "text": new_text
                 }))
                 
                 if "```" in accumulated_output:
                     break
             
             # Stop Monitor
             metrics = self.monitor.stop()
             await self._send_log(websocket, f"[Baseline] Generation done. Metrics collected.")
             
             # Process SQL & Verify
             sql = self._extract_sql(accumulated_output)
             await self._send_log(websocket, f"[Baseline] Extracted SQL: {sql[:50]}...")
             eval_result = self._evaluate_sql(sql, q["gold_query"], q["db_id"])
             await self._send_log(websocket, f"[Baseline] Evaluation: {eval_result['valid']} (Match: {eval_result['match']})")
             
             await websocket.send(json.dumps({
                 "event": "baseline_complete",
                 "metrics": metrics,
                 "eval": eval_result
             }))

        elif msg_type == "run_anticipatory":
             idx = data.get("index", 0)
             q = self.questions[idx]
             
             self.monitor.start()
             
             # Notify frontend to start typing IMMEDIATELY
             await self._send_log(websocket, f"[APITS] Ready for user input.")
             await websocket.send(json.dumps({
                 "event": "apits_ready",
                 "user_text": q['user_part'] # Tell frontend what to type
             }))
             

        elif msg_type == "keystroke":
             # Pass through to controller
             text = data.get("text", "")
             await self.controller.on_text_update(text)
             
        elif msg_type == "submit_apits":
            # Commit and Generate
            idx = data.get("index", 0)
            q = self.questions[idx]
            
            await self._send_log(websocket, f"[APITS] Submitting final input...")
            
            accumulated_output = ""
            first_token_sent = False
            
            async for event in self.controller.on_submit():
                evt = event.get("event")
                if evt == "generation_token":
                    token = event.get("token", "")
                    accumulated_output += token
                    
                    if not first_token_sent:
                        await self._send_log(websocket, f"[APITS] First token received!")
                        await websocket.send(json.dumps({"event": "apits_first_token"}))
                        first_token_sent = True
                    
                    await websocket.send(json.dumps({
                        "event": "apits_token",
                        "text": token
                    }))
                    
                    if "```" in accumulated_output:
                        break
            
            # Stop Monitor
            metrics = self.monitor.stop()
            await self._send_log(websocket, f"[APITS] Generation complete.")
            
            # Eval
            sql = self._extract_sql(accumulated_output)
            await self._send_log(websocket, f"[APITS] Extracted SQL: {sql[:50]}...")
            eval_result = self._evaluate_sql(sql, q["gold_query"], q["db_id"])
            await self._send_log(websocket, f"[APITS] Evaluation: {eval_result['valid']} (Match: {eval_result['match']})")
             
            await websocket.send(json.dumps({
                 "event": "apits_complete",
                 "metrics": metrics,
                 "eval": eval_result
            }))
            
    def _extract_sql(self, text: str) -> str:
        """Extract SQL from model output. 
           Reliably handles cases where output is just the code, or wrapped in backticks (start/middle/end).
        """
        text = text.strip()
        
        # 1. If we have the closing block ```, cut everything after it.
        if "```" in text:
            # Check if it STARTS with ```sql or similar
            if text.startswith("```"):
                parts = text.split("```")
                if len(parts) >= 3:
                     # ```sql ... ``` -> parts[0]='', parts[1]='sql ...', parts[2]=''
                     candidate = parts[1]
                     if candidate.startswith("sql"): 
                         candidate = candidate[3:]
                     return candidate.strip()
                elif len(parts) == 2:
                     # ```sql ... (incomplete?)
                     candidate = parts[1]
                     if candidate.startswith("sql"): 
                         candidate = candidate[3:]
                     return candidate.strip()
            
            # It might just End with ``` (because we prompt with ```sql opened)
            else:
                 # Take everything BEFORE the first ```
                 return text.split("```")[0].strip()
        
        # 2. No backticks found? Just return text (maybe cleaned up)
        if ";" in text:
            text = text.split(";")[0].strip() + ";"
        return text

    def _evaluate_sql(self, pred_sql: str, gold_sql: str, db_id: str) -> dict:
        """Run Correctness Check."""
        if not self.executor:
            return {"valid": False, "match": False, "error": "Executor not connected"}
            
        gold_sql = gold_sql.replace('"', "'")
        
        res_gold = self.executor.execute(gold_sql, schema=db_id)
        res_pred = self.executor.execute(pred_sql, schema=db_id)
        
        match = False
        if res_pred.success and res_gold.success:
            match = compare_results(res_pred, res_gold)
            
        return {
            "valid": res_pred.success,
            "match": match,
            "error": res_pred.error,
            "gold_rows": res_gold.row_count if res_gold.success else 0,
            "pred_rows": res_pred.row_count
        }

    async def _async_streamer(self, streamer):
        """Yield tokens from sync streamer asynchronously."""
        import queue
        while True:
            try:
                # Use to_thread to wait for the queue without blocking the event loop
                # TextIteratorStreamer uses an internal queue
                token = await asyncio.to_thread(lambda: next(streamer))
                yield token
            except StopIteration:
                break

    async def run(self):
        """Start the server."""
        print(f"\nE2E Evaluation Server on ws://{self.host}:{self.port}")
        print(f"   Open demo/index.html in browser\n")
        
        # Disable ping_interval to prevent timeouts during heavy blocking inference
        async with serve(self.handle_connection, self.host, self.port, ping_interval=None):
            await asyncio.Future()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    
    server = E2EServer(host=args.host, port=args.port)
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
