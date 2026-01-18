"""
E2E Evaluation Server.

Supports client-driven evaluation where TTFT is measured in the browser.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "eval"))

from websockets.server import serve
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

from src.stream_controller import KVCacheManager, StreamController
from schema_loader import SchemaLoader
from eval import load_questions


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
    
    def _build_prompt(self, question: str, schema: str) -> str:
        return f"""You are an expert SQL assistant. Generate a SQL query for the question.

### Schema:
{schema}

IMPORTANT: Output ONLY the SQL query in a code block.

### Question:
{question}

### Answer:
```sql
"""
    
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
    
    async def handle_message(self, data: dict, websocket):
        """Handle incoming message."""
        msg_type = data.get("type")
        
        if msg_type == "load_questions":
            # Load model and questions
            model_name = data.get("model", "Qwen/Qwen2.5-Coder-3B-Instruct")
            db_filter = data.get("db", "car_1")
            limit = data.get("limit", 5)
            
            self.load_model(model_name)
            
            # Load questions with prompts
            data_dir = PROJECT_ROOT / "eval" / "data"
            schema_loader = SchemaLoader(data_dir)
            raw_questions = load_questions(db_filter=db_filter)[:limit]
            
            # Build prompts
            self.questions = []
            for idx, q in enumerate(raw_questions):
                try:
                    schema = schema_loader.get_schema_prompt(q.get("db_id", "car_1"))
                    prompt = self._build_prompt(q["question"], schema)
                    # Use question_id if present, otherwise create one
                    q_id = q.get("question_id") or f"{db_filter}_{idx+1}"
                    self.questions.append({
                        "question_id": q_id,
                        "question": q["question"],
                        "prompt": prompt
                    })
                except:
                    pass
            
            # Initialize cache manager for anticipatory mode
            self.cache_manager = KVCacheManager(self.model, self.tokenizer, device=self.device, verbose=False)
            self.controller = StreamController(self.cache_manager, debounce_ms=30.0)
            
            await websocket.send(json.dumps({
                "event": "questions_loaded",
                "questions": self.questions
            }))
        
        elif msg_type == "keystroke":
            # Client is typing - update KV cache (anticipatory mode)
            text = data.get("text", "")
            question_idx = data.get("question_idx", 0)
            
            # Reset if new question (check if we changed question index)
            if not hasattr(self, '_last_question_idx') or self._last_question_idx != question_idx:
                self.cache_manager.is_first_session = True
                await self.controller.start_session(base_prompt="")
                self._last_question_idx = question_idx
            
            await self.controller.on_text_update(text)
            
            await websocket.send(json.dumps({"event": "keystroke_ack"}))
        
        elif msg_type == "submit":
            # Client pressed enter - generate
            mode = data.get("mode", "anticipatory")
            question_idx = data.get("question_idx", 0)
            
            if mode == "anticipatory":
                # Use the pre-built KV cache
                first_token_sent = False
                accumulated_output = ""
                async for event in self.controller.on_submit():
                    if event.get("event") == "generation_token":
                        token = event.get("token", "")
                        accumulated_output += token
                        
                        if not first_token_sent:
                            await websocket.send(json.dumps({"event": "first_token"}))
                            first_token_sent = True
                        await websocket.send(json.dumps({
                            "event": "token",
                            "text": token
                        }))
                        
                        # Stop on closing code block
                        if "```" in accumulated_output:
                            break
                    elif event.get("event") == "generation_complete":
                        break
                
                await websocket.send(json.dumps({"event": "generation_complete"}))
                
            else:
                # Baseline - cold start generation
                prompt = self.questions[question_idx]["prompt"]
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
                
                first_token_sent = False
                accumulated_output = ""
                for new_text in streamer:
                    accumulated_output += new_text
                    
                    if not first_token_sent:
                        await websocket.send(json.dumps({"event": "first_token"}))
                        first_token_sent = True
                    await websocket.send(json.dumps({
                        "event": "token",
                        "text": new_text
                    }))
                    
                    # Fallback stop check
                    if "```" in accumulated_output:
                        break
                
                await websocket.send(json.dumps({"event": "generation_complete"}))
        
        elif msg_type == "generate_report":
            # Generate markdown report
            results = data.get("results", {})
            report_path = self._generate_report(results)
            await websocket.send(json.dumps({
                "event": "report_saved",
                "path": report_path
            }))
    
    def _generate_report(self, results: dict) -> str:
        """Generate markdown report from client-side results."""
        reports_dir = PROJECT_ROOT / "eval" / "data" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"e2e_eval_{timestamp}.md"
        
        ant = results.get("anticipatory", [])
        base = results.get("baseline", [])
        
        ant_avg = sum(r["ttft"] for r in ant) / len(ant) if ant else 0
        base_avg = sum(r["ttft"] for r in base) / len(base) if base else 0
        improvement = ((base_avg - ant_avg) / base_avg * 100) if base_avg > 0 else 0
        speedup = base_avg / ant_avg if ant_avg > 0 else 0
        
        content = f"""# E2E Evaluation Report (Client-Side Metrics)

**Model**: {self.model_name}  
**Date**: {datetime.now().isoformat()}  
**Questions**: {len(ant)}

## Summary (True E2E Latency)

| Metric | Anticipatory | Baseline | Speedup |
|--------|--------------|----------|---------|
| **Avg TTFT** | {ant_avg:.0f} ms | {base_avg:.0f} ms | **{speedup:.1f}x** |
| **Improvement** | - | - | **{improvement:.1f}%** |

> Note: These metrics are measured client-side (browser) and include network latency.

## Per-Question Results

| Question | Anticipatory TTFT | Baseline TTFT | Speedup |
|----------|-------------------|---------------|---------|
"""
        for i, a in enumerate(ant):
            b = base[i] if i < len(base) else {"ttft": 0}
            spd = b["ttft"] / a["ttft"] if a["ttft"] > 0 else 0
            content += f"| {a['question_id']} | {a['ttft']} ms | {b['ttft']} ms | {spd:.1f}x |\n"
        
        content += "\n*Generated by OT1-APITS E2E Evaluation Dashboard*\n"
        
        with open(report_path, "w") as f:
            f.write(content)
        
        return str(report_path)
    
    async def run(self):
        """Start the server."""
        print(f"\nE2E Evaluation Server on ws://{self.host}:{self.port}")
        print(f"   Open demo/index.html in browser\n")
        
        async with serve(self.handle_connection, self.host, self.port):
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
