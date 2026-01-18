"""
SQL Correctness Evaluation

Uses production code paths to evaluate Text-to-SQL performance.

Usage:
    python eval/eval.py [--limit N]
"""

import argparse
import asyncio
import json
import sys
import os
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# prod imports
from src.utils import load_model_and_tokenizer
from src.stream_controller import KVCacheManager

# eval imports
from sql_executor import SQLExecutor, compare_results
from metrics import MetricsCollector, EvalResult
from report_generator import generate_report


from schema_loader import SchemaLoader


# =============================================================================
# Generator (Production Code)
# =============================================================================

class SQLGenerator:
    """SQL generator using production KVCacheManager."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct", schema: str = "", verbose: bool = False):
        self.model_name = model_name
        self.schema = schema
        self.verbose = verbose
        self.cache_manager = None
        self._loaded = False
    
    def load(self):
        if self._loaded:
            return
        
        print(f"Loading model: {self.model_name}...")
        model, tokenizer, device = load_model_and_tokenizer(self.model_name)
        
        self.cache_manager = KVCacheManager(
            model=model,
            tokenizer=tokenizer,
            device=device,
            verbose=self.verbose
        )
        
        self._loaded = True
        print(f"Model loaded on {device}")
    
    async def generate(self, question: str, max_tokens: int = 256, schema: str = None) -> str:
        if not self._loaded:
            self.load()
        
        # Use provided schema or fallback to self.schema
        current_schema = schema if schema else self.schema
        prompt = self._build_prompt(question, current_schema)
        
        if self.verbose:
            print("Prompt:", prompt[:100] + "...") # Log briefly
        
        # Use production cache manager
        await self.cache_manager.initialize(prompt)
        
        output = ""
        async for token in self.cache_manager.generate(max_new_tokens=max_tokens):
            output += token
            # STOP Condition: If we see the closing code block, stop.
            if "```" in output:
                break
        
        # Reset for next query
        self.cache_manager.is_first_session = True
        
        return self._extract_sql(output)
    
    def _build_prompt(self, question: str, schema: str) -> str:
        return f"""You are an expert SQL assistant. Generate a SQL query for the question.

### Schema:
{schema}

### Example:
```sql
SELECT c.Name, COUNT(ci.ID) as city_count
FROM country c
JOIN city ci ON c.Code = ci.CountryCode
GROUP BY c.Name
ORDER BY city_count DESC
LIMIT 5;
```

IMPORTANT: Output ONLY the SQL query, nothing else.

### Question:
{question}

### SQL:
```sql
"""
    
    def _extract_sql(self, text: str) -> str:
        text = text.strip()
        if "```" in text:
            text = text.split("```")[0].strip()
        if ";" in text:
            text = text.split(";")[0].strip() + ";"
        return text


# =============================================================================
# Evaluation
# =============================================================================

def load_questions(limit: int = None, db_filter: str = None):
    """Load questions from ALL db directories, or a specific one if filter is set."""
    data_dir = Path(__file__).parent / "data"
    all_questions = []

    # Determine which directories to check
    if db_filter:
        # Direct lookup if filter is provided
        target_dir = data_dir / db_filter
        if not target_dir.exists() or not target_dir.is_dir():
            print(f"⚠️ Warning: Database directory '{db_filter}' not found at {target_dir}")
            return []
        dirs_to_scan = [target_dir]
    else:
        # Scan all subdirectories
        dirs_to_scan = [d for d in data_dir.iterdir() if d.is_dir() and d.name != "reports"]
    
    for db_dir in dirs_to_scan:
        db_id = db_dir.name
        q_file = db_dir / f"{db_id}_questions.json"
        
        if q_file.exists():
            with open(q_file) as f:
                qs = json.load(f)
                # Inject db_id if missing
                for q in qs:
                    q['db_id'] = db_id
                all_questions.extend(qs)
    
    if limit:
        all_questions = all_questions[:limit]
    
    return all_questions


async def run_eval(limit: int = None, model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct", db: str = None):
    """Run evaluation."""
    print("=" * 60)
    print("SQL Correctness Evaluation")
    print("=" * 60)
    
    # Setup
    print("\n📦 Connecting to PostgreSQL...")
    executor = SQLExecutor()
    test = executor.execute("SELECT 1")
    if not test.success:
        print(f"❌ PostgreSQL failed: {test.error}")
        print("Run: docker-compose -f docker-compose.eval.yml up -d")
        return
    print("✅ Connected")
    
    # Load schema loader
    schema_loader = SchemaLoader(Path(__file__).parent / "data")

    # Load model
    generator = SQLGenerator(model_name, verbose=False)
    generator.load()
    print("✅ Model loaded successfully.")
    
    # Load questions
    print(f"\n📋 Loading questions from all databases...")
    questions = load_questions(limit, db)
    print(f"Loaded {len(questions)} cases.")
    
    if not questions:
        print("No questions found!")
        return

    # Results tracking
    metrics_collector = MetricsCollector()
    
    print(f"\n🚀 Starting evaluation of {len(questions)} questions...")
    print("-" * 60)
    
    start_time = time.time()
    
    # Use tqdm for progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(questions), total=len(questions), desc="Evaluating", file=sys.stdout)
    except ImportError:
        iterator = enumerate(questions)
        print("Note: tqdm not found, using simple print.")

    for i, case in iterator:
        q_id = case.get('id', i)
        question = case['question']
        gold_sql = case['gold_query']
        db_id = case.get('db_id', 'world_1') # Fallback if missing
        
        # Get schema prompt for this specific DB
        try:
            prompt_schema = schema_loader.get_schema_prompt(db_id)
        except Exception as e:
            if hasattr(iterator, "write"):
                iterator.write(f"⚠️ Schema load failed for {db_id}: {e}")
            else:
                print(f"⚠️ Schema load failed for {db_id}: {e}")
            metrics_collector.add(EvalResult(
                question_id=q_id, question=question, gold_query=gold_sql, predicted_query="",
                syntax_valid=False, syntax_error=f"Schema Load Error: {e}",
                result_match=False, gold_row_count=0, predicted_row_count=0
            ))
            continue

        # Generate SQL
        try:
            pred_sql = await generator.generate(question, schema=prompt_schema)
        except Exception as e:
            msg = f"⚠️ Generation failed: {e}"
            if hasattr(iterator, "write"):
                iterator.write(msg)
            else:
                print(msg)
            pred_sql = f"-- ERROR: {e}"

        # Execute & Compare
        gold_sql_pg = gold_sql.replace('"', "'")
        
        # Execute against Specific Schema
        res_gold = executor.execute(gold_sql_pg, schema=db_id)
        res_pred = executor.execute(pred_sql, schema=db_id)
        
        # Compare
        match = compare_results(res_pred, res_gold) if res_pred.success else False
        
        # Record
        result = EvalResult(
            question_id=q_id,
            question=question,
            gold_query=gold_sql,
            predicted_query=pred_sql,
            syntax_valid=res_pred.success,
            syntax_error=res_pred.error,
            result_match=match,
            gold_row_count=res_gold.row_count if res_gold.success else 0,
            predicted_row_count=res_pred.row_count
        )
        metrics_collector.add(result)
        
        # Progress
        status = "✅" if result.is_correct() else ("⚠️" if res_pred.success else "❌")
        
        # Update logs
        log_msg = f"[{i+1:3d}/{len(questions)}] {status} {q_id}"
        if hasattr(iterator, "write"):
             iterator.write(log_msg)
             # Optionally update description?
             # iterator.set_description(f"Eval {q_id}: {status}")
        else:
             print(log_msg)
    
    elapsed = time.time() - start_time
    executor.close()
    
    # Results
    metrics = metrics_collector.compute()
    
    print("-" * 60)
    print(f"\n📊 Results:")
    print(f"   Execution Accuracy: {metrics.execution_accuracy:.1f}%")
    print(f"   Exact Match:        {metrics.exact_match_accuracy:.1f}%")
    print(f"   Time: {elapsed:.1f}s ({elapsed/len(questions):.2f}s per query)")
    
    # Report
    report_path = generate_report(metrics, metrics_collector.results)
    print(f"\n📄 Report: {report_path}")
    
    return report_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit number of questions (default: all)")
    parser.add_argument("--model", "-m", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--db", type=str, help="Filter by database ID (e.g. car_1)")
    args = parser.parse_args()
    
    asyncio.run(run_eval(limit=args.limit, model_name=args.model, db=args.db))

if __name__ == "__main__":
    main()