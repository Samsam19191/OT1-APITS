# OT1-APITS Evaluation Framework

This directory contains the framework for evaluating Text-to-SQL models against the [Spider](https://yale-lily.github.io/spider) dataset, adapted for **PostgreSQL**.

## Quick Start

### 1. Seed the Database
Before running evaluations, you must populate the local PostgreSQL database with the test datasets (`car_1`, `world_1`, etc.).

```bash
# This script does 3 things:
# 1. Creates/Wipes PostgreSQL tables.
# 2. Migrates data from original SQLite files.
# 3. Extracts Foreign Keys and enforces Lowercase naming in JSON metadata.
python eval/seed_eval_db.py --force
```

### 2. Run Evaluation
Run the evaluation script on a specific dataset.

```bash
# Run on car_1 dataset, limit to first 100 questions
python eval/eval.py --db car_1 --limit 100 --model Qwen/Qwen2.5-Coder-3B-Instruct
```

**Arguments:**
- `--db`: Database name (e.g., `car_1`, `world_1`, `dog_kennels`).
- `--limit`: Number of questions to evaluate (default: all).
- `--model`: Hugging Face model ID (default: `Qwen/Qwen2.5-Coder-3B-Instruct`).
- `--verbose`: Enable detailed KV cache logging (default: False).

## 📂 Structure

- **`eval.py`**: Main entry point. Loads model, generates SQL, executes on Postgres, and computes accuracy.
- **`seed_eval_db.py`**: ETL script. 
    - **Crucial Feature**: It normalizes all table/column names to **lowercase** to ensure PostgreSQL compatibility and extracts **Foreign Keys** from SQLite to enrich the model's prompt.
- **`schema_loader.py`**: Generates the Prompt Context. It reads `*.json` metadata to build the SQL schema description, including Foreign Key constraints.
- **`sql_executor.py`**: Handles PostgreSQL queries and result normalization.
- **`data/`**: Contains dataset folders (`car_1/`, etc.) with their JSON metadata and original SQLite files.
- **`data/reports/`**: Generated Markdown reports of evaluation runs.

### Metrics
- **Execution Accuracy**: Compares the *result set* of the generated SQL vs. Gold SQL.
- **Exact Match**: Compares the *string* of the generated SQL vs. Gold SQL (less reliable due to equivalent styling).