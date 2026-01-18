# Evaluation Dashboard

Web-based evaluation dashboard for comparing Anticipatory Prefill vs Cold Start performance.

## Quick Start

1. **Start the WebSocket server:**
   ```bash
   .venv/bin/python eval/e2e/server.py
   ```

2. **Open the dashboard:**
   Open `eval/e2e/index.html` in your browser (double-click or drag to browser).

3. **Configure and run:**
   - Select model, database, and number of questions
   - Set typing speed (chars/sec)
   - Click "Launch Evaluation"

4. **View results:**
   - See TTFT comparison metrics
   - Per-question speedup table
   - Report auto-saved to `eval/data/reports/`