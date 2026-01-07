# Anticipatory Prefill with Keystroke Streaming

**Goal**: Reduce Time-To-First-Token (TTFT) in interactive Text-to-SQL by overlapping prefill compute with user typing time, maintaining a single evolving KV-cache.

## Quickstart

### Prerequisites

- **Python 3.10+**
- **Docker Desktop** installed and running
- **Node.js 20+** (for frontend)

### Running the Application

```bash
# Setup environment and run the full application
make run-app
```

This will:

1. Create a Python virtual environment (`.venv`)
2. Install backend dependencies
3. Start PostgreSQL database (Docker)
4. Seed the database with sample data
5. Start the backend WebSocket server
6. Start the frontend development server

The application will be available at:

- **Frontend**: <http://localhost:5173>
- **Backend WebSocket**: ws://localhost:8000/ws/session

Press `Ctrl+C` to stop all processes.

### Running Evaluations

```bash
# Run the evaluation pipeline
make eval
```

This will:

1. Setup the evaluation environment
2. Start the evaluation PostgreSQL database
3. Seed it with test data
4. Run evaluation scripts
5. Generate reports in `eval/reports/`

## Database Setup

## Available Commands

```bash
make help          # Show all available commands
make run-app       # Run the full application (backend + frontend + DB)
make eval          # Run evaluation pipeline
make rebuild       # Clean and rebuild everything
make clean         # Stop all containers and remove virtual environment
```

## Architecture

### Application Environment

- **Database**: `ot1_apits` (PostgreSQL on port 5432)
- **Backend**: WebSocket server (`run_server.py` on port 8000)
- **Frontend**: React UI (Vite dev server on port 5173)

### Evaluation Environment

- **Database**: `ot1_apits_eval` (PostgreSQL on port 5433)
- **Scripts**: CLI evaluation tools in `eval/`
- **Reports**: Generated in `eval/data/reports/`

## Repo Layout

```
repo/
├── src/
│   ├── config.py            # Global constants (Model name, quantization, etc.)
│   ├── stream_controller.py # KV-cache management and event handling
│   ├── websocket_server.py  # WebSocket API server
│   └── __init__.py
├── ui/                       # React frontend (keystroke streaming UI)
├── docs/
│   ├── colab.md             # Browser instructions
│   └── vscode_colab.md      # VS Code instructions
├── requirements.txt          # Pinned dependencies
├── .gitignore
├── Makefile
├── eval/
│   ├── eval.py              # Evaluation script
│   ├── seed_eval_db.py      # Evaluation database seeder
│   ├── README.md            # Evaluation README
│   └── data/
│       ├── *.json           # Database and evaluation data
│       └── reports/         # Evaluation reports
├── docker-compose.app.yml    # Application database setup
├── docker-compose.eval.yml   # Evaluation database setup
└── README.md
```

## Team Workflow

- **Do not edit** `00_onboarding.ipynb` unless fixing a setup bug.
- Create your own directory for feature work.
