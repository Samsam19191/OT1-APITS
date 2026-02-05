# OT1-APITS: Anticipatory Prefill with Keystroke Streaming

> **Project Goal**: Dramatically reduce **Time-To-First-Token (TTFT)** in interactive Text-to-SQL tasks.
> This is achieved by overlapping the prefill compute phase with user typing time, maintaining a single evolving KV-cache for high-performance inference.

---

## 🚀 Overview

OT1-APITS is a high-performance research prototype designed to optimize LLM responsiveness in streaming scenarios. By pre-calculating the KV-cache as the user types, we try to eliminate the traditional "wait-then-generate" bottleneck.

### Key Features

- **Keystroke Streaming**: Real-time synchronization between typing and server-side prefill.
- **Anticipatory Prefill**: Leverages the "idle" time during user input to prepare the model state.
- **Optimized KV-Cache**: Efficient management of model memory to ensure instant generation upon submission.
- **Dual Flow Architecture**: Dedicated environments for both real-time interaction and automated evaluation.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

- **Python**: 3.10+
- **Node.js**: 20+ (for the UI)
- **Docker & Docker Compose**: For database orchestration

---

## 🏗️ Quick Start

The project uses a `Makefile` to simplify orchestration across the backend, database, and frontend.

### 1. Launch the Application

To set up the environment, seed the database, and start both the server and UI:

```bash
make run-app
```

**Access points:**

- **Frontend**: [http://localhost:5173](http://localhost:5173)
- **WebSocket API**: `ws://localhost:8000/ws/session`
- **API Health**: [http://localhost:8000/health](http://localhost:8000/health)

### 2. Run Evaluations

To execute the performance measurement pipeline:

```bash
make eval
```

*Reports will be generated in `eval/data/reports/`.*

---

## 📂 Project Structure

```bash
.
├── src/                  # Core Engine
│   ├── stream_controller.py  # KV-cache management & logic
│   ├── websocket_server.py   # FastAPI WebSocket implementation
│   └── utils.py              # Model loading & helper functions
├── ui/                   # React/Vite Frontend
├── eval/                 # Evaluation Suite
│   ├── eval.py               # Main evaluation script
│   └── seed_eval_db.py       # Evaluation-specific data seeding
├── scripts/              # Utility scripts (seeding, etc.)
├── Makefile              # Central orchestration
└── docker-compose.*.yml  # Database services configuration
```

---

## ⚙️ Advanced Commands

| Command | Action |
| :--- | :--- |
| `make help` | Show all available commands |
| `make rebuild` | Wipe environment and perform a fresh build |
| `make clean` | Stop docker containers and remove virtual env |
| `make setup-db-app` | Reset and re-seed the application database |

---

## 🧪 Architecture Notes

- **Model**: Powered by `Qwen/Qwen2.5-Coder-1.5B-Instruct`.
- **Backend**: FastAPI with Uvicorn, utilizing asynchronous WebSocket handling.
- **Frontend**: React (Vite) with real-time keystroke tracking and performance metrics display.
- **Database**: PostgreSQL (Docker-isolated) for both app and evaluation contexts.

---

## 👨‍💻 Workflow Guidelines

- **Environment**: Always use the virtual environment created in `.venv`.
- **Database**: Ensure Docker is running before executing any `make` targets.
- **Modifications**: If you update the model configuration, check `src/websocket_server.py` and `src/config.py`.

---
*Created by the OT1-APITS Research Team.*
