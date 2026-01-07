# Anticipatory Prefill with Keystroke Streaming

**Goal**: Reduce Time-To-First-Token (TTFT) in interactive Text-to-SQL by overlapping prefill compute with user typing time, maintaining a single evolving KV-cache.

## Quickstart

### Option A: VS Code + Google Colab (Recommended)

You can code in VS Code but run on Colab's GPU.

1. Install [Google Colab Extension](https://marketplace.visualstudio.com/items?itemName=google.colab).
2. Open `notebooks/00_onboarding.ipynb`.
3. Connect Kernel to **Google Colab**.
4. Run all cells. (The notebook will auto-configure the environment).
   - [Detailed Instructions](docs/vscode_colab.md)

### Option B: Browser

1. Open `notebooks/00_onboarding.ipynb` in [Google Colab](https://colab.research.google.com/).
   - [Click here for instructions](docs/colab.md)

## Database Setup

The project uses PostgreSQL for data storage. A Docker Compose configuration is provided for easy setup.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop) installed and running

### Quick Start

```bash
# Start the database
docker-compose up -d

# Verify it's running
docker-compose ps

# Stop the database
docker-compose down

# Stop and remove all data
docker-compose down -v
```

### Connection Details

- **Host**: `localhost`
- **Port**: `5432`
- **Database**: `ot1_apits`
- **Username**: `postgres`
- **Password**: `password`

Data is persisted in a Docker volume and will survive container restarts.

## Repo Layout

```
repo/
├── notebooks/
│   └── 00_onboarding.ipynb  # ENTRYPOINT: Run this first!
├── src/
│   ├── config.py            # Global constants (Model name, quantization, etc.)
│   ├── stream_controller.py # KV-cache management and event handling
│   ├── websocket_server.py  # WebSocket API server
│   └── __init__.py
├── ui/                       # React frontend (keystroke streaming UI)
├── docs/
│   ├── colab.md             # Browser instructions
│   └── vscode_colab.md      # VS Code instructions
├── docker-compose.yml        # PostgreSQL database setup
├── requirements.txt          # Pinned dependencies
├── .gitignore
└── README.md
```

## Team Workflow

- **Do not edit** `00_onboarding.ipynb` unless fixing a setup bug.
- Create your own directory for feature work.
