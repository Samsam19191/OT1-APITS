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

## Repo Layout

```
repo/
├── notebooks/
│   └── 00_onboarding.ipynb  # ENTRYPOINT: Run this first!
├── src/
│   ├── config.py            # Global constants (Model name, quantization, etc.)
│   └── __init__.py
├── docs/
│   ├── colab.md             # Browser instructions
│   └── vscode_colab.md      # VS Code instructions
├── requirements.txt         # Pinned dependencies
├── .gitignore
└── README.md
```

## Team Workflow
- **Do not edit** `00_onboarding.ipynb` unless fixing a setup bug.
- Create your own directory for feature work.
