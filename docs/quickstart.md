# Quick Start Guide

This tutorial shows how to install the prerequisites, run the Colab notebook and launch the demo either offline or with API credentials.

## Installing prerequisites

1. Install **Python 3.11 or 3.12** and create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip pre-commit
   ```
2. Install **Docker** and **Docker Compose** (Compose ≥2.5).
3. Install **Node.js 20** for the web client. Run `nvm use` to activate the version from `.nvmrc`.
4. Ensure `git` is available. Verify the tools:
   ```bash
   python --version
   docker --version
   docker compose version
   git --version
   ```

## Running the Colab notebook

Open [`colab_alpha_agi_insight_demo.ipynb`](../alpha_factory_v1/demos/alpha_agi_insight_v0/colab_alpha_agi_insight_demo.ipynb) in Google Colab and execute the following steps:

1. Run the first cell to clone the repository and install dependencies.
2. Optionally set `OPENAI_API_KEY` in the second cell.
3. Execute the demo cell to launch the Insight search loop.

The notebook works entirely offline when no API key is provided.

## Launching the demo

From the repository root, verify the environment and start the demo:

```bash
python check_env.py --auto-install  # may take several minutes
# Press Ctrl+C to abort and rerun with '--timeout 300' to limit waiting
./quickstart.sh
```

Copy `.env.sample` to `.env` and add your API keys to enable cloud features. Without keys, the program falls back to the local Meta‑Agentic Tree Search:

```bash
alpha-agi-insight-v1 --episodes 5  # with or without OPENAI_API_KEY
```

Run `pre-commit run --all-files` once the setup completes to ensure formatting.

## OpenAI Agents SDK and Google ADK integration

Expose the search loop via the **OpenAI Agents SDK**:

```bash
python alpha_factory_v1/demos/meta_agentic_agi/openai_agents_bridge.py
# → http://localhost:5001/v1/agents
```

Enable the optional **Google ADK** gateway for federation:

```bash
pip install google-adk
ALPHA_FACTORY_ENABLE_ADK=true \
  python alpha_factory_v1/demos/meta_agentic_agi/openai_agents_bridge.py
```

The bridge automatically falls back to local execution when the packages or API keys are missing.
