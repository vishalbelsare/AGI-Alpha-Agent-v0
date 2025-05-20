# Quick Start – Alpha‑AGI Business v1 Demo

This short guide summarises how to launch the business demo either locally or in Google Colab.

## Local Launch
1. **Clone the repository**
   ```bash
   git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
   cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/alpha_agi_business_v1
   ```
2. **Check dependencies**
   ```bash
   python ../../check_env.py --auto-install
   ```
3. **Run the demo**
   ```bash
   python run_business_v1_local.py --bridge
   # optional: choose a different port
   python run_business_v1_local.py --bridge --port 9000
   ```
   The dashboard is available at [http://localhost:<port>/docs](http://localhost:<port>/docs), where `<port>` is the port number used (default is `8000`).

Set `OPENAI_API_KEY` to enable cloud models. Offline mode works automatically when the key is absent.
Set `YFINANCE_SYMBOL` (e.g. `YFINANCE_SYMBOL=SPY`) to fetch a live price when `yfinance` is available.

## Colab Notebook
Open [`colab_alpha_agi_business_v1_demo.ipynb`](colab_alpha_agi_business_v1_demo.ipynb) and run all cells. The notebook checks requirements, starts the orchestrator, and exposes helper tools via the OpenAI Agents SDK.

## ADK Bridge
To expose the helper agent via Google's Agent Development Kit (ADK), install the
`google-adk` package and enable the bridge:
```bash
pip install google-adk  # optional dependency
ALPHA_FACTORY_ENABLE_ADK=true python openai_agents_bridge.py --host http://localhost:8000
```
The ADK gateway listens on `ALPHA_FACTORY_ADK_PORT` (default `9000`) and supports
Agent-to-Agent (A2A) messaging.
