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
   ```
   The dashboard is available at [http://localhost:8000/docs](http://localhost:8000/docs).

Set `OPENAI_API_KEY` to enable cloud models. Offline mode works automatically when the key is absent.
Set `YFINANCE_SYMBOL` (e.g. `YFINANCE_SYMBOL=SPY`) to fetch a live price when `yfinance` is available.

## Colab Notebook
Open [`colab_alpha_agi_business_v1_demo.ipynb`](colab_alpha_agi_business_v1_demo.ipynb) and run all cells. The notebook checks requirements, starts the orchestrator, and exposes helper tools via the OpenAI Agents SDK.
