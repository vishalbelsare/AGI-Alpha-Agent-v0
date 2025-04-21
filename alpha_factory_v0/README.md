
# Alpha‑Factory v0 👁️✨ — Multi‑Agent AGENTIC α‑AGI

**Out‑learn | Out‑think | Out‑design | Out‑strategise | Out‑execute**

*A cross‑industry “Alpha Factory” with production‑grade agents for Finance, Policy, Manufacturing and Biotech. Ships data‑feed & broker adapters, live trace‑graph UI, swarm‑ready A2A mesh, CI‑driven evaluation harness, SPIFFE‑secured Helm chart and reproducible Docker images.*

---

## Folder structure

```text
alpha_factory_v0/
├── backend/                      # Python source
│   ├── __init__.py               # ASGI entry‑point (/api/logs, /ws/trace, /metrics, /api/csrf)
│   ├── finance_agent.py          # Production trading agent (factor model + VaR / DD guard‑rails)
│   ├── market_data.py            # Async Polygon / Binance / Yahoo adapter
│   ├── broker/                   # Alpaca, IBKR & simulated brokers
│   ├── portfolio.py              # Tiny append‑only trade ledger
│   ├── policy_agent.py           # GPT‑RAG over statutes (FAISS + OpenAI Agents SDK)
│   ├── manufacturing_agent.py    # OR‑Tools shop‑floor optimiser
│   ├── biotech_agent.py          # Bio knowledge‑graph RAG agent
│   ├── a2a_client.py             # gRPC / WebSocket remote‑swarm client
│   ├── trace_ws.py               # WebSocket hub (+ CSRF) → Trace UI
│   ├── governance.py             # Prompt & output moderator
│   └── …
├── ui/                           # Vite / D3 trace‑graph front‑end
├── helm/alpha-factory-remote/    # Kubernetes chart (SPIFFE‑aware)
├── tests/                        # pytest + red‑team prompts
├── Dockerfile                    # Multi‑stage (UI build → CPU/GPU runtime)
├── docker-compose.yml            # base compose
├── docker-compose.override.yml   # dev overrides (bind‑mount, hot‑reload)
├── requirements.txt              # upper bounds (dev)
├── requirements-lock.txt         # reproducible lock (prod)
└── .github/workflows/            # CI – build, SBOM, push, sign
```

## Quick‑start (local)

```bash
# 1. clone
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v0

# 2. run everything via Compose (CPU)
docker compose -f docker-compose.yml -f docker-compose.override.yml up --build

# Open:
#   http://localhost:3000/      ← live trace‑graph UI
#   http://localhost:8000/api/logs
#   http://localhost:8000/metrics
```

### GPU build

```bash
docker build -t alphafactory:cuda   --build-arg BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04 .
```

Run with `--gpus all`.

## Installation (bare‑metal dev)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
uvicorn backend:app --reload
```

Optional: set `OPENAI_API_KEY`, `POLYGON_API_KEY`, `BINANCE_API_KEY`, `ALPACA_KEY_ID`, `ALPACA_SECRET_KEY`.

## Helm (remote swarm)

```bash
helm repo add alpha-factory https://montrealai.github.io/alpha-factory-charts
helm install af alpha-factory/alpha-factory-remote   --set image.tag=cuda-latest   --set spiffe.enabled=true
```

## Security

* SPIFFE/SPIRE side‑car (opt‑in) for mTLS identity inside K8s.
* CSRF token required for `/ws/trace` handshake.
* Governance module blocks extremist, illicit‑finance or profane requests.

## Tests & CI

```bash
pytest -q                  # unit + red‑team moderation tests
pytest --runbench          # optional benchmark suite
```

GitHub Actions builds `cpu‑slim` and `cuda` images, attaches SBOM (SPDX) and signs with Cosign before pushing to GHCR.

## Contributing

1. Fork → feature branch → PR.
2. Run `pre-commit run -a`.
3. Ensure **CI is green** (pytest, smoke, lint).

## Licence

MIT © 2025 MONTREAL.AI  —  This repo ships model weights only via public URLs; check each model licence separately.
