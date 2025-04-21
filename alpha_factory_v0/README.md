
# Alpha‑Factory v0 👁️✨ — Multi‑Agent AGENTIC α‑AGI

**Out‑learn | Out‑think | Out‑design | Out‑strategise | Out‑execute**

<!--
  α‑Factory • Multi‑Agents AGENTIC α‑AGI 👁️✨
  Out‑learn · Out‑think · Out‑design · Out‑strategise · Out‑execute
  © 2025 MONTREAL.AI   MIT License
-->

# Alpha‑Factory v0 – Cross‑Industry **Agentic** AGI Stack

**α‑Factory** is a reference‑quality, end‑to‑end implementation of a
multi‑agent system that **identifies live alpha opportunities** and
**converts them into value** across Finance, Policy, Manufacturing and
Biotech.  
It is built on the latest best‑practices from

* **[OpenAI Agents SDK]** (2024 preview)  
* **Google [ADK] Agent Development Kit**  
* **Agent‑to‑Agent (A2A) Protocol** & **Model Context Protocol**  
* Guidance from *“[A Practical Guide to Building Agents]”* (OpenAI, 2025)

The stack runs **with or without** an `OPENAI_API_KEY` — offline fallback
models keep everything usable when the cloud is unavailable.

<div align="center">
  <img src="docs/trace_demo.gif" width="680"/>
  <br/><em>Live trace‑graph streaming from the planner ➝ tool calls.</em>
</div>

---

## ✨ Why α‑Factory?

* **Agentic First** – planner + tools pattern baked in everywhere.  
* **Cross‑Domain** – Finance / Policy / Manufacturing / Biotech agents
  share infrastructure & governance.  
* **Production‑Grade** – Kubernetes Helm chart, SPIFFE zero‑trust side‑cars,
  SBOM, Cosign signatures, Prometheus / Grafana dashboards.  
* **Extensible** – swap a data‑feed, add a tool, or plug a brand‑new agent
  with three lines of code.  
* **Reg‑Tech Ready** – governance guard‑rails, audit logs, antifragile
  design to withstand regulatory scrutiny.

---

## 🏗️ Project Tree (TL;DR)

```text
alpha_factory_v0/
├── backend/                      # Python source
│   ├── __init__.py               # ASGI entry‑point  (/api/logs, /ws/trace, /metrics)
│   ├── finance_agent.py          # Factor & VaR‑aware trading agent
│   ├── market_data.py            # Polygon / Binance / Yahoo async adapter
│   ├── broker/                   # Alpaca, IBKR & Sim brokers
│   ├── portfolio.py              # Tiny on‑disk trade ledger
│   ├── policy_agent.py           # GPT‑RAG over statute corpus
│   ├── manufacturing_agent.py    # OR‑Tools shop‑floor optimiser
│   ├── biotech_agent.py          # Bio knowledge‑graph RAG agent
│   ├── a2a_client.py             # gRPC / WebSocket remote‑agent mesh
│   ├── trace_ws.py               # WebSocket hub (+ CSRF) → Trace UI
│   ├── governance.py             # Prompt & output guard‑rails
│   └── …
├── ui/                           # Vite + D3 trace‑graph front‑end
├── helm/alpha-factory-remote/    # SPIFFE‑aware Kubernetes Helm chart
├── tests/                        # pytest + red‑team prompts
├── Dockerfile                    # Multi‑stage (UI build → CPU / CUDA runtime)
├── docker-compose.yml            # single‑node stack
├── docker-compose.override.yml   # dev overrides (bind‑mount code, hot‑reload)
├── requirements.txt              # dev upper bounds
├── requirements-lock.txt         # reproducible prod lock
└── .github/workflows/            # CI – build, SBOM, push, Cosign sign
```

A **complete file‑by‑file tour** is provided later in this README.

---

## 🚀 Quick Start (5 minutes)

### 1 · Clone & run pre‑built container (no code required)

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v0

# CPU‑slim image (works on any Docker host)
docker compose up          # builds UI if needed & starts API + UI

# Visit:
#   http://localhost:3000   ← D3 Trace‑graph UI
#   http://localhost:8000   ← OpenAPI schema / live API
```

> **GPU / CUDA** – build with:
>
> ```bash
> docker compose --profile cuda up --build
> ```

### 2 · Run **without** an `OPENAI_API_KEY`

All agents fall back to local models (Φ‑2.gguf for text; nomic‑embed for
embeddings).  
Simply omit the env‑var – everything keeps working (slower inference).

### 3 · Bring your own OpenAI key (optional – unlock GPT‑4o, text‑embedding‑3)

```bash
export OPENAI_API_KEY="sk‑…"
docker compose up
```

The agents auto‑switch to cloud quality while offline fallbacks remain
as hot‑spare resilience.

---

## 📦 Advanced Install (devs & prod ops)

<details>
<summary><strong>▼ Using <code>conda + poetry</code> (dev workflow)</strong></summary>

```bash
# one‑liner
make dev

# manual steps
conda env create -f env.yml   # uses the lock file
conda activate alpha-factory
poetry install --sync --with dev
uvicorn backend:app --reload  # hot‑reload ASGI
```
</details>

<details>
<summary><strong>▼ Kubernetes (remote swarm)</strong></summary>

```bash
helm repo add alpha-factory https://montrealai.github.io/helm-charts
helm install af remote/alpha-factory-remote \
     --set image.tag=cpu-slim-latest \
     --set spiffe.enabled=true \
     --namespace alpha-factory --create-namespace
```

The chart automatically injects a **SPIFFE/SPIRE side‑car** for
mutual‑TLS between pods and enables the **A2A gRPC mesh**.
</details>

---

## 🛠️ Agents & Tools

| Agent | Key Technologies | Top‑Line Features |
|-------|------------------|-------------------|
| **FinanceAgent** | numpy / pandas · Cornish‑Fisher VaR · OpenAI Agents planner | Factor‑model alpha, async market data, VaR + max‑draw‑down guard‑rails, Prom‑metrics |
| **PolicyAgent** | OpenAI Agents SDK · local Llama‑cpp fallback · FAISS RAG | Answers legal / regulatory queries, A2A service, governance moderation |
| **ManufacturingAgent** | OR‑Tools · Google ADK | Optimises shop‑floor schedule, constraint modelling, agentic “what‑if” |
| **BiotechAgent** | KG‑RAG (RDFLib) · text‑embedding‑3 | Links pathways / compounds; surfaces gene–drug hypotheses |
| **A2A Remote Swarm** | gRPC‑WebSocket hybrid · SPIFFE | Spin‑up remote workers that self‑register; secure by default |

All planners emit **trace events** which travel via `trace_ws.hub` to the
front‑end. The D3 panel visualises the decision graph in real‑time.

---

## 🧩 Architecture

```text
flowchart TD
    subgraph Browser
        UI[Trace‑graph UI<br/>(Vite + D3)]
        UI -- WebSocket : /ws/trace --> API
    end

    subgraph API["backend/__init__.py<br/>FastAPI (+ fallback ASGI)"]
        Logs[/api/logs]
        Metrics[/metrics]
        CSRF[/api/csrf]
    end

    API -- ASGI mount --> Finance(🏦 FinanceAgent)
    API -- ASGI mount --> Policy(⚖️ PolicyAgent)
    API -- ASGI mount --> Mfg(🏭 ManufacturingAgent)
    API -- ASGI mount --> Bio(🧬 BiotechAgent)

    Finance <-- gRPC / A2A --> Remote[Remote Pods<br/>(Helm chart)]
    Policy  <-- trace events --> API
    Mfg -. tool calls .-> Finance
```

---

## 📚 File‑by‑File Reference

*(collapsed for brevity – expand if needed in GitHub)*

<details><summary><strong>backend/ – key modules</strong></summary>

| File | Purpose |
|------|---------|
| **`__init__.py`** | ASGI root; routes `/api/logs`, `/api/csrf`, mounts `/metrics`, wires `/ws/trace` |
| **`finance_agent.py`** | Vectorised factor model, VaR + DD limits, Prom‑metrics |
| **`market_data.py`** | Async Polygon / Binance / Yahoo feed auto‑select |
| **`broker/`** | `alpaca.py`, `ibkr.py`, `sim.py`; exponential back‑off |
| **`trace_ws.py`** | In‑memory hub ➝ WebSocket; CSRF token handshake |
| **`policy_agent.py`** | GPT‑RAG statutes; OpenAI Agents or offline Llama |
| **`manufacturing_agent.py`** | OR‑Tools job‑shop schedule optimiser |
| **`biotech_agent.py`** | Pathway KG RAG; sparkline hypothesis |
| **`a2a_client.py`** | Remote mesh connector (gRPC + WebSocket) |
| **`governance.py`** | Bad‑prompt moderation (red‑team tests) |
| **tests/** | Red‑team prompts, smoke‑tests, CI gate |
</details>

<details><summary><strong>CI / CD</strong></summary>

* `.github/workflows/container-publish.yml` – Buildx matrix ➝ CPU & CUDA
  images, SBOM, Cosign signature to **ghcr.io**, multi‑arch.
* SBOM is exported as SPDX and attached as build artefact.
</details>

---

## 📈 Dashboards & Metrics

* `/metrics` – Prometheus exposition (FinanceAgent VaR, P&L, draw‑down)
* `grafana/finance.json` – ready‑made dashboard
* Helm chart auto‑labels pods for Prometheus ServiceMonitor.

---

## 🧪 Tests & Eval Harness

```bash
pytest -q
```

* Continuous red‑team prompts check governance filters.  
* OpenAI “evals” JSONL harness (nightly in CI).  
* Coverage > 90 % on core business logic.

---

## 🔒 Security Notes

* SPIFFE side‑car (opt‑in) issues `/run/spire/sock` for mTLS identity.  
* WebSocket CSRF – first frame must echo one‑time token from `/api/csrf`.  
* SBOM + Cosign signature on every container.  
* Agent guard‑rails powered by `backend/governance.py` (Better‑Profanity, custom red‑team rules).

---

## 📜 Further Reading

* **OpenAI Agents SDK (Python)** – <https://openai.github.io/openai-agents-python/>  
* **A Practical Guide to Building Agents** (OpenAI, 2025)  
  <https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf>  
* **Google Agent Development Kit** – <https://google.github.io/adk-docs/>  
* **Agent‑to‑Agent Protocol (A2A)** – <https://github.com/google/A2A>  
* **Model Context Protocol** – <https://www.anthropic.com/news/model-context-protocol>

---

## ✨ Roadmap

1. **Reinforcement Learning on Execution Alpha** (live slippage minimiser)  
2. **Self‑Play Stress‑Test Harness** – antifragile improvement loop  
3. **Reg‑Tech Audit Trail Export** – OpenTelemetry + W3C VCs  
4. **Plug‑&‑Play Industry Packs** – Energy, Logistics, Health‑Care

---

> **Let’s out‑think the future.**

## CURRENTLY UNDER ACTIVE DEVELOPMENT.
