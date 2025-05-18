# 👁️ Alpha-Factory v1 — Cross-Industry **AGENTIC α-AGI** Demo
*Out-learn • Out-think • Out-design • Out-strategise • Out-execute*

---

### 1 · Why we built this
Alpha-Factory stitches together **five flagship agents** (Finance, Biotech, Climate, Manufacturing, Policy) under a
zero-trust, policy-guarded orchestrator. 
It closes the full loop:

> **alpha discovery → uniform real-world execution → continuous self-improvement**

and ships with:

* **Automated curriculum** (Ray PPO trainer + reward rubric) 
* **Uniform adapters** (market data, PubMed, Carbon-API, OPC-UA, GovTrack) 
* **DevSecOps hardening** — SBOM + _cosign_, MCP guard-rails, ed25519 prompt signing 
* Runs **online (OpenAI)** or **offline** via bundled Mixtral-8×7B local-LLM 
* One-command Docker installer **_or_** one-click Colab notebook for non-technical users

The design follows the “AI-GAs” recipe for open-ended systems, 
embraces Sutton & Silver’s “Era of Experience” doctrine, and borrows
MuZero-style model-based search to stay sample-efficient.

---

### 2 · Two-click bootstrap

| Path | Audience | Time | Hardware |
|------|----------|------|----------|
| **Docker script**<br>`deploy_alpha_factory_cross_industry_demo.sh` | dev-ops / prod | 8 min | any Ubuntu with Docker 24 |
| **Colab notebook**<br>`colab_deploy_alpha_factory_cross_industry_demo.ipynb` | analysts / no install | 4 min | free Colab CPU |

Both flows autodetect `OPENAI_API_KEY`; when absent they inject a **Mixtral 8×7B**
local LLM container so the demo works **fully offline**.

> **Prerequisite**: Docker 24+ with the `docker compose` plugin (or the
> legacy `docker-compose` binary) must be installed.

### Quick Start
```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/cross_industry_alpha_factory
./deploy_alpha_factory_cross_industry_demo.sh
```


---

### 3 · Live endpoints after install

| Service | URL (default ports) |
|---------|---------------------|
| Grafana dashboards | `http://localhost:9000` `admin/admin` |
| Prometheus | `http://localhost:9090` |
| Trace-Graph (A2A spans) | `http://localhost:7860` |
| Ray dashboard | `http://localhost:8265` |
| REST orchestrator | `http://localhost:8000` (`GET /healthz`) |

All ports are configurable: set `DASH_PORT / PROM_PORT` etc before running the
installer.

---

### 4 · Architecture at a glance
```
┌──────────────────────────────────────────────────────────────────────────────┐
│ docker-compose (network: alpha_factory)                   │
│                                       │
│  Grafana ◄── Prometheus ◄── metrics ───────┐                │
│     ▲                 │                │
│ Trace-Graph ◄─ A2A spans ─ Orchestrator ──┴─► Knowledge-Hub (RAG + vec-DB) │
│           ▲      ▲                      │
│           │ ADK RPC  │ REST                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │      Five industry agents (side-car adapters in *italics*)    │ │
│  │ Finance   Biotech   Climate    Mfg.    Policy       │ │
│  │ broker,   *PubMed*   *Carbon*   *OPC-UA*  *GovTrack*     │ │
│  │ factor α  RAG-ranker  intensity   scheduler  bill watch    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```
_Edit the Visio diagram under `assets/diagram_architecture.vsdx`._

---

### 5 · The five flagship agents

| Agent | Core libraries | Live adapter | Reward | Key env vars |
|-------|---------------|--------------|--------|--------------|
| FinanceAgent | `pandas-ta`, `cvxpy` | broker / market-data | Δ P&L − λ·VaR | `BROKER_API_KEY` |
| BiotechAgent | `langchain`, `biopython` | *PubMed* mock | novelty-weighted citations | `PUBMED_EMAIL` |
| ClimateAgent | `prophet` | *carbon-api* mock | − tCO₂eq / $ | `CARBON_API_KEY` |
| ManufacturingAgent | `ortools` | *OPC-UA* bridge | cost-to-produce ↓ | `OPC_HOST` |
| PolicyAgent | `networkx`, `sentence-transformers` | *GovTrack* | sentiment × p(passage) | `GOVTRACK_KEY` |

All inherit `BaseAgent(plan, act, learn)` and register with the orchestrator
via ADK’s `AgentDescriptor`.

---

### 6 · Continuous-learning pipeline (15 min cadence)
1. **Ray RLlib PPO** trainer spins in its own container (`alpha-trainer`).
2. Rewards are computed by `continual/rubric.json` (edit live; hot-reload).
3. Best checkpoint is zipped and `POST /agent/<id>/update_model` → agents swap
  weights **with zero downtime**.
4. CI smoke-tests (`.github/workflows/ci.yml`) validate orchestration on every
  PR; failures block merge.

---

### 7 · Security, compliance & transparency

| Layer | Control | Verification |
|-------|---------|--------------|
| Software Bill of Materials | **Syft** emits SPDX JSON | attested with **cosign** and pushed to the **Rekor** transparency log |
| Policy enforcement | **MCP** side-car runs `redteam.json` deny-rules | unit test: `make test:policy` |
| Prompt integrity | ed25519 signature embedded in every request header | Grafana panel “Signed Prompts %” |
| Container hardening | read-only FS, dropped caps, seccomp | passes *Docker Bench* & *Trivy* |

---

### 8 · Performance & heavy-load benchmarking
A **k6** scenario (`bench/k6_load.js`) and a matching Grafana dashboard are
included. On a 4-core VM the stack sustains **🌩 550 req/s** across agents
with p95 latency < 180 ms.

---

### 9 · Extending & deploying at scale
* **New vertical** → subclass `BaseAgent`, add adapter container, append to
 `AGENTS_ENABLED` in `.env`.
* **Custom LLM** → point `OPENAI_API_BASE` to your endpoint.
* **Kubernetes** → `make helm && helm install alpha-factory chart/`.

---

### 10 · Roadmap
* Production Helm chart (HA Postgres + Redis event-bus) 
* Replace mock PubMed / Carbon adapters with real connectors 
* Grafana auto-generated dashboards from OpenTelemetry spans 

Community PRs welcome!

---

### References
Clune 2019 · Sutton & Silver 2024 · MuZero 2020 

© 2025 MONTREAL.AI — MIT License
