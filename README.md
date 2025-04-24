# Alpha‑Factory v1 👁️✨ — Multi‑Agent **AGENTIC** α‑AGI  
*Out‑learn | Out‑think | Out‑design | Out‑strategise | Out‑execute*

> “[Vincent Boucher](https://www.linkedin.com/in/montrealai/), a pioneer in AI and President of [MONTREAL.AI](https://www.montreal.ai/) since 2003, reshaped the landscape by dominating the [OpenAI Gym](https://web.archive.org/web/20170929214241/https://gym.openai.com/read-only.html) with **AI Agents** in 2016 and unveiling the game‑changing [**“Multi‑Agent AI DAO”**](https://www.quebecartificialintelligence.com/priorart) blueprint in 2017 (“*The Holy Grail of Foundational IP at the Intersection of AI Agents and Blockchain*” — watch the 2018 reveal [🎥](https://youtu.be/Y4_6aZbVlo4), [read](https://www.huffpost.com/archive/qc/entry/blockchain-et-lintelligence-artificielle-une-combinaison-puis_qc_5ccc6223e4b03b38d6933d24)).

**AGI ALPHA AGENT (ALPHA.AGENT.AGI.Eth) Powered by $AGIALPHA.** 

> Our **AGI ALPHA AGENT**—fuelled by the strictly‑utility **$AGIALPHA CA: tWKHzXd5PRmxTF5cMfJkm2Ua3TcjwNNoSRUqx6Apump** token—now harnesses that visionary foundation—*arguably the world’s most valuable, impactful & important IP*—to unleash the ultimate alpha‑signal engine.

Alpha‑Factory v1 is a reference‑grade, cross‑industry **Multi‑Agent AGENTIC α‑AGI** that **detects live alpha** and **converts it into value**—autonomously, safely and auditable—across every vertical.

Built on the leading agent frameworks:

* **OpenAI Agents SDK** (2024‑25)  
* **Google ADK — Agent Development Kit**  
* **Agent‑to‑Agent (A2A) Protocol**  
* **Model Context Protocol (MCP)**  
* Best‑practice guidance from *“A Practical Guide to Building Agents”* (OpenAI, 2025)

…and engineered to operate **with or without** an `OPENAI_API_KEY` (graceful offline fall‑back).

<!-- ----------------------------------------------------------------- -->
## 📜 Table of Contents
1. [Design Philosophy](#design-philosophy)  
2. [Capability Graph 🌐](#capability-graph)  
3. [Backend Agents 🖼️](#backend-agents)  
4. [Demo Suite 🎮](#demo-suite)  
5. [5‑Minute Quick‑Start 🚀](#5-minute-quick-start)  
6. [Deployment Recipes 🍳](#deployment-recipes)  
7. [Runtime Topology 🗺️](#runtime-topology)  
8. [Governance & Compliance ⚖️](#governance--compliance)  
9. [Observability 🔭](#observability)  
10. [Extending the Mesh 🔌](#extending-the-mesh)  
11. [Troubleshooting 🛠️](#troubleshooting)  
12. [Roadmap 🛣️](#roadmap)  
13. [License](#license)  

<!-- ----------------------------------------------------------------- -->
## Design Philosophy
> “We’ve moved from **big‑data hoarding** to **big‑experience compounding**.” — Sutton & Silver, *Era of Experience*

Each agent runs an **experience loop**:

| Sense 👂 | Imagine 🧠 | Act 🤖 | Adapt 🔄 |
|----------|-----------|--------|---------|
| Stream real‑time data (Kafka, MQTT, Webhooks) | Plan on a *learned world‑model* (MuZero‑style where useful) | Execute tool‑calls & external actions — every artefact wrapped in MCP | Online learning, antifragile to dependency loss |

Heavy extras (GPU, FAISS, OR‑Tools, OpenAI) are **optional**; agents **degrade gracefully** to heuristics while preserving audit artefacts.

<!-- ----------------------------------------------------------------- -->
## Capability Graph 🌐
```text
graph TD
  subgraph α‑Mesh
    finance["💰 Finance"]
    biotech["🧬 Biotech"]
    manufacturing["⚙️ Manufacturing"]
    policy["📜 Policy"]
    energy["🔋 Energy"]
    supply["📦 Supply‑Chain"]
    marketing["📈 Marketing"]
    research["🔬 Research"]
    cyber["🛡️ Cyber‑Sec"]
    climate["🌎 Climate"]
    stub["🫥 Stub"]
  end
  classDef n fill:#0d9488,color:#ffffff,stroke-width:0px;
  class finance,biotech,manufacturing,policy,energy,supply,marketing,research,cyber,climate,stub n;
```
Call `GET /capabilities` to discover skills at run‑time.

<!-- ----------------------------------------------------------------- -->
## Backend Agents 🖼️
| # | Agent | Core Super‑powers | Heavy Deps | Key Env |
|---|-------|------------------|-----------|---------|
| 1 | **Finance** 💰 | Multi‑factor alpha, CVaR 99 %, RL execution & OMS bridge | `pandas`, `lightgbm`, `ccxt` | `ALPHA_UNIVERSE`, `ALPHA_MAX_VAR_USD` |
| 2 | **Biotech** 🧬 | UniProt/PubMed KG‑RAG, CRISPR off‑target design | `faiss`, `rdflib`, `openai` | `BIOTECH_KG_FILE` |
| 3 | **Manufacturing** ⚙️ | CP‑SAT optimiser, CO₂ forecast | `ortools`, `prometheus_client` | `ALPHA_MAX_SCHED_SECONDS` |
| 4 | **Policy** 📜 | Statute QA, ISO‑37301 risk tags | `faiss`, `rank_bm25` | `STATUTE_CORPUS_DIR` |
| 5 | **Energy** 🔋 | Demand‑response bidding | `numpy` + APIs | `ENERGY_API_TOKEN` |
| 6 | **Supply‑Chain** 📦 | VRP routing, ETA prediction | `networkx`, `scikit-learn` | `SC_DB_DSN` |
| 7 | **Marketing** 📈 | Multi‑touch attribution, RL tuning | `torch`, `openai` | `MARKETO_KEY` |
| 8 | **Research** 🔬 | Literature RAG, hypothesis ranking | `faiss` | — |
| 9 | **Cyber‑Sec** 🛡️ | CVE triage, MITRE ATT&CK reasoning | `faiss`, threat‑intel APIs | `VIRUSTOTAL_KEY` |
|10| **Climate** 🌎 | Emission forecasting | `xarray`, `numpy` | `NOAA_TOKEN` |
|11| **Stub** 🫥 | Placeholder when deps missing | — | — |

Each agent registers as an **OpenAI Agents SDK tool** and can be orchestrated from any LLM prompt or another agent.

<!-- ----------------------------------------------------------------- -->
## Demo Suite 🎮
| Demo | Purpose | Alpha Impact | Start |
|------|---------|--------------|-------|
| **AIGA Meta Evolution** 🧬 | Agents evolve new agents/unit‑tests | Compounding discovery speed | `docker compose -f demos/docker-compose.aiga_meta.yml up` |
| **Era Tutor** 🏛️ | Memory‑graph personal AI | Turns tacit memory into signals | `docker compose -f demos/docker-compose.era.yml up` |
| **Finance Alpha** 💹 | Live momentum + risk parity bot | Real P&L > baseline | `docker compose -f demos/docker-compose.finance.yml up` |
| **Macro Sentinel** 🌐 | News scanner → CTA hedge | Draw‑down hedge alpha | `docker compose -f demos/docker-compose.macro.yml up` |
| **MuZero Lab** ♟️ | Planning under uncertainty | Execution alpha | `docker compose -f demos/docker-compose.muzero.yml up` |
| **Self‑Healing Repo** 🩹 | Auto‑patch failing tests | Uptime alpha | `docker compose -f demos/docker-compose.selfheal.yml up` |

<!-- ----------------------------------------------------------------- -->
## 5‑Minute Quick‑Start 🚀
```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1
pip install -r requirements.txt             # full‑fat

export ALPHA_KAFKA_BROKER=localhost:9092    # empty → stdout
# optional
export OPENAI_API_KEY=sk-...

python -m backend.orchestrator
```
Browse **http://localhost:8000** (Swagger) and **http://localhost:3000** (trace‑graph UI).

<!-- ----------------------------------------------------------------- -->
## Deployment Recipes 🍳
| Platform | One‑liner | Highlights |
|----------|-----------|------------|
| Docker Compose | `docker compose up -d orchestrator` | Kafka + Prometheus |
| Kubernetes | `helm install af alpha-factory/stack` | SPIFFE mTLS, HPA |
| AWS Fargate | `infra/deploy_fargate.sh` | Spot‑friendly SQS‑shim |
| Bare‑Metal Edge | `python edge_runner.py --agents manufacturing,energy` | Zero external deps |

<!-- ----------------------------------------------------------------- -->
## Runtime Topology 🗺️
```text
flowchart LR
  subgraph α‑Mesh
    ORC([🛠️ Orchestrator])
    FIN(💰) BIO(🧬) MFG(⚙️) POL(📜) ENE(🔋) SUP(📦) MKT(📈) RES(🔬) CYB(🛡️) CLI(🌎)
  end
  ORC -- A2A / SDK --> FIN & BIO & MFG & POL & ENE & SUP & MKT & RES & CYB & CLI
  ORC -- Kafka --> DATALAKE[(🗄️ Data Lake)]
  FIN -.->|Prometheus| GRAFANA{{📊}}
```

<!-- ----------------------------------------------------------------- -->
## Governance & Compliance ⚖️
* **Model Context Protocol** wraps every artefact (SHA‑256, ISO‑8601 ts, determinism seed).  
* Agents self‑label `COMPLIANCE_TAGS` (`gdpr_minimal`, `sox_traceable` …).  
* `DISABLED_AGENTS=finance,policy` → regulator‑friendly boot.  
* Full audit chain logged to Sigstore Rekor.

<!-- ----------------------------------------------------------------- -->
## Observability 🔭
| Signal | Sink | Example |
|--------|------|---------|
| Heart‑beats | Kafka `agent.heartbeat` | `latency_ms` |
| Metrics | Prometheus | `af_job_lateness_seconds` |
| Traces | OpenTelemetry → Jaeger | `alpha_factory.trace_id` |

Grafana dashboards live in `infra/grafana/`.

<!-- ----------------------------------------------------------------- -->
## Extending the Mesh 🔌
```bash
pip install my_super_agent
```
```toml
# pyproject.toml
[project.entry-points."alpha_factory.agents"]
super = my_pkg.super_agent:MySuperAgent
```
Restart orchestrator — the agent self‑registers and appears on the graph.

<!-- ----------------------------------------------------------------- -->
## Troubleshooting 🛠️
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `ImportError: faiss` | Native lib missing | `pip install faiss-cpu` or rely on **StubAgent** |
| Agent quarantined | ≥3 failures | Fix bug → restart (state persisted) |
| Kafka timeout | Broker down | Unset `ALPHA_KAFKA_BROKER` → stdout |
| 402 (OpenAI) | Quota exhausted | Remove `OPENAI_API_KEY` → offline models |

<!-- ----------------------------------------------------------------- -->
## Roadmap 🛣️
1. Execution RL (live slippage minimiser)  
2. Self‑play stress harness (antifragile loops)  
3. Verifiable credentials for audit (OpenTelemetry × W3C VC)  
4. Plug‑&‑Play Industry Packs (Energy, Logistics, Health‑Care)  

<!-- ----------------------------------------------------------------- -->
## License
**MIT** © 2025 MONTREAL.AI — forging the tools that forge tomorrow.
