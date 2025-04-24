# Alpha‑Factory v1 👁️✨ — Backend Agents Suite  
*Out‑learn · Out‑think · Out‑design · Out‑strategise · Out‑execute*

Welcome to **Alpha‑Factory’s** beating heart: **eleven** self‑improving, cross‑domain agents working in concert to detect *alpha* and transform it into decisive action — continuously, securely, and under full governance.

---

## 📜 Contents  
1. [Design Philosophy](#design-philosophy)  
2. [Agent Gallery 🖼️](#agent-gallery)  
3. [5‑Minute Quick‑Start 🚀](#5‑minute-quick-start)  
4. [Deployment Recipes 🍳](#deployment-recipes)  
5. [Per‑Agent Playbooks 📘](#per‑agent-playbooks)  
6. [Runtime Topology 🗺️](#runtime-topology)  
7. [Governance & Compliance ⚖️](#governance--compliance)  
8. [Health‑Beats & Quarantine 💓](#health‑beats--quarantine)  
9. [Extending the Mesh 🔌](#extending-the-mesh)  
10. [Troubleshooting 🛠️](#troubleshooting)

---

## Design Philosophy  

> “We’ve moved from *big‑data hoarding* to *big‑experience compounding*.”  

Each agent embodies an **experience‑first loop** inspired by Sutton & Silver’s *Era of Experience* and Clune’s AI‑GA pillars:

1. **Sense 👂** — ingest streaming data (Kafka, MQTT, Web sockets, REST hooks).  
2. **Imagine 🧠** — plan on a learned world‑model (MuZero style where useful).  
3. **Act 🤖** — execute, monitor, log — all wrapped in a Model Context Protocol (MCP) envelope.  
4. **Adapt 🔄** — online learning, antifragile to stress & dependency loss.

Heavy extras (GPU, OR‑Tools, FAISS, OpenAI) are *optional*; agents **degrade gracefully** to heuristics while preserving audit artefacts.

---

## Agent Gallery  

| # | Agent (file) | Emoji | Core Super‑powers | Status | Heavy Deps | Key Env Vars |
|---|--------------|-------|------------------|--------|-----------|--------------|
| 1 | `finance_agent.py` | 💰 | Multi‑factor alpha, VaR 99 %, RL execution | **Prod** | `pandas`, `lightgbm` | `ALPHA_UNIVERSE`, `ALPHA_MAX_VAR_USD` |
| 2 | `biotech_agent.py` | 🧬 | KG‑RAG, CRISPR/assay design, pathway maps | **Prod** | `faiss`, `rdflib` | `BIOTECH_KG_FILE`, `OPENAI_API_KEY` |
| 3 | `manufacturing_agent.py` | ⚙️ | CP‑SAT job‑shop optimiser, CO₂ forecast | **Prod** | `ortools`, `prometheus_client` | `ALPHA_MAX_SCHED_SECONDS` |
| 4 | `policy_agent.py` | 📜 | Statute QA, red‑line diff, ISO‑37301 risk tags | **Prod** | `faiss`, `rank_bm25` | `STATUTE_CORPUS_DIR` |
| 5 | `energy_agent.py` | 🔋 | Demand‑response bidding, price elasticity | **Beta** | `numpy`, external API | `ENERGY_API_TOKEN` |
| 6 | `supply_chain_agent.py` | 📦 | VRP routing, ETA prediction, delay heat‑map | **Beta** | `networkx`, `scikit-learn` | `SC_DB_DSN` |
| 7 | `marketing_agent.py` | 📈 | Multi‑touch attribution, campaign RL tuning | **Beta** | `torch`, `openai` | `MARKETO_KEY` |
| 8 | `research_agent.py` | 🔬 | Literature RAG, hypothesis ranking | **Beta** | `faiss` | — |
| 9 | `cybersec_agent.py` | 🛡️ | CVE triage, MITRE ATT&CK reasoning | **Beta** | `faiss`, threat‑intel APIs | `VIRUSTOTAL_KEY` |
|10 | `climate_agent.py` | 🌎 | Emission forecasting, scenario stress tests | **Beta** | `xarray`, `numpy` | `NOAA_TOKEN` |
|11 | `stub_agent.py` | 🫥 | Auto‑generated placeholder when deps missing | **Auto** | — | — |

---

## 5‑Minute Quick‑Start 🚀  

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1

# (optional) full‑fat install incl. OR‑Tools, FAISS, Kafka‑Python, etc.
pip install -r requirements.txt

# minimal env for a demo
export OPENAI_API_KEY=<sk‑...>          # optional
export ALPHA_KAFKA_BROKER=localhost:9092

python -m backend.orchestrator
```

**Voilà! 🎉** Agents auto‑discover, self‑register, and stream:

* **`agent.manifest`** — signed JSON metadata per agent  
* **`agent.heartbeat`** — latency + health status  
* domain‑specific topics, e.g. `bt.experience`, `mf.schedule`, `fx.alpha`

---

## Deployment Recipes 🍳  

| Environment | Command | Highlights |
|-------------|---------|------------|
| **Docker Compose** | `docker compose up -d orchestrator` | Kafka + Prometheus stack |
| **Kubernetes** | `helm install alpha-factory ./charts/alpha-factory` | HPA, PodMonitor, secrets via SealedSecrets |
| **AWS Fargate** | `./infra/deploy_fargate.sh` | Spot‑instance friendly, SQS in place of Kafka |
| **Bare‑metal Edge** | `python edge_runner.py --agents manufacturing,energy` | Zero external deps, SQLite persistence |

---

## Per‑Agent Playbooks 📘  

<details>
<summary>Finance 💰</summary>

```python
from backend.agents import get_agent
fin = get_agent("finance")
alpha = fin.generate_signals()          # returns DataFrame
fin.execute_portfolio(alpha)
```

*Requires:* market data feed (Kafka topic `px.tick`) or CSV fallback  
*Governance:* VaR hard‑limit enforced via `ALPHA_MAX_VAR_USD`
</details>

<details>
<summary>Biotech 🧬</summary>

```python
bio = get_agent("biotech")
print(await bio.answer("Role of p53 in DNA repair?"))
```

*Offline mode* if no OpenAI key — SBERT embeddings + bullet summary.
</details>

<details>
<summary>Manufacturing ⚙️</summary>

```python
mf = get_agent("manufacturing")
jobs = [[{"machine":"M1","proc":10}, {"machine":"M2","proc":5}]]
sched_json = mf.build_schedule({"jobs": jobs, "horizon": 480})
```

Prometheus exports `af_job_lateness_seconds` metric for every schedule run.
</details>

…and so on for the remaining agents.  
(See `/examples` notebook for full interactive demos.)

---

## Runtime Topology 🗺️  

```text
flowchart LR
    subgraph Mesh
        ORC([🛠️ Orchestrator])
        F(💰) B(🧬) M(⚙️) P(📜) E(🔋) S(📦)
        C(🛡️) R(🔬) CL(🌎) MK(📈)
    end
    ORC -- A2A / OpenAI SDK --> F & B & M & P & E & S & C & R & CL & MK
    ORC -- Kafka bus --> DL[(🗄️ Data Lake)]
    F -.->|Prometheus| GRAFANA{{📊}}
```

---

## Governance & Compliance ⚖️  

* **Model Context Protocol (MCP)** wraps every outbound artefact (SHA‑256 digest, ISO‑8601 ts, determinism seed).  
* Agents self‑label with `COMPLIANCE_TAGS` such as `gdpr_minimal`, `iso22400`, `sox_traceable`.  
* Set `DISABLED_AGENTS=finance,policy` to boot without restricted domains.  

---

## Health‑Beats & Quarantine 💓  

A daemon thread monitors each `run_cycle`:

* **Latency** pushed to `agent.heartbeat`  
* **Failure streak** ≥ `AGENT_ERR_THRESHOLD` → auto‑quarantine & swap with **StubAgent**

Quarantined agents remain visible in the capability graph as `"quarantined": true`.

---

## Extending the Mesh 🔌  

1. **Create** `my_super_agent.py` subclassing `backend.agent_base.AgentBase`.  
2. **Declare** class constants:

```python
class MySuperAgent(AgentBase):
    NAME = "super"
    CAPABILITIES = ["telemetry_fusion"]
    COMPLIANCE_TAGS = ["gdpr_minimal"]
```

3. **Expose** via entry‑point:

```toml
[project.entry-points."alpha_factory.agents"]
super = my_pkg.super_agent:MySuperAgent
```

4. `pip install .` & restart orchestrator — no further wiring needed.

---

## Troubleshooting 🛠️  

| Symptom | Likely Cause | Remedy |
|---------|--------------|--------|
| `ImportError: faiss` | FAISS not in env | `pip install faiss-cpu` *or* rely on StubAgent |
| Agent in *quarantined* state | repeated exceptions | review logs, fix root cause, then `DISABLED_AGENTS=` restart |
| Kafka timeouts | Broker unreachable | set `ALPHA_KAFKA_BROKER=` to empty for stdout fallback |

---

Made with ❤️ by the Alpha‑Factory core team — *forging the tools that forge tomorrow*.
