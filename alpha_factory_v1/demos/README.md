# Alpha‑Factory v1 👁️✨ — **Interactive Demo & Agent Gallery**
*Out‑learn | Out‑think | Out‑design | Out‑strategise | Out‑execute*

> “Intelligence is **experience** distilled through relentless self‑play.” — inspired by Sutton & Silver’s *Era of Experience* citeturn32file3

---

## 🗺️ Navigator
| Section | Why read it? |
|---------|--------------|
|[1 • Welcome](#1-welcome) | Grand vision & quick launch |
|[2 • Demo Showcase 🎮](#2-demo-showcase-) | What each demo does & how to run it |
|[3 • Agent Roster 🖼️](#3-agent-roster-) | How every backend agent creates alpha |
|[4 • Deploy Cheat‑Sheet 🚀](#4-deploy-cheat-sheet-) | One‑liners for laptop ↔ cloud |
|[5 • Governance & Safety ⚖️](#5-governance--safety-) | Zero‑trust, audit trail, fail‑safes |
|[6 • Extending the Factory 🔌](#6-extending-the-factory-) | Plug‑in new demos & agents |
|[7 • Credits ❤️](#7-credits-) | Legends & support |

---

## 1 • Welcome  
**Alpha‑Factory v1** is the **cross‑industry agentic engine** that captures live α‑signals and turns them into value across Finance, Policy, Manufacturing, Biotech & beyond.  This gallery lets you *touch* that power:

```bash
# one‑command immersive tour (CPU‑only)
curl -sSL https://raw.githubusercontent.com/MontrealAI/AGI-Alpha-Agent-v0/main/alpha_factory_v1/demos/quick_start.sh | bash
```
Opens **http://localhost:7860** with a Gradio portal to every demo. Works on macOS, Linux, WSL 2 & Colab.

---

## 2 • Demo Showcase 🎮
| # | Demo Folder | Emoji | Lightning Pitch | Alpha Contribution | Start Locally |
|---|-------------|-------|-----------------|--------------------|---------------|
|1|`aiga_meta_evolution`|🧬|Agents that *evolve* new agents; genetic unit tests auto‑score fitness.|Exponentially expands strategy space, surfacing fringe alpha pockets.|`docker compose -f docker-compose.aiga_meta.yml up`|
|2|`era_of_experience`|🏛️|Streams of lifelong events feed an autobiographical memory‑graph tutor.|Transforms tacit SME knowledge into tradable signals.|`docker compose -f docker-compose.era.yml up`|
|3|`finance_alpha`|💹|Live momentum + risk‑parity bot on Binance test‑net.|Generates real P&L; stress‑tested against CVaR limits.|`docker compose -f docker-compose.finance.yml up`|
|4|`macro_sentinel`|🌐|GPT‑RAG news scanner auto‑hedges with CTA futures.|Shields portfolios from macro shocks.|`docker compose -f docker-compose.macro.yml up`|
|5|`muzero_planning`|♟️|MuZero plans in synthetic markets → learns optimal execution curves.|Validates world‑model planning in noisy domains. (MuZero core from Schrittwieser et al. 2020) citeturn34file3|`docker compose -f docker-compose.muzero.yml up`|
|6|`selfheal_repo`|🩹|CI fails → agent crafts patch ⇒ PR green again.|Keeps critical pipelines up, sustaining uptime alpha.|`docker compose -f docker-compose.selfheal.yml up`|

> **Colab?** Each folder ships an `*.ipynb` that mirrors the Docker flow with free GPUs.

---

## 3 • Agent Roster 🖼️
Each backend agent is callable as an **OpenAI Agents SDK** tool *and* as a REST endpoint (`/v1/agents/<name>`).  

| # | Agent File | Emoji | Secret Sauce | Deploy Solo |
|---|------------|-------|--------------|-------------|
|1|`finance_agent.py`|💰|LightGBM multi‑factor α → RL execution bridge.|`AF_AGENT=finance python -m backend.orchestrator`|
|2|`biotech_agent.py`|🧬|UniProt × PubMed KG‑RAG; CRISPR off‑target scorer.|`AF_AGENT=biotech …`|
|3|`manufacturing_agent.py`|⚙️|OR‑Tools CP‑SAT scheduler + CO₂ predictor.|`AF_AGENT=manufacturing …`|
|4|`policy_agent.py`|📜|Statute QA + ISO‑37301 risk tagging.|`AF_AGENT=policy …`|
|5|`energy_agent.py`|🔋|Demand‑response optimiser for ISO‑NE.|`AF_AGENT=energy …`|
|6|`supply_chain_agent.py`|📦|VRP solver & ETA forecaster.|`AF_AGENT=supply_chain …`|
|7|`marketing_agent.py`|📈|RL campaign tuner with multi‑touch attribution.|`AF_AGENT=marketing …`|
|8|`research_agent.py`|🔬|Literature RAG + hypothesis ranking.|`AF_AGENT=research …`|
|9|`cybersec_agent.py`|🛡️|CVE triage & honeypot director.|`AF_AGENT=cybersec …`|
|10|`climate_agent.py`|🌎|Emission forecasting under scenario stress.|`AF_AGENT=climate …`|
|11|`stub_agent.py`|🫥|Auto‑spawn placeholder when deps missing.|n/a (auto) |

**Playbooks** live in `/examples/<agent_name>.ipynb` — copy‑paste ready.

---

## 4 • Deploy Cheat‑Sheet 🚀
| Platform | One‑liner | Notes |
|----------|-----------|-------|
|Docker Compose|`docker compose up -d orchestrator`|Spins Kafka, Prometheus, Grafana, agents, demos.|
|Kubernetes|`helm repo add alpha-factory https://montrealai.github.io/helm && helm install af alpha-factory/full`|mTLS via SPIFFE, HPA auto‑scales.|
|Colab|Launch notebook ⇒ click *“Run on Colab”* badge.|GPU‑accelerated demos.|
|Bare‑metal Edge|`python edge_runner.py --agents manufacturing,energy`|Zero external deps; SQLite state.|

---

## 5 • Governance & Safety ⚖️
* **Model Context Protocol** envelopes every artefact (SHA‑256 + ISO‑8601).  
* **SPIFFE + mTLS** across the mesh → zero‑trust.  
* **Cosign + Rekor** immutable supply chain.  
* Live bias & harm evals via model‑graded tests each night.

---

## 6 • Extending the Factory 🔌
```toml
[project.entry-points."alpha_factory.demos"]
my_demo = my_pkg.cool_demo:app
```
1. Ship a Gradio or Streamlit `app` returning a FastAPI router.  
2. Add Helm annotation `af.maturity=beta` → appears in UI.  
3. Submit PR — CI auto‑runs red‑team prompts.

---

## 7 • Credits ❤️
*Jeff Clune* for **AI‑GA** inspiration; *Sutton & Silver* for the *Era of Experience* pillars; *Schrittwieser et al.* for **MuZero** foundations.

Special salute to **[Vincent Boucher](https://www.linkedin.com/in/montrealai/)** — architect of the 2017 [Multi‑Agent AI DAO](https://www.quebecartificialintelligence.com/priorart) and steward of the **$AGIALPHA** utility token powering this venture.

---

© 2025 MONTREAL.AI — MIT License

