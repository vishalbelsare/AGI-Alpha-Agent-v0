
<!-- 2025‑04‑23 — α‑Factory v1 README -->
<p align="center">
  <img src="https://raw.githubusercontent.com/MontrealAI/brand-assets/main/alpha-eye.svg" alt="α‑AGI Eye" width="120">
</p>

<h1 align="center">AGI‑Alpha‑Agent‑v0</h1>
<h3 align="center">CA: tWKHzXd5PRmxTF5cMfJkm2Ua3TcjwNNoSRUqx6Apump</h3>
<h2 align="center">AGI ALPHA AGENT (<a href="https://app.ens.domains/name/alpha.agent.agi.eth">ALPHA.AGENT.AGI.Eth</a>) ⚡ Powered by $AGIALPHA</h2>
<p align="center"><strong>Seize the Alpha. Transform the world.</strong></p>

> **Vincent Boucher** — President of <a href="https://www.montreal.ai">MONTREAL.AI</a> — reshaped AI history by conquering the <a href="https://web.archive.org/web/20170929214241/https://gym.openai.com/read-only.html">OpenAI Gym</a> (2016) and by publishing the seminal <a href="https://www.quebecartificialintelligence.com/priorart">“Multi‑Agent AI DAO”</a> paper (2017).  
> Today that lineage culminates in the **AGENTIC α‑AGI 👁️✨**: a cross‑industry factory that Out‑learns, Out‑thinks, Out‑designs, Out‑strategises, and Out‑executes.

<p align="center">
  <a href="https://htmlpreview.github.io/?https://raw.githubusercontent.com/MontrealAI/AGI-Alpha-Agent-v0/main/deploy_sovereign_agentic_agialpha_agent_v0.html">
    🔱 ∞ AGENTIC ALPHA EXPLORER (INFINITE MODE) ∞ 🔱
  </a>
</p>

---

## ✨ Why α‑Factory?

```
┌───────────────────────────────────────────┐
│   α‑Factory =  Multi‑Agent  Signal Forge  │
│            ────────────────────           │
│  • FinanceAgent        • PolicyAgent      │
│  • BiotechAgent        • Manufacturing    │
│                                           │
│  Each agent ↔ autonomously discovers,     │
│  validates & exploits α (alpha signals)   │
│  under unified governance & observability │
└───────────────────────────────────────────┘
```

* **State‑of‑the‑art agent tool‑chain** — OpenAI Agents SDK, Google ADK, Anthropic MCP, A2A protocol  
* **Plug‑and‑profit vertical adapters** — live market feeds, genomics pipelines, factory‑floor schedulers  
* **Built‑in safety & compliance** — SBOM, SPIFFE/SPIRE identities, model‑graded evals, audit API  
* **Antifragile architecture** — every stressor becomes training data; agents self‑fine‑tune continuously  
* **Runs **with or without** `OPENAI_API_KEY`** — falls back to local Φ‑2 (from Ollama) or any HF model

---

## 🚀 Quick Start — one command

```bash
curl -sL https://raw.githubusercontent.com/MontrealAI/AGI-Alpha-Agent-v0/main/deploy_live_alpha.sh | bash
```

The launcher will:

1. 🔍 Check GPU & memory, install Docker / Ollama if missing  
2. 🛠️ Clone or update **AGI‑Alpha‑Agent‑v0** and build/pull images  
3. 📈 Run `alpha_finder.py` → pick today’s best live momentum alpha (finance demo)  
4. 🧩 Spin up all agents + web UI (Docker Compose or K8s‑kind)  
5. 🌐 Open <http://localhost:8080> — watch the trace‑graph UI in real‑time  

_No API key? No problem — the launcher automatically switches models and disables remote calls._

---

## 🏗️ Architecture at a Glance

```mermaid
flowchart LR
    subgraph Runtime
        direction TB
        AM[AlphaManager<br>(Ray Actor)]
        FA[FinanceAgent] --> AM
        PA[PolicyAgent]  --> AM
        BA[BiotechAgent] --> AM
        MA[ManufacturingAgent] --> AM
    end
    UI[Trace‑Graph&nbsp;UI] --- AM
    Grafana:::obs --- AM
    subgraph Control‑Plane
        Git[GitHub Actions<br/>+ Cosign] -->|SBOM| REG[GHCR Registry]
        REG -->|Signed images| Runtime
    end
    classDef obs fill:#fffbdd;
```

---

## 🧬 Vertical Agents

| Agent | Data source | Core model | Unique skills |
|-------|-------------|-----------|---------------|
| **FinanceAgent** | Polygon.io, Binance, FRED | GPT‑4o or Φ‑2 | Factor discovery, risk parity, on‑chain execution |
| **BiotechAgent** | Ensembl REST, PDB, Lab notebook RAG | GPT‑4o | Protein‑target match, CRISPR guide scoring |
| **PolicyAgent** | govinfo.gov, RegHub API | GPT‑4o | Bill summarisation, lobbying pathfinder |
| **ManufacturingAgent** | OPC‑UA stream, MES export | GPT‑4o | OR‑Tools schedule optimiser, downtime root‑cause |

Every agent implements `IAgent` from OpenAI Agents SDK and speaks A2A messages (`.a2a.json`). Messages are signed and logged for audit.

---

## 🛡️ Security & Compliance

* **Zero‑trust mesh** — SPIFFE IDs, mTLS everywhere  
* **Cosign‑signed containers** — verified at startup  
* **Model‑Graded Eval** — OpenAI *bias / defamation* evals run on every new model checkpoint  
* **Reg‑Ready** — full trace, reproducible builds, SOC2‑style controls

---

## 🎮 Demos (`alpha_factory_v1/demos/`)

1. **`macro_sentinel`** — horizon‑scans macro‑econ events and back‑tests hedge positioning  
2. **`muzero_planning`** — uses MuZero ♟️ against synthetic markets to stress‑test agent reasoning  
3. **`aiga_meta_evolution`** — evolutionary meta‑agent that writes new agents via code‑gen  

Run any demo:

```bash
docker compose -f demos/docker-compose.muzero.yml up
```

---

## 🛠️ Developer Guide

```bash
# run tests (+ red‑team prompts)
docker compose exec orchestrator pytest -q /app/tests

# hot‑reload backend
docker compose exec orchestrator reflex run --reload

# generate SBOM
docker compose exec orchestrator syft packages /app -o json > sbom.json
```

---

## 📜 License

MIT (c) 2025 Montreal.AI — see `LICENSE`.

---

<p align="center"><sub>α‑Factory v1 • Out‑learn · Out‑think · Out‑design · Out‑strategise · Out‑execute</sub></p>
