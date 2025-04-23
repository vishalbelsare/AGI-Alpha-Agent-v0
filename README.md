
<!-- =======================================================================
     2025‑04‑23 — α‑Factory v1 README (Ultimate Edition)
     This README is auto‑generated.  Drop the file at repo root.
     ======================================================================= -->

# AGI‑Alpha‑Agent‑v0

## CA: tWKHzXd5PRmxTF5cMfJkm2Ua3TcjwNNoSRUqx6Apump

## AGI ALPHA AGENT ([ALPHA.AGENT.AGI.Eth](https://app.ens.domains/name/alpha.agent.agi.eth)) ⚡ Powered by $AGIALPHA

### Seize the Alpha. Transform the World.

> **Vincent Boucher** — President of [MONTREAL.AI](https://www.montreal.ai) — conquered the [OpenAI Gym](https://web.archive.org/web/20170929214241/https://gym.openai.com/read-only.html) in 2016 and authored the seminal [Multi‑Agent AI DAO](https://www.quebecartificialintelligence.com/priorart) blueprint (2017).    
> That foundation now fuels **AGENTIC α‑AGI 👁️✨** — a cross‑industry **Alpha Factory** built to **Out‑learn · Out‑think · Out‑design · Out‑strategise · Out‑execute**.

<p align="center">
  <a href="https://htmlpreview.github.io/?https://raw.githubusercontent.com/MontrealAI/AGI-Alpha-Agent-v0/main/deploy_sovereign_agentic_agialpha_agent_v0.html">
    <img src="https://img.shields.io/static/v1?label=LIVE&message=ALPHA%20EXPLORER&color=blue&style=for-the-badge">
  </a>
</p>

---

## ✨ Why α‑Factory?

```
┌───────────────────────────────────────────────────────────┐
│ α‑Factory  →  Autonomous Alpha Signal Foundry            │
│                                                         │
│  Finance  •  Policy  •  Biotech  •  Manufacturing  • 🧬 │
│                  Meta‑Evolution Lab                     │
│                                                         │
│ Discovery  →  Backtest  →  Governance  →  Execution     │
└───────────────────────────────────────────────────────────┘
```

* **Best‑in‑class Agent Stack** — OpenAI Agents SDK, Google ADK, Anthropic MCP, Agent2Agent (A2A)  
* **Plug‑and‑Play Adapters** — live markets, genomics APIs, legislative feeds, OPC‑UA factory streams  
* **Reg‑grade Security** — SPIFFE identities, Cosign‑signed containers, SBOM, model‑graded evals  
* **Antifragile Feedback Loop** — stressors → metrics → self‑fine‑tuning under governance guard‑rails  
* **API‑Key Optional** — fully offline with local Φ‑2 (Ollama) or φ‑3b HF models when `OPENAI_API_KEY` is absent  

---

## 🚀 Quick Start (one‑liner)

```bash
curl -sL https://raw.githubusercontent.com/MontrealAI/AGI-Alpha-Agent-v0/main/deploy_live_alpha.sh | bash
```

The bootstrap script:

1. Installs Docker (and Ollama if needed).  
2. Clones **AGI‑Alpha‑Agent‑v0** & verifies Cosign signatures.  
3. Detects today’s best finance momentum alpha (`/runtime/best_alpha.json`).  
4. Starts Ray‑based agent cluster + trace‑graph UI on **localhost:8080**.  
5. Falls back to local Φ‑2 if OpenAI credentials are missing.

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    subgraph Security
        Spire[SPIRE<br/>Workload Attestations]
        Cosign[Cosign + Rekor]
    end
    subgraph DevSecOps
        GH[GitHub Actions] --> Cosign
        GH --> Syft[Syft SBOM]
    end
    subgraph Runtime
        AlphaMgr[AlphaManager<br/>(Ray Actor)]
        Finance[FinanceAgent]
        Policy[PolicyAgent]
        Biotech[BiotechAgent]
        Mfg[ManufacturingAgent]
        Meta[MetaEvolutionAgent]
        UI[Trace‑Graph UI]
        Prom[Prometheus / Grafana]
    end
    Cosign -->|verified images| Runtime
    Spire --> Runtime
    Finance -->|A2A| AlphaMgr
    Policy -->|A2A| AlphaMgr
    Biotech -->|A2A| AlphaMgr
    Mfg -->|A2A| AlphaMgr
    Meta -->|A2A| AlphaMgr
    AlphaMgr --> Prom
    UI --- AlphaMgr
```

---

## 🎮 Demo Showcase (`alpha_factory_v1/demos/`)

| Demo | Emoji | Essence | Command |
|------|-------|---------|---------|
| AIGA Meta Evolution | 🧬 | Evolutionary code‑gen lab where agents *write* & unit‑test new agents. | `docker compose -f demos/docker-compose.aiga_meta.yml up` |
| Era of Experience | 🏛️ | Narrative RAG that fuses personal memory graph into tutoring agent. | `docker compose -f demos/docker-compose.era.yml up` |
| Finance Alpha | 💹 | Live factor momentum + risk parity; outputs JSON trade blotter. | `docker compose -f demos/docker-compose.finance.yml up` |
| Macro Sentinel | 🌐 | Macro horizon scanner + CTA hedge back‑tester. | `docker compose -f demos/docker-compose.macro.yml up` |
| MuZero Planning | ♟️ | MuZero planning against synthetic markets to probe reasoning depth. | `docker compose -f demos/docker-compose.muzero.yml up` |
| Self‑Healing Repo | 🩹 | Watches GitHub webhooks; patches failing tests via Agents SDK. | `docker compose -f demos/docker-compose.selfheal.yml up` |

---

## 🔎 Vertical Agents Deep Dive

| Agent | Core Model | Connectors | Algoritmic Edge | Governance Guard‑rails |
|-------|------------|------------|-----------------|------------------------|
| FinanceAgent | GPT‑4o / Φ‑2 | Polygon, Binance, DEX Screener, FRED | Factor momentum, risk parity, Monte‑Carlo VAR | Max VAR, explain‑before‑trade |
| BiotechAgent | GPT‑4o | Ensembl REST, PubChem, PDB | Protein‑target mapping, CRISPR guide scoring | 3‑layer bio‑safety filter |
| PolicyAgent | GPT‑4o | govinfo.gov, RegHub, Global‑Voices | Bill lineage graph, lobbying pathfinder | Conflict‑of‑interest log, bias eval |
| ManufacturingAgent | GPT‑4o | OPC‑UA, MQTT, CSV | OR‑Tools MILP scheduling, downtime root‑cause | Safety FMEA, SLA violation flag |
| MetaEvolutionAgent | GPT‑4o | GitHub API, HuggingFace | Genetic programming, auto‑unit test harness | SBOM diff + Cosign gate |

---

## 🛡️ Security & Compliance Stack

```
[ Dev Laptop ]
      |
      v
[ Cosign‑signed image ] → [ Rekor log ] → [ K8s Cluster ]
      |                         |
      +── verified by ──────────+
      |
 SPIFFE ID ↔ mTLS ↔ Agents
```

* **Zero‑Trust** — Each container gets a SPIFFE ID; all RPCs mTLS.  
* **Supply‑Chain** — Syft SBOM → GitHub Release; Cosign signature verified at boot.  
* **Model‑Graded Evaluations** — nightly bias / safety evals with OpenAI evals framework.  
* **Auditability** — Hash of every A2A message & prompt stored (BLAKE3) and query‑able.

---

## 🛠️ Developer Workflow

```bash
# Run CI locally
make test            # pytest + red‑team prompts
make eval            # model‑graded eval suite

# Hot‑reload orchestrator
docker compose exec orchestrator reflex run --reload

# Produce SBOM & sign image
make sbom && make sign
```

### Offline / Local LLM

```bash
export LLM_ENDPOINT=http://localhost:11434
export LLM_MODEL=phi
```

---

## 📜 License

**MIT** © 2025 Montreal.AI

---

<p align="center"><sub>α‑Factory v1 — Out‑learn · Out‑think · Out‑design · Out‑strategise · Out‑execute</sub></p>
