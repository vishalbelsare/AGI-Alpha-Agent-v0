
<!-- 2025‑04‑23 — α‑Factory v1 README (Extended Edition) -->

<h1 align="center">AGI‑Alpha‑Agent‑v0</h1>
<h3 align="center">CA: tWKHzXd5PRmxTF5cMfJkm2Ua3TcjwNNoSRUqx6Apump</h3>
<h2 align="center">AGI ALPHA AGENT (<a href="https://app.ens.domains/name/alpha.agent.agi.eth">ALPHA.AGENT.AGI.Eth</a>) ⚡ Powered by $AGIALPHA</h2>
<p align="center"><strong>Seize the Alpha. Transform the world.</strong></p>

> **Vincent Boucher** — President of <a href="https://www.montreal.ai">MONTREAL.AI</a> — dominated the <a href="https://web.archive.org/web/20170929214241/https://gym.openai.com/read-only.html">OpenAI Gym</a> in 2016 and authored the landmark <a href="https://www.quebecartificialintelligence.com/priorart">“Multi‑Agent AI DAO”</a> blueprint in 2017.  
> That heritage now powers **AGENTIC α‑AGI 👁️✨** — a cross‑industry **Alpha Factory** that Out‑learns · Out‑thinks · Out‑designs · Out‑strategises · Out‑executes.

<p align="center">
  <a href="https://htmlpreview.github.io/?https://raw.githubusercontent.com/MontrealAI/AGI-Alpha-Agent-v0/main/deploy_sovereign_agentic_agialpha_agent_v0.html"><strong>🔱 ∞ AGENTIC ALPHA EXPLORER (INFINITE MODE) ∞ 🔱</strong></a>
</p>

---

## ✨ Why α‑Factory?

```
┌───────────────────────────────────────────┐
│   α‑Factory  ➜   Multi‑Agent  α‑Signal   │
│                 Discovery & Conversion   │
│ ──────────────────────────────────────── │
│   Finance   •   Policy   •   Biotech    │
│   Manufacturing   •   Meta‑Evolution    │
└───────────────────────────────────────────┘
```

* **Best‑in‑class agent stack** — OpenAI Agents SDK, Google ADK, Anthropic MCP, A2A messaging  
* **Plug‑and‑profit adapters** — live markets, genomics, legislative DBs, factory OPC‑UA streams  
* **Reg‑grade security** — SPIFFE + Cosign + SBOM + model‑graded evals  
* **Antifragile learning loop** — stressors → metrics → self‑fine‑tuning  
* **Offline‑friendly** — runs fully‑air‑gapped via local Φ‑2/φ‑3b models if no `OPENAI_API_KEY`  

---

## 🚀 Quick Start — one command

```bash
curl -sL https://raw.githubusercontent.com/MontrealAI/AGI-Alpha-Agent-v0/main/deploy_live_alpha.sh | bash
```

_No API key? The launcher auto‑installs Ollama + Φ‑2 and disables network calls._

---

## 🏗️ System Diagram

```text
flowchart TD
    subgraph Infra
        Spire[SPIRE Server]
        GHCR[Signed Images (GHCR)]
        Prom[Prometheus]
    end
    subgraph Runtime
        AM[AlphaManager<br>(Ray Actor)]
        FA[FinanceAgent]
        PA[PolicyAgent]
        BA[BiotechAgent]
        MA[ManufacturingAgent]
        ME[MetaEvolutionAgent]
    end
    Spire --> Runtime
    FA -->|A2A| AM
    PA -->|A2A| AM
    BA -->|A2A| AM
    MA -->|A2A| AM
    ME -->|A2A| AM
    AM --> Prom
    classDef default fill:#f9f9ff,stroke:#333,stroke-width:1px;
```

---

## 🎮 Demo Showcase (`alpha_factory_v1/demos/`)

| Demo | Emoji | What it proves | How to run |
|------|-------|----------------|------------|
| **AIGA Meta Evolution** | 🧬 | Agents that **write new agents** — evolutionary code‑gen with self‑evaluation. | `docker compose -f demos/docker-compose.aiga_meta.yml up` |
| **Era of Experience** | 🏛️ | Narrative engine that blends user memories into chain‑of‑thought for personalised tutoring. | `docker compose -f demos/docker-compose.era.yml up` |
| **Finance Alpha** | 💹 | Live factor‑momentum model with risk parity and on‑chain execution stub. | `docker compose -f demos/docker-compose.finance.yml up` |
| **Macro Sentinel** | 🌐 | Macro‑economic horizon scanner + hedge back‑tester (CTA style). | `docker compose -f demos/docker-compose.macro.yml up` |
| **MuZero Planning** | ♟️ | Stress‑test the reasoning loop with MuZero vs synthetic markets. | `docker compose -f demos/docker-compose.muzero.yml up` |
| **Self‑Healing Repo** | 🩹 | Agent watches Git events and auto‑patches failing tests using OpenAI Agents SDK. | `docker compose -f demos/docker-compose.selfheal.yml up` |

<img src="https://raw.githubusercontent.com/MontrealAI/brand-assets/main/demo-collage.png" alt="Demo collage" width="100%"/>

---

## 🧬 Vertical Agents — Deep Dive

| Agent | Core Model ▼ | Data Connectors | Governance Guard‑rails |
|-------|--------------|-----------------|------------------------|
| **FinanceAgent** | GPT‑4o · Φ‑2 | Polygon, Binance, FRED, DEX Screener | VaR cap, max drawdown, explain‑before‑trade |
| **BiotechAgent** | GPT‑4o | Ensembl REST, PubChem, PDB | 3‑layer bio‑safety filter, CRISPR off‑target check |
| **PolicyAgent** | GPT‑4o | govinfo.gov, RegHub, Global‑Voices | Conflict‑of‑interest log, bias eval |
| **ManufacturingAgent** | GPT‑4o | OPC‑UA, csv, IoT MQTT | Safety FMEA, downtime SLA |
| **MetaEvolutionAgent** | GPT‑4o | GitHub API, Hugging Face | Unit‑test pass gate, SBOM diff |

---

## 🛡️ Security & Compliance

* **Zero‑Trust IDs** — Every container gets a SPIFFE identity signed by SPIRE.  
* **Cosign** — Images are signed; the bootstrap script refuses unsigned layers.  
* **SBOM** — Syft auto‑generates JSON CycloneDX; uploaded to GH Releases.  
* **Model‑Graded Evals** — Bias, hate, defamation tests run nightly (`make eval`).  
* **Audit API** — All A2A messages & prompts are hashed (BLAKE3) and query‑able.

---

## 🛠️ Developer Guide

```bash
# 1️⃣  Run full test & eval suite
docker compose exec orchestrator pytest -q && make eval

# 2️⃣  Hot‑reload backend while hacking
docker compose exec orchestrator reflex run --reload

# 3️⃣  Generate SBOM + sign image
make sbom && cosign sign --key cosign.key ghcr.io/montrealai/alphafactory_pro:latest
```

### Local LLM override

```bash
export LLM_ENDPOINT=http://localhost:11434 # e.g. Ollama
export LLM_MODEL=phi
```

Set those vars and every agent swaps to local inference.

---

## 📜 License

MIT © 2025 MONTREAL.AI

---

<p align="center"><sub>α‑Factory v1 • Out‑learn · Out‑think · Out‑design · Out‑strategise · Out‑execute</sub></p>
