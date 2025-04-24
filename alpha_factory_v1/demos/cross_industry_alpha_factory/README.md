# Cross‑Industry **α‑Factory** Demo

> **Alpha‑Factory v1 👁️✨ — Multi‑Agent AGENTIC α‑AGI**
>
> *Out‑learn • Out‑think • Out‑design • Out‑strategise • Out‑execute*

---

## 1 · What this demo is
A **one‑command, production‑grade showcase** that spins up the Alpha‑Factory runtime plus **five flagship agents**—Finance, Biotech, Climate, Manufacturing, and Policy—under a hardened orchestrator.  It proves an end‑to‑end loop: **alpha discovery → automated execution → continuous self‑improvement** across industries.

The installer (`deploy_alpha_factory_cross_industry_demo.sh`) takes **≈ 8 min** on any modern Ubuntu machine and needs only Docker + Compose. If `OPENAI_API_KEY` is absent, it falls back to a bundled LLama‑3‑8B local model, guaranteeing the demo runs **offline**.

---

## 2 · Quick start
```bash
# 1. fetch the repo & run the script (sudo only if you’re not in the docker group)
chmod +x deploy_alpha_factory_cross_industry_demo.sh
./deploy_alpha_factory_cross_industry_demo.sh

# 2. open the local dashboards
http://localhost:9000   # Grafana – metrics & traces
http://localhost:7860   # α‑Factory trace‑graph UI
```
<i>Tip — re‑run the script at any time; it’s idempotent.</i>

---

## 3 · Architecture snapshot
```text
┌────────────────────────────────────────────────────────────────────────────┐
│  docker‑compose (alpha_factory network)                                   │
│                                                                            │
│  ┌─────────────┐        ┌───────────────┐                                   │
│  │  Grafana    │◄──────►│  Prometheus   │◄─ metrics from every container   │
│  └─────────────┘        └───────────────┘                                   │
│          ▲                         ▲                                        │
│          │                         │                                        │
│  ┌─────────────┐        ┌───────────────┐          ┌────────────────────┐  │
│  │ Trace‑Graph │◄──────►│ Orchestrator  │◄────────►│  Knowledge‑Hub     │  │
│  └─────────────┘        └───────────────┘          └────────────────────┘  │
│          ▲                         ▲                 (RAG + embeddings)     │
│          │   A2A / ADK / REST     │                                        │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │            Industry Agents  (side‑car adapters shown dimmed)          │  │
│  │                                                                       │  │
│  │  Finance        Biotech        Climate        Mfg.         Policy     │  │
│  │  ‾‾‾‾‾‾‾        ‾‾‾‾‾‾‾        ‾‾‾‾‾‾‾         ‾‾‾‾‾         ‾‾‾‾‾     │  │
│  │  • broker ◄───► • pubmed       • carbon       • opc‑ua      • govtrack │  │
│  │  • market data   crawler         API            bridge        API      │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```
*See `assets/diagram_architecture.vsdx` for an editable version.*

---

## 4 · Meet the 5 flagship agents
| Agent | Key skills | Value contribution | Learning signals |
|-------|------------|--------------------|------------------|
| **FinanceAgent** | Factor discovery, position sizing, risk guard (VaR/MDD) | Real‑time P&L via Alpaca or simulated broker | Reward = trade P&L, risk penalties |
| **BiotechAgent** | PubMed RAG, candidate molecule ranking | Short‑lists drug‑gene hypotheses | Reward = novelty‑weighted PubMed citations |
| **ClimateAgent** | Carbon intensity forecasting, policy impact analysis | Recommends carbon arbitrage opportunities | Reward = ∆CO₂ reduction × ROI |
| **ManufacturingAgent** | OR‑Tools scheduling, OPC‑UA shop‑floor bridge | Lowers makespan & energy cost | Reward = cost‑to‑produce savings |
| **PolicyAgent** | Bill tracking, impact simulation | Flags regulatory alpha & lobbying windows | Reward = sentiment‑adjusted passage probability |

Each agent **implements the same Adapter → Skill → Governance stack**, so new verticals plug‑in with minimal code.

---

## 5 · Why it matters
- **Automated learning loops** (Ray evaluator) fine‑tune rewards & prompts every 15 min → continuous improvement.
- **Uniform execution adapters** mean *any* industry gets live data + actuation parity.
- **DevSecOps hardening**: SBOM via Syft, cosign signatures, MCP policy engine.
- **Regulator ready**: ed25519 prompt signing, red‑team deny‑patterns, full audit trail in Grafana.
- **Antifragile**: chaos‑monkey container restarts are logged & trigger curriculum ramps per *AI‑GAs* pillar three citeturn1file0.

---

## 6 · Extending / hacking
1. **Add a new agent** → copy `backend/agents/template_agent.py`, implement three abstract methods, add ENV‑var to `.env`.
2. **Swap LLM** → set `OPENAI_API_BASE` to your endpoint or leave blank for local‑llm.
3. **Deploy to k8s** → run `make helm && helm install alpha-factory chart/` (charts included).

---

## 7 · Troubleshooting
| Symptom | Fix |
|---------|-----|
| Port 9000 already in use | set `DASH_PORT=9091` before running script |
| Orchestrator health‑check fails | `docker compose logs orchestrator` – check missing GPU drivers |
| Local‑llm pulls slowly | `docker pull ollama/ollama:latest` beforehand |

---

## 8 · References & inspiration
- **AI‑GAs** paradigm (Clune 2020) citeturn1file0
- **Era of Experience** vision (Sutton & Silver 2024) citeturn1file1
- **MuZero** planning archetype (Schrittwieser et al. 2020) citeturn1file2  
These ideas shaped the automated curriculum, continual evaluation, and model‑based search embedded here.

---

© 2025 Montreal.AI   Licensed under **MIT**
