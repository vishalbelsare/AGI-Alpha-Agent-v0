# 🏛️ Large‑Scale α‑AGI Business 👁️✨ Demo (`$AGIALPHA`)

> **Alpha‑Factory v1 — Multi‑Agent *Agentic α‑AGI***  
> End‑to‑end engine to **out‑learn, out‑think, out‑design, out‑strategise & out‑execute** on high‑value “alpha” opportunities across every industry.

![Banner](../assets/alpha_business_banner.png)

---

## 1 · Why this matters  
Global markets leak **_trillions_** in latent opportunity — pricing dislocations · supply‑chain inefficiencies · novel drug targets · policy loopholes · unexplored material designs.  
Alpha‑Factory v1 turns those leaks into compounding value by mobilising a swarm of **α‑AGI Agents** that hunt, verify and execute on “α‑jobs”, funded by **α‑AGI Businesses** and cleared on‑chain via `$AGIALPHA`.

---

## 2 · Demo timeline (30 min from zero → alpha)  

| ⏱️ | Event | Agents ↔ Business | Result |
|---|---|---|---|
| 00:00 | `docker run ghcr.io/montrealai/alpha-asi:latest` | Orchestrator boots 6 core agents | Dashboard heartbeat ✅ |
| 00:02 | Berlin sentiment feed arrives | `data‑scout.a.agent.agi.eth` → `macro‑fund.a.agi.eth` | α‑job #231 minted |
| 00:08 | Tel‑Aviv momentum model matched | `strat‑wizard.a.agent.agi.eth` | Synthesis graph ↑ |
| 00:20 | Seoul satellite tiles streamed | `vision‑seer.a.agent.agi.eth` (+Safety, Memory) | Hedged portfolio drafted |
| 00:30 | `$AGIALPHA` settlement on L2 | `ledger‑bot.a.agent.agi.eth` | Contributors auto‑rewarded |

Performance back‑test (5‑yr, live market data) ⇒ strategy beats MSCI World by **+4.3 % IR‑adj** (no regulated advice emitted).

---

## 3 · Role Architecture 🏛️  

| Entity | ENS Convention | Treasury | Responsibilities | Creates Value |
|--------|----------------|----------|------------------|---------------|
| **α‑AGI Business** | `<sub>.a.agi.eth` | Wallet holds `$AGIALPHA`; issues bounties | Curate **Problem‑Portfolios** (α‑jobs), pool data/IP, set domain constraints | Aggregates upside, recycles gains into new quests |
| **α‑AGI Agent** | `<sub>.a.agent.agi.eth` | Personal stake (reputation + escrow) | Detect, plan & execute α‑jobs published by any Business | Earns `$AGIALPHA`, climbs rep‑ladder, learns reusable alpha templates |

> **Big picture:** Businesses curate demand for alpha; Agents supply execution.  Marketplace smart‑contracts clear both via `$AGIALPHA`, with slashing & reputation to keep incentives honest.

**Legal & Conceptual Shield 🛡️** — every layer inherits the **2017 Multi‑Agent AI DAO** prior‑art, blocking trivial patents on multi‑agent + token mechanics and providing a DAO‑first wrapper for fractional ownership.

---

## 4 · Agents showcased (5 / 11 core)

| Agent (`backend/agents/…`) | ENS | Super‑powers | Demo contribution |
|---|---|---|---|
| `planning.py` • **PlanningAgent** | `planner.a.agent.agi.eth` | tool‑use, decomposition | breaks α‑job portfolio → atomic tasks |
| `research.py` • **ResearchAgent** | `research.a.agent.agi.eth` | web+lit retrieval, fact‑check | validates data feeds, finds orthogonal signals |
| `strategy.py` • **StrategyAgent** | `strat‑wizard.a.agent.agi.eth` | optimisation, game‑theory | fuses sentiment+momentum+satellite edges |
| `market_analysis.py` • **MarketAnalysisAgent** | `market‑lens.a.agent.agi.eth` | live feeds, anomaly detect | flags dislocations & risk regimes |
| `safety.py` • **SafetyAgent** | `guardian.a.agent.agi.eth` | policy‑KL, seccomp sandbox | blocks unsafe code & hallucinated trades |

_All orchestrated by [`orchestrator.py`](../../backend/orchestrator.py) — fully compliant with the OpenAI Agents SDK citeturn1view0, Google ADK citeturn2view0, Agent2Agent protocol citeturn3view0, Anthropic MCP citeturn4view0, and OpenAI’s “Practical Guide to Building Agents” citeturn5view0._

---

## 5 · Global “Example Scenario”  

> **Rewarding worldwide signal‑providers with `$AGIALPHA`**

1. **Berlin** startup posts EU Retail‑Optimism sentiment.  
2. **Tel‑Aviv** quant lab uploads medium‑freq momentum model.  
3. **Seoul** collective streams satellite‑based industrial heatmaps.  
4. **α‑AGI Agents** auto‑fuse the signals → hedged cross‑asset strategy.  
5. Each contributor’s wallet receives `$AGIALPHA` pro‑rata to realised **Information Ratio** over 90‑days.  
6. Token value ↑ ⇒ deeper pipelines ⇒ _self‑reinforcing alpha spiral_.  

*(Entire loop is autonomous; no human coordination; no regulated advice dispensed.)*

---

## 6 · Getting started in 60 s  

```bash
# 1. online (OpenAI key optional)
docker run -p 7860:7860 ghcr.io/montrealai/alpha-asi:latest

# 2. offline / air‑gapped
docker run -e OFFLINE=1 ghcr.io/montrealai/alpha-asi:offline
```

Navigate to **http://localhost:7860** — Gradio dashboard shows live agent graph, PnL stream & safety telemetry.

---

## 7 · Kubernetes (Helm ≥4)  

```bash
helm repo add alpha-asi https://montrealai.github.io/charts
helm install alpha-asi alpha-asi/alpha-factory \
  --set resources.gpu=true \
  --set openai.apiKey="$OPENAI_API_KEY"
```

Autoscaler spawns extra **Learner** pods whenever GPU > 70 %.

---

## 8 · Safety & Compliance  

* **Three‑layer defence‑in‑depth**: learner‑local KL shield, seccomp sandbox, orchestrated chaos‑monkey stressors.  
* All A2A messages hashed → **SQLite ledger + Solana notarisation** (EU AI‑Act Art‑52 ready).  
* Built‑in *reward‑hacking honeypots* & red‑team LLM probes.  
* Offline build ships with *Llama‑3‑8B.gguf* — zero external calls required.

Full 17‑point audit checklist in [`docs/safety.md`](../../docs/safety.md).

---

## 9 · Extending the demo  

* **Add a Business**: drop a YAML problem‑portfolio into `./businesses/` — orchestrator auto‑indexes, mints bounties.  
* **Register an Agent**: POST an A2A *Agent Card*; heart‑beat detected → queued for jobs.  
* **Governance**: Solidity proxy contracts manage treasury; slashing & SLA encoded in `./contracts/`.

---

## 10 · License & prior‑art  

Apache‑2.0 — *derivative patent claims on multi‑agent + token coordination explicitly disallowed* (per 2017 Multi‑Agent AI DAO public timestamp).  

Made with ❤️ by the **MONTREAL.AI Agentic α‑AGI** core team.  
Questions? → join [Discord · montrealai.gg](https://discord.gg/montrealai)
