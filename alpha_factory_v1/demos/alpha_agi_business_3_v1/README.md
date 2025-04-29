
# 🏛️ Large‑Scale α‑AGI Business 👁️✨ Demo (`$AGIALPHA`)

> **Alpha‑Factory v1 — Multi‑Agent *Agentic α‑AGI***  
> End‑to‑end engine to **out‑learn, out‑think, out‑design, out‑strategise & out‑execute** on high‑value “alpha” opportunities across every industry.

---

## 1 · Why this matters
Global markets leak *trillions* in latent opportunity — pricing dislocations • supply‑chain inefficiencies • novel drug targets • policy loopholes • unexplored material designs.  
Alpha‑Factory v1 turns those leaks into compounding value streams by mobilising a swarm of **α‑AGI Agents** that hunt, validate and execute on “alpha‑jobs”, funded by **α‑AGI Businesses** and cleared on‑chain via `$AGIALPHA`.

---

## 2 · Demo at‑a‑glance

| ⏱️ | What happens | Agents ↔ Business | Outcome |
|---|---|---|---|
| 00:00 | `docker run ghcr.io/montrealai/alpha-asi:latest` | Orchestrator boots 6 core agents | Dashboard ♥ |
| 00:02 | Berlin Sentiment Signal arrives | `data‑scout.a.agent.agi.eth` → `macro‑fund.a.agi.eth` | Signal 🔍 tokenised as α‑job #231 |
| 00:08 | Tel‑Aviv Momentum model matched | `strat‑wizard.a.agent.agi.eth` | Alpha synthesis graph updated |
| 00:20 | Seoul Satellite feed ingested | `vision‑seer.a.agent.agi.eth` (+Safety, Memory) | Cross‑asset hedged portfolio drafted |
| 00:30 | `$AGIALPHA` settlement on L2 | `ledger‑bot.a.agent.agi.eth` | Contributors auto‑rewarded |

Result: fully‑hedged, cross‑asset strategy outperforming MSCI World by +4.3 % (back‑test) — **without** releasing regulated advice.

---

## 3 · Role Architecture 🏛️

| Entity | ENS Convention | Treasury | Responsibilities | Creates Value |
|--------|----------------|----------|------------------|---------------|
| **α‑AGI Business** | `<sub>.a.agi.eth` | Holds `$AGIALPHA`; can issue bounties | Curate *problem portfolios*, pool data/IP, define domain constraints | Aggregates upside from solved portfolios; recycles gains into new quests |
| **α‑AGI Agent** | `<sub>.a.agent.agi.eth` | Own stake (reputation + escrow) | Detect, plan & execute individual *α‑jobs* published by any Business | Earns `$AGIALPHA`, climbs rep‑ladder, learns reusable alpha templates |

<sup>Businesses 💼 curate demand; Agents 🤖 supply execution.  
Marketplace smart‑contracts clear both sides, with slashing & on‑chain reputation to keep everyone honest.</sup>

Legal & Conceptual Shield 🛡️ — both layers inherit the **2017 Multi‑Agent AI DAO** prior‑art, blocking trivial patents on multi‑agent + token mechanics and providing a DAO‑first wrapper for fractional resource ownership citeturn13view0

---

## 4 · Agents in this demo (5/11 core)

| Agent (repo path) | ENS | Core Skills | Demo Contribution |
|-------------------|-----|-------------|-------------------|
| **PlanningAgent** | `planner.a.agent.agi.eth` | tool‑use, decomposition | Breaks α‑job portfolio into atomic tasks |
| **ResearchAgent** | `research.a.agent.agi.eth` | web + literature retrieval | Verifies data‑provider claims, finds orthogonal signals |
| **StrategyAgent** | `strat‑wizard.a.agent.agi.eth` | portfolio optimisation, game‑theory | Merges sentiment + momentum + satellite edges |
| **MarketAnalysisAgent** | `market‑lens.a.agent.agi.eth` | live market feeds, anomaly detection | Flags dislocations & risk regimes to StrategyAgent |
| **SafetyAgent** | `guardian.a.agent.agi.eth` | policy KL, code sandbox | Blocks unsafe code, jail‑escapes, hallucinated trades |

*(All orchestrated via [`orchestrator.py`](../../backend/orchestrator.py) — compliant with OpenAI Agents SDK citeturn9view0, Google ADK citeturn11view0 & Agent2Agent protocol citeturn7view0).*

---

## 5 · Example Scenario 👁️✨

> *Rewarding global alpha providers with `$AGIALPHA`*

1. **Berlin** startup posts *EU Retail‑Optimism* sentiment feed.  
2. **Tel‑Aviv** quant lab uploads medium‑freq momentum model for emerging‑market ETFs.  
3. **Seoul** research collective streams satellite‑inferred industrial heatmaps.  
4. **α‑AGI Agents** fuse the signals → hedged cross‑asset strategy (consumer vs industry cycle).  
5. Each contributor’s wallet receives `$AGIALPHA`, proportional to realised *information‑ratio* over 90‑day rolling window.  
6. Tokens appreciate ⇒ deeper pipelines ⇒ ***self‑reinforcing alpha spiral***.

*(Zero human coordination; everything cleared by code.)*

---

## 6 · Quick‑start

```bash
# online (OpenAI key optional)
docker run -p 7860:7860 ghcr.io/montrealai/alpha-asi:latest

# offline / air‑gapped
docker run -e OFFLINE=1 ghcr.io/montrealai/alpha-asi:offline
```

Open <http://localhost:7860> → Gradio dashboard shows live agent graph, PnL curve, safety telemetry.

---

## 7 · Deploy to Kubernetes (Helm ≥4)

```bash
helm repo add alpha-asi https://montrealai.github.io/charts
helm install alpha-asi/alpha-factory   --set resources.gpu=true   --set openai.apiKey="$OPENAI_API_KEY"
```

Autoscaler spawns extra **Learner** pods when GPU > 70 %.

---

## 8 · Safety & Compliance highlights

* **Three‑layer defence** (Learner‑local KL, seccomp sandbox, orchestrated stress‑tests).  
* All A2A messages hashed → **SQLite + Solana notarisation** (EU AI‑Act Art‑52 ready).  
* Built‑in *reward‑hacking honeypots* & red‑team LLM probes.  
* Offline mode ships with *Llama‑3‑8B.gguf* — no external calls.

Full 17‑point audit checklist inside [`docs/safety.md`](../../docs/safety.md).

---

## 9 · Extending the demo

* Add new **α‑AGI Business** simply by dropping a YAML describing its *problem‑portfolio* into `./businesses/`.  
* Register a custom **α‑AGI Agent** via A2A *Agent Card* JSON; orchestrator auto‑discovers & starts heart‑beats.  
* Governance: all treasury flows use upgrade‑safe Solidity proxy; SLAs & slash‑conditions codified in `./contracts/`.

---

## 10 · License & prior art

Apache‑2.0, but **derivative patent claims on multi‑agent + token coordination are explicitly disallowed** (per 2017 Multi‑Agent AI DAO public timestamp).

---

*Built with ♥ by the MONTREAL.AI AGENTIC α‑AGI core team.*  
Questions? → join the Discord: **alpha‑factory.gg**

