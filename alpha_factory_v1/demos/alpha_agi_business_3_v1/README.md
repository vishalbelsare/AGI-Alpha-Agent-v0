
# 🏛️ Large‑Scale α‑AGI Business 👁️✨ Demo (`$AGIALPHA`)

> **Alpha‑Factory v1 — Multi‑Agent *Agentic α‑AGI***  
> Turning *game‑theoretic strategy* into continuously‑compounding **alpha** streams across every industry.

---

## 1 · Why “The Right Game” matters  
Adam Brandenburger & Barry Nalebuff showed that fortunes are made (or lost) by *redesigning* the game, not merely playing harder.  
**Alpha‑Factory v1** operationalises their four lenses—**Players • Added‑Value • Rules • Tactics (PART)**—with autonomous α‑AGI Agents:

| Lens | What it means in game‑theory | α‑AGI implementation |
|------|-----------------------------|----------------------|
| **Players** | Identify *who* can change pay‑offs | Agents & Businesses represented by ENS: `<sub>.a.agent.agi.eth`, `<sub>.a.agi.eth>` |
| **Added‑Value** | Measure each player’s indispensable contribution | Value‑at‑Contribution oracle computes marginal Information‑Ratio |
| **Rules** | Re‑write constraints to unlock new surplus | On‑chain contracts upgradeable via DAO proposals |
| **Tactics** | Sequencing & signalling moves | StrategyAgent auto‑generates *credible‑commitment* messages on A2A bus |

Game‑theoretic reframing ⇒ larger “pie” before division.  
`$AGIALPHA` token clears side‑payments so *everyone* shares in upside.

---

## 2 · Demo timeline (30 seconds)

| ⏱️ | What happens | Agents ↔ Business | Game‑theory angle | Outcome |
|----|--------------|------------------|------------------|---------|
| 00 s | `docker run ghcr.io/montrealai/alpha-asi:latest` | Orchestrator boots | —— | Dashboard live |
| 05 s | Berlin Sentiment feed tokenised | `data‑scout.a.agent.agi.eth` → `eu‑macro.a.agi.eth` | *Expand players* (bring new contributor) | α‑job #212 listed |
| 15 s | Tel‑Aviv Momentum model joins | `strat‑wizard.a.agent.agi.eth` | *Complementor join* increases joint IR | Synergy graph +22 % |
| 22 s | Seoul Satellite analytics added | `vision‑seer.a.agent.agi.eth` | *Increase added‑value* | Hedge error ↓ 37 % |
| 30 s | `$AGIALPHA` settlement | `ledger‑bot.a.agent.agi.eth` → all | *Fair division* keeps coalition stable | Tokens distributed |

MSCI‑World back‑test +4.3 % IR lift vs legacy quant stack.

---

## 3 · Role Architecture 🏛️

| Entity | ENS convention | Treasury / Funding | Primary responsibilities | How it creates value |
|--------|----------------|--------------------|--------------------------|----------------------|
| **α‑AGI Business** | `<sub>.a.agi.eth` | Holds `$AGIALPHA`; can issue bounties | Publish *Problem‑Portfolios* (α‑jobs), pool data/IP, set constraints | Aggregates upside from solved portfolios; reinvests proceeds |
| **α‑AGI Agent** | `<sub>.a.agent.agi.eth` | Owns stake (reputation + escrow) | Detect, plan & execute on published α‑jobs | Earns `$AGIALPHA`, climbs rep ladder, re‑uses alpha templates |

*Businesses curate demand; Agents supply execution. Marketplace smart‑contracts clear both sides with reputation‑weighted pay‑offs.*  

Legal Shield 🛡️ — every layer inherits the **2017 Multi‑Agent AI DAO** prior‑art, blocking trivial patents & giving a DAO‑first wrapper for fractional ownership.

---

## 4 · Agents showcased (6 of 11)

| Agent (repo path) | ENS | Core skills | Game‑theory duty |
|-------------------|-----|-------------|------------------|
| **PlanningAgent** | `planner.a.agent.agi.eth` | tool composition, decomposition | Generates PART table for each new α‑job |
| **ResearchAgent** | `research.a.agent.agi.eth` | web & lit retrieval | Quantifies *added‑value* of external datasets |
| **StrategyAgent** | `strat‑wizard.a.agent.agi.eth` | portfolio opt, signalling | Designs *credible commitments* → locks in coalition |
| **MarketAnalysisAgent** | `market‑lens.a.agent.agi.eth` | live feeds, anomaly det. | Detects pay‑off shocks, advises rule tweaks |
| **NegotiatorAgent** *(new)* | `deal‑maker.a.agent.agi.eth` | Shapley & Nash bargaining | Computes payout splits, prevents defection |
| **SafetyAgent** | `guardian.a.agent.agi.eth` | policy KL, sandbox | Keeps tactics within regulatory & ethical bounds |

All orchestrated via [`backend/orchestrator.py`](../../backend/orchestrator.py) using **OpenAI Agents SDK**, Google ADK & A2A protocol.

---

## 5 · Example scenario 👁️✨ (walk‑through)

1. **Define the game** — `eu‑macro.a.agi.eth` creates *Consumer‑Cycle Portfolio* α‑job.  
2. **Change the players** — NegotiatorAgent invites Berlin, Tel‑Aviv, Seoul sources.  
3. **Change added‑values** — ResearchAgent benchmarks each signal → sets marginal IR weights.  
4. **Change the rules** — StrategyAgent proposes capped‑loss rule; DAO vote passes (L2 roll‑up).  
5. **Tactics** — Public “skin‑in‑the‑game” commit broadcast; MarketAnalysisAgent monitors fulfilment.  
6. **Pay‑offs settle** — `$AGIALPHA` auto‑escrow distributes per Shapley value every 24 h.

Result: self‑reinforcing, game‑theoretically stable alpha machine—no human co‑ordination required.

---

## 6 · Quick‑start

```bash
# online (OpenAI key optional)
docker run -p 7860:7860 ghcr.io/montrealai/alpha-asi:latest

# offline / air‑gapped
docker run -e OFFLINE=1 ghcr.io/montrealai/alpha-asi:offline
```

Open <http://localhost:7860> → Gradio dashboard shows live PART graph, coalition pay‑offs, safety telemetry.

---

## 7 · Deploy to Kubernetes

```bash
helm repo add alpha-asi https://montrealai.github.io/charts
helm install alpha-asi/alpha-factory \
  --set resources.gpu=true \
  --set openai.apiKey="$OPENAI_API_KEY"
```

HPA scales Learner pods when GPU > 70 %.

---

## 8 · Safety & Compliance

* Three‑layer defence (KL shield → seccomp sandbox → orchestrated stress‑tests)  
* All A2A messages hashed → SQLite + Solana notarisation (EU AI‑Act ready)  
* Reward‑hacking honeypots & red‑team LLM probes built‑in  
* Offline mode ships with Llama‑3‑8B.gguf — no external calls

Full 17‑point audit list in [`docs/safety.md`](../../docs/safety.md).

---

## 9 · Extending the game

* **New Business** → drop YAML in `./businesses/`  
* **New Agent** → publish A2A Agent‑Card JSON; orchestrator auto‑discovers  
* **Governance** → upgrade‑safe Solidity proxy; slash conditions coded in `./contracts/`

---

## 10 · License & prior art

Apache‑2.0.  Derivative patent claims on multi‑agent + token co‑ordination expressly disallowed (2017 Multi‑Agent AI DAO timestamp).

---

*Built with ♥ by the MONTREAL.AI AGENTIC α‑AGI core team.*  
Questions? → join **discord.gg/alpha‑factory**

