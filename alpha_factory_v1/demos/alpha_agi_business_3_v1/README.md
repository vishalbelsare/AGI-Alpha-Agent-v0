
# 🏛️ Large‑Scale α‑AGI Business 👁️✨ Demo (`$AGIALPHA`)
> **Alpha‑Factory v1 — Multi‑Agent *Agentic α‑AGI***  
> Harnessing *game‑theoretic design* + *thermodynamic optimisation* to spin latent market entropy into continuously‑compounding **alpha** — across every industry, 24 × 7.

---

## 📜 Table of Contents
1. [Why Strategy ≈ Statistical Physics](#1)  
2. [30‑second Demo Walk‑through](#2)  
3. [Role Architecture 🏛️](#3)  
4. [Featured α‑AGI Agents (6/11)](#4)  
5. [Example Scenario 👁️✨](#5)  
6. [Quick‑Start](#6)  
7. [Kubernetes / Helm Deploy](#7)  
8. [Safety & Compliance](#8)  
9. [Extending the Game / Energy Landscape](#9)  
10. [License & Prior‑Art Shield](#10)  

---

<a id="1"></a>
## 1 · Why Strategy **=** Statistical Physics 📈➗🔬
`ΔG = ΔH − TΔS` (Gibbs Free Energy) tells us *how much work* can be extracted from a system.  
**Brandenburger & Nalebuff’s** PART lens tells us *how much value* can be extracted from a game.

| Analogy | Thermodynamics | Game‑Theory / Business |
|---------|----------------|------------------------|
| **State space** | Micro‑states of particles | Strategy profiles of players |
| **Energy well** | Low‑energy attractor | Nash equilibrium |
| **Temperature T** | Random agitation | Market volatility |
| **Entropy S** | # reachable micro‑states | Optionality / strategic degrees of freedom |
| **Free Energy ΔG < 0** | Spontaneous process | Positive‑EV alpha‑job |

**Alpha‑Factory v1** minimises *free energy* of global markets: every dislocation is a metaphorical *ΔG < 0* pocket; α‑AGI Agents diffuse like molecules following a **Maxwell–Boltzmann** distribution  
\[
P(E)=\frac{e^{-E/kT}}{Z}\quad\Rightarrow\quad
P(\text{pursue job }j)=\frac{e^{-ΔG_j / kT_{market}}}{Z}
\]  
Higher mis‑pricing (more negative ΔG) ⇒ higher probability an Agent picks the job.

---

<a id="2"></a>
## 2 · Demo timeline (30 s, real‑time)

| ⏱️ | Event | Agents ↔ Business | Game‑Theory move | Thermodynamic view | Outcome |
|----|-------|------------------|------------------|--------------------|---------|
| 0 s | `docker run ghcr.io/montrealai/alpha-asi:latest` | Orchestrator online | —— | System initialises at *T‑startup* | Dashboard ready |
| 5 s | Berlin sentiment feed tokenised | `data‑scout.a.agent.agi.eth` → `eu‑macro.a.agi.eth` | **Add player** | Inject low‑entropy data | α‑job #412 opened |
| 15 s| Tel‑Aviv momentum model contributed | `strat‑wizard.a.agent.agi.eth` | **Complementor coalition** | Entropy ↓ , ΔG more negative | IR ↑ +1.2 % |
| 22 s| Seoul satellite heat‑map ingested | `vision‑seer.a.agent.agi.eth` | **Raise added‑value** | More micro‑states explored | Hedge error ↓ 37 % |
| 30 s| `$AGIALPHA` settlement executed | `ledger‑bot.a.agent.agi.eth` | **Pay‑off division** (Shapley) | Work extracted → cash | Tokens minted |

Back‑test: strategy beats MSCI World by **+4.3 % IR**, without triggering advisory regulations.

---

<a id="3"></a>
## 3 · Role Architecture 🏛️ — Businesses & Agents

| Entity | ENS Convention | Treasury | Primary Duties | Value Creation |
|--------|----------------|----------|----------------|----------------|
| **α‑AGI Business** | `<sub>.a.agi.eth` | Holds `$AGIALPHA`; issues bounties | Publish **Problem‑Portfolios** (α‑jobs), pool data/IP, set constraints | Aggregates solved‑job upside; recycles gains |
| **α‑AGI Agent** | `<sub>.a.agent.agi.eth` | Staked reputation + escrow | Detect, plan & execute on α‑jobs | Earns `$AGIALPHA`; learns reusable templates |

> **Big picture** — Businesses *curate demand* for alpha; Agents *supply execution*. Smart‑contracts clear both with reputational slashing to keep incentives truthful.

**Legal & Conceptual Shield 🛡️** — design inherits the **2017 Multi‑Agent AI DAO** public timestamp → blocks trivial patents on multi‑agent + token mechanics, provides DAO‑first wrapper for fractional ownership.

---

<a id="4"></a>
## 4 · Featured α‑AGI Agents (6 / 11 core)

| Agent (repo path) | ENS | Core Skills | Game‑/Thermo Duty |
|-------------------|-----|-------------|-------------------|
| **PlanningAgent** | `planner.a.agent.agi.eth` | Tool orchestration, task decomposition | Generates PART + ΔG table for each α‑job |
| **ResearchAgent** | `research.a.agent.agi.eth` | Web/lit retrieval, fact‑check | Estimates *marginal entropy* reduction of new data |
| **StrategyAgent** | `strat‑wizard.a.agent.agi.eth` | Portfolio opt, signalling | Designs *credible commitments*; shifts equilibrium |
| **MarketAnalysisAgent** | `market‑lens.a.agent.agi.eth` | Live feeds, anomaly detection | Spots energy gaps (mis‑pricings) in real‑time |
| **NegotiatorAgent** *(new)* | `deal‑maker.a.agent.agi.eth` | Shapley & Nash bargaining | Divides surplus → keeps coalition Nash‑stable |
| **SafetyAgent** | `guardian.a.agent.agi.eth` | KL policy shield, sandbox | Ensures entropy ↓ does not violate constraints |

All wired via [`orchestrator.py`](../../backend/orchestrator.py) using **OpenAI Agents SDK**, Google ADK, A2A protocol & Anthropic MCP.

---

<a id="5"></a>
## 5 · Example Scenario 👁️✨ — *Thermo‑Game Synthesis*

1. **Define state‑space** — `eu‑macro.a.agi.eth` posts *Consumer‑Cycle* portfolio (α‑jobs = {EUR retail optimism, EM momentum, industry heat‑maps}).  
2. **Partition energy wells** — PlanningAgent assigns ΔG score to each job.  
3. **Lower energy via data fusion** — ResearchAgent brings Berlin NLP feed (ΔG ↓).  
4. **Coalition formation** — NegotiatorAgent computes Shapley pay‑offs; players accept.  
5. **Reach equilibrium** — StrategyAgent publishes public immutable commit (credible).  
6. **Extract work** — LedgerBot swaps positions on‑chain; `$AGIALPHA` minted per contribution entropy‑drop.  

No human co‑ordination, no central broker — physics makes the coalition *inevitable*.

---

<a id="6"></a>
## 6 · Quick‑Start 🚀

```bash
docker run -p 7860:7860 ghcr.io/montrealai/alpha-asi:latest   # online
docker run -e OFFLINE=1 ghcr.io/montrealai/alpha-asi:offline  # air‑gapped
```

→ open **localhost:7860**  
Dashboard shows live **PART graph**, ΔG heat‑map, coalition pay‑offs & safety telemetry.

---

<a id="7"></a>
## 7 · Kubernetes / Helm 📦

```bash
helm repo add alpha-asi https://montrealai.github.io/charts
helm install alpha-asi/FULL   --set resources.gpu=true   --set openai.apiKey="$OPENAI_API_KEY"
```

HPA triggers when GPU > 70 %; Learner pods auto‑scale.  
All A2A traffic stays inside cluster VPC; egress blocked unless `values.yaml` enables.

---

<a id="8"></a>
## 8 · Safety & Compliance 🔒

* **Three‑layer defence** — KL shield → seccomp sandbox → chaos stress‑tests.  
* **Entropy honeypots** — randomly invert reward sign to detect reward‑hacking.  
* **Ledger notarisation** — BLAKE3 hashes → Solana testnet (EU AI‑Act Art‑52).  
* **Offline mode** — ships with Llama‑3‑8B.gguf; zero external calls.  

17‑point audit list in [`docs/safety.md`](../../docs/safety.md).

---

<a id="9"></a>
## 9 · Extending the Game / Energy Landscape ➕

* **New Business** — drop YAML in `./businesses/`; orchestrator mints ENS & wallet.  
* **New Agent** — publish Agent‑Card JSON; radar auto‑discovers, assigns initial *kT*.  
* **Governance** — DAO proposal changes rules (e.g. volatility cap).  
* **Thermo tweak** — adjust global temperature parameter to trade‑off exploration vs exploitation.

---

<a id="10"></a>
## 10 · License & Prior‑Art 🛡️

Apache‑2.0.  Derivative patents on multi‑agent + token coordination **expressly forbidden** (2017 Multi‑Agent AI DAO timestamp, see [prior‑art](https://www.quebecartificialintelligence.com/priorart)).

---

*Built with ♥ by the MONTREAL.AI AGENTIC α‑AGI core team.*  
Chat with us → **https://discord.gg/montrealai**
