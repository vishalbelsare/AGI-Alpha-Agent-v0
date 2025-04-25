
<!--
───────────────────────────────────────────────────────────────────────────────
 Alpha-Factory v1 👁️✨  ·  α-ASI World-Model Demo
───────────────────────────────────────────────────────────────────────────────
-->

<h1 align="center">
  Alpha-Factory v1 👁️✨<br>
  <sub>Multi-Agent AGENTIC α-AGI · World-Model Demo</sub>
</h1>

<p align="center">
  <em>Outlearn · Outthink · Outdesign · Outstrategize · Outexecute</em>
</p>

---

## 🚀 Why this demo matters

This repository shows **how** we get there — building on  
**`alpha_asi_world_model_demo.py`**, a fully-agentic prototype that

1. **Generates endless worlds** (POET-style 🗺️)  
2. **Learns a general policy** (MuZero-style 🧠)  
3. **Self-improves forever** via a constellation of cooperating agents 🤝  
4. Runs **locally, offline** – cloud LLMs are optional 🌐

Together these ingredients light the path to **α-ASI**, the *alpha* of artificial
super-intelligence.

---

## 🧩 Architectural Super-Snapshot

```mermaid
graph LR
  subgraph Orchestrator 🛰️
     O(Orchestrator)
  end
  subgraph Agents
     PL[Planning Agent]
     RS[Research Agent]
     ST[Strategy Agent]
     MK[Market Analysis Agent]
     CG[Code Gen Agent]
     SA[Safety Agent]
  end
  subgraph Engine
     ENV[POET Env Generator]
     LRN[MuZero Learner]
  end
  user[(User / API)]
  UI[Web UI 📈]
  
  user --REST/WS--> UI
  UI --A2A--> O
  O -->|cmd| ENV
  O -->|data| LRN
  ENV --> LRN
  LRN --metrics--> SA
  SA -. guard .-> O
  
  O <--strategy--> PL
  O <--insights--> RS
  O <--roadmap--> ST
  O <--alpha-signals--> MK
  O <--patches--> CG
```

*Drawn with :heart:&nbsp;in Mermaid – renders natively on GitHub.*

---

## 🤖 The starring agents (≥5 already wired‐in)

| Agent | Purpose | Typical Messages |
|-------|---------|------------------|
| **PlanningAgent** 🧭 | Breaks high-level goals into executable sub-plans. Optional LLM reasoning via OpenAI Agents SDK. | `{"goal":"navigate maze"}` → returns ordered tasks |
| **ResearchAgent** 🔍 | Scans papers, code & data to surface new techniques or fixes. | `{"topic":"better exploration"}` |
| **StrategyAgent** ♟️ | Chooses *which* worlds to tackle next for maximum generalisation. | `{"metrics":{…}}` |
| **MarketAnalysisAgent** 💹 | Looks for monetisable “alpha” opportunities across industries and feeds them back. | `{"request":"latest alpha"}` |
| **CodeGenAgent** 🛠️ | Safely patches the code-base (tests + lint) when new capabilities are required. | `{"diff":"…"}` |
| **SafetyAgent** 🛡️ | Monitors losses/NaNs & enforces constitutional-AI rules. | `{"loss":1234}` triggers halt |

> **Need more?** Drop a `.py` in `backend/agents/` – it auto-loads via the
> dynamic bootstrapper (A2A-compliant).

---

## 🏎️ Quick Start (30 seconds)

```bash
# ❶ Clone repo (or pull latest)
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/alpha_asi_world_model

# ❷ Install deps (CPU-only; GPU auto-detected)
python -m pip install -r requirements.txt  # tiny list, all OSS

# ❸ Launch!
python -m alpha_asi_world_model_demo --demo
```

Open <http://localhost:7860> – watch worlds generate and metrics stream in real-time.  
No API keys? No problem. Provide `OPENAI_API_KEY` env-var **only** if you want LLM
planning.

---

## 🐳 One-liner Docker

```bash
docker run --pull=always -p 7860:7860 ghcr.io/montrealai/alpha-asi-world-model:latest
```

Kubernetes fan?  
```bash
helm repo add alpha-asi https://montrealai.github.io/helm-charts
helm install asi alpha-asi/alpha-asi-world-model
```

---

## 🧪 CI + CD highlights

* Matrix tests (Py 3.10-3.12) & 100 % coverage gate  
* Ruff + Mypy for style & types  
* Trivy vulnerability scan on the final image  
* Helm chart lint & smoke-deploy (kind)  
* Optional **tag-push = automatic GHCR release**  

See `.github/workflows/ci.yml` for full glam ✨.

---

## 🔬 Reproducing research claims

| Paper / Talk | Feature in demo | Where |
|--------------|-----------------|-------|
| **MuZero** (Schrittwieser et al., 2019) | Representation + Dynamics + Prediction network | `MuZeroTiny` class |
| **POET** (Wang et al., 2019) | Open-ended env generation | `POETGenerator` |
| **World-Model Foundation** (Rocktäschel 2024 talk) | Single learner across diverse tasks | orchestrator loop |
| **Era of Experience** (Silver & Sutton 2021) | Self-growing curriculum | Curriculum agent (built-in) |
| **AI-GA** (Clune 2020) | Quality-Diversity search | obstacle mutation & novelty check |

---

## 🛠️ Extending the system

1. **Add a new world**  
   Implement `step()`, `reset()` in a MiniWorld-like env and register in
   `POETGenerator`.

2. **Plug a stronger learner**  
   Swap `MuZeroTiny` for your fancy model (e.g. Dreamer-V4) – the RL loop is
   interface-stable.

3. **Introduce a new agent**  
   ```python
   # backend/agents/my_agent.py
   from alpha_factory_v1.demos.alpha_asi_world_model.alpha_asi_world_model_demo import BaseAgent
   class MyAgent(BaseAgent):
       def __init__(self): super().__init__("my_agent")
       def handle(self, msg): ...
   ```
   The bootstrapper auto-loads it at start-up.

---

## 🛡️ Safety & antifragility

* Loss spikes > 1 000 or NaNs ⇒ **instant halt** by `SafetyAgent`.
* Each agent runs **sandboxed**, no network unless whitelisted.
* Optional Constitutional-AI prompts protect LLM usage.
* Replay-buffer & checkpoints saved every 1 000 steps – crash-safe.

---

## 🌐 Cross-Industry “Alpha” in action

*Demo scenario shipped:*

1. **MarketAnalysisAgent** pulls live FX volatility (open-source feed).  
2. Identifies a *mean-reversion alpha* opportunity in EUR-USD.  
3. Sends plan → **PlanningAgent**, which decomposes tasks:  
   “simulate, stress-test, deploy micro-hedge bot”.  
4. **CodeGenAgent** patches a strategy stub; tests pass.  
5. **StrategyAgent** verifies risk limits; **SafetyAgent** green-lights.  
6. Orchestrator spins a **live mini-sim** environment; learner trains & rolls out.  
7. **Profit metrics stream** to UI — voilà, *alpha captured* ✨.

---

## ❤️ Contributing

PRs welcome (tests + lint = green).  
Join the conversation in **#alpha-factory** on the Montreal.AI Discord.

---

<p align="center">
  <b>Alpha-Factory v1 👁️✨ – forging the alpha of ASI.</b><br>
  <sub>© 2025 MONTREAL.AI · MIT-licensed · Made with passion for open-ended innovation</sub>
</p>
