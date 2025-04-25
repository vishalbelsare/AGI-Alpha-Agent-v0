
# 🚀 **α‑ASI World‑Model Demo 👁️✨**  
*Alpha‑Factory v1 — fully‑agentic, open‑ended curriculum + MuZero learner*  

![Alpha‑Factory banner](https://placehold.co/1200x250?text=Alpha‑Factory+v1+%F0%9F%91%81%E2%9C%A8+%E2%80%94+World+Model+Demo)  

> *“Imagination will often carry us to worlds that never were. But without it, we go nowhere.”* — **Carl Sagan**

---

## 🧭 Quick Navigation  
- [Why this demo?](#why)   |   [Architecture](#arch)   |   [Getting Started](#start)   |   [Controls](#controls)  
- [Agents](#agents)   |   [Safety & Trust](#safety)   |   [Extending](#extend)   |   [FAQ & Help](#faq)

---

<a id="why"></a>
## 1 Why does this demo exist? 🤔  
**Alpha‑Factory** aims to **Out‑learn, Out‑think, Out‑design, Out‑strategise & Out‑execute** across *all* industries.  
To do that we need an AI that 1️⃣ **grows its own worlds**, 2️⃣ **learns general skills** inside them, and 3️⃣ **turns those skills into Alpha (α) opportunities**.

This folder delivers a **single‑file, production‑deployable proof‑of‑concept** showing:

| 🔄 | Component | Highlight |
|----|-----------|-----------|
| 🌱 | **POET‑style generator** | births endless, diverse grid‑worlds |
| 🧠 | **MuZero‑style learner** | plans with a learned model (no rules given) |
| 🤝 | **≥ 5 Alpha‑Factory agents** | Planning, Research, Strategy, Market, CodeGen … plus Safety guard‑rails |
| 📴 | **Offline‑first** | no keys required; optional LLM helpers auto‑activate if `OPENAI_API_KEY` present |

---

<a id="arch"></a>
## 2 High‑level Architecture 🏗️  

```mermaid
flowchart LR
    subgraph Agents  🔌
        P(Planning) ---|A2A| O[Orchestrator]
        R(Research) ---|A2A| O
        S(Strategy) ---|A2A| O
        M(Market)   ---|A2A| O
        C(CodeGen)  ---|A2A| O
        G(Safety)   ---|A2A| O
    end
    O -->|spawns| ENV{{POET Generator}}
    O -->|trains| LRN[MuZero Learner]
    ENV -. new world .-> LRN
    LRN -. telemetry .-> O
    O --> API[FastAPI + WS UI]
```

*The Orchestrator is the “macro‑sentinel” quietly running in the background.*  
Agents talk over **Agent‑2‑Agent (A2A)** topics; external tools are wrapped via **MCP**.

---

<a id="start"></a>
## 3 Getting Started ⚡️  

| Mode | Command | Notes |
|------|---------|-------|
| **Local (Python)** | `pip install -r requirements.txt`<br>`python -m alpha_asi_world_model_demo --demo` | Opens UI at <http://127.0.0.1:7860> |
| **Docker** | `python -m alpha_asi_world_model_demo --emit-docker`<br>`docker build -t alpha_asi_world .`<br>`docker run -p 7860:7860 alpha_asi_world` | Fully self‑contained |
| **Kubernetes** | `python -m alpha_asi_world_model_demo --emit-helm`<br>`helm install asi ./helm_chart` | Scales to a cluster |
| **Notebook** | `python -m alpha_asi_world_model_demo --emit-notebook` | Interactive playground |

> **Tip:** *No GPU?* The demo auto‑detects and falls back to CPU.

---

<a id="controls"></a>
## 4 Runtime Controls 🎮  

| Action | REST / CLI | Description |
|--------|------------|-------------|
| Spawn new world | `POST /command {"cmd":"new_env"}` | Curriculum jump‑start |
| Pause learning | `POST /command {"cmd":"stop"}` | Halts main loop (Safety will also do this on anomaly) |
| List agents | `GET /agents` | Verify at least 5 topics alive |
| Stream metrics | WebSocket `/ws` | JSON every `ui_tick` steps |

Swagger docs auto‑mount at `/docs`.

---

<a id="agents"></a>
## 5 Meet the Agents 👥  

| Topic | Role in α‑Factory | Fallback if module missing |
|-------|-------------------|----------------------------|
| `planning_agent` | Breaks business goals into RL objectives | Stub logger |
| `research_agent` | Injects background knowledge via MCP | Stub logger |
| `strategy_agent` | Detects lucrative α‑opportunities, signals env swap | Stub logger |
| `market_agent` | Streams synthetic market signals for cross‑domain learning | Stub logger |
| `codegen_agent` | Hot‑patches learner architecture (AutoML) | Stub logger |
| `safety_agent` | Watches for NaN spikes & reward hacking | **Always active** |

*Guarantee:* **≥ 5** agent topics remain alive, preserving orchestration integrity.

---

<a id="safety"></a>
## 6 Safety, Trust & Antifragility 🛡️  

- **Loss & NaN sentinel** — learner auto‑pauses on divergence.  
- **Replay cap** — prevents memory explosions (`buffer_limit=50 k`).  
- **Opt‑in cloud** — no external calls unless keys are exported.  
- **Role‑scoped messages** — agents can’t mutate each other’s internals directly.

---

<a id="extend"></a>
## 7 Extending the Demo 🛠️  

1. **Add environment** → subclass `MiniWorld`, register in `POETGenerator`.  
2. **Swap learner** → implement `.act / .remember / .train`.  
3. **Plug real agent** → drop file in `backend/agents/`, class `.name` = topic.

---

<a id="faq"></a>
## 8 FAQ ❓  

<details><summary>“Does this *really* prove α‑ASI?”</summary>  
<b>No demo by itself proves ASI 😅</b>.  
It *does* prove the Alpha‑Factory runtime can autonomously generate worlds, learn, and self‑improve without human tasks. That’s a necessary (but not sufficient) step toward α‑ASI.</details>

<details><summary>“I only have a laptop — will it melt?”</summary>  
The default grid‑world is tiny and CPU‑friendly. For serious scale, enable GPU or spawn multiple learner pods in K8s.</details>

---

## 9 License & Citation 📜  

MIT (inherited).  
If you use this work, please cite:

> MONTREAL.AI (2025) *Alpha‑Factory v1 — Multi‑Agent AGENTIC α‑AGI.*

---

*Enjoy exploring the frontier!* 🚀
