# Meta‑Agentic α‑AGI 👁️✨ Demo v3 — **AZR‑Powered Production‑Grade v0.2.0**

Identical to **v1** plus **two synergistic upgrades**  
1. *Statistical‑physics wrapper* — logs & minimises **Gibbs / variational free‑energy** for every candidate agent.  
2. *Absolute Zero Reasoner (AZR) self‑curriculum* — a **reinforced self‑play engine** that perpetually invents and solves its own tasks, unlocking *open‑ended* cross‑domain reasoning.

> **Official definition — Meta‑Agentic (adj.)**  
> *Describes an agent whose **primary role** is to **create, select, evaluate, or re‑configure other agents** and the rules governing their interactions, thereby exercising **second‑order agency** over a population of first‑order agents.*  
> *The term was **pioneered by Vincent Boucher, President of MONTREAL.AI**.*

---

## 🚀 Why AZR Matters
`Absolute Zero Reasoner` (Zhao *et al.* 2025) discards the last human bottleneck: **task curation**.  
It **proposes**, **validates**, **solves**, and **learns from** its own code‑reasoning problems — then feeds the distilled knowledge back into the evolutionary loop.  
*Result:* steeper learning curves, bolder exploration, and broad generalisation across math, code, and strategic planning — all while remaining vendor‑agnostic.

---

mermaid
%% 𝗚𝗿𝗮𝗻𝗱 𝗦𝘆𝗻𝗮𝗽𝘀𝗲 𝗚𝗿𝗮𝗽𝗵 – Meta‑Agentic α‑AGI (v3 AZR‑powered)
graph LR
  classDef meta       fill:#6425ff,stroke:#eee,color:#fff
  classDef layer      fill:#1e1e2e,stroke:#ddd,color:#fff
  classDef agent      fill:#0f9d58,stroke:#fff,color:#fff
  classDef tool       fill:#fbbc05,stroke:#000,color:#000
  classDef physics    fill:#ff6d00,stroke:#000,color:#fff
  classDef curriculum fill:#d81b60,stroke:#eee,color:#fff

  A0["🧠 Meta‑Programmer"]:::meta
  A1["📈 Evolution Archive"]:::layer
  A2["⚖️ Multi‑Objective Scorer"]:::layer
  Aφ["♾️ Free‑Energy Monitor"]:::physics
  AZ["🧮 AZR Self‑Curriculum Engine"]:::curriculum
  A3["🧩 Agent Population"]:::layer

  subgraph " "
    direction TB
    D1["🔍 Researcher"]:::agent
    D2["👷 Builder"]:::agent
    D3["🧪 Evaluator"]:::agent
    D4["⚙️ Auto‑Tuner"]:::agent
    D5["🛡 Guardian"]:::agent
  end

  subgraph " "
    direction TB
    T1["GPT‑4o"]:::tool
    T2["Claude‑3"]:::tool
    T3["Llama‑3 ∞"]:::tool
  end

  subgraph " "
    direction LR
    V1["🌐 Industry Data Streams"]
    V2["💎 Extracted Alpha"]
    V3["🚀 Deployed Solutions"]
  end

  %% control‑flow
  A0 -->|spawn| A3
  A0 -->|bootstrap tasks| AZ
  AZ -->|invent curricula| A3
  A3 -->|solver traces| AZ
  A3 -->|select| A2
  A2 -->|rank| A1
  A1 -- feedback --> A0

  %% free‑energy link
  A3 -.state logits.-> Aφ
  Aφ -->|F‑metric| A2
  Aφ -- entropy grad --> A0
  AZ -- expected‑learning‑gain --> Aφ

  %% providers
  D1 -.uses.-> T1
  D2 -.uses.-> T3
  D3 -.uses.-> T2
  D4 -.uses.-> T3
  D5 -.uses.-> T1

  %% value loop
  A3 -->|iterate| V1
  V1 -->|signals| D1
  D2 --> V2
  V2 --> D3
  D4 --> V3
  D5 -.audit.-> V3

---

## 📌 Purpose & Positioning
This demo operationalises **Automated Design of Agentic Systems (ADAS)** and adds:

* **AZR‑driven open‑ended learning** — tasks invented on‑the‑fly, tuned for maximal learning gain.
* **True multi‑objective optimisation** — accuracy, cost, latency, risk, carbon **& free‑energy**.
* **Open‑weights *or* API FMs** — swap GPT‑4o, Claude‑3, Llama‑3, Mistral .gguf at will.
* **Provable lineage & provenance** — every agent / artefact traceable via the Lineage UI.
* **Battle‑tested safeguards** — sandboxing, taint‑tracking, chaos‑testing.

Together, they lift **Alpha‑Factory v1** into a *self‑improving*, cross‑industry **Alpha Factory** that systematically  
> **Out‑Learn · Out‑Think · Out‑Design · Out‑Strategize · Out‑Execute**  

— with zero dependence on any single model or vendor.

---

## 1 Quick‑start 🏁

```bash
# 1️⃣ Clone & enter
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/meta_agentic_agi_v3

# 2️⃣ Environment (CPU‑only default)
micromamba create -n metaagi python=3.11 -y
micromamba activate metaagi
pip install -r requirements.txt   # ≤ 40 MiB wheels

# 3️⃣ Run – zero‑API mode (pulls a gguf via Ollama)
python meta_agentic_agi_demo.py --provider mistral:7b-instruct.gguf

#   …or point to any provider
OPENAI_API_KEY=sk‑… python meta_agentic_agi_demo.py --provider openai:gpt-4o

# 4️⃣ Launch the lineage UI
streamlit run ui/lineage_app.py
```

*Tip:* `llama‑cpp‑python` auto‑quantises to 4‑bit; < 6 GB RAM is enough.

---

## 2 Folder Structure 📁
```
meta_agentic_agi/
├── core/                # provider‑agnostic primitives
│   ├── fm.py            # unified FM wrapper
│   ├── prompts.py       # reusable prompt fragments
│   └── tools.py         # exec sandbox, RAG, vector store
├── meta_agentic_search/ # ⬅ evolutionary loop
│   ├── archive.py       # stepping‑stone JSONL log
│   ├── search.py        # NSGA‑II + Reflexion
│   └── scorer.py        # multi‑objective metrics (+ free‑energy)
├── azr/                 # ⬅ Absolute Zero Reasoner
│   ├── proposer.py      # task invention
│   ├── solver.py        # task execution
│   ├── buffers.py       # triplet storage
│   └── rewards.py       # learnability & accuracy
├── agents/
│   ├── agent_base.py    # runtime interface
│   └── seeds.py         # bootstrap population
├── ui/
│   ├── lineage_app.py   # Streamlit dashboard
│   └── assets/
├── configs/
│   └── default.yml      # editable in‑UI
└── meta_agentic_agi_demo.py
```

---

## 3 Provider Abstraction ➡️ open‑weights 🏋️‍♀️
`configs/default.yml` excerpt:
```yaml
provider: mistral:7b-instruct.gguf   # any ollama / llama.cpp id
context_length: 8192
rate_limit_tps: 4
retry_backoff: 2
```

Change **provider** to:

| Value                       | Notes                    |
|-----------------------------|--------------------------|
| openai:gpt-4o               | needs `OPENAI_API_KEY`   |
| anthropic:claude-3-sonnet   | needs `ANTHROPIC_API_KEY`|
| mistral:7b-instruct.gguf    | default local model      |

Wrapper normalises chat/completions, streams via **MCP**, and window‑slides tokens.

---

## 4 AZR Self‑Curriculum 🎒
*Task space:* deterministic Python triplets **(program p, input i, output o)**  
*Modes:* **abduction** (`infer i`), **deduction** (`infer o`), **induction** (`synthesise p`)  

AZR jointly trains **proposer** & **solver** roles with **Task‑Relative REINFORCE++**.  
Learnability reward peaks when tasks are “just‑hard‑enough”, driving an *automatic syllabus* that adapts as agents improve.

---

## 5 Multi‑Objective Search 🎯
Objective vector → **[accuracy, cost, latency, hallucination‑risk, carbon, free‑energy]**

* NSGA‑II elitist selection  
* Behaviour descriptor = SHA‑256 of candidate AST  
* Optional human‑in‑the‑loop thumbs ↑/↓  

---

## 6 Security & Antifragility 🛡
* Firejail `--seccomp` + 512 MiB mem‑cgroup sandbox  
* Static analysis (`bandit`) + dynamic taint tracking  
* Live watchdog kills rogue processes > 30 s CPU  
* Chaos‑tests inject tool failures; reward graceful degradation  
* Curriculum pruning auto‑drops unsafe proposals.

---

## 7 Extending 🛠
1. **New dataset** — drop `my.pkl` into `data/`, run `--dataset my`.  
2. **New metric** — subclass `scorer.BaseMetric`, list in `configs/default.yml`.  
3. **New AZR reward** — edit `azr/rewards.py`, plug into `buffers.py`.

---

## 8 Roadmap 🗺
* ☐ Hierarchical meta‑meta search  
* ☐ GPU batch infer (Flash‑infer v3)  
* ☐ Offline RL fine‑tune search policy with lineage replay  
* ☐ Multimodal AZR (image ↔ code ↔ math)  

---

## 9 References 📚
* A. Zhao *et al.* “Absolute Zero: Reinforced Self‑play Reasoning with Zero Data” arXiv 2025  
* S. Hu *et al.* “Automated Design of Agentic Systems” ICLR 2025  
* OpenAI “A Practical Guide to Building Agents” (2024)  
* Google ADK docs (2025)

---

© 2025 MONTREAL.AI — Apache‑2.0
