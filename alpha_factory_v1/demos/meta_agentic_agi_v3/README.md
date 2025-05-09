
# **Meta‑Agentic α‑AGI 👁️✨ Demo v3 — AZR‑Powered “Alpha‑Factory v1” (Production‑Grade v0.3.0)**

```mermaid
%% 𝗚𝗿𝗮𝗻𝗱 𝗢𝗽𝗲𝗿𝗮𝘁𝗶𝗼𝗻𝗮𝗹 𝗦𝘆𝗻𝗮𝗽𝘀𝗲 — Alpha‑Factory v1 (AZR + POET + MuZero++)
flowchart TD
  subgraph LAYER0["Meta‑Agency"]
    direction LR
    MP["🧠 Meta‑Programmer"]:::meta -->|spawns| POP
    AZR["🔄 AZR Self‑Curriculum"]:::curri
    MP --> AZR
  end

  subgraph LAYER1["Evolutionary Loop"]
    POP["📈 Evolution Archive"]:::layer
    SCORE["⚖️ Multi‑Objective Scorer"]:::layer
    FE["♾️ Free‑Energy Monitor"]:::phys
    POP --> SCORE --> FE --> MP
    AZR --> POP
  end

  subgraph LAYER2["Agent Population"]
    direction TB
    R["🔍 Researcher"]:::agent
    B["👷 Builder"]:::agent
    E["🧪 Evaluator"]:::agent
    T["⚙️ Auto‑Tuner"]:::agent
    G["🛡 Guardian"]:::agent
  end
  MP --> R & B & E & T & G

  subgraph LAYER3["Foundation Model Tools"]
    GPT["GPT‑4o"]:::tool
    CLAUDE["Claude‑3 Opus"]:::tool
    L3["Llama‑3‑70B ∞"]:::tool
  end
  R -.uses.-> GPT
  B -.uses.-> L3
  E -.uses.-> CLAUDE
  T -.uses.-> L3
  G -.uses.-> GPT

  subgraph VALUE_LOOP["Industry Value‑Loop"]
    DATA["🌐 Market & Web Streams"]
    ALPHA["💎 Extracted Alpha"]
    SOL["🚀 Deployed Solutions"]
  end
  R --> DATA
  B --> ALPHA
  E --> SOL
  T --> ALPHA
  G -.audit.-> SOL

classDef meta fill:#6425ff,stroke:#eee,color:#fff
classDef layer fill:#2b2b40,stroke:#ddd,color:#fff
classDef phys fill:#ff6d00,stroke:#000,color:#fff
classDef agent fill:#0f9d58,stroke:#fff,color:#fff
classDef tool fill:#fbbc05,stroke:#000,color:#000
classDef curri fill:#d81b60,stroke:#eee,color:#fff
```

> *“Alpha‑Factory v1 transforms raw data‑streams into deployable solutions that **Out‑Learn · Out‑Think · Out‑Design · Out‑Strategize · Out‑Execute** the market — autonomously.”*

---

## 📌 Purpose & Scope
This repository packages a **self‑improving, cross‑industry Alpha Factory**:  
a multi‑agent system that combines **Absolute Zero Reasoner (AZR)**, **POET‑style open‑ended evolution**, and **MuZero++ world‑model planning** into a single orchestrated runtime.  
It is designed to:

* **Continuously generate synthetic tasks** → open‑ended curriculum (AZR).  
* **Evolve & rank specialised agents** ↔ multi‑objective Pareto frontier (NSGA‑II).  
* **Minimise thermodynamic free‑energy** → statistical‑physics wrapper for robust convergence.  
* **Extract live alpha** from market + web streams, and **deploy automated strategies** under strict risk & compliance guards.

The default demo includes a **live NVDA‑earnings momentum alpha** (see *section 10*).

---

## 1 Quick‑Start 🏁
```bash
# 1️⃣ Clone
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/meta_agentic_agi_v3

# 2️⃣ Environment
micromamba create -n alpha_factory python=3.11 -y
micromamba activate alpha_factory
pip install -r requirements.txt      # ≤ 60 MiB wheels

# 3️⃣ Run – open‑weights default (no API key)
python src/main.py --provider mistral:7b-instruct.gguf --curriculum azr

#  …or switch to GPT‑4o
OPENAI_API_KEY=sk-... python src/main.py --provider openai:gpt-4o --curriculum azr

# 4️⃣ UI dashboards
streamlit run ui/lineage_app.py           # provenance graph
streamlit run ui/alpha_monitor.py         # live alpha dashboard
```

*Hardware:* CPU‑only works (llama‑cpp 4‑bit); GPU speeds things up. 8 GB RAM minimum.

---

## 2 Folder Structure 📁
```
meta_agentic_agi_v3/
├── src/
│   ├── main.py              # entry‑point CLI
│   ├── orchestrator.py      # A2A bus & agent scheduler
│   ├── curriculum/
│   │   └── azr_engine.py    # Absolute Zero implementation
│   ├── evolution/
│   │   ├── archive.py       # stepping‑stone log
│   │   └── search.py        # NSGA‑II + novelty
│   ├── agents/
│   │   ├── alpha_finder.py  # identifies live alphas
│   │   ├── execution.py     # trade execution stub
│   │   ├── risk_manager.py  # position sizing & VaR stops
│   │   └── guardian.py      # safety & compliance checks
│   ├── physics/
│   │   └── gibbs.py         # free‑energy metric
│   └── utils/
│       └── provider.py      # OpenAI / Anthropic / llama‑cpp
├── ui/
│   ├── lineage_app.py       # provenance visualisation
│   └── alpha_monitor.py     # live PnL & risk
├── configs/
│   └── default.yml          # editable at runtime
└── requirements.txt
```

---

## 3 Provider‑Agnostic FM Wrapper ➡ Open‑Weights 🏋️‍♀️
`configs/default.yml` excerpt:
```yaml
provider: mistral:7b-instruct.gguf     # any ollama / llama.cpp id
context_length: 16_384
rate_limit_tps: 4
retry_backoff: 2
```

| Value                         | Note                            |
|-------------------------------|---------------------------------|
| `openai:gpt-4o`               | needs `OPENAI_API_KEY`          |
| `anthropic:claude-3-sonnet`   | needs `ANTHROPIC_API_KEY`       |
| `mistral:7b-instruct.gguf`    | local default via llama‑cpp      |

All chats stream via **MCP** & window‑slide for long contexts.

---

## 4 Multi‑Objective Evolution 🎯
**Objective vector** = `[accuracy, cost, latency, hallu‑risk, carbon, free‑energy]`  
* NSGA‑II elitist selection  
* Behaviour descriptor = SHA‑256(AST)  
* Human‑in‑the‑loop thumbs ↑/↓ (web UI)  

---

## 5 Security & Antifragility 🛡
* Firejail `--seccomp` + 512 MiB cgroup sandbox  
* Static (`bandit`) + dynamic taint tracking  
* Live watchdog terminates rogue proc > 30 s CPU  
* Chaos‑monkey fault injections each epoch  
* Curriculum filter auto‑drops unsafe proposals  

---

## 6 Lineage & Observability 📊
Run `streamlit run ui/lineage_app.py` → DAG of every agent, prompt, tool‑call, metric, and deployment artefact.  
OpenTelemetry exporters emit traces; Prometheus scrapes runtime metrics; Grafana dashboards included.

---

## 7 Extending 🛠
1. **New dataset** — drop `foo.pkl` → auto‑RAG ingest.  
2. **New metric** — subclass `evolution.metrics.BaseMetric`.  
3. **Custom curriculum** — register engine in `curriculum/__init__.py`.  
4. **Real exchange adapter** — implement `execution.broker.BaseBroker` (see IBKR stub).  

---

## 8 Roadmap 🗺
- ☑ AZR integration & POET outer‑loop  
- ☑ Free‑Energy minimisation  
- ☑ Live alpha demo (NVDA earnings)  
- ☐ Multimodal (image ↔ code ↔ math) AZR  
- ☐ Hierarchical meta‑meta search  
- ☐ Flash‑Infer v3 GPU batched inference  
- ☐ RL fine‑tune search policy w/ lineage replay  

---

## 9 Key References 📚
* Zhao *et al.* “Absolute Zero: Reinforced Self‑Play Reasoning with Zero Data” (2025) citeturn1file0  
* Hu *et al.* “Automated Design of Agentic Systems” ICLR 2025 citeturn1file1  
* Clune “AI‑Generating Algorithms” (2020) citeturn1file3  
* Schrittwieser *et al.* “MuZero” (2020) citeturn1file4  
* Silver & Sutton “Era of Experience” (2025) citeturn1file5  

---

## 10 Live Alpha Demo 🚀

**Signal:** Anticipated upside surprise in NVIDIA (NVDA) Q1‑FY2026 earnings on **28 May 2025**, driven by record data‑center demand and Blackwell GPU ramp.  
*Evidence:* 78 % YoY revenue jump to \$39.3 B and guidance for \$43 B next quarter citeturn0search0turn0search4. Deloitte projects continued semiconductor boom via generative‑AI build‑outs citeturn0search1.

**Strategy:**  
1. **Entry window**: T‑10 → T‑2 trading days before earnings.  
2. **Position**: 40 % capital → 30‑delta call options (30 Jun expiry) + 60 % delta‑one shares.  
3. **Risk**: Max 2 % account equity; stop‑loss at ATR‑2× below entry; risk manager enforces VaR<1 %.  
4. **Exit**: 50 % gamma‑scalp on IV crush at T + 1; remainder trail‑stop @ EMA‑21.  

The included `agents/alpha_finder.py` continuously scans news/API feeds and triggers the **ExecutionAgent** when criteria match.  Sources are injected into the lineage graph for auditability.

---

© 2025 MONTREAL.AI — Apache‑2.0
