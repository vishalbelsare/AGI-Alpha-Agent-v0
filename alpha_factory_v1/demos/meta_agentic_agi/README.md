# Meta-Agentic α-AGI 👁️✨ Demo — **Production-Grade v0.1.0**

> **Meta-Agentic (adj.)**  
> An agent whose *primary* role is to **create, select, evaluate, or re-configure other agents** and the rules that bind them, exercising *second-order agency* over a population of first-order agents.  
> *Coined by Vincent Boucher (MONTREAL.AI).*

> **Mission** – Elevate **Alpha-Factory v1** into a self-improving, cross-industry *Alpha Factory* that systematically  
> **Out-Learn · Out-Think · Out-Design · Out-Strategize · Out-Execute** — while remaining provider-agnostic, regulator-ready, and antifragile.

---

## 📌 Why this demo exists
This repository fuses three recent breakthroughs:

1. **Automated Design of Agentic Systems (ADAS)** – meta-agents that *program* better agents via evolutionary search.  
2. **Foundation-model unification layer** – seamless swap-in of **OpenAI**, **Anthropic**, or *any* local `gguf` weight (Mistral, Llama 3, etc.).  
3. **Full lineage provenance** – every candidate agent, its code, metrics, and Pareto rank rendered live in a **Streamlit dashboard**.

The result is a turnkey sandbox for experimenting with **Meta-Agentic α-AGI** that can run entirely offline *or* scaled up with cloud APIs.

---

## 1 Quick-start 🏁
\`\`\`bash
# 1️⃣ Clone & enter
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/meta_agentic_agi

# 2️⃣ Minimal environment (CPU-only by default)
micromamba create -n metaagi python=3.11 -y
micromamba activate metaagi
pip install -r requirements.txt          # ≤ 40 MiB wheels

# 3️⃣ Run – zero-API mode (pulls a 7-B gguf via Ollama)
python meta_agentic_agi_demo.py --provider mistral:7b-instruct.gguf

#   …or point to any provider
OPENAI_API_KEY=sk-...   python meta_agentic_agi_demo.py --provider openai:gpt-4o
ANTHROPIC_API_KEY=tok…  python meta_agentic_agi_demo.py --provider anthropic:claude-3-sonnet

# 4️⃣ Launch the real-time lineage UI
streamlit run ui/lineage_app.py
\`\`\`

*Tip – no GPU?* \`llama-cpp-python\` autoselects 4-bit quantisation → runs in < 6 GiB RAM.

---

## 2 Folder Structure 📁
\`\`\`
meta_agentic_agi/
├── core/                # provider-agnostic primitives
│   ├── fm.py            # unified FM wrapper (OpenAI / Anthropic / gguf)
│   ├── prompts.py       # reusable prompt fragments
│   └── tools.py         # exec sandbox, RAG, vector store, chaos-monkey
├── meta_agentic_search/ # evolutionary loop
│   ├── archive.py       # lineage store & analytics
│   ├── scorer.py        # multi-objective metrics plug-ins
│   └── search.py        # NSGA-II + Reflexion + self-repair
├── agents/
│   ├── agent_base.py    # runtime interface
│   └── seeds.py         # bootstrap population (vanilla LLM, RAG, planner…)
├── ui/
│   ├── lineage_app.py   # Streamlit dashboard
│   └── assets/          # SVGs, CSS
├── configs/
│   └── default.yml      # editable live in UI
└── meta_agentic_agi_demo.py  # orchestration entry-point
\`\`\`

---

## 3 Architecture 🔍

### 3.1 Meta-Agent Search Loop
\`\`\`mermaid
graph TD
  MLLM["🧠 Meta-LLM<br/>"Programmer""]
  Cand["📝 Candidate Agent<br/>Python fn"]
  Eval["🔒 Sandboxed Evaluator"]
  Arch["📚 Archive<br/>Pareto + Novelty"]
  UI["📈 Streamlit UI"]

  MLLM -->|generates| Cand
  Cand  --> Eval
  Eval  -->|scores| Arch
  Arch  -->|prompt context| MLLM
  Arch  -->|websocket| UI
\`\`\`

### 3.2 Integration with Alpha-Factory v1
\`\`\`mermaid
flowchart LR
  AFV1("Alpha-Factory v1 Core")
  MLayer("Meta-Agentic Layer<br/>(this demo)")
  Providers("FM Providers<br/>OpenAI / Anthropic / gguf")
  Data("Domain Datasets & RAG stores")
  AFV1 --> MLayer
  MLayer --> Providers
  MLayer --> Data
\`\`\`

---

## 4 Configuring Providers 🏋️

`configs/default.yml` (excerpt):

\`\`\`yaml
provider: mistral:7b-instruct.gguf   # any ollama / llama.cpp id
context_length: 8192
rate_limit_tps: 4
retry_backoff: 2
\`\`\`

| Value                          | Notes                              |
|--------------------------------|------------------------------------|
| \`openai:gpt-4o\`                | needs \`OPENAI_API_KEY\`             |
| \`anthropic:claude-3-sonnet\`    | needs \`ANTHROPIC_API_KEY\`          |
| \`mistral:7b-instruct.gguf\`     | default local model (no API key)   |
| \`llama3:70b-instruct.Q4_K_M.gguf\` | drop into \`~/.ollama\` and reference |

The unified wrapper normalises chat/completions, streams via **Model Context Protocol**, and slides windows automatically.

---

## 5 Multi-Objective Search 🎯

**Objective Vector**  
\`[−accuracy, latency, cost, hallucination-risk, carbon]\`   *(lower is better)*

* *NSGA-II* elitist ranking + crowding distance  
* Novelty score = Shannon entropy of AST + Jaccard vs. archive  
* Live *thumbs-up / down* in UI feeds a human-in-the-loop reward slice

---

## 6 Security & Antifragility 🛡

| Layer                   | Mechanism                                               |
|-------------------------|---------------------------------------------------------|
| **Sandbox**             | \`firejail --seccomp\` + 512 MiB cgroup + net-ns isolation |
| **Static analysis**     | \`bandit\` + AST-level policy linter                      |
| **Dynamic guards**      | taint tracking, syscall whitelist                       |
| **Chaos testing**       | random API failure / latency injection                  |
| **Watchdog**            | kills rogue proc > 30 s CPU or > 256 MB tmp             |
| **Audit trail**         | every prompt/response hashed & timestamped to lineage   |

---

## 7 Extending 🛠

| Goal                       | How-to                                                                 |
|----------------------------|------------------------------------------------------------------------|
| **Add dataset**            | Drop \`my.pkl\` into \`data/\`, run \`--dataset my\`                         |
| **Inject new metric**      | Subclass \`scorer.BaseMetric\`, list in \`configs/default.yml\`            |
| **Create new agent seed**  | Add a function into \`agents/seeds.py\` returning \`python\` code string   |
| **Add external tool**      | Implement in \`core/tools/\`, register via entry-point                   |

---

## 8 Roadmap 🗺

* ☐ Hierarchical *meta-meta* evolutionary layer  
* ☐ Batch GPU inference via Flash-Infer v3 kernels  
* ☐ Offline RL fine-tuning of the search policy from lineage replay  
* ☐ Live “agent marketplace” plug-in protocol (A2A-compatible)

---

## 9 Key Metrics 📊

| KPI                              | Why it matters                       |
|----------------------------------|--------------------------------------|
| *Design cycle time*              | Shorter loops ⇒ faster compounding   |
| *Cross-domain adaptability Δ*    | Measures generality across verticals |
| *Surplus $/GPU-hour*             | Direct economic efficiency signal    |
| *Pareto front size*              | Diversity of top solutions           |

---

## 10 References 📚

* Hu S. *et al.* **“Automated Design of Agentic Systems”** (ICLR 2025)  
* OpenAI **“A Practical Guide to Building Agents”** (2024)  
* Google **ADK Documentation** (2025)  
* Silver D. & Sutton R. **“Welcome to the Era of Experience”** (MIT Press preprint, 2025)  
* Schrittwieser J. *et al.* **“Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model”** (2019)

---

© 2025 **MONTREAL.AI** — Apache-2.0  
*Built with ❤️ for open, provider-agnostic innovation.*
