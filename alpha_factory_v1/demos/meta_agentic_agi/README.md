# Meta‑Agentic α‑AGI 👁️✨ Demo – **Production‑Grade v0.1.0**

---

## 📌 Purpose & Positioning

This demo extends **Alpha‑Factory v1** into a *self‑improving*, cross‑industry “Alpha Factory” able to **Out‑Learn · Out‑Think · Out‑Design · Out‑Strategize · Out‑Execute** — *without hard‑wiring a single vendor or model*.

It operationalises the **Automated Design of Agentic Systems** paradigm from Hu *et al.* ICLR‑25 and layers true **multi‑objective search**, open‑weights support, automated lineage documentation, and antifragile safeguards on top of the existing α‑Factory.

> **Goal:** Provide a **completely deployable, audited‑by‑design reference stack** that a non‑technical stakeholder can run on a laptop *or* scale up in Kubernetes, then immediately surface alpha‑grade opportunities in any vertical.

---

## 1 Quick‑start 🏁

```bash
# ❶ Clone Alpha‑Factory v1 and enter demo folder
$ git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
$ cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/meta_agentic_agi

# ❷ Create & activate env
$ micromamba create -n metaagi python=3.11 -y
$ micromamba activate metaagi
$ pip install -r requirements.txt          # ↓ pure‑Python, CPU‑only default

# ❸ Run in 🗲 zero‑API mode (open‑weights) – pulls a gguf model via Ollama
$ python meta_agentic_agi_demo.py --provider mistral:7b-instruct.gguf

#  …or plug any provider (OpenAI, Anthropic, LM‑Studio‑local)
$ OPENAI_API_KEY=sk‑… python meta_agentic_agi_demo.py --provider openai:gpt-4o

# ❹ Launch visual lineage UI
$ streamlit run ui/lineage_app.py           # http://localhost:8501
```

No GPU? No problem — default settings use dynamic low‑RAM quantisation and the search loop throttles to respect laptop thermals.

---

## 2 High‑Level Architecture 🏗

```
alpha_factory_v1/
└── demos/meta_agentic_agi/
    ├── core/                       # Provider‑agnostic primitives
    │   ├── fm.py                  # Unified FM wrapper (OpenAI / Anthropic / llama‑cpp)
    │   ├── tools.py               # Search, exec sandbox, RAG, vector store helpers
    │   └── prompts.py             # Shared prompt fragments (COT, Reflexion, MCP streaming)
    ├── meta_search/               # Meta‑agentic search loop
    │   ├── archive.py             # JSONL stepping‑stone log (Pareto + novelty hashes)
    │   ├── search.py              # NSGA‑II evolutionary loop w/ reflexive LLM programmer
    │   └── scorer.py              # Accuracy • Latency • Cost • Hallucination • Carbon
    ├── agents/                    # Runtime agent interface
    │   ├── agent_base.py          # forward(task: Info) → Info | str
    │   └── seeds.py               # Bootstrap population (COT, Self‑Refine, etc.)
    ├── ui/
    │   ├── lineage_app.py         # Streamlit dashboard – graph lineage & metrics
    │   └── assets/                # Icons / D3 templates / Tailwind sheet
    ├── configs/
    │   └── default.yml            # Editable in‑UI – objectives & provider switch
    ├── requirements.txt           # ≤ 40 MiB wheels; pure‑py numpy‑lite default
    └── meta_agentic_agi_demo.py   # Entry‑point CLI
```

---

## 3 Architecture 🔍

1. **Agent spec = Python function** – *Turing‑complete search‑space*.
2. **Meta Agent (gpt‑4o, Claude 3‑Sonnet, or local Llama 3‑70B)\_**
   ↳ builds candidate agents **in‑code**, guided by archive & multi‑objective score.
3. **Evaluator** spawns sandboxed subprocesses → returns metrics (accuracy 📈, latency ⏱, cost 💰, risk 🛡).
4. **Archive** (Quality–Diversity map) retains Pareto‑front & behavioural novelty hashes.
5. **UI** streams lineage graph (D3.js) + spark‑lines; click any node to expand full source & run.

<p align="center"><img src="https://raw.githubusercontent.com/MontrealAI/AGI-Alpha-Agent-v0/main/docs/img/meta_search_flow.svg" width="640"></p>

---

## 4 Replacing vendors ➡️ open‑weights 🏋️‍♀️

```yaml
# configs/default.yml (excerpt)
provider: mistral:7b-instruct.gguf   # any ollama / llama.cpp id
context_length: 8192
rate_limit_tps: 4
retry_backoff: 2
```

Set `provider:` to:

* `openai:gpt-4o`  – env `OPENAI_API_KEY`
* `anthropic:claude-3-sonnet` – env `ANTHROPIC_API_KEY`
* `mistral:7b-instruct.gguf` (default) – auto‑pull via **llama‑cpp‑python**
  The wrapper normalises chat/completions and automatically chunks > context tokens via **MCP** streams.

---

## 5 True multi‑objective search 🎯

> **Objective vector** = `[accuracy, cost, latency, hallucination‑risk, carbon]`

* **NSGA‑II** selection (fast elitist) implemented in `meta_search/search.py`.
* Behaviour descriptor = SHA‑256 of AST of candidate agent — encourages divergent program shapes (§2, Hu *etal.*).
* Optional *human‑in‑the‑loop* override — thumbs up/down in UI feeds reward shaping.

---

## 6 Security & antifragility 🛡

* All generated code executed in a **petting‑zoo**: firejail + `--seccomp` + 512 MiB memcg.
* Mandatory static analysis via `bandit` and dynamic taint tracking before promotion to archive.
* Live monitors shoot rogue processes > 30 s CPU.

---

## 7 Extending 🛠

1. **Add domain dataset** → drop `my_dataset.pkl` into `data/` and reference in CLI.
2. **Custom metric** → implement `scorer.MyMetric` and list under `configs/default.yml/objectives`.
3. **Plug tool** (browser, SQL, vector‑RAG) → write `core/tools/my_tool.py` (must expose `__call__(self, query)`), then import in seeds.

---

## 8 Roadmap 🗺

* \[ ] Hierarchical Meta‑Meta search (self‑improving meta‑agent).
* \[ ] Native CUDA kernel for batch eval of tens of lightweight models (Flash‑infer).
* \[ ] Offline RL fine‑tuning of search policy using lineage replay.

---

## 9 References 📚

* Hu *et al.* “Automated Design of Agentic Systems” ICLR 2025
* OpenAI “Practical Guide to Building Agents” (2024)
* Google ADK docs (2025)

---

© 2025 MONTREAL.AI   Licensed Apache‑2.0
