# Era‑of‑Experience Demo 🧠⏩

A hands‑on illustration of Silver & Sutton’s **“Welcome to the Era of Experience”**
inside the Alpha‑Factory agent stack.

> **Four pillars** → *streams · actions · grounded rewards · non‑human reasoning* :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}

---

## 🚀 Quick start (local Docker)

```bash
cd alpha_factory_v1/demos/era_of_experience
chmod +x run_experience_demo.sh
./run_experience_demo.sh
```

Open <http://localhost:7860> — you’ll see the agent’s real‑time trace‑graph,
reward curves and an interactive chat panel so you can inject new experiences.

*   **With an OpenAI key** Drop it into `config.env` → GPT‑4/o3 drives the
    reasoning.
*   **Offline** Leave the key blank and the stack starts an **Ollama** container
    with `mixtral‑instruct`. Same UI, just slower.

---

## 🎓 Google Colab (no install)

Click the badge below. The first code cell creates the demo folder, writes the
same files you see here and launches a Gradio tunnel.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)
](https://colab.research.google.com/github/MontrealAI/AGI-Alpha-Agent-v0/blob/main/alpha_factory_v1/demos/era_of_experience/colab_stub.ipynb)

---

## 🛠️ How it works

| Component | Library | Why |
|-----------|---------|-----|
| **Agent core** | [`openai‑agents‑python`](https://openai.github.io/openai-agents-python) | battle‑tested tool‑calling & memory |
| **Inter‑agent protocol** | [`A2A`](https://github.com/google/A2A) | extension‑ready for swarm runs |
| **World model & planning** | internal MCTS helper | shows non‑human reasoning |
| **Deployment** | Docker Compose | 100 % reproducible |
| **Fallback LLM** | Ollama ✕ Mixtral | zero external API requirement |

The demo is kept **under 200 LoC** so you can grok the paradigm and then plug
in real environments (e.g. gymnasium ⇢ robotics lab, Bloomberg ⇢ live markets).

---

## 📚 Reading list

* Silver & Sutton, *The Era of Experience* (2024).  
* OpenAI, *A Practical Guide to Building Agents* (2024).  
* OpenAI Agents SDK docs.  
* Google ADK & A2A protocol specs.

