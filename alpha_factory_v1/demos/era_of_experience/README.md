<!--
  Era‑of‑Experience Demo
  Alpha‑Factory v1 👁️✨ — Multi‑Agent **AGENTIC α‑AGI**
  Out‑learn · Out‑think · Out‑strategise · Out‑execute
  © 2025 MONTREAL.AI   MIT License
-->

# 🌌 Welcome to the **Era of Experience** — Run it locally in *one* command

> “AI will eclipse the limits of human‑authored data only when agents **act, observe, and adapt** in the world.”  
> — *David Silver & Richard S. Sutton* citeturn12file0

This demo fuses their vision with **Alpha‑Factory v1** — a production‑ready, multi‑agent AGI spine.  
Within 60 seconds you’ll watch an agent **evolve in real time**: grounded rewards, long‑range memory and non‑human planning… all on your laptop.

---

## 🚀 1‑click quick‑start (macOS / Windows / Linux)

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/era_of_experience
chmod +x run_experience_demo.sh
./run_experience_demo.sh        # ←  THAT’S IT
```

1. **Docker Desktop** builds the image (≈1 min cold start).  
2. Your browser auto‑opens **http://localhost:7860**:  
   * live trace‑graph 🪄  
   * reward curves 📈  
   * interactive chat 💬

> **Tip – offline mode**   Leave `OPENAI_API_KEY=` blank in `config.env`: the stack boots **Ollama ✕ Mixtral** and stays air‑gapped.

---

## 🎓 Run in Google Colab (no Docker required)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MontrealAI/AGI-Alpha-Agent-v0/blob/main/alpha_factory_v1/demos/era_of_experience/colab_era_of_experience.ipynb)

The notebook:

* installs a lean Python stack (<120 s)  
* launches the agent & tunnels Gradio  
* lets you call tools directly from Python cells.

---

## ✨ Why this matters

| Silver & Sutton pillar | How the demo brings it alive |
|------------------------|------------------------------|
| **Streams of experience** | Infinite generator feeds months‑long synthetic logs |
| **Sensor‑motor actions** | `web_search`, `plan_meal`, & user Chat mutate the environment |
| **Grounded rewards** | Dual back‑ends `fitness_reward` & `education_reward` |
| **Non‑human reasoning** | MCTS planner + vector memory (no CoT imitation) |

The agent **rewrites its playbook every few seconds** — the leap beyond static prompt libraries.

---

## 🛠️ Architecture

```text
┌────────────┐   streams    ┌──────────────┐
│ Experience │ ───────────▶ │ Orchestrator │───┐
└────────────┘              └──────────────┘   │ tool‑calls
        ▲                           │          ▼
 rewards │                    ┌────────────┐ ┌─────────────┐
        │                    │   Planner   │ │  Tools/API  │
        └────────────────────┴─────────────┴───────────────┘
```

* **openai‑agents‑python** → battle‑tested tool‑calling & memory  
* **A2A hooks** → multi‑agent swarms out‑of‑the‑box  
* **Single Dockerfile** → deterministic, air‑gapped builds  

---

## 🗂️ Repo tour

| Path | Purpose |
|------|---------|
| `agent_experience_entrypoint.py` | boots the orchestrator & Gradio UI |
| `run_experience_demo.sh` | 1‑liner production launcher |
| `docker-compose.experience.yml` | services: orchestrator + Ollama |
| `colab_era_of_experience.ipynb` | cloud notebook twin |
| `reward_backends/` | plug‑in reward functions |

---

## 🧩 Extending the demo

* **New reward** – drop a file in `reward_backends/`; it hot‑loads.  
* **Add a tool**

```python
from openai_agents import Tool

@Tool(name="place_trade", description="execute an order on Alpaca")
async def place_trade(ticker:str, qty:int, side:str):
    ...
```

* **Scale out** – `docker compose --scale orchestrator=4 ...` → emergent cooperation.

---

## 🛡️ Production hygiene

* Container runs **non‑root**, no Docker‑in‑Docker.  
* Secrets stay in `config.env`.  
* `/__live` HTTP probe returns **200 OK** for K8s.

---

## 🆘 Troubleshooting cheat‑sheet

| Symptom | Fix |
|---------|-----|
| Docker not installed | [Download Docker Desktop](https://docs.docker.com/get-docker) |
| Port 7860 busy | Edit `ports:` in YAML |
| ARM Mac build slow | Enable *Rosetta* in Docker settings |
| Need GPU | Change base image to CUDA & add `--gpus all` |

---

## 🤝 Credits

* Engineered by **Montreal.AI**.  
* Inspired by **David Silver & Richard S. Sutton**.  
* Powered by the open‑source community — thank you!

> **Alpha‑Factory** — forging intelligence that *out‑learns, out‑thinks, out‑executes*.
