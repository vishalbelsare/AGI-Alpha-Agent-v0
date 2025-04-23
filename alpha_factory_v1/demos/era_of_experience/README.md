<!--
 Era‑of‑Experience Demo
 Alpha‑Factory v1 👁️✨ — Multi‑Agent **AGENTIC α‑AGI**
 Out‑learn · Out‑think · Out‑strategise · Out‑execute
 © 2025 MONTREAL.AI   MIT License
-->

<h1 align="center">🌌 Era of Experience &mdash; Your personal lifelong‑RL sandbox</h1>
<p align="center">
 <em>Spin up a self‑improving agent in <strong>one command</strong>.<br>
 Watch it learn, plan, and act in real‑time &mdash; entirely on your laptop.</em>
</p>

> “AI will eclipse the limits of human‑authored data only when agents <strong>act, observe, and adapt</strong> in the world.”  
> — David Silver &amp; Richard S. Sutton citeturn12file0

This demo forges their vision into **Alpha‑Factory v1**, a production‑ready multi‑agent spine that — within a single minute — lets you **experience** an agent evolving before your eyes: grounded rewards, episodic memory, and non‑human planning… all on commodity hardware.

---

## 🚀 Zero‑friction quick‑start

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/era_of_experience
chmod +x run_experience_demo.sh
./run_experience_demo.sh              # ← THAT’S IT
```

1. **Docker Desktop** builds a 300 MB image (≈1 min cold start).  
2. Your browser pops open **http://localhost:7860** with  
   * **live trace‑graph** 🪄  
   * **reward curves** 📈  
   * **interactive chat** 💬  

> **Tip &ndash; offline mode**   Leave `OPENAI_API_KEY=` blank in `config.env`:  
> the stack boots **Ollama ✕ Mixtral‑8x7B** and stays fully air‑gapped.

---

## 🎓 One‑click Colab

| Notebook | Runtime | Badge |
|----------|---------|-------|
| *Colab twin* | CPU / GPU | <a href="https://colab.research.google.com/github/MontrealAI/AGI-Alpha-Agent-v0/blob/main/alpha_factory_v1/demos/era_of_experience/colab_era_of_experience.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a> |

The notebook:

* installs a lean Python stack in &lt;120 s, no Docker needed;  
* launches the agent &amp; tunnels Gradio;  
* exposes tools (`web_search`, `plan_meal`, …) straight from Python cells.

---

## ✨ Why this matters

| Silver &amp; Sutton pillar | How the demo brings it alive |
|---------------------------|------------------------------|
| **Streams of experience** | Endless generator spitting months‑long synthetic logs |
| **Sensor‑motor actions**  | Internet search + meal‑planner mutate the world state |
| **Grounded rewards**      | Dual back‑ends `fitness_reward` &amp; `education_reward` |
| **Non‑human reasoning**   | Monte‑Carlo Tree Search planner + vector memory &mdash; no CoT echo |

The agent **rewrites its playbook every few seconds** &ndash; a glimpse beyond static prompt engineering.

---

## 🛠️ Under the hood

```text
┌────────────┐   streams    ┌────────────────┐
│ Experience │ ───────────▶ │ Orchestrator ⚙ │───┐
└────────────┘              └────────────────┘   │  tool calls
        ▲                           ▲           ▼
        │                     ┌──────────┐  ┌────────────┐
 rewards│                     │ Planner ♟ │  │  Tools/API │
        └─────────────────────┴──────────┴───────────────┘
```

* **openai‑agents‑python** – battle‑tested tool‑calling, vector memory, recursion‑safe.  
* **A2A protocol** – future‑proof hooks for multi‑agent swarms.  
* **Single Dockerfile** – deterministic, air‑gapped builds; no base‑image roulette.

---

## 🗂️ Repo map

| Path | Purpose |
|------|---------|
| `agent_experience_entrypoint.py` | boots orchestrator + Gradio UI |
| `run_experience_demo.sh` | 1‑liner production launcher (health‑gated) |
| `docker-compose.experience.yml` | orchestrator + Ollama services |
| `colab_era_of_experience.ipynb` | cloud notebook twin |
| `reward_backends/` | plug‑in reward functions |
| `simulation/` | tiny Gym‑like env stubs (future work) |

---

## 🧩 Hack me!

* **New reward** &rarr; drop a file into `reward_backends/`, hot‑reloaded.  
* **Add a tool**

```python
from openai_agents import Tool

@Tool(name="place_trade", description="Execute an order on Alpaca")
async def place_trade(symbol:str, qty:int, side:str="BUY"):
    ...
```

* **Scale out** – `docker compose --profile gpu --scale orchestrator=4 up`  
  → emergent cooperation and shared memory.

---

## 🛡️ Production hygiene

* Container runs **non‑root**, no Docker‑in‑Docker.  
* Secrets isolated in `config.env`, never baked into images.  
* `/__live` health probe returns **200 OK** for K8s &amp; Traefik.

---

## 🆘 Trouble‑shoot in 30″

| Symptom | Quick fix |
|---------|-----------|
| *Docker missing* | Install ➜ <https://docs.docker.com/get-docker> |
| *Port 7860 busy* | Edit `ports:` in YAML |
| *ARM Mac build slow* | Enable **Rosetta** emulation in Docker settings |
| *Need GPU* | Switch base image to CUDA &amp; add `--gpus all` |

---

## 🤝 Credits & License

* Crafted with ❤️ by **Montreal.AI**.  
* Homage to the legends of RL – **David Silver &amp; Richard S. Sutton**.  
* MIT‑licensed. Use it, fork it, break it, improve it — just share the love.

> **Alpha‑Factory** — forging intelligence that *out‑learns, out‑thinks, out‑executes*.
