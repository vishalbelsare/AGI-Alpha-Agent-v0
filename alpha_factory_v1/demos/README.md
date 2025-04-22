<!--
  Era‑of‑Experience Demo
  Alpha‑Factory v1 👁️✨ — Multi‑Agent **AGENTIC α‑AGI**
  Out‑learn · Out‑think · Out‑strategise · Out‑execute
  © 2025 MONTREAL.AI   MIT License
-->

# 🌌 Welcome to the **Era of Experience** — Run it locally in ***one*** command

> “AI will eclipse the limits of human‑authored data only when agents **act, observe, and adapt** in the world.”  
> — *David Silver & Richard S. Sutton*

This demo fuses their vision with **Alpha‑Factory v1** — a production‑grade, multi‑agent AGI spine.  
Within 60 seconds you’ll watch an agent **evolve in real time**, guided by grounded rewards, long‑range memory and non‑human planning. No GPU required.

---

## 🚀 Zero‑friction quick‑start (macOS / Windows / Linux)

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/era_of_experience
chmod +x run_experience_demo.sh
./run_experience_demo.sh        #  ← THAT’S IT
```

1. **Docker Desktop** will build the image (≈ 1 min cold start).  
2. Your browser auto‑opens **http://localhost:7860**:  
   * live trace‑graph 🪄  
   * reward curves 📈  
   * interactive chat to inject new experience events 💬

> **No OpenAI key?** Leave `OPENAI_API_KEY` blank in `config.env` — the stack spins up *Ollama ✕ Mixtral* and stays fully offline.

---

## ✨ Why this matters

| Silver & Sutton’s Pillar | How the demo brings it alive |
|--------------------------|------------------------------|
| **Streams of experience** | Infinite generator feeds months‑long synthetic logs |
| **Sensor‑motor actions** | API calls & “plan meal” simulator mutate the environment |
| **Grounded rewards** | Fitness & Education signals — measurable, bias‑free |
| **Non‑human reasoning** | MCTS planner + vector memory, not chain‑of‑thought imitation |

The result: an agent that **rewrites its own playbook** every few seconds — exactly the leap beyond static prompt engineering the authors foresee.

---

## 🛠️ Architecture at a glance

```text
┌────────────┐   streams    ┌──────────────┐
│ Experience │ ───────────▶ │  Orchestrator│─────┐
└────────────┘              └──────────────┘     │ tool‑calls
        ▲                           │            ▼
 grounded│                    ┌────────────┐  ┌─────────────┐
 rewards │                    │  Planner   │  │  Tools/API  │
        │                      └────────────┘  └─────────────┘
        └──────────────────────────────────────────┘
```

* **openai‑agents‑python** → battle‑tested tool‑calling & memory  
* **A2A protocol hooks** → multi‑agent swarms ready out‑of‑the‑box  
* **Single Dockerfile** → deterministic, air‑gapped builds  

---

## 📚 Deep‑dive links

* Silver & Sutton, *The Era of Experience (2024)*  
* OpenAI, *A Practical Guide to Building Agents (2024)*  
* Google ADK & A2A specifications

---

## 🧩 Extending

* Drop a new reward backend into `reward_backends/` — it auto‑mounts.  
* Register a sensor‑motor tool with one decorator:  

  ```python
  @Tool(name="place_trade", description="execute an order on Alpaca")
  async def place_trade(ticker:str, qty:int, side:str): ...
  ```

* Scale‑out: `docker compose --scale orchestrator=4 …` for emergent cooperation.

---

## 🛡️ Security & Production notes

* The container runs **non‑root**, no exposed Docker socket.  
* Secrets stay in `config.env` (never committed).  
* Offline fallback eliminates third‑party data egress.  
* Health‑check endpoint `GET /__live` returns **200 OK** for Kubernetes probes.

---

## 🆘 Troubleshooting (30‑second cheat‑sheet)

| Symptom | Fix |
|---------|-----|
| “Docker not installed” | [Download Docker Desktop](https://docs.docker.com/get-docker) |
| Port 7860 already in use | Edit `ports:` in `docker-compose.experience.yml` |
| Build timeout on ARM Mac | Enable *“Use Rosetta for x86/amd64 emulation”* in Docker settings |
| Want GPU speed‑up | Replace base image tag with `nvidia/cuda:12.4.0-runtime-ubuntu22.04` and add `--gpus all` |

---

## 🤝 Credits

* Demo engineered by **Montreal.AI**.  
* Inspired by the legends of Reinforcement Learning, **David Silver & Richard S. Sutton**.  
* Powered by the open‑source community — thank you!

> **Alpha‑Factory** — forging intelligence that **out‑learns, out‑thinks, out‑executes**.
