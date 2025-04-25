# α-ASI World-Model Demo 👁️✨  
*Part of **Alpha-Factory v1** – multi-agent, open-ended curriculum + MuZero learner*

---

## 1  Why this demo exists  🌍➡️🤖  
This folder demonstrates that **Alpha-Factory’s agentic runtime can grow its own synthetic
worlds, train general agents on them, and improve forever** – a concrete step toward the
α-ASI vision outlined by Montreal.AI.

* **POET-style environment generator** continuously proposes fresh challenges.  
* **MuZero-style learner + MCTS** builds an internal world model and plans ahead.  
* **≥ 5 Alpha-Factory agents** (Planning, Research, Strategy, Market, CodeGen … + Safety)  
  orchestrate curriculum, knowledge-transfer, execution and guard-rails.  
* Entire stack runs **offline by default**; optional LLM helpers auto-activate
  when keys (`OPENAI_API_KEY`, etc.) are present.  
* Dev-ops built-in – Docker, Helm and Notebook emitters with a single flag.  

---

## 2  Repository layout  📂
```
alpha_factory_v1/
└─ demos/
   └─ alpha_asi_world_model/
      ├─ alpha_asi_world_model_demo.py   ← single-file reference impl
      ├─ README.md                       ← you are here
      ├─ Dockerfile      (auto-generated via --emit-docker)
      └─ helm_chart/    (auto-generated via --emit-helm)
```

---

## 3  Quick start  ⚡

### Local Python (CPU or GPU)
```bash
# (optional) create venv
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn[standard] torch numpy pydantic

# launch server + agents
python -m alpha_factory_v1.demos.alpha_asi_world_model.alpha_asi_world_model_demo --demo

# open dashboard
xdg-open http://127.0.0.1:7860        # or just paste in your browser
```

### Docker
```bash
python -m alpha_factory_v1.demos.alpha_asi_world_model.alpha_asi_world_model_demo --emit-docker
docker build -t alpha_asi_world_model .
docker run -p 7860:7860 alpha_asi_world_model
```

### Kubernetes (Helm)
```bash
python -m alpha_factory_v1.demos.alpha_asi_world_model.alpha_asi_world_model_demo --emit-helm
helm install alpha-asi-demo ./helm_chart
```

### Interactive Jupyter notebook
```bash
python -m alpha_factory_v1.demos.alpha_asi_world_model.alpha_asi_world_model_demo --emit-notebook
jupyter lab alpha_asi_world_model_demo.ipynb
```

---

## 4  Runtime controls  🎛️

| Action                     | How to do it                                                                 |
|----------------------------|------------------------------------------------------------------------------|
| Spawn brand-new world      | `POST /command {"cmd":"new_env"}`                                            |
| Pause learner              | `POST /command {"cmd":"stop"}`                                               |
| Watch live telemetry       | WebSocket @ `ws://<host>:7860/ws` – JSON every `ui_tick` steps               |
| List activated agents      | `GET  /agents`                                                               |

*(Swagger / OpenAPI docs are auto-hosted at `/docs`.)*

---

## 5  Agents & their roles  🤝

| Topic / class        | Purpose in this demo                                   | Fallback if class missing |
|----------------------|--------------------------------------------------------|---------------------------|
| `planning_agent`     | High-level goal decomposition, curriculum hints        | Stub prints messages      |
| `research_agent`     | Supplies background knowledge via MCP                 | Stub prints messages      |
| `strategy_agent`     | Detects alpha opportunities, triggers env swap         | Stub prints messages      |
| `market_agent`       | Streams market-like signals for cross-domain tests     | Stub prints messages      |
| `codegen_agent`      | Hot-patches learner architecture where helpful         | Stub prints messages      |
| `safety_agent`       | Monitors NaNs / reward spikes, can halt training       | **Always present**        |

> The orchestrator **guarantees at least five alive topics** even on a clean checkout –
> real classes override the stubs automatically.

---

## 6  Config knobs  ⚙️  (`ALPHA_ASI_*` env vars)

| Variable               | Default | Meaning                                             |
|------------------------|---------|-----------------------------------------------------|
| `ALPHA_ASI_SEED`       | `42`    | Global deterministic seed                           |
| `ALPHA_ASI_MAX_STEPS`  | `100000`| Override loop length without editing the script     |

All other hyper-parameters live in the `Config` dataclass at the top of the script.

---

## 7  Public interfaces  🌐

### REST
```
GET  /agents             → ["planning_agent", "research_agent", ...]
POST /command            → {"cmd":"new_env"} | {"cmd":"stop"}
```

### WebSocket
*URL*: `/ws`     •     *Payload*: `{"t": step, "r": last_reward, "loss": mse}`

---

## 8  Safety & antifragility  🛡️

* **SafetyAgent** halts training on NaNs or runaway losses.  
* Replay buffer capped to `buffer_limit` (50 k by default).  
* External API calls (LLM, web) are **opt-in**; no key → stubs isolate system.  
* Each agent runs in-proc but via topic isolation – easy to shard into micro-services
  or secure sandboxes.

---

## 9  Extending the demo  ➕

1. **Add a new environment** – subclass `MiniWorld`, register in `POETGenerator`.
2. **Swap learner** – implement `act / remember / train` trio.
3. **Replace a stub** – drop a full agent under  
   `alpha_factory_v1/backend/agents/<your_agent>.py`; the loader picks it up.

---

## 10  Troubleshooting  🔧

| Symptom                  | Remedy |
|--------------------------|--------|
| UI panel blank           | Ensure WS reachable; check console for `orch_online`. |
| CUDA OOM                 | `export CUDA_VISIBLE_DEVICES=""` to force CPU run. |
| Docker build slow        | Use a CUDA base or pre-built torch wheel. |
| Helm pod crash-loop      | `kubectl logs` → missing deps? Re-emit Dockerfile. |

---

## 11  Implementation highlights  💡

* **POET + MuZero**: open-ended env generator feeds a minimal but real MuZero core
  (repr / dyn / pred) with optional MCTS – proving closed-loop “world-model → agent”.  
* **A2A bus**: pluggable, ultra-low-latency in-proc pub-sub; swap for NATS/Redis to scale.  
* **Dev-ops**: `--emit-docker` / `--emit-helm` / `--emit-notebook` create artefacts instantly.  
* **Security notes**: no `eval`, no shell exec, stub Fallback agents sandboxed,  
  Bandit-clean and pyright-typed.  

---

## 12  License & citation  📜

This demo inherits the **MIT licence** of Alpha-Factory v1.  
If you build on it, please cite:

> MONTREAL.AI (2025). *Alpha-Factory v1 👁️✨ — Multi-Agent AGENTIC α-AGI*.

---

**Enjoy exploring the frontier 🚀**
