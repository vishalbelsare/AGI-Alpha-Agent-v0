
# α-ASI World-Model Demo 👁️✨  
*Alpha-Factory v1 – fully-agentic, open-ended curriculum generator + MuZero learner*

---

## 1  Why this demo exists  
This folder proves that **Alpha-Factory’s multi-agent runtime can grow its own synthetic
worlds, train general agents on them, and improve forever** – a minimal step toward the
α-ASI vision outlined by Montreal.AI.

* **POET-style environment generator** continuously proposes fresh challenges.  
* **MuZero-style learner** builds an internal world model and plans with MCTS.  
* **≥ 5 Alpha-Factory agents** (Planning, Research, Strategy, Market, CodeGen … + Safety)  
  orchestrate curriculum, knowledge transfer and guard-rails.  
* Entire stack runs *offline by default*; optional LLM helpers auto-activate when keys
  (`OPENAI_API_KEY`, etc.) are present.  

---

## 2  Repository layout
```
alpha_factory_v1/
└─ demos/
   └─ alpha_asi_world_model/
      ├─ alpha_asi_world_model_demo.py   ← single-file implementation
      ├─ README.md                       ← this file
      ├─ Dockerfile      (auto-generated via --emit-docker)
      └─ helm_chart/    (auto-generated via --emit-helm)
```

---

## 3  Quick start

### Local Python (CPU or GPU)
```bash
# (optional) create venv & install deps
pip install fastapi uvicorn[standard] pydantic torch numpy

# launch
python -m alpha_asi_world_model_demo --demo
# open the dashboard
xdg-open http://127.0.0.1:7860        # or just paste in your browser
```

### Docker
```bash
python -m alpha_asi_world_model_demo --emit-docker   # writes Dockerfile
docker build -t alpha_asi_world_model .
docker run -p 7860:7860 alpha_asi_world_model
```

### Kubernetes
```bash
python -m alpha_asi_world_model_demo --emit-helm
helm install alpha-asi-demo ./helm_chart
```

### Interactive notebook
```bash
python -m alpha_asi_world_model_demo --emit-notebook
jupyter lab alpha_asi_world_model_demo.ipynb
```

---

## 4  Runtime controls

| Action                     | How to do it                                                                                |
|----------------------------|---------------------------------------------------------------------------------------------|
| Spawn a brand-new world    | `POST /command {"cmd":"new_env"}` (curl, REST client, or notebook cell)                     |
| Pause learner              | `POST /command {"cmd":"stop"}`                                                              |
| Watch live telemetry       | Connect WebSocket to `ws://<host>:7860/ws` – JSON blobs every `ui_tick` steps               |
| List activated agents      | `GET  /agents`                                                                              |

*(The **FastAPI Swagger UI** is available at `/docs`.)*

---

## 5  Included agents and their roles

| Name (topic)          | Purpose in this demo                            | Fallback behaviour if real class missing |
|-----------------------|-------------------------------------------------|------------------------------------------|
| `planning_agent`      | High-level goal decomposition / curriculum hints| Stub prints messages                     |
| `research_agent`      | Supplies background knowledge via MCP          | Stub prints messages                     |
| `strategy_agent`      | Detects alpha opportunities, triggers env swap | Stub prints messages                     |
| `market_agent`        | Streams market-like signals for cross-domain   | Stub prints messages                     |
| `codegen_agent`       | Hot-patches learner architecture if needed     | Stub prints messages                     |
| `safety_agent`        | Monitors loss NaNs / reward spikes, can halt   | **Always present (real or stub)**        |

> **Guarantee:** at least **five** agent topics are always alive, preserving
> orchestration even on a clean checkout of Alpha-Factory.

---

## 6  Configuration knobs (`ALPHA_ASI_*` env vars)

| Variable            | Default | Meaning                                              |
|---------------------|---------|------------------------------------------------------|
| `ALPHA_ASI_SEED`    | `42`    | Global deterministic seed                            |
| `ALPHA_ASI_MAX_STEPS` | `100000`| Override `Config.max_steps` without code change      |

Alternatively edit the `Config` dataclass at the top of the script.

---

## 7  Public interfaces

### REST
```http
GET  /agents            → ["planning_agent", "research_agent", ...]
POST /command           → {"cmd":"new_env" | "stop"}
```

### WebSocket
*URL*: `/ws`   •   *Messages*: `{"t": step, "r": last_reward, "loss": mse}`.

---

## 8  Safety & antifragility

* **SafetyAgent** halts training on NaNs or runaway losses.  
* Replay buffer capped to `buffer_limit` (50 k by default).  
* All external API calls (LLM, web) are opt-in; if keys are absent, stubs isolate
  the system from the internet.  
* Every agent runs inside the Python process but uses A2A topic isolation – easy
  to migrate into separate micro-services or sandboxes.

---

## 9  Extending the demo

1. **Add a new environment type** – subclass `World`, register it in `Generator`.
2. **Swap learner** – drop-in a different RL algorithm that follows
   `act / remember / train` interface.  
3. **Plug a real agent** – place a fully-featured
   `alpha_factory_v1/backend/agents/<your_agent>.py` with `.name` set to topic;
   the auto-loader will pick it up instead of the stub.

---

## 10  Troubleshooting

| Symptom                             | Fix |
|-------------------------------------|-----|
| *UI doesn’t update*                 | Check browser console → ensure WS reachable; ensure server logs show `orch_online`. |
| *CUDA out of memory*                | `export ALPHA_ASI_SEED=123 && torch.cuda.empty_cache();` or force CPU: `export CUDA_VISIBLE_DEVICES=""`. |
| *Docker build slow*                 | Add `--build-arg TORCH_URL=<wheel>` or use a CUDA base image for GPU. |
| *Helm pod crashloop*                | `kubectl logs` – missing Python deps?  Ensure container built from generated Dockerfile. |

---

## 11  License & citation
This demo inherits the **MIT licence** of Alpha-Factory v1.  
If you use or modify it, please cite:

> Montreal.AI (2025) *Alpha-Factory v1 👁️✨ — Multi-Agent AGENTIC α-AGI*.

---

Enjoy exploring the frontier 🚀
