# ⚡ **α‑ASI World‑Model Helm Chart** 📦💫

> **Alpha‑Factory v1 👁️✨** — Cross‑Industry *Alpha Factory* Demo  
> Launch a self‑contained constellation of agents that evolves open‑ended
> worlds and trains a MuZero‑style learner toward **α‑ASI**.

| Chart ID | `alpha-asi-demo` |
|----------|------------------|
| Works on | K8s 1.25+ (Kind, GKE, EKS, AKS, Openshift OKD) |
| Ports    | `80/TCP` (REST + WS) |

---

## 🌟 Features
* **Multi‑agent micro‑services** (≥5 agents) wired by the Agent2Agent bus
* **POET environment‑factory** + **MuZero learner** in one pod
* **FastAPI** UI & OpenAPI docs out‑of‑the‑box
* **HPA‑ready** probes, PodDisruptionBudget, resource templates
* **Air‑gapped capable** — no external calls unless you set `OPENAI_API_KEY`

---

## 🚀 Quick start

```bash
helm repo add alpha-factory https://montrealai.github.io/alpha-factory-charts
helm install my-asi alpha-factory/alpha-asi-demo \
  --set image.tag=$(git rev-parse --short HEAD)
kubectl port-forward svc/my-asi 7860:80
open http://localhost:7860/docs   # Swagger UI
```

---

## 🔧 Important `values.yaml` knobs

| Key | Default | Why you might change it |
|-----|---------|-------------------------|
| `replicaCount` | `1` | Scale learner horizontally (stateless). |
| `image.repository` | `alpha_asi_world_model` | Use your own registry. |
| `service.type` | `ClusterIP` | `LoadBalancer` for cloud ELB, `NodePort` for bare‑metal. |
| `resources` | – | Request GPU (e.g. `nvidia.com/gpu: 1`). |
| `env.OPENAI_API_KEY` | – | Enable LLM‑assisted planning. |

See full schema inside **values.yaml**.

---

## 🩺 Health checks

| Probe      | Path     | Success criterion |
|------------|----------|-------------------|
| Liveness   | `/agents` | HTTP 200 JSON with ≥5 agents |
| Readiness  | `/agents` | same |

Helm **tests** run a curl + WebSocket sanity script.

---

## 🤖 Five highlighted agents

| Agent | Mission | Example signal |
|-------|---------|----------------|
| **PlanningAgent** | Break complex goals into sub‑tasks | Receives `llm_plan` JSON |
| **ResearchAgent** | Inject external knowledge | Emits `memory.add` |
| **StrategyAgent** | Adaptive curriculum scheduling | Sends `orch` → `new_env` |
| **MarketAnalysisAgent** | Spot cross‑industry Alpha | Publishes `alpha.opportunity` |
| **SafetyAgent** | Shutdown on unsafe gradients | Publishes `orch` → `stop` |

---

## 🧩 Extending the chart

1. Build your image → `docker build -t myrepo/alpha_asi:latest .`  
2. `helm upgrade --install` with `--set image.repository=myrepo/alpha_asi`  
3. Add volumes/env for new agents under `templates/deployment.yaml.j2`

PRs welcome — let’s push **α‑AGI** forward together! ✨
