# `alpha_factory_v1/scripts` — Zero‑to‑Alpha in One Command ⚡️  

> **Part of [Alpha‑Factory v1 👁️✨](https://github.com/MontrealAI/AGI-Alpha-Agent-v0/tree/main/alpha_factory_v1)** – the multi‑agent AGENTIC α‑AGI that  
> *Out‑learns · Out‑thinks · Out‑designs · Out‑strategises · Out‑executes*  

Welcome!  
This folder contains the **bootstrap scripts** that turn any Docker‑enabled
machine into a **running, self‑testable α‑Factory skeleton in under a minute** –
even when you’re totally offline.

---

## 🚀 Quick start

```bash
# 1 · make the installer executable
chmod +x scripts/install_alpha_factory_v1.sh

# 2 · run it (add --no-cache for a clean build)
./scripts/install_alpha_factory_v1.sh

# 3 · self‑check
docker compose -p alpha_factory exec orchestrator pytest -q /app/tests
# → .
#   1 passed in X.XX s
```

The script prints two URLs on success:

| Service | URL | What you’ll see |
|---------|-----|-----------------|
| **Backend API** | `http://localhost:<backend‑port>/docs` | Swagger / Redoc |
| **Trace UI** | `http://localhost:<ui‑port>` | “α‑Factory ✔” banner |

*(Ports auto‑shift if 8080/3000 are busy.)*

---

## 🧐 What happens under the hood

1. **Prereq guardrails** – checks Docker, Git, Curl, Unzip, `ss` / `lsof`.  
2. **Shallow clone** of the repo (ZIP fallback).  
3. **Port scan** – finds free host ports for backend, proxy, mesh, UI.  
4. **Self‑scaffolds** FastAPI servers & Dockerfiles if missing.  
5. **Secrets prompt** – asks for `OPENAI_API_KEY`; spawns offline Ollama fallback when empty.  
6. **Generates** `docker‑compose.yaml` with dynamic ports.  
7. **Builds & starts** containers under project `alpha_factory`.  
8. **Runs** a pytest health‑check inside the backend.

---

## 🔧 Next steps

| Task | Command |
|------|---------|
| Follow logs | `docker compose -p alpha_factory logs -f orchestrator ui` |
| Hot‑reload backend | edit code → `docker compose -p alpha_factory build backend && docker compose -p alpha_factory up -d backend` |
| Swap default LLM | change `OLLAMA_MODEL` in `docker-compose.yaml` |
| Add a new agent | create `backend/my_agent.py` + router; rebuild backend |
| CI smoke‑test | run this script + pytest inside GitHub Actions |

---

## 🛡️ Security

* `.env` is git‑ignored – secrets stay on your machine.  
* Each container has a Docker **HEALTHCHECK**; backend exposes Prometheus metrics.

---

## 📜 License

MIT – © 2025 [MONTREAL.AI](https://montreal.ai).  
