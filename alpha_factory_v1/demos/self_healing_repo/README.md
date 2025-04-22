<!--
  Self‑Healing Repo Demo
  Alpha‑Factory v1 👁️✨ — Multi‑Agent **AGENTIC α‑AGI**
  Out‑learn · Out‑think · Out‑debug · Out‑ship
  © 2025 MONTREAL.AI   MIT License
-->

# 🔧 **Self‑Healing Repo** — when CI fails, agents patch

Imagine a codebase that diagnoses its own wounds, stitches the bug, and walks
back onto the production floor—all before coffee drips.  
This demo turns that fantasy into a clickable reality inside **Alpha‑Factory v1**.

---

## 🚀 Quick start (Docker)

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/self_healing_repo
chmod +x run_selfheal_demo.sh
./run_selfheal_demo.sh
```

Browse **http://localhost:7863** → hit **“Heal Repository”**.

* No config needed; the agent clones a tiny repo with a deliberate bug.
* **With an OpenAI key** the agent uses GPT‑4o to reason about stack‑traces.  
* **Offline?** Leave the key blank—Mixtral via Ollama drafts the patch.

---

## 🎓 Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MontrealAI/AGI-Alpha-Agent-v0/blob/main/alpha_factory_v1/demos/self_healing_repo/colab_self_healing_repo.ipynb)

Runs the same flow with a public Gradio link.

---

## 🛠️ What happens under the hood

| Step | Tool call | Outcome |
|------|-----------|---------|
| **1** | `run_tests` | Pytest reveals a failure |
| **2** | `suggest_patch` | LLM converts stack‑trace → unified diff |
| **3** | `apply_patch_and_retst` | Diff applied atomically → tests pass |

* Powered by **OpenAI Agents SDK v0.4** tool‑calling.  
* **A2A protocol** ready: spin up multiple healers across micro‑repos.  
* **Model Context Protocol** streams only the diff—not the whole file—for
  context‑efficient reasoning.

---

## 🛡️ Security & Ops

* Container runs as **non‑root UID 1001**.  
* Patch application sandboxed to `/tmp/demo_repo`.  
* Rollback on any `patch` failure; originals restored.  
* **/__live** endpoint returns **200 OK** for readiness probes.

---

## 🆘 Troubleshooting

| Symptom | Remedy |
|---------|--------|
| “patch: command not found” | `apt-get update && apt-get install -y patch` inside Dockerfile |
| Port 7863 busy | Edit `ports:` in `docker-compose.selfheal.yml` |
| LLM exceeds context | Patch diff is now chunked; increase `OPENAI_CONTEXT_WINDOW` env if needed |

---

## 🤝 Credits

* Inspired by the *Self‑Healing Software* vision (S. Brun et al., 2023).  
* Built on **Agents SDK**, **A2A**, and the ever‑wise open‑source community.

> **Alpha‑Factory** — shipping code that ships itself.
