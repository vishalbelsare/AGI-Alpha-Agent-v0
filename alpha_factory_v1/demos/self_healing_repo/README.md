<!--
  Self‑Healing Repo Demo
  Alpha‑Factory v1 👁️✨ — Multi‑Agent **AGENTIC α‑AGI**
  Out‑learn · Out‑think · Out‑debug · Out‑ship
  © 2025 MONTREAL.AI   Apache‑2.0 License
-->

# 🔧 **Self‑Healing Repo** — when CI fails, agents patch

Imagine a codebase that diagnoses its own wounds, stitches the bug, and walks
back onto the production floor—all before coffee drips.  
This demo turns that fantasy into a clickable reality inside **Alpha‑Factory v1**.

## Disclaimer
This demo is a conceptual research prototype. References to "AGI" and
"superintelligence" describe aspirational goals and do not indicate the presence
of a real general intelligence. Use at your own risk. Nothing herein constitutes
financial advice. MontrealAI and the maintainers accept no liability for losses
incurred from using this software.

---

## 🚀 Quick start (Docker)

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/self_healing_repo
sudo apt-get update && sudo apt-get install -y patch  # install GNU patch
chmod +x run_selfheal_demo.sh
./run_selfheal_demo.sh
```

Before launching the dashboard or running tests, run `python alpha_factory_v1/scripts/preflight.py` (or `python check_env.py --auto-install`) from the repository root to confirm all dependencies.

Browse **http://localhost:7863** → hit **“Heal Repository”**.

* No config needed; the agent clones a tiny repo with a deliberate bug.
* **With an OpenAI key** the agent uses GPT‑4o to reason about stack‑traces.
* **Offline?** Leave the key blank—Mixtral via Ollama drafts the patch.
* If the remote clone fails, the demo falls back to the bundled
  `sample_broken_calc` repository.

> **Note:** `run_selfheal_demo.sh` copies `config.env.sample` to `config.env` on the
> first run. Edit this file to add your `OPENAI_API_KEY`, choose a `MODEL_NAME`,
> tweak `TEMPERATURE`, and set other options.

```bash
OPENAI_API_KEY=
MODEL_NAME="gpt-4o-mini"
TEMPERATURE=0.3
GRADIO_SHARE=0
```

Set `USE_LOCAL_LLM=true` in `config.env` to force the agent to run the
local Mixtral model when no API key is provided. The same file also lets
you override `MODEL_NAME` and `TEMPERATURE` for custom models or
tuning.

### Quick start (Python)
Prefer a local run without Docker?
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../../..
pip install -r ../../backend/requirements.txt
sudo apt-get update && sudo apt-get install -y patch  # install GNU patch if missing
python agent_selfheal_entrypoint.py
```
Then open **http://localhost:7863** and trigger **“Heal Repository”**.

Set `GRADIO_SHARE=1` to expose a public link (useful on Colab).
Set `TEMPERATURE=0.3` (0‑2) to tune patch creativity.

### Offline workflow

When the host has no internet access, `agent_selfheal_entrypoint.py`
automatically clones the included `sample_broken_calc` repository
instead of pulling from GitHub. Install dependencies from a local
wheelhouse and run the entrypoint directly:

```bash
WHEELHOUSE=/media/wheels python agent_selfheal_entrypoint.py
```

The dashboard behaves the same, but all code comes from the bundled repo.
See [../../scripts/README.md](../../scripts/README.md) for details on building a wheelhouse.

### Before running tests

Verify your environment first:

```bash
python scripts/check_python_deps.py
python alpha_factory_v1/scripts/preflight.py  # or python check_env.py --auto-install
```

Missing dependencies will cause tests to be skipped or fail.

---

## 🎓 Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MontrealAI/AGI-Alpha-Agent-v0/blob/main/alpha_factory_v1/demos/self_healing_repo/colab_self_healing_repo.ipynb)

Runs the same flow with a public Gradio link.
The notebook sets `GRADIO_SHARE=1` so the dashboard URL appears automatically.

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
