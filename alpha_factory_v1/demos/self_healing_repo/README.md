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

## 🛠 Requirements

The demo expects a few extra packages:

- [`openai_agents`](https://openai.github.io/openai-agents-python/) (required unless you rely on the offline fallback)
- [`gradio`](https://gradio.app/)
- [`pytest`](https://docs.pytest.org/)
- GNU `patch`

`run_selfheal_demo.sh` verifies that `patch` is installed but does not check for `openai_agents`. The library is required for normal online usage. When the package is missing, the entrypoint automatically falls back to a minimal stub that calls the bundled offline model via `llm_client.call_local_model`.

## 🚀 Quick start (Docker)

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/self_healing_repo
sudo apt-get update && sudo apt-get install -y patch  # install GNU patch
chmod +x run_selfheal_demo.sh
./run_selfheal_demo.sh
```

Alternatively run `af demo heal` from the repository root after installation.

Before launching the dashboard or running tests, run `python alpha_factory_v1/scripts/preflight.py` (or `python check_env.py --auto-install`) from the repository root to confirm all dependencies. If the machine has no internet access, build a wheelhouse and run `python check_env.py --auto-install --wheelhouse <dir>` first (see **Build a wheelhouse & run offline**).

Browse **http://localhost:7863** → hit **“Heal Repository”**.

* No config needed; the agent clones a tiny repo with a deliberate bug.
* **With an OpenAI key** the agent uses GPT‑4o to reason about stack‑traces.
* **Offline?** Leave the key blank—Mixtral via Ollama drafts the patch.
* If the remote clone fails, the demo falls back to the bundled
  `sample_broken_calc` repository.

> **Note:** `run_selfheal_demo.sh` copies `config.env.sample` to `config.env` on the
> first run. Edit this file to configure OpenAI or your local model.
> Key settings include:

```bash
OPENAI_API_KEY=
OPENAI_MODEL="gpt-4o-mini"
TEMPERATURE=0.3
GRADIO_SHARE=0
USE_LOCAL_LLM=true
OLLAMA_BASE_URL="http://ollama:11434/v1"
CLONE_DIR="/tmp/demo_repo"  # sandbox for patched repo
```

When `OPENAI_API_KEY` is blank the agent falls back to the local model
via Ollama. Set `USE_LOCAL_LLM=true` to force this behaviour even when
a key is present. Set `OLLAMA_BASE_URL` to the base URL of your Ollama server (defaults to `http://ollama:11434/v1`) when the model runs on a remote host. The same file also lets you override `OPENAI_MODEL` and
`TEMPERATURE` for custom tuning. **`OPENAI_MODEL` controls both the
remote API model and the local one when `USE_LOCAL_LLM=true`.** Set
`CLONE_DIR` if you want the repository clone to live elsewhere.

### Windows (PowerShell)
Run the same container with PowerShell:

```powershell
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/self_healing_repo
winget install --id GnuWin32.Patch -e # install GNU patch (or `choco install patch`)
docker compose -p alpha_selfheal -f docker-compose.selfheal.yml up -d --build
```

Before launching the dashboard or running tests, run `python alpha_factory_v1/scripts/preflight.py` (or `python check_env.py --auto-install`) from the repository root to confirm all dependencies. Stop the stack with `docker compose -p alpha_selfheal down`.

### Quick start (Python)
Prefer a local run without Docker?
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../../..
pip install -r ../../backend/requirements.txt
sudo apt-get update && sudo apt-get install -y patch  # install GNU patch if missing
python -m alpha_factory_v1.demos.self_healing_repo.agent_selfheal_entrypoint
```
Then open **http://localhost:7863** and trigger **“Heal Repository”**.

Set `GRADIO_SHARE=1` to expose a public link (useful on Colab).
Set `TEMPERATURE=0.3` (0‑2) to tune patch creativity.

### Offline workflow

When the host has no internet access, `agent_selfheal_entrypoint.py`
clones the bundled `sample_broken_calc` repository instead of pulling
from GitHub.

#### Build a wheelhouse & run offline

Create a wheelhouse so Python packages install without the network:

```bash
cd /path/to/AGI-Alpha-Agent-v0
mkdir -p /media/wheels
pip wheel -r requirements.lock -w /media/wheels
pip wheel -r requirements-dev.txt -w /media/wheels
```

Before each run install from the wheelhouse and set the required
environment variables:

```bash
USE_LOCAL_LLM=true \
OLLAMA_BASE_URL=http://localhost:11434/v1 \
WHEELHOUSE=/media/wheels \
python check_env.py --auto-install --wheelhouse $WHEELHOUSE
```

Launch the demo with the same variables:

```bash
USE_LOCAL_LLM=true OLLAMA_BASE_URL=http://localhost:11434/v1 \
WHEELHOUSE=/media/wheels python -m alpha_factory_v1.demos.self_healing_repo.agent_selfheal_entrypoint
```

The dashboard behaves the same, but all code comes from the bundled repo.
Run `python alpha_factory_v1/scripts/preflight.py` (or `python check_env.py --auto-install --wheelhouse /media/wheels`) from the repository root to confirm dependencies before each run.

### Manual healing

You can run the patcher directly on any repository:

```bash
python patcher_core.py --repo <path>
```

Install the optional `openai_agents` package and the `patch` utility beforehand so the script can suggest and apply fixes.

When the library is missing the CLI automatically falls back to the offline model via
`agent_core.llm_client.call_local_model`. Configure the environment variables to
match your local setup:

```bash
OPENAI_MODEL=mixtral-8x7b \
TEMPERATURE=0.3 \
OLLAMA_BASE_URL=http://localhost:11434/v1 \
python patcher_core.py --repo <path>
```

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

```
clone repo → [sandbox pytest] → error log
                    ↑             ↓
        LLM diff ← [sandbox patch] ←┘
                    ↓
          [sandbox pytest] → commit+PR
```

---

## 🛡️ Security & Ops

* Container runs as **non‑root UID 1001**.  
* Patch application sandboxed to `/tmp/demo_repo`.
* Rollback on any `patch` failure; originals restored.
* Diff paths are validated relative to the cloned repository; any patch
  touching files outside this tree is rejected.
* **/__live** endpoint returns **200 OK** for readiness probes.

---

## 🆘 Troubleshooting

| Symptom | Remedy |
|---------|--------|
| “patch: command not found” | Rebuild the Docker image; the Dockerfile now installs `patch` |
| Port 7863 busy | Edit `ports:` in `docker-compose.selfheal.yml` |
| LLM exceeds context | Patch diff is now chunked; increase `OPENAI_CONTEXT_WINDOW` env if needed |

---

## 🤝 Credits

* Inspired by the *Self‑Healing Software* vision (S. Brun et al., 2023).  
* Built on **Agents SDK**, **A2A**, and the ever‑wise open‑source community.

> **Alpha‑Factory** — shipping code that ships itself.
