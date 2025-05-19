<!--
 AI‑GA Meta‑Evolution Demo
 Alpha‑Factory v1 👁️✨ — Multi‑Agent **AGENTIC α‑AGI**
 Out‑learn · Out‑think · Out‑strategise · Out‑evolve
 © 2025 MONTREAL.AI   MIT License
 -------------------------------------------------------------------------------
 Exhaustive README: quick‑start, deep‑dive, SOC‑2 rails, CI/CD, K8s,
 observability, SBOM notice. Rendered as GitHub‑flavoured Markdown.
-->


# 🌌 Algorithms That Invent Algorithms — <br>**AI‑GA Meta‑Evolution Demo**

> *“Why hand‑craft intelligence when evolution can author it for you?”* 
> — Jeff Clune, *AI‑GAs: AI‑Generating Algorithms* (2019) 

A single‑command, browser‑based showcase of Clune’s **Three Pillars**:

| Pillar | Demo realisation |
| :-- | :-- |
| **Meta‑learning architectures** | Genome encodes a **typed list** of hidden sizes & activation; mutates via neuro‑evolution |
| **Meta‑learning the learning algorithms** | Runtime flag flips between *SGD* and a *fast Hebbian plasticity* inner‑loop |
| **Generating learning environments** | `CurriculumEnv` self‑mutates → *Line* → *Zig‑zag* → *Gap* → *Maze* |

Within **&lt; 60 s** you’ll watch neural nets **rewrite their own blueprint** *while the world itself mutates to stay challenging*.

---

<details open>
<summary>📑 Table of contents — click to jump</summary>

- [🚀 Quick‑start (Docker)](#-quick‑start-docker)
- [🎓 Run in Colab](#-run-in-colab)
- [🚀 Production deployment](#-production-deployment)
- [🔑 Online vs offline LLMs](#-online-vs-offline-llms)
- [🛠 Architecture deep‑dive](#-architecture-deep‑dive)
- [📈 Observability & metrics](#-observability--metrics)
- [🧪 Tests & CI](#-tests--ci)
- [☁️ Kubernetes deploy](#-kubernetes-deploy)
- [🛡 SOC‑2 & supply‑chain](#-soc‑2--supply‑chain)
- [🧩 Tinker guide](#-tinker-guide)
- [🆘 FAQ](#-faq)
- [🤝 Contributing](#-contributing)
- [⚖️ License & credits](#-license--credits)
</details>

---

## 🚀 Quick‑start (Docker)

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/aiga_meta_evolution

# optional: --pull (signed image) --gpu (NVIDIA runtime)
./run_aiga_demo.sh
```

The service automatically resumes from the latest checkpoint if one exists,
so you can stop and restart the container without losing progress.

| Endpoint | URL | Purpose |
| --- | --- | --- |
| **Gradio** UI | <http://localhost:7862> | Click **Evolve 5 Generations** |
| **FastAPI** docs | <http://localhost:8000/docs> | Programmatic control |
| **Prometheus** | <http://localhost:8000/metrics> | `aiga_*` gauges & counters |

> 🧊 **Cold build** ≈ 40 s (900 MB). Subsequent runs are instant (cache).

Minimal host reqs → Docker 24, ≥ 4 GB RAM, **no GPU** needed.

## 🚀 Quick‑start (Python)

Prefer running natively? The service also launches directly from the
repository without Docker. This path is handy for quick experiments or
when Docker is unavailable.

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0
AUTO_INSTALL_MISSING=1 python check_env.py  # verify deps offline/online
pip install -r alpha_factory_v1/requirements.txt
# ensures `openai-agents` and friends are installed
python alpha_factory_v1/demos/aiga_meta_evolution/agent_aiga_entrypoint.py
# optional cross‑platform launcher
python alpha_factory_v1/demos/aiga_meta_evolution/start_aiga_demo.py --help
# or via module entrypoint
python -m alpha_factory_v1.demos.aiga_meta_evolution --help
```

Set `OPENAI_API_KEY` in your environment to enable cloud models. Without
it the demo falls back to the bundled offline mixtral model.

### 🤖 OpenAI Agents bridge

Expose the evolver to the **OpenAI Agents SDK** runtime:

```bash
python alpha_factory_v1/demos/aiga_meta_evolution/openai_agents_bridge.py
```

Requires the `openai-agents` package (installed via requirements).

The bridge registers an `aiga_evolver` agent exposing four tools:
`evolve` (run N generations), `best_alpha` (return the champion),
`checkpoint` (persist state), and `reset` (fresh population).
It works offline by routing to the local Mixtral server when no API key
is configured.

### 🛰️ Google ADK gateway

Set `ALPHA_FACTORY_ENABLE_ADK=true` to expose the same agent via a local
Google **Agent Development Kit** gateway:

```bash
ALPHA_FACTORY_ENABLE_ADK=true python openai_agents_bridge.py &
```

This publishes the tools over the **A2A protocol** so other agents can
orchestrate evolution remotely.
Set `ALPHA_FACTORY_ENABLE_ADK=1` in `config.env` to auto-start the gateway
when running `./run_aiga_demo.sh`.

## 🔐 API authentication

Export `AUTH_BEARER_TOKEN` to require a static token on every API request. For
JWT-based auth, provide `JWT_PUBLIC_KEY` (PEM) and optional `JWT_ISSUER`.
The `/health` and `/metrics` endpoints remain public.

---
## 🚀 Production deployment

For step-by-step instructions on running the service in a production or workshop environment, see [PRODUCTION_GUIDE.md](PRODUCTION_GUIDE.md).


## 🎓 Run in Colab

| | |
| :-- | :-- |
| <a href="https://colab.research.google.com/github/MontrealAI/AGI-Alpha-Agent-v0/blob/main/alpha_factory_v1/demos/aiga_meta_evolution/colab_aiga_meta_evolution.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> | Launches the same dashboard with an automatic public URL. Ideal for workshops & quick demos. |

---

## 🔑 Online vs offline LLMs

| Environment variable | Effect |
| --- | --- |
| `OPENAI_API_KEY` **set** | Tools routed through **OpenAI Agents SDK** |
| **unset / empty** | Drops to **Mixtral‑8x7B‑Instruct** via local Ollama side‑car – *zero network egress* |

LLMs supply *commentary & analysis* only – **core evolution is deterministic**.

## 🔍 Alpha discovery stub

For a bite-size illustration of agent-driven opportunity scanning, run:

```bash
python alpha_factory_v1/demos/aiga_meta_evolution/alpha_opportunity_stub.py
```

To query a specific domain once without starting the full runtime:

```bash
python alpha_factory_v1/demos/aiga_meta_evolution/alpha_opportunity_stub.py \
  --domain supply-chain --once
```

The `alpha_discovery` agent exposes a single `identify_alpha` tool that asks the LLM to suggest three inefficiencies in a chosen domain. It works offline when `OPENAI_API_KEY` is unset.

## ♻️ Alpha conversion stub

Turn a discovered opportunity into a short execution plan:

```bash
python alpha_factory_v1/demos/aiga_meta_evolution/alpha_conversion_stub.py --alpha "Battery arbitrage"
```

The tool outputs a three‑step JSON plan and logs it to `alpha_conversion_log.json`. When `OPENAI_API_KEY` is configured, it queries an OpenAI model; otherwise a sample plan is returned.

## 🤝 End‑to‑end workflow

Combine discovery and conversion into a single agent:

```bash
python alpha_factory_v1/demos/aiga_meta_evolution/workflow_demo.py
```

The `alpha_workflow` agent lists opportunities in the chosen domain, selects the
first suggestion and returns a short execution plan. When Google ADK is enabled
(via `ALPHA_FACTORY_ENABLE_ADK=1` and successful import of the ADK module), the same
agent is published over the A2A protocol for orchestration by external controllers.


---

## 🛠 Architecture deep‑dive

```text
┌── docker‑compose ─────────────┐
│ orchestrator (FastAPI + UI) │◀─────────┐
│ ollama (Mixtral fallback)  │     │ WebSocket
│ prometheus (opt)       │     │
└───────────────────────────────┘     │
    ▲ REST / Ray RPC          │
┌────────────────────────────────────────┐ │
│ MetaEvolver    checkpoint.json  │ │
│  ├─ Ray / mp evaluation workers   │ │
│  └─ EvoNet(nn.Module) ──┐      │ │ obs/reward
│              ▼      │ │
│ CurriculumEnv (Gymnasium)       │◀┘
└────────────────────────────────────────┘
```

* **MetaEvolver** – pop 24, tournament‑k 3, elitism 2, novelty bonus toggle 
* **EvoNet** – arbitrary hidden layers, activation ∈ {relu,tanh,sigmoid}, optional Hebbian ΔW 
* **CurriculumEnv** – 12 × 12 grid, DFS solvability check, energy budget, genome auto‑mutation 

---

## 📈 Observability & metrics

| Metric | Meaning |
| :-- | :-- |
| `aiga_avg_fitness` | Mean generation fitness |
| `aiga_best_fitness` | Elite fitness |
| `aiga_generations_total` | Counter |
| `aiga_curriculum_stage` | 0–3 |

Enable profile `telemetry` to autopush → Prometheus → Grafana. 
`docker compose --profile telemetry up`.

---

## 🧪 Tests & CI

* **Coverage ≥ 90 %** in < 0.5 s (`pytest -q`) 
* GitHub Actions → lint → test → build → Cosign sign 
* **SBOM** via *Syft* (SPDX v3) per release

---

## ☁️ Kubernetes deploy

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: aiga-demo }
spec:
 replicas: 1
 selector: { matchLabels: { app: aiga-demo } }
 template:
  metadata: { labels: { app: aiga-demo } }
  spec:
   containers:
   - name: orchestrator
    image: ghcr.io/montrealai/alpha-aiga:latest@sha256:<signed>
    ports:
    - { containerPort: 8000 }  # API
    - { containerPort: 7862 }  # UI
    readinessProbe:
     httpGet: { path: /health, port: 8000 }
    envFrom: [{ secretRef: { name: aiga-secrets } }]
```

*Helm chart* → `infra/helm/aiga-demo/`.

---

## 🛡 SOC‑2 & supply‑chain

* Cosign‑signed images (`cosign verify …`) 
* Runs **non‑root UID 1001**, read‑only code volume 
* Secrets via K8s / Docker *secrets* (never baked into layers) 
* Dependencies hashed (Poetry lock) & validated at runtime 
* SBOM exported; SLSA level 2 pipeline

---

## 🧩 Tinker guide

| Goal | File | Hint |
| --- | --- | --- |
| Bigger populations | `meta_evolver.py` → `pop_size` | Add `--profile gpu` |
| Faster novelty | `Genome.novelty_weight` | Try 0.2 |
| New curriculum stage | `curriculum_env.py` | Extend `_valid_layout` |
| Swap LLM | `config.env` | Any OpenAI model id |
| Automate experimentation | FastAPI → `/evolve/{n}` | Deterministic SHA checkpoint id |
| Manual reset | FastAPI → `POST /reset` | Fresh population |
| Persist progress | FastAPI → `POST /checkpoint` | Atomic save |

---

## 🆘 FAQ

| Symptom | Fix |
| :-- | :-- |
| “Docker not installed” | <https://docs.docker.com/get-docker> |
| Port collision 7862 | Edit host port in compose |
| ARM Mac slow build | Enable **Rosetta** or `./run_aiga_demo.sh --pull` |
| GPU unseen | `sudo apt install nvidia-container-toolkit` & restart Docker |
| Colab URL missing | Re‑run launch cell (ngrok quirk) |

---

## ⚖️ License & credits

*Source & assets* © 2025 Montreal.AI, released under the **MIT License**. 
Huge thanks to:

* **Jeff Clune** – visionary behind AI‑GAs 
* **OpenAI, Anthropic, Google** – open‑sourcing crucial agent tooling 
* OSS maintainers – you make this possible ♥

> **Alpha‑Factory** — forging intelligence that *invents* intelligence.
