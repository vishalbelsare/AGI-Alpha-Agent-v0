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
> — Jeff Clune, *AI‑GAs: AI‑Generating Algorithms* (2019) citeturn3file0

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

# optional: --pull (signed image)  --gpu (NVIDIA runtime)
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

---

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

---

## 🛠 Architecture deep‑dive

```text
┌── docker‑compose ─────────────┐
│ orchestrator  (FastAPI + UI)  │◀─────────┐
│ ollama  (Mixtral fallback)    │          │ WebSocket
│ prometheus  (opt)             │          │
└───────────────────────────────┘          │
        ▲ REST / Ray RPC                   │
┌────────────────────────────────────────┐ │
│  MetaEvolver        checkpoint.json    │ │
│    ├─ Ray / mp evaluation workers      │ │
│    └─ EvoNet(nn.Module) ──┐            │ │ obs/reward
│                           ▼            │ │
│  CurriculumEnv (Gymnasium)             │◀┘
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
        - { containerPort: 8000 }   # API
        - { containerPort: 7862 }   # UI
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
