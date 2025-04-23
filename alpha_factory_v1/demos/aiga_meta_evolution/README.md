<!--
  AI‑GA Meta‑Evolution Demo
  Alpha‑Factory v1 👁️✨ — Multi‑Agent **AGENTIC α‑AGI**
  Out‑learn · Out‑think · Out‑strategise · Out‑evolve
  © 2025 MONTREAL.AI   MIT License
  -------------------------------------------------------------------------------
  This README is intentionally exhaustive: quick-start, deep-dive, SOC-2 rails,
  CI/CD, K8s, observability, troubleshooting, contributor guide, SBOM notice.
-->

# 🌌 Algorithms That Invent Algorithms — **AI-GA Meta-Evolution Demo**

> *“Why hand-craft intelligence when evolution can author it for you?”*  
> — Jeff Clune, **AI-GAs: AI-Generating Algorithms** (2019)  

This is a one-command, browser-based showcase of Clune’s **Three Pillars**:

| Pillar | Demo realisation |
|--------|------------------|
| **Meta-learning architectures** | Genome encodes *variable hidden-layer list* & activation; mutates via neuro-evolution |
| **Meta-learning the learning algorithms** | Flag toggles **SGD** ↔ **fast Hebbian plasticity** inside *EvoNet* |
| **Generating learning environments** | `CurriculumEnv` self-mutates through 4 stages (Line → Zig-zag → Gap → Maze) |

Within 60 s you’ll watch neural networks **rewrite their own blueprint**
*while the world itself mutates to keep them sharp*.

---

*Table of contents • [(↑ back to top)](#)*

- [🚀 Quick start (local Docker)](#-quick-start-local-docker)
- [🔑 OpenAI vs offline Mixtral](#-openai-vs-offline-mixtral)
- [🛠 Architecture deep-dive](#-architecture-deep-dive)
- [📈 Observability & metrics](#-observability--metrics)
- [🧪 Tests & CI](#-tests--ci)
- [☁️ Deploying to Kubernetes](#-deploying-to-kubernetes)
- [🛡 SOC-2 / supply-chain rails](#-soc2--supply-chain-rails)
- [🧩 Tinker guide](#-tinker-guide)
- [🆘 Troubleshooting FAQ](#-troubleshooting-faq)
- [🤝 Contributing](#-contributing)
- [⚖️ License & credits](#-license--credits)

---

## 🚀 Quick start (local Docker)

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/aiga_meta_evolution

# optional flags:  --pull  use signed image from GHCR
#                  --gpu   enable NVIDIA runtime (CUDA ≥ 12)
./run_aiga_demo.sh
```

| Component | URL / Port | What you get |
|-----------|-----------|--------------|
| Gradio dashboard | <http://localhost:7862> | Click **Evolve 5 Generations** to iterate |
| FastAPI | <http://localhost:8000/docs> | OpenAPI JSON API |
| Prometheus scrape | <http://localhost:8000/metrics> | `aiga_*` metrics (avg fitness, stage, gen count) |

> **Cold build** ≤ 40 s on modern laptop (≈ 900 MB image).  
> Re-runs are instant (cached layers).

### Minimal prerequisites

* Docker 24 + compose plug-in  
* ≈ 4 GB RAM (8 GB if you bump `pop_size`)  
* **No GPU required** – runs CPU-only by default.

---

## 🔑 OpenAI vs offline Mixtral

The stack auto-detects `OPENAI_API_KEY` in `config.env`.

| Scenario | Behaviour |
|----------|-----------|
| `OPENAI_API_KEY=` **set** | LLM commentary & planning via OpenAI Agents SDK |
| **blank / unset** | Falls back to **Mixtral-8x7B-Instruct** served by Ollama side-car – runs 100 % offline |

Neither path changes core evolution logic; LLMs are *assistants*, not oracles.

---

## 🛠 Architecture deep-dive

```
┌────────── Docker compose ──────────┐
│ orchestrator  (FastAPI + Gradio)   │
│ ollama       (Mixtral fallback)    │
│ prometheus   (optional profile)    │
└────────────────────────────────────┘
        ▲ REST / WebSocket
┌────────────────────────────────────┐
│ MetaEvolver  ← population JSON ckpt│
│  ├─ Ray / mp evaluation pool       │
│  └─ EvoNet (torch)   ──┐           │
│                        ▼ obs       │
│ CurriculumEnv (Gymnasium)          │
└────────────────────────────────────┘
```

* **MetaEvolver**  
  * Population 24, tournament-k 3, elitism 2, novelty search option  
  * Checkpoint every generation (`/data/checkpoints/evolver_gen####.json`)  
  * SHA-256 digest of population for audit  

* **EvoNet**  
  * Variable hidden layers (tuple), activation from `{relu, tanh, sigmoid}`  
  * Optional per-step Hebbian update (`Δw = η · h xᵀ`)  

* **CurriculumEnv**  
  * Grid-world size 12×12, solvability checked via DFS, energy budget  
  * Mutates genome when mastered (< 50 % steps for 5 consecutive episodes)  

---

## 📈 Observability & metrics

Metric | Type | Description
-------|------|------------
`aiga_avg_fitness` | gauge | Mean fitness of last generation
`aiga_best_fitness` | gauge | Elite fitness (stage-independent)
`aiga_generations_total` | counter | Total generations evolved
`aiga_curriculum_stage` | gauge | 0–3 (Line → Maze)
`process_*` | gauge | Standard Prometheus process metrics

Enable full stack (`--profile telemetry`) to auto-scrape with Prometheus +
OpenTelemetry Collector. Graph fitness in Grafana or hit the `/metrics` endpoint
directly.

---

## 🧪 Tests & CI

* **Branch coverage ≥ 90 %** < 0.5 s (`pytest -q`)  
* GitHub Actions (`.github/workflows/ci.yml`) runs: lint → tests → Docker build  
* **SBOM** generated via Syft, uploaded as job artifact.

Run locally:

```bash
pip install -r ../../requirements-dev.txt
pytest -q
coverage html  # optional report
```

---

## ☁️ Deploying to Kubernetes

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

*Helm chart* lives under `infra/helm/aiga-demo/`.

Horizontal scaling:

```bash
kubectl scale deploy aiga-demo --replicas=4     # Ray will auto-cluster
```

---

## 🛡 SOC-2 / supply-chain rails

* **Cosign-signed image** (`cosign verify …`) – enforced by `docker-compose.aiga.yml`  
* Non-root UID `1001`, read-only code volume, `/data` for checkpoints only  
* Secrets via Docker/K8s *secrets* (NOT env baked into layers)  
* SBOM (SPDX v3) published per release tag  
* Dependency list locked with Poetry & hash-checked at runtime  

---

## 🧩 Tinker guide

| Goal | Touch-point | Hint |
|------|-------------|------|
| Bigger populations | `MetaEvolver(pop_size=…)` | Increase Ray workers or `--profile gpu` |
| Faster convergence | Tune mutation rates (`Genome.mutate`) | Try `novelty_weight ≈ 0.2` |
| New curriculum stage | Append in `CurriculumEnv._valid_layout` | Guarantee solvability via `_is_reachable` |
| Swap LLM | edit `config.env` → `MODEL_NAME=` | Any OpenAI Agents-SDK model ID |
| Plug into trading bot | Use JSON API `/evolve/{n}` + `/checkpoint/latest` | Deterministic SHA id for compliance |

---

## 🆘 Troubleshooting FAQ

| Symptom | Remedy |
|---------|--------|
| “Docker not installed” | <https://docs.docker.com/get-docker> |
| Port 7862 already in use | Change host port in `docker-compose.aiga.yml` |
| Build slow on ARM Mac | Enable **Rosetta** or use `./run_aiga_demo.sh --pull` |
| GPU not detected | `sudo apt install nvidia-container-toolkit` → restart Docker |
| Colab public URL missing | Re-run launch cell; ngrok occasionally throttles |

---

## ⚖️ License & credits

*Code & assets* MIT-licensed. Refer to `LICENSE` for full text.  
Heavy thanks to:

* **Jeff Clune** – for the audacious AI-GA roadmap  
* **OpenAI / Anthropic / Google** – open-sourcing pivotal agent tooling  
* Every OSS maintainer whose work this demo stands on

> **Alpha-Factory** — forging intelligence that *invents* intelligence.
