<!--
  AI‑GA Meta‑Evolution Demo
  Alpha‑Factory v1 👁️✨ — Multi‑Agent **AGENTIC α‑AGI**
  Out‑learn · Out‑think · Out‑strategise · Out‑evolve
  © 2025 MONTREAL.AI   MIT License
  ===============================================================================
  This README is intentionally exhaustive: quick‑start, deep‑dive, SOC‑2 rails,
  CI/CD, K8s, observability, troubleshooting, contributor guide, SBOM notice.
-->

# 🌌 Algorithms That Invent Algorithms — **AI‑GA Meta‑Evolution Demo**

> *“Why hand‑craft intelligence when evolution can author it for you?”*  
> — Jeff Clune, <cite>AI‑GAs: AI‑Generating Algorithms</cite> (2019) citeturn1file0

Welcome to the world’s first **one‑command, browser‑based showcase** of Clune’s
*Three Pillars* — meta‑learning architectures, meta‑learning algorithms and
self‑generating curricula — all woven into the **Alpha‑Factory** agent spine.

In **\<60 s** you’ll watch a population of neural networks **rewrite their own
blueprint** while the world itself mutates to keep them sharp.

---

## 📜 Table of Contents
1. [🚀 Quick start](#-quick-start)
2. [🔑 LLM back‑ends](#-llm-back-ends)
3. [🛠 Architecture deep‑dive](#-architecture-deep-dive)
4. [📈 Observability](#-observability)
5. [🧪 Tests & CI/CD](#-tests--cicd)
6. [☁️ Kubernetes & Cloud Run](#-kubernetes--cloud-run)
7. [🛡 SOC‑2 & Supply‑chain](#-soc-2--supply-chain)
8. [🧩 Tinker guide](#-tinker-guide)
9. [🆘 FAQ](#-faq)
10. [🤝 Contributing](#-contributing)
11. [⚖️ License & credits](#-license--credits)

---

## 🚀 Quick start

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/aiga_meta_evolution

# Optional flags:
#   --pull    ⟶ use signed image from GHCR
#   --gpu     ⟶ enable CUDA runtime
./run_aiga_demo.sh --pull
```

| Component | Port / URL | What you get |
|-----------|------------|--------------|
| **Gradio dashboard** | `http://localhost:7862` | Visualise population fitness, evolve generations |
| **FastAPI JSON API** | `http://localhost:8000/docs` | OpenAPI 3 spec & interactive playground |
| **Prometheus** | `http://localhost:8000/metrics` | `aiga_*` gauges + counters |

<details>
<summary>Minimal prerequisites</summary>

* Docker 24 + compose plug‑in  
* ≥ 4 GB RAM (8 GB if you bump `pop_size`)  
* **GPU not required** (CUDA 12 optional)  
</details>

---

## 🔑 LLM back‑ends

| Setting | Behaviour |
|---------|-----------|
| `OPENAI_API_KEY=<your‑key>` | Agents SDK tool‑calling with GPT‑5.5‑Turbo |
| *unset / blank* | Private **Mixtral‑8x7B‑Instruct** served via Ollama side‑car |
| `MODEL_NAME=claude-3-opus` | Anthropic MCP compliant calls |

LLMs provide commentary, code‑assist & speculative planning; **core evolution
logic is fully deterministic and offline‑capable.**

---

## 🛠 Architecture deep‑dive

```text
┌──────────────── Docker Compose ───────────────┐
│ orchestrator  (FastAPI + Gradio UI)           │
│ evolution‑worker  (Ray actor pool)            │
│ ollama‑mixtral  (optional offline LLM)        │
│ prometheus & grafana (optional profile)       │
└───────────────────────────────────────────────┘
        ▲ REST / WebSocket            │
        │                              ▼ checkpoints/*.json
┌─────────────────────────────────────────────────────────────┐
│ MetaEvolver ↔↔ Ray pool ↔ EvoNet (torch)  ──┐               │
│    ├── tournament‑selection, novelty search │ obs           │
│    └── Genome: [layers, activations, plasticity coeffs]     │
│                                    ▲ Hebbian Δw             │
│ CurriculumEnv  (Gymnasium) ────────┘ self‑mutates map/goal   │
└──────────────────────────────────────────────────────────────┘
```

* **MetaEvolver** — population 24, novelty‑weighted ES, elitism 2  
* **EvoNet** — variable‑depth MLP with per‑edge plasticity mask  
* **CurriculumEnv** — procedurally mutating 12 × 12 grid‑world  
* **Checkpoints** — SHA‑256 digests, resume on boot, Prom‑scrape fitness

---

## 📈 Observability

Metric | Type | Description
-------|------|------------
`aiga_avg_fitness` | gauge | Generation mean fitness
`aiga_best_fitness` | gauge | Elite champion
`aiga_curriculum_stage` | gauge | 0–3 (Line → Maze)
`aiga_generations_total` | counter | Life‑time generations

Enable Grafana dashboard via:  
```bash
docker compose --profile telemetry up -d
```

---

## 🧪 Tests & CI/CD

* **pytest** branch‑cov ≥ 90 % in < 0.5 s  
* **GitHub Actions**: lint → tests → Docker build → cosign sign  
* **SLSA‑3** provenance, **SBOM** via Syft, **Trivy** scan gate

```bash
pip install -r ../../requirements-dev.txt
pytest -q
coverage html
```

---

## ☁️ Kubernetes & Cloud Run

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: aiga-demo }
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: orchestrator
        image: ghcr.io/montrealai/alpha-aiga:latest@sha256:<signed>
        ports:
        - { containerPort: 8000 }
        - { containerPort: 7862 }
        readinessProbe: { httpGet: { path: /health, port: 8000 } }
        envFrom: [{ secretRef: { name: aiga-secrets } }]
```

Scale Ray workers horizontally:  
```bash
kubectl scale deploy aiga-demo --replicas=8
```

**Audit rails**: OPA policies, Falco runtime sensors, CloudTrail‑equiv logs.

---

## 🛡 SOC‑2 & Supply‑chain

* **Non‑root UID 1001**, read‑only code volume  
* **Cosign**‑signed images; `docker pull --verify=cosign` enforced  
* **SBOM** (SPDX v3) published per tag  
* Dependencies pinned via **Poetry.lock** & hash‑verified at run‑time

---

## 🧩 Tinker guide

| Goal | Touch‑point | Hint |
|------|-------------|------|
| Larger population | `MetaEvolver(pop_size=…)` | Also raise `RAY_worker_envs` |
| New curriculum stage | `CurriculumEnv._gen_map()` | Ensure DFS‑reachable goal |
| Faster convergence | Lower `mutation_sigma`, add `novelty_weight` | Observe diversity metric |
| Swap optimiser | `EvoNet.forward()` | SGD ↔ Hebbian plasticity toggle |
| Integrate trading bot | `/evolve/{n}` then `/checkpoint/latest` | Same SHA across nodes |

---

## 🆘 FAQ

| ❓ Symptom | 💡 Remedy |
|-----------|-----------|
| “Docker not installed” | <https://docs.docker.com/get-docker> |
| Port 7862 busy | Edit host mapping in `docker-compose.aiga.yml` |
| Colab tunnel missing | Re‑run “Launch” cell — ngrok throttles occasionally |
| GPU not detected | `sudo apt install nvidia-container-toolkit && sudo systemctl restart docker` |
| Build slow on ARM Mac | Enable Rosetta 2 emulation or use `./run_aiga_demo.sh --pull` |

---

## ⚖️ License & credits

**Code & assets**: MIT.  
Huge thanks to:

* **Jeff Clune** — audacious AI‑GA roadmap  
* **OpenAI / Anthropic / Google DeepMind** — agent tooling & research  
* **Open‑source community** — every library that made this possible

> **Alpha‑Factory** — forging intelligence that *invents* intelligence.
