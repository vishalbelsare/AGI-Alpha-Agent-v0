<!--
  AI‑GA Meta‑Evolution Demo
  Alpha‑Factory v1 👁️✨ — Multi‑Agent **AGENTIC α‑AGI**
  Out‑learn · Out‑think · Out‑strategise · Out‑evolve
  © 2025 MONTREAL.AI   MIT License
-->

# 🌌 Algorithms That Invent Algorithms

> *“Why hand‑craft intelligence when evolution can author it for you?”*  
> — Jeff Clune, **AI‑GAs: AI‑Generating Algorithms** (2019)

Welcome to the first browser‑based, one‑command showcase of Clune’s **Three
Pillars**—meta‑learning architectures, meta‑learning algorithms, and
self‑generating curricula—woven into the **Alpha‑Factory** agent spine.

In < 60 s you will watch a population of neural networks **rewrite their own
blueprint** while the world itself mutates to keep them sharp.

---

## 🚀 Launch locally (zero config)

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/aiga_meta_evolution
chmod +x run_aiga_demo.sh
./run_aiga_demo.sh
```

1. **Docker Desktop** builds the image (≈ 40 s cold).  
2. Open **http://localhost:7862** → the dashboard appears.  
3. Press **Evolve 5 Generations** and witness fitness ascend.

> **No OpenAI key?** Leave `OPENAI_API_KEY` blank in `config.env`.  
> The stack drops seamlessly to **Mixtral** via Ollama—fully offline.

---

## ✨ Inside the magic

| AI‑GA Pillar | Demo realisation |
|--------------|------------------|
| **Architectures that learn to learn** | Genome `[n_hidden, activation]` mutates via neuro‑evolution |
| **Learning rules that learn** | Flag toggles **SGD** ↔ **fast Hebbian plasticity** |
| **Worlds that teach** | CurriculumEnv evolves from *Line‑follow* → *Maze‑nav* |

Population size 20, tournament selection k = 3, elitism by curriculum stage.

---

## 🛠️ Architecture snapshot

```text
┌────────────────┐   (obs)   ┌────────────────────┐
│ CurriculumEnv  │──────────▶│   EvoNet (genome)  │
└────────────────┘           └─────────┬──────────┘
                    mutate genome ▲    │ Hebbian ΔW
                                  └────┴───────────┐
                tournament, mutate, cross‑seed     │
┌───────────────────────────────────────────────────┘
│ MetaEvolver (outer loop) — 5 gens / click
└────────────────────────────────────────────────────
```

* **MetaEvolver ≤ 150 LoC** — clean, CPU‑friendly neuro‑evolution.  
* **CurriculumEnv ≤ 120 LoC** — self‑mutating Gymnasium task factory.  
* **openai‑agents‑python** — optional LLM commentary via tool‑calling.  
* **Docker + Gradio** — deterministic, air‑gapped UX.

---

## 🎓 Google Colab (two clicks)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MontrealAI/AGI-Alpha-Agent-v0/blob/main/alpha_factory_v1/demos/aiga_meta_evolution/colab_aiga_meta_evolution.ipynb)

Colab spins up the same dashboard with a public link—perfect for workshops.

---

## 🧩 Tinker & explore

| What to tweak | Where |
|---------------|-------|
| Population size / mutation rate | `MetaEvolver.__init__` |
| Add a curriculum stage | `CurriculumEnv._gen_map()` |
| Swap optimiser | `EvoNet.forward()` |
| Multi‑agent swarm | `docker compose --scale orchestrator=4 …` |

---

## 🛡️ Production‑grade safeguards

* Runs as **non‑root UID 1001**.  
* Secrets isolated in `config.env`; never baked into images.  
* Offline fallback ⇒ zero third‑party data egress.  
* Health‑check endpoint `/__live` for k8s and Docker Swarm.

---

## 🆘 Quick fixes

| Symptom | Remedy |
|---------|--------|
| “Docker not installed” | Install via <https://docs.docker.com/get-docker> |
| Port 7862 busy | Edit `ports:` in `docker-compose.aiga.yml` |
| ARM Mac slow build | Enable *Rosetta* for x86/amd64 emulation in Docker settings |
| Want GPU | Change base image to `nvidia/cuda:12.4.0-runtime-ubuntu22.04` & add `--gpus all` |

---

## 🤝 Credits

* **Jeff Clune** for the bold blueprint toward open‑ended AI evolution.  
* **Montreal.AI** for distilling the vision into runnable code.  
* The open‑source community for every library that made this possible.

> **Alpha‑Factory** — forging intelligence that **invents** intelligence.
