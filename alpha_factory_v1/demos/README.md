<!--
  Alpha‑Factory v1 👁️✨ — Interactive Demo Gallery
  Multi‑Agent **AGENTIC α‑AGI** • Out‑learn · Out‑think · Out‑design · Out‑strategise · Out‑execute
  © 2025 MONTREAL.AI   MIT License
-->

<div align="center">

# 🏛️ **Alpha‑Factory Demo Gallery**

A living cabinet of curiosities where seminal ideas from Sutton, Silver, Schrittwieser and Clune awaken inside a single agentic engine.

[Era of Experience](./era_of_experience) · [MuZero Planning](./muzero_planning) · [AI‑GA Meta‑Evolution](./aiga_meta_evolution)

</div>

---

## ✨ What makes these demos special?

* **One‑command launch** — every vignette is wrapped in a self‑contained Docker script.  
* **Cloud or laptop** — Colab notebooks replicate the full experience when Docker isn’t available.  
* **Offline‑capable** — omit your `OPENAI_API_KEY` and the stack falls back to Mixtral via Ollama.  
* **Research‑grade DNA** — each demo traces directly to peer‑reviewed breakthroughs.

---

## 🎬 Choose your adventure

| Demo | What you’ll witness | Research lineage | Pillars on display |
|------|--------------------|------------------|--------------------|
| **Era of Experience** | Streams of life events, sensor‑motor tool calls and MCTS planning — unfolding in real time. | Sutton & Silver (2024) | Streams · Actions · Grounded rewards · Non‑human reasoning |
| **MuZero Planning** | A model that *imagines* CartPole physics, then beats it with 64‑node search. | Schrittwieser et al. (2020) | World‑model learning · Joint value/policy · Planning |
| **AI‑GA Meta‑Evolution** | Neural networks that redesign themselves while the curriculum evolves to keep pace. | Clune (2019) | Meta‑NAS · Meta‑optimiser · Environment generator |

---

## 🚀 Run locally

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/<demo_folder>
chmod +x run_<demo_folder>_demo.sh
./run_<demo_folder>_demo.sh
```

*Builds in < 60 s on CPU. Open the printed URL (ports 7860‑7862).*

---

## 🎓 Run in Google Colab

Every folder contains an `.ipynb` that:

1. Clones this repository  
2. Installs lean CPU‑only dependencies  
3. Launches Gradio and prints a public link

Ideal for workshops, classrooms and mobile devices.

---

## 🛡️ Unified safety & ops baseline

* Runs as **non‑root UID 1001**  
* Secrets isolated in `config.env`  
* Offline fallback ⇒ zero data egress  
* `/__live` health endpoint for Kubernetes / Swarm  
* Deterministic builds from a single multi‑stage Dockerfile

---

## 🤝 Lore & acknowledgements

* Silver & Sutton for the **Era of Experience** vision  
* Schrittwieser et al. for **MuZero**  
* Jeff Clune for **AI‑Generating Algorithms**  
* The open‑source community for every brick in this cathedral citeturn3file0

> **Alpha‑Factory** — forging intelligence that forges itself.

