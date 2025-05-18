# Alpha‑Factory Demos 📊

Welcome! These short demos let **anyone – even if you’ve never touched a
terminal – spin up Alpha‑Factory, watch a live trade, and explore the
planner trace‑graph in *under 2 minutes*.  

*(Runs with or without an `OPENAI_API_KEY`; the image auto‑falls back to
a local Φ‑2 model.)*

---

## 🚀 Instant CLI demo

```bash
curl -L https://raw.githubusercontent.com/MontrealAI/AGI-Alpha-Agent-v0/main/alpha_factory_v1/demos/finance_alpha/deploy_alpha_factory_demo.sh | bash
```

**What happens**

1. Docker pulls the signed `alphafactory_pro:cpu-slim` image.
2. Container starts with the *BTC / GLD* momentum strategy.
3. The script verifies the API port is free and waits for the health endpoint.
4. The script prints JSON tables for **positions** and **P&L**.
5. You get a link to the live **trace‑graph UI** (`http://localhost:8088`).
6. Container stops automatically when you close the terminal.

_No installation beyond Docker, `curl`, and `jq`._

**Customize**  
`STRATEGY=my_pair PORT_API=8001 bash deploy_alpha_factory_demo.sh` runs a
different momentum pair on an alternate port.

---

## 📒 Interactive notebook demo

> Perfect for analysts who love Pandas or anyone on Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MontrealAI/AGI-Alpha-Agent-v0/blob/main/alpha_factory_v1/demos/finance_alpha/finance_alpha.ipynb)

The first code cell checks for Docker and installs it automatically when running on Colab. Simply run each cell in order to launch Alpha‑Factory, view positions, and open the live trace‑graph.

An additional cell now embeds the trace‑graph UI directly inside the notebook so you can follow the planner's decisions without leaving Colab.

```bash
git clone --depth 1 https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1
jupyter notebook demos/finance_alpha/finance_alpha.ipynb
```

Run the cells to spin up Alpha‑Factory and render positions & P&L as
Pandas tables right inside the notebook. Tweak `STRATEGY` or `PORT_API`
in the parameter cell if you need a custom pair or port.

---


## 🛠️ Troubleshooting

| Symptom | Resolution |
|---------|------------|
| **“docker: command not found”** | Install Docker Desktop or Docker Engine |
| Port 8000 already used | The script aborts; re-run with `PORT_API=8001` |
| Corporate proxy blocks image pull | Pull image on a VPN, `docker save` → `docker load` |
| Want GPU speed | `PROFILE=cuda ./scripts/install_alpha_factory_pro.sh --deploy` |

---

## 🔐 Security

* No secrets leave your machine. `.env` (optional) is git‑ignored.  
* Image is **Cosign‑signed**; SBOM available in GitHub Releases.

Enjoy exploring **α‑Factory** – and out‑think the future! 🚀

---

## WARNING: Disclaimer

This demo and the included trading strategy are **for research and
educational purposes only**. They operate on a simulated exchange by
default and **should not be used with real funds**. Nothing here
constitutes financial advice. MontrealAI and the maintainers accept no
liability for losses incurred from using this software.
