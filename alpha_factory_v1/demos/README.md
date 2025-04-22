# Alpha‑Factory Demos 🚀

**Zero‑to‑Alpha for every skill level – from copy‑pasting a single command
in Windows PowerShell to interactive Pandas analysis in Google Colab.**

↳ Runs with or without an **OPENAI_API_KEY** — offline Φ‑2 fallback is automatic.

---

## 1 · One‑liner CLI demo *(60 s)*

```bash
curl -L https://raw.githubusercontent.com/MontrealAI/AGI-Alpha-Agent-v0/main/alpha_factory_v1/demos/deploy_alpha_factory_demo.sh \
     | bash
```

| Step | What happens | ⏱️ |
|------|--------------|----|
| 1 | Pull Cosign‑signed `alphafactory_pro:cpu-slim` | 20 s |
| 2 | Boot container with *BTC / GLD* momentum strategy | 10 s |
| 3 | Print **positions** & **P&L** via jq | instant |
| 4 | Link to live **trace‑graph UI** `http://localhost:8088` | — |
| 5 | Container stops when terminal closes | — |

*Change alpha on the fly:*  
`STRATEGY=eth_usd PORT_API=8001 bash <(curl -fsSL …/deploy_alpha_factory_demo.sh)`

---

## 2 · Interactive notebook demo

```bash
git clone --depth 1 https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1
jupyter notebook demos/finance_alpha.ipynb
```

Run two cells – Alpha‑Factory boots, Pandas tables appear.

Colab: drag‑and‑drop the notebook, click **Run all** (free tier OK).

---

## 3 · Codespaces / Dev Container

Click **Code → Codespaces** on GitHub; the devcontainer auto‑opens the trace UI.

---

## 4 · Preview GIF

![Trace demo](../docs/trace_demo.gif)

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| Docker not installed | <https://docs.docker.com/get-docker/> |
| Port 8000 busy | `PORT_API=8100 …deploy_alpha_factory_demo.sh` |
| Proxy blocks image pull | Use VPN, then `docker save` / `docker load` |
| Need GPU | `PROFILE=cuda ./scripts/install_alpha_factory_pro.sh --deploy` |

---

## 🔐 Security

* `.env` is git‑ignored; no secrets leave your machine.  
* Image is Cosign‑signed; SBOM in GitHub Releases.  

Enjoy exploring **α‑Factory** – and out‑think the future! 🚀
