# Documentation Overview

This directory hosts reference files and configuration snippets for **Alpha‑Factory v1**.

- `alpha_agi_agent.md` – technical blueprint of the α‑AGI Agent runtime.
- `alpha_agi_business.md` – architecture of a production α‑AGI Business.
- `REMOTE_SWARM.md` – quick‑start to launch remote agents via Helm.
- `download.html` – minimal installer guide for Alpha‑Factory Pro.
- `grafana/` – Grafana dashboards and datasource provisioning.
- `prometheus.yml` – sample Prometheus scraper configuration.

When running the Insight demo the `/simulate` endpoint and CLI accept `energy`
and `entropy` parameters to control the initial state of generated sectors.

All container images are [Cosign](https://github.com/sigstore/cosign) signed. Verify signatures before running:
```bash
cosign verify ghcr.io/montrealai/alpha-factory:latest
```
