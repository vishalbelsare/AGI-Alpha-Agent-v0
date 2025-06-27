# Insight Demo GitHub Pages Sprint

This document outlines the minimal tasks required to publish the **α‑AGI Insight v1** demo to GitHub Pages so that users can experience the full browser-based simulation, including the animated meta‑agentic tree search.

## 1. Prepare the Environment
- Install **Python 3.11+**, **Node.js 20+**, `mkdocs`, and `mkdocs-material`.
- Run `node alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1/build/version_check.js` to confirm the Node version.
- Execute `python scripts/check_python_deps.py` and `python check_env.py --auto-install` to install optional dependencies.

## 2. Build the Static Demo
- From the repository root, run `./scripts/build_insight_docs.sh`.
  - This fetches assets, builds the browser bundle, and refreshes `docs/alpha_agi_insight_v1/`.
- Verify that `docs/alpha_agi_insight_v1/index.html` loads locally using:
  ```bash
  python -m http.server --directory site 8000
  ```
- Browse to <http://localhost:8000/alpha_agi_insight_v1/> and confirm the charts and the animated tree search function correctly.

## 3. Deploy to GitHub Pages
- Execute `./scripts/publish_insight_pages.sh` to build and deploy the `gh-pages` branch.
- Once the workflow completes, verify the live site at:
  <https://montrealai.github.io/AGI-Alpha-Agent-v0/alpha_agi_insight_v1/>
- The repository’s `docs/index.html` already redirects to this path, making the demo accessible from the project root URL.

## 4. Maintain the Demo
- Update `forecast.json` or `population.json` to change the scenario, then re-run `build_insight_docs.sh`.
- Run `scripts/verify_insight_offline.py` to ensure offline caching works before publishing.

These steps keep the demo production-ready and easily reproducible for non‑technical users.
