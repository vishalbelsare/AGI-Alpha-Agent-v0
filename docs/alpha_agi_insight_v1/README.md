[See docs/DISCLAIMER_SNIPPET.md](../DISCLAIMER_SNIPPET.md)

# α-AGI Insight v1

This directory hosts the static α‑AGI Insight demo used in the documentation. Build the docs with `mkdocs build` and open `alpha_agi_insight_v1/index.html` from the generated `site/` folder. To preview the files directly from the repository, run:

```bash
python -m http.server --directory docs/alpha_agi_insight_v1 8000
```

and navigate to <http://localhost:8000/>. Direct `file://` access is unsupported due to browser security restrictions.

**Live demo:** <https://montreal-ai.github.io/AGI-Alpha-Agent-v0/alpha_agi_insight_v1/>

The project’s GitHub Pages site redirects its root URL to this directory for
convenience. Non‑technical users can simply open
<https://montreal-ai.github.io/AGI-Alpha-Agent-v0/> and they will be forwarded
to the demo automatically.

For details on publishing the site automatically, see [Quick Deployment](../HOSTING_INSTRUCTIONS.md#quick-deployment).

The charts rely on synthetic data for illustration. Refer to the project disclaimer for important usage information.

### Prerequisites

* **Python ≥3.11**
* **Node.js ≥20** (checked via `node build/version_check.js`)
* **MkDocs**

## One-Command Build

Run the helper script to build the Insight progressive web app (PWA) and generate the `site/` directory:

```bash
./scripts/build_insight_docs.sh
```

The script installs Node dependencies, builds the browser bundle and runs `mkdocs build`. When executed in CI, it also publishes the resulting `site/` directory to GitHub Pages.
The bundled site registers a service worker so the demo remains available offline once loaded. The PWA includes `workbox-sw.js` and a `manifest.json` file to enable caching. Serve the files with a simple HTTP server (e.g. `python -m http.server`) so the service worker can register; opening `index.html` directly with `file://` will not work.

Preview the generated site locally with:

```bash
python -m http.server --directory site 8000
```

For convenience, run `./scripts/preview_insight_docs.sh` to build the demo and immediately serve it on `http://localhost:8000/`.

The [`📚 Docs` workflow](../../.github/workflows/docs.yml) runs the same script and publishes the contents of `site/` to GitHub Pages on every push to `main`.

### Testing offline mode

To verify that the PWA works without an internet connection:

1. Build the documentation with `./scripts/build_insight_docs.sh`.
2. Serve the `site/` directory locally:
   `python -m http.server --directory site 8000`.
3. Open <http://localhost:8000/> in a browser.
4. After the page loads, disable your network connection and reload.
   The demo should still display correctly.

Contributors are encouraged to run this check before publishing changes.
