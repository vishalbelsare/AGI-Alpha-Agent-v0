[See docs/DISCLAIMER_SNIPPET.md](../docs/DISCLAIMER_SNIPPET.md)

# Demo Gallery – GitHub Pages Sprint for Codex

This short guide outlines how Codex can publish the full Alpha‑Factory demo gallery as a static site hosted on GitHub Pages. The steps build the α‑AGI Insight interface, include all Markdown documentation and ensure non‑technical users can explore the demos visually.

## 1. Environment Setup

1. Install **Python 3.11+** and **Node.js 20+**.
2. Install `mkdocs` and `mkdocs-material` via `pip`:
   ```bash
   pip install mkdocs mkdocs-material
   ```
3. Verify Node version:
   ```bash
   node alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1/build/version_check.js
   ```
4. Install optional Python packages:
   ```bash
   python scripts/check_python_deps.py
   python check_env.py --auto-install
   ```

## 2. Build and Deploy the Gallery

Run the helper from the repository root:

```bash
./scripts/gallery_sprint.sh
```

The script checks the environment, compiles the Insight browser bundle, refreshes `docs/alpha_agi_insight_v1`, builds the MkDocs site and publishes it to the `gh-pages` branch.

Preview locally:

```bash
python -m http.server --directory site 8000
```

Browse to <http://localhost:8000/> and verify the index page links to the Insight demo and the full demo gallery.

## 3. Verification Checklist

- ✅ `docs/alpha_agi_insight_v1/` contains `index.html`, `manifest.json` and `lib/workbox-sw.js`.
- ✅ `scripts/verify_workbox_hash.py site/alpha_agi_insight_v1` passes.
- ✅ Landing page links to the Demo Gallery.
- ✅ The GitHub Pages site loads without errors in an incognito window.

Following these steps ensures the entire demo suite is accessible via GitHub Pages with a single command.
