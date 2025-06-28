[See docs/DISCLAIMER_SNIPPET.md](../docs/DISCLAIMER_SNIPPET.md)

# Edge-of-Human-Knowledge Demo Tasks Sprint for Codex

This sprint describes how Codex can publish the **Alpha-Factory v1** demo gallery to GitHub Pages in a polished subdirectory. The goal is a user-friendly deployment where every demo plays in real time with rich visuals and smooth interactions.

Run `./scripts/edge_of_knowledge_sprint.sh` from the repository root for an automated workflow.

## 1. Validate the Environment
1. Install **Python 3.11+** and **Node.js 20+**.
2. Execute the preflight script:
   ```bash
   python alpha_factory_v1/scripts/preflight.py
   ```
3. Verify the browser demo requirements:
   ```bash
   node alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1/build/version_check.js
   ```
4. Install optional dependencies so verification tools succeed:
   ```bash
   python scripts/check_python_deps.py
   python check_env.py --auto-install
   ```
5. Confirm each README contains the standard disclaimer:
   ```bash
   python scripts/verify_disclaimer_snippet.py
   ```
6. Validate the demo packages:
   ```bash
   python -m alpha_factory_v1.demos.validate_demos
   ```

## 2. Build the Insight Demo
Compile the progressive web app and verify the service worker hash:
```bash
./scripts/build_insight_docs.sh
```
If lineage logs are available, this exports `tree.json` so the **Meta-Agentic Tree Search** animates organically.

## 3. Generate the Demo Gallery
From the repository root run:
```bash
./scripts/deploy_gallery_pages.sh
```
The script fetches assets, rebuilds documentation and compiles the MkDocs site under `site/`.

## 4. Preview Locally
Serve the pages and check that animations are fluid:
```bash
python -m http.server --directory site 8000
```
Navigate to <http://localhost:8000/> and step through `gallery.html`. Confirm that each README showcases preview media so viewers can follow along in real time.

Verify offline support:
```bash
python scripts/verify_workbox_hash.py site/alpha_agi_insight_v1
python scripts/verify_insight_offline.py
```

## 5. Deploy to GitHub Pages
Publish the gallery when satisfied:
```bash
mkdocs gh-deploy --force
```
The resulting URL typically looks like:
```
https://<org>.github.io/AGI-Alpha-Agent-v0/
```
The landing page redirects to `alpha_agi_insight_v1/` while `gallery.html` links to every demo.

## 6. Maintenance Tips
- Re-run the helper whenever demo docs or assets change.
- Capture short GIFs or screenshots under `docs/<demo>/assets/` for a highly visual experience.
- Test with `mkdocs build --strict` before deploying and ensure `pre-commit` hooks pass.
