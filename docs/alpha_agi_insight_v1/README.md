[See docs/DISCLAIMER_SNIPPET.md](../DISCLAIMER_SNIPPET.md)

# α-AGI Insight v1

This directory hosts the static α‑AGI Insight demo used in the documentation. Build the docs with `mkdocs build` and open `alpha_agi_insight_v1/index.html` from the generated `site/` folder. To preview the files directly from the repository, run:

```bash
python -m http.server --directory docs/alpha_agi_insight_v1 8000
```

and navigate to <http://localhost:8000/>. Direct `file://` access is unsupported due to browser security restrictions. You can also visit the [hosted page](https://montreal-ai.github.io/AGI-Alpha-Agent-v0/alpha_agi_insight_v1/).

For details on publishing the site automatically, see [HOSTING_INSTRUCTIONS.md](../HOSTING_INSTRUCTIONS.md).

The charts rely on synthetic data for illustration. Refer to the project disclaimer for important usage information.

## One-Command Build

Run the helper script to generate the Insight demo and the `site/` directory:

```bash
./scripts/build_insight_docs.sh
```

The script expects **Python ≥3.11**, **Node.js ≥20** and **MkDocs** to be installed. It installs Node dependencies, builds the browser bundle and runs `mkdocs build`.

Preview the generated site locally with:

```bash
python -m http.server --directory site 8000
```

The `📚 Docs` workflow runs the same script and publishes the contents of `site/` to GitHub Pages on every push to `main`.
