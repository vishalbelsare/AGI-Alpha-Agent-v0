[See docs/DISCLAIMER_SNIPPET.md](../docs/DISCLAIMER_SNIPPET.md)

# Hosting Instructions

This project uses [MkDocs](https://www.mkdocs.org/) to build the static documentation.

## Prerequisites

- **Python 3.11 or 3.12**
- `mkdocs` and `mkdocs-material`
- **Node.js 20+** *(optional, only for building the React dashboard)*

Install MkDocs:

```bash
pip install mkdocs mkdocs-material
```

## Build the Insight Demo

The static browser bundle lives under
`alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1`. Install the
Node dependencies then create the distribution archive:

```bash
cd alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1
npm install
npm run build:dist
```

`npm run build:dist` produces `insight_browser.zip`. Extract the archive and copy
its contents into `docs/alpha_agi_insight_v1` so MkDocs can include the files:

```bash
unzip -o insight_browser.zip -d ../../../docs/alpha_agi_insight_v1
```


## Building the Site

Run the following from the repository root:

```bash
mkdocs build
```

This generates the HTML under `site/`. Verify that the Insight demo was copied
correctly:

```bash
ls site/alpha_agi_insight_v1
```

Serve the site locally to test it:

```bash
python -m http.server --directory site 8000
```

Then browse to <http://localhost:8000/alpha_agi_insight_v1/>. Direct `file://`
access is unsupported due to the service worker; use a minimal HTTP server or
GitHub Pages.

The workflow
[`.github/workflows/docs.yml`](../.github/workflows/docs.yml) runs the same
command, copies `docs/alpha_agi_insight_v1` into `site/` and then deploys.

## Publishing to GitHub Pages

When changes land on `main` or a release is published, `docs.yml` pushes the
`site/` directory to the `gh-pages` branch. GitHub Pages serves the result at
`https://montreal-ai.github.io/AGI-Alpha-Agent-v0/alpha_agi_insight_v1/`.
The standard [project disclaimer](DISCLAIMER_SNIPPET.md) applies.

## Verifying Deployment

Confirm the workflow is enabled under **Actions** and that
[`docs.yml`](../.github/workflows/docs.yml) specifies
`permissions: contents: write`. Run the "📚 Docs" workflow from the GitHub UI or
push to `main` to trigger it. The initial run creates the `gh-pages` branch.
After it finishes, browse to
<https://montreal-ai.github.io/AGI-Alpha-Agent-v0/alpha_agi_insight_v1/> and
check that the insight demo loads.
