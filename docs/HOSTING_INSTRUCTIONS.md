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

## Building the Site

Run the following from the repository root:

```bash
mkdocs build
```

This generates the HTML under `site/`. The workflow
[`.github/workflows/docs.yml`](../.github/workflows/docs.yml) runs the same
command, copies `docs/alpha_agi_insight_v1` into `site/` and then deploys.

## Publishing to GitHub Pages

When changes land on `main` or a release is published, `docs.yml` pushes the
`site/` directory to the `gh-pages` branch. GitHub Pages serves the result at
`https://montreal-ai.github.io/AGI-Alpha-Agent-v0/alpha_agi_insight_v1/`.
The standard [project disclaimer](DISCLAIMER_SNIPPET.md) applies.
