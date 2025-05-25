# React Web Client

This directory contains the Vite-based React dashboard for the α‑AGI Insight demo. It communicates with the FastAPI backend and visualises progress in real time.

## Setup

```bash
cd alpha_factory_v1/demos/alpha_agi_insight_v1/src/interface/web_client
pnpm install
pnpm dev        # start the development server
pnpm build      # production build in `dist/`
```

The app expects the API server on `http://localhost:8000`. When building the Docker image from the project root, ensure `pnpm --dir alpha_factory_v1/demos/alpha_agi_insight_v1/src/interface/web_client run build` completes so that `dist/` exists. The infrastructure Dockerfile copies this directory automatically.
