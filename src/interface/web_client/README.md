# React Web Client

This directory contains a small React interface built with [Vite](https://vitejs.dev/) and TypeScript. It fetches disruption forecast results from the API and renders a Plotly chart.

## Setup

```bash
cd src/interface/web_client
pnpm install
pnpm dev        # start the development server
pnpm build      # build production assets in `dist/`
```

Set `VITE_API_BASE_URL` to change the API path prefix at build time:

```bash
# prepend '/api' to all requests
VITE_API_BASE_URL=/api pnpm build
```

The app expects the FastAPI server on `http://localhost:8000` by default. After running `pnpm build`, open `dist/index.html` or copy the `dist/` folder into your container image.

When building the Docker image from the project root, ensure `pnpm --dir src/interface/web_client run build` completes so that `src/interface/web_client/dist/` exists. The `infrastructure/Dockerfile` copies this directory automatically.

A basic smoke test simply runs `npm test`, which exits successfully if the project dependencies are installed.

## Usage with Docker Compose

Set the variable when launching containers:

```yaml
services:
  web:
    environment:
      VITE_API_BASE_URL: /api
```
