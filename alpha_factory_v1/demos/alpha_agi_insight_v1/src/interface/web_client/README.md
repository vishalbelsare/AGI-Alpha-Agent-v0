# Alpha AGI Insight Web Client

This directory contains a minimal React application built with [Vite](https://vitejs.dev/).
It connects to the API server at `/ws/progress` and logs messages to the console.

## Development

```bash
cd alpha_factory_v1/demos/alpha_agi_insight_v1/src/interface/web_client
pnpm install
pnpm dev       # open http://localhost:5173
```

## Build

```bash
pnpm build  # outputs static files in dist/
```

The production bundle lives under `dist/` and is served automatically by the API
server when present.
