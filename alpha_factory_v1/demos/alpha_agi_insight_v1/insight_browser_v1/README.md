### 🔬 Browser-only Insight demo
A zero-backend Pareto explorer lives in
`demos/alpha_agi_insight_v1/insight_browser_v1/`.

## Prerequisites
- **Node.js ≥20** must be installed.

## Build & Run
```bash
npm ci           # deterministic install
npm run build    # compile to dist/ and embed env vars
```
Copy `.env.sample` to `.env` and fill in variables like `PINNER_TOKEN` and
`OTEL_ENDPOINT` before building or running.
The build script reads `.env` automatically and writes `window.PINNER_TOKEN`,
`window.OPENAI_API_KEY` and `window.OTEL_ENDPOINT` to `dist/index.html`.
Place the Pyodide 0.25 files in `wasm/` before building. The script copies them
to `dist/wasm` so the demo can run offline.
```bash
PINNER_TOKEN=<token> npm start
```
`npm start` serves the `dist/` folder on `http://localhost:3000` by default.
Set `PINNER_TOKEN` to your [Web3.Storage](https://web3.storage/) token so
exported JSON results can be pinned.

If `OPENAI_API_KEY` is saved in `localStorage`, the demo uses the OpenAI API for
chat prompts. When the key is absent a lightweight GPT‑2 model under
`wasm_llm/` runs locally.

Open `index.html` directly in your browser or pin the folder to IPFS
(`ipfs add -r insight_browser_v1`) and share the CID.
The URL fragment encodes parameters such as `#/s=42&p=120&g=80`.

## Toolbar & Controls
- **CSV** – export the current population as `population.csv`.
- **PNG** – download a `frontier.png` screenshot of the chart.
- **Share** – copy a permalink to the clipboard. When `PINNER_TOKEN` is set,
  exported JSON is pinned to Web3.Storage and the CID appears in a toast.
- **Theme** – toggle between light and dark mode.

Drag a previously exported JSON state onto the drop zone to restore a
simulation.

Environment variables can be configured in `.env` (see `.env.sample`).
