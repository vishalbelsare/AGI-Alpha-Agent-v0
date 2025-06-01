### 🔬 Browser-only Insight demo
A zero-backend Pareto explorer lives in
`demos/alpha_agi_insight_v1/insight_browser_v1/`.

## Prerequisites
- **Node.js ≥20** must be installed.
- **Python ≥3.11** is required when using `manual_build.py`.

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
to `dist/wasm` so the demo can run offline. When preparing the environment
offline run:

```bash
python ../../../scripts/fetch_assets.py
```

This downloads the Pyodide runtime and `wasm-gpt2` model from IPFS into
`wasm/` and `wasm_llm/`.
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

## Manual Build
When Node.js or network access isn't available, run `manual_build.py`
instead of `npm run build`:

```bash
python manual_build.py
```

The script requires Python ≥3.11 and writes the bundled output to `dist/`.
`dist/index.html` loads `dist/app.js`, `bundle.esm.min.js` and
`pyodide.js` with integrity hashes. Any `wasm/` or `wasm_llm/` directories
are copied as-is so the demo can run fully offline.
Run `python ../../../scripts/fetch_assets.py` beforehand to download the
Pyodide runtime and GPT‑2 WASM model when building without internet access.
Serve the folder with `npm start` or open `dist/index.html` directly.

## Toolbar & Controls
- **CSV** – export the current population as `population.csv`.
- **PNG** – download a `frontier.png` screenshot of the chart.
- **Share** – copy a permalink to the clipboard. When `PINNER_TOKEN` is set,
  exported JSON is pinned to Web3.Storage and the CID appears in a toast.
- **Theme** – toggle between light and dark mode.

Drag a previously exported JSON state onto the drop zone to restore a
simulation.

## Darwin-Archive
Every completed run is stored locally using IndexedDB. The **Evolution** panel
lists archived runs with their score and novelty. Click **Re-spawn** next to a
row to restart the simulation using that seed.

## Running a Simulation
Use the **Simulator** panel to launch a full run. Adjust the seed list, population
size and number of generations, then press **Start**. When `PINNER_TOKEN` is set,
the resulting `replay.json` is pinned to Web3.Storage and the CID appears once
the run finishes. Keep this CID handy to share or reload the simulation later.

## Load via CID
Append `#/cid=&lt;CID&gt;` to the URL (or use the **Share** permalink) to replay a
previous run. The simulator fetches the JSON from IPFS and populates the chart.

Environment variables can be configured in `.env` (see `.env.sample`).
