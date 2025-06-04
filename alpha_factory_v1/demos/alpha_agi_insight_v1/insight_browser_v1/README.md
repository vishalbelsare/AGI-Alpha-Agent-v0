### 🔬 Browser-only Insight demo
A zero-backend Pareto explorer lives in
`demos/alpha_agi_insight_v1/insight_browser_v1/`.

## Prerequisites
- **Node.js ≥20** is required for offline PWA support and by `manual_build.py`  
  to generate the service worker.
- **Python ≥3.11** is required when using `manual_build.py`.

Verify your Node.js version before running the build script:

```bash
node build/version_check.js
```
The output should be empty for a valid setup. Only run `manual_build.py` when
this requirement is met. The `package.json` also enforces Node.js 20 or newer
via the `engines` field.

## Environment Setup
Copy [`.env.sample`](.env.sample) to `.env` then review the variables:

- `PINNER_TOKEN` – Web3.Storage token used to pin results.
- `OPENAI_API_KEY` – optional OpenAI key for chat prompts.
- `IPFS_GATEWAY` – base URL of the IPFS gateway used to fetch pinned runs.
- `OTEL_ENDPOINT` – OTLP/HTTP endpoint for anonymous telemetry (leave blank to disable).
- `WEB3_STORAGE_TOKEN` – build script token consumed by `npm run build`.
- Browsers with WebGPU can accelerate the local model using the ONNX runtime.
  Use the GPU toggle in the power panel to switch between WebGPU and WASM.

See [`.env.sample`](.env.sample) for the full list of supported variables.

## Build & Run
Before compiling the app you **must** replace the placeholder WebAssembly
artifacts. Run the helper below **before** `npm run build` or
`python manual_build.py` to download the Pyodide runtime and
`wasm-gpt2` model:

```bash
python ../../../scripts/fetch_assets.py
```

Once the wasm files are in place run:
```bash
npm install
npm run build    # compile to dist/ and embed env vars
```
The build script reads `.env` automatically and injects the values into
`dist/index.html` as `window.PINNER_TOKEN`, `window.OPENAI_API_KEY`,
`window.IPFS_GATEWAY` and `window.OTEL_ENDPOINT`.
It also copies `dist/sw.js` to `dist/service-worker.js` which `index.html`
registers for offline support.
The unbuilt `index.html` falls back to `'self'` for the IPFS and telemetry
origins, but running `npm run build` (or `python manual_build.py`) replaces
these defaults with the real values from `.env`.
Place the Pyodide 0.25 files in `wasm/` before building. The script copies them
to `dist/wasm` so the demo can run offline. When preparing the environment
offline run:

```bash
python ../../../scripts/fetch_assets.py
```

This downloads the Pyodide runtime and `wasm-gpt2` model from IPFS into
`wasm/` and `wasm_llm/`.
It also retrieves `lib/bundle.esm.min.js` from the mirror. The build and
`manual_build.py` scripts scan every downloaded asset for the word
`"placeholder"` and abort when any file still contains that marker.
`scripts/fetch_assets.py` also downloads `lib/workbox-sw.js` from
Workbox 6.5.4 so the service worker can operate offline.
Run `scripts/fetch_assets.py` if you encounter this error.
```bash
PINNER_TOKEN=<token> npm start
```
`npm start` serves the `dist/` folder on `http://localhost:3000` by default.
Set `PINNER_TOKEN` to your [Web3.Storage](https://web3.storage/) token so
exported JSON results can be pinned.

If `OPENAI_API_KEY` is saved in `localStorage`, the demo uses the OpenAI API for
chat prompts. When the key is absent a lightweight GPT‑2 model under
`wasm_llm/` runs locally.

Open `index.html` directly or pin the built `dist/` directory to IPFS
(`ipfs add -r dist`) and share the CID.
The URL fragment encodes parameters such as `#/s=42&p=120&g=80`.

See [docs/insight_browser_quickstart.pdf](docs/insight_browser_quickstart.pdf) for a short walkthrough. The
build script copies this file to `dist/insight_browser_quickstart.pdf` so
the guide is available alongside `dist/index.html`.

## Manual Build
Use `manual_build.py` for air‑gapped environments:

1. `cp .env.sample .env` and edit the values if you haven't already.
2. `python ../../../scripts/fetch_assets.py` to fetch Pyodide and the GPT‑2 model.
   The build scripts verify these files no longer contain the word `"placeholder"`.
   Failing to replace placeholders will break offline mode.
3. Run `node build/version_check.js` to ensure Node.js **v20** or newer is
   installed. `manual_build.py` exits if this check fails.
4. `python manual_build.py` – bundles the app, generates `dist/sw.js` and embeds
   your `.env` settings.
5. `npm start` or open `dist/index.html` directly to run the demo.

The script requires Python ≥3.11. It loads `.env` automatically and injects
`PINNER_TOKEN`, `OPENAI_API_KEY`, `IPFS_GATEWAY` and `OTEL_ENDPOINT` into
`dist/index.html`, mirroring `npm run build`.

### Offline Build

Follow these steps when building without internet access:

1. Run `python ../../../scripts/fetch_assets.py`.
2. Verify checksums match `build_assets.json`.
3. Confirm no files under `wasm/` or `lib/` contain the word "placeholder".
4. Execute `python manual_build.py` to generate the PWA in `dist/`.
5. Launch with `npm start` or pin the directory with `ipfs add -r dist`.

Failing to replace placeholders will break offline mode.

### Offline build checklist

1. Run `python ../../../scripts/fetch_assets.py`.
2. Confirm no placeholder text remains in `lib/` or `wasm*/`.
3. Execute `npm run build` or `python manual_build.py`.

Failing to run the fetch script leaves offline mode disabled.

### Fetching Assets Offline

Set `WEB3_STORAGE_TOKEN` before running the helper script:

```bash
WEB3_STORAGE_TOKEN=<token> python ../../../scripts/fetch_assets.py
```


The script retrieves the WebAssembly runtime and supporting files from IPFS,
verifying checksums to ensure each asset is intact.

## Distribution
Run `npm run build:dist` to generate `insight_browser.zip` in this directory.
The archive bundles the production build along with the service worker and
assets. It stays under **3&nbsp;MiB** so the entire demo can be shared easily.
Extract the zip and open `index.html` directly—no additional dependencies are
required. New users can review
[dist/insight_browser_quickstart.pdf](dist/insight_browser_quickstart.pdf) after
extraction for a brief walkthrough.

## Locale Support
The interface automatically loads French, Spanish or Chinese translations based
on your browser preferences. Set `localStorage.lang` to override the detected
language.

## Toolbar & Controls
- **CSV** – export the current population as `population.csv`.
- **PNG** – download a `frontier.png` screenshot of the chart.
- **Share** – copy a permalink to the clipboard. When `PINNER_TOKEN` is set,
  exported JSON is pinned to Web3.Storage and the CID appears in a toast.
- **Theme** – toggle between light and dark mode.
- **GPU** – enable or disable WebGPU acceleration. The app sends a
  `{type:'gpu', available:<flag>}` message to the evolver worker which
  forwards the flag to mutation functions.

  Set `localStorage.setItem('USE_GPU','1')` to force the GPU backend when
  WebGPU is available.

Drag a previously exported JSON state onto the drop zone to restore a
simulation.

## Darwin-Archive
Every completed run is stored locally using IndexedDB. When storage access is
unavailable, the archive falls back to an in-memory store and data is lost on
refresh. The **Evolution** panel
lists archived runs with their score and novelty. Click **Re-spawn** next to a
row to restart the simulation using that seed.

## Arena & Meme Cloud
The **Arena panel** allows quick debates between roles on any candidate in the
frontier. Results appear ranked within the panel. The **Meme Cloud** below the
archive table visualizes common strategy transitions across runs.

## Running a Simulation
Use the **Simulator** panel to launch a full run. Adjust the seed list, population
size and number of generations, then press **Start**. When `PINNER_TOKEN` is set,
the resulting `replay.json` is pinned to Web3.Storage and the CID appears once
the run finishes. Keep this CID handy to share or reload the simulation later.

## Load via CID
Append `#/cid=&lt;CID&gt;` to the URL (or use the **Share** permalink) to replay a
previous run. The simulator fetches the JSON from IPFS and populates the chart.

## Privacy
Anonymous telemetry is optional. On first use a random ID is generated and
hashed with SHA-256 using the static salt `"insight"`. Only this salted hash and
basic usage metrics are sent to the OTLP endpoint. Clearing browser storage
resets the identifier.

Use the **Analytics** panel to enable or disable telemetry at any time.

See **Environment Setup** above for the list of supported variables.

## Safari/iOS Support
Pyodide is disabled on Safari and iOS devices because the runtime fails to load
reliably. The demo automatically falls back to the JavaScript engine instead of
executing Python code in the browser. Expect noticeably slower performance for
LLM tasks and the absence of features that rely on the Python bridge, such as
the local GPT‑2 critic.

## Running Browser Tests

The demo includes a small Playwright and Pytest suite. **Node.js ≥20** is
required. After fetching the WebAssembly assets simply run:

```bash
npm test
```

This command now builds the bundle automatically before running the tests. It
launches Playwright to exercise `dist/index.html` and then runs the Python
checks. Offline setups can point Playwright at pre‑downloaded browsers by
exporting `PLAYWRIGHT_BROWSERS_PATH=/path/to/browsers` or skip the download step
with `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1`.
