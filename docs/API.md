# API and CLI Usage

This page documents the REST endpoints provided by the demo API server and the available command line commands.

## REST endpoints

The API is implemented with FastAPI in `src/interface/api_server.py` and exposes three routes when running.
The orchestrator boots in the background when the server starts and is gracefully shut down on exit:

- `POST /simulate` – start a new simulation. The payload accepts the forecast horizon, population size and number of evolutionary generations. A unique simulation ID is returned immediately.
- `GET /results/{sim_id}` – retrieve the final forecast data and Pareto front once the simulation finishes.
- `WS  /ws/{sim_id}` – a websocket that streams progress logs while the simulation runs.

Start the server with:

```bash
python -m src.interface.api_server --host 0.0.0.0 --port 8000
```

## Command line interface

The CLI lives in `alpha_factory_v1/demos/alpha_agi_insight_v1/src/interface/cli.py`. It groups several commands under one entry point:

```bash
python cli.py [COMMAND] [OPTIONS]
```

For example, to run a three‑generation simulation with six agents for a
five‑year horizon:

```bash
python cli.py simulate --horizon 5 --pop-size 6 --generations 3
```

Available commands are:

- `simulate` – run a forecast and launch the orchestrator. Key options include `--horizon`, `--curve`, `--pop-size` and `--generations`.
- `show-results` – display the latest ledger entries recorded by the orchestrator.
- `agents-status` – list currently registered agents.
- `replay` – replay ledger entries with a small delay for analysis.

The CLI is also installed as the `alpha-agi-bhf` entry point for convenience.

### Endpoint Details

**POST `/simulate`**

Start a new simulation run. The JSON body accepts:

- `horizon` – forecast horizon in years.
- `pop_size` – number of individuals per generation.
- `generations` – number of evolutionary steps.

Returns a JSON object containing the simulation `id`.

**GET `/results/{sim_id}`**

Retrieve the final forecast and Pareto front for a previously launched simulation.

**WebSocket `/ws/{sim_id}`**

Provides a live feed of progress messages during a running simulation. Clients should close the socket once the final `done` message is received.

The server honours environment variables defined in `.env` such as `PORT` (HTTP port), `OPENAI_API_KEY` and `RUN_MODE`. When `RUN_MODE=web`, a small frontend served at `/` consumes these endpoints via JavaScript.
