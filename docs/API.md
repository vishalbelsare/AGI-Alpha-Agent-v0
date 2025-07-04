# API and CLI Usage

This page documents the REST endpoints provided by the demo API server and the available command line commands.

## REST endpoints

The API is implemented with FastAPI in `src/interface/api_server.py`.  Three routes
are available.  The orchestrator boots in the background when the server starts and
is gracefully shut down on exit:

- `POST /simulate` – start a new simulation.
- `GET /results/{sim_id}` – fetch final forecast data.
- `WS  /ws/{sim_id}` – stream progress logs while the simulation runs.

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

Start a new simulation. Send a JSON payload with the following fields:

- `horizon` – forecast horizon in years
- `pop_size` – number of individuals per generation
- `generations` – number of evolutionary steps

```json
{
  "horizon": 5,
  "pop_size": 6,
  "generations": 3
}
```

The response contains the generated simulation identifier:

```json
{"id": "<sim_id>"}
```

**GET `/results/{sim_id}`**

Return the final forecast and Pareto front for an earlier run.

Example response:

```json
{
  "forecast": [{"year": 1, "capability": 0.1}],
  "pareto": [[0.0, 0.0], [0.5, 0.2]],
  "logs": ["Year 1: 0 affected"]
}
```

**WebSocket `/ws/{sim_id}`**

Streams progress messages during a running simulation. Messages are plain text lines
such as `"Year 1: 0 affected"` or `"Generation 2"`. Close the socket once all
messages have been received.

The server honours environment variables defined in `.env` such as `PORT` (HTTP port), `OPENAI_API_KEY` and `RUN_MODE`. When `RUN_MODE=web`, a small frontend served at `/` consumes these endpoints via JavaScript.
