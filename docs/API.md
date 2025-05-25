# API and CLI Usage

This page documents the REST endpoints provided by the demo API server and the available command line commands.

## REST endpoints

The API is implemented with FastAPI in `src/interface/api_server.py` and exposes three routes when running:

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

Available commands are:

- `simulate` – run a forecast and launch the orchestrator. Key options include `--horizon`, `--curve`, `--pop-size` and `--generations`.
- `show-results` – display the latest ledger entries recorded by the orchestrator.
- `agents-status` – list currently registered agents.
- `replay` – replay ledger entries with a small delay for analysis.

The CLI is also installed as the `alpha-agi-bhf` entry point for convenience.
