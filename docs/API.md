# API and CLI Usage

This page documents the REST endpoints provided by the demo API server and the available command line commands.

## REST endpoints

The API is implemented with FastAPI in `src/interface/api_server.py`. Three
routes are available. The orchestrator boots in the background when the server
starts and is gracefully shut down on exit:

- `POST /simulate` – start a new simulation.
- `GET /results/{sim_id}` – fetch final forecast data.
- `WS  /ws/progress` – stream progress logs while the simulation runs.

### Authentication

All requests must include an `Authorization: Bearer $API_TOKEN` header. Set
`API_TOKEN` inside your `.env` file or pass `-e API_TOKEN=yourtoken` when
launching the Docker image.

Start the server with:

```bash
python -m src.interface.api_server --host 0.0.0.0 --port 8000
```

Once the server is running you can trigger a forecast using `curl`:

```bash
curl -X POST http://localhost:8000/simulate \
  -H "Authorization: Bearer $API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"horizon": 5, "pop_size": 6, "generations": 3}'
```

Retrieve the results when the run finishes:

```bash
curl -H "Authorization: Bearer $API_TOKEN" \
  http://localhost:8000/results/<sim_id>
```

## Command line interface

The CLI lives in `alpha_factory_v1/demos/alpha_agi_insight_v1/src/interface/cli.py`. It groups several commands under one entry point:

```bash
python cli.py [COMMAND] [OPTIONS]
```

Display all available commands and options:

```bash
python cli.py --help
```

For example, to run a three‑generation simulation with six agents for a
five‑year horizon:

```bash
python cli.py simulate --horizon 5 --pop-size 6 --generations 3
```

The orchestrator starts automatically and persists a ledger under `./ledger/`.
Use the `show-results` command to display the latest forecast:

```bash
python cli.py show-results
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

Return the final forecast for an earlier run. The returned list contains one
object per simulated year with the capability value reached by the model.

Example response:

```json
{
  "id": "<sim_id>",
  "forecast": [{"year": 1, "capability": 0.1}]
}
```

**WebSocket `/ws/progress`**

Streams progress messages during a running simulation. Messages are plain text lines
such as `"Year 1: 0 affected"` or `"Generation 2"`. Close the socket once all
messages have been received.

```bash
wscat -c "ws://localhost:8000/ws/progress" \
  -H "Authorization: Bearer $API_TOKEN"
```

The server honours environment variables defined in `.env` such as `PORT` (HTTP port), `OPENAI_API_KEY` and `RUN_MODE`. When `RUN_MODE=web`, a small frontend served at `/` consumes these endpoints via JavaScript.
