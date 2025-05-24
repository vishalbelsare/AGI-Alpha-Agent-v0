# Alpha-Factory Backend

This directory contains the production backend services for the Alpha-Factory stack. The primary entry point is `orchestrator.py`, which bootstraps agents, exposes REST and gRPC interfaces, and integrates optional components when the corresponding dependencies are installed.

## Key Modules

- `orchestrator.py` – main service supervisor and API gateway.
- `agents/` – individual domain agents auto-discovered by the orchestrator.
- `memory_*.py` – pluggable memory fabric implementations (vector & graph).
- `broker*/` – market/broker integrations used by finance-oriented agents.

## Environment Variables

The orchestrator honours several variables to tune runtime behaviour:

| Variable | Purpose | Default |
|----------|---------|---------|
| `DEV_MODE` | Enable lightweight in-memory mode | `false` |
| `LOGLEVEL` | Logging level (e.g. `DEBUG`) | `INFO` |
| `PORT` | REST API port | `8000` |
| `METRICS_PORT` | Prometheus metrics port (0 = disabled) | `0` |
| `A2A_PORT` | gRPC A2A port (0 = disabled) | `0` |
| `INSECURE_DISABLE_TLS` | Disable TLS for gRPC | `false` |
| `ALPHA_KAFKA_BROKER` | Kafka bootstrap servers | *(unset)* |
| `ALPHA_CYCLE_SECONDS` | Default agent cycle period | `60` |
| `MAX_CYCLE_SEC` | Hard limit per agent run in seconds | `30` |
| `ALPHA_ENABLED_AGENTS` | Comma-separated list of agents to run | *(all)* |

## Running Locally

```bash
cd alpha_factory_v1
python -m backend.orchestrator --dev
```

This starts the orchestrator in development mode, using in-memory stubs for Kafka and databases. Agents can be triggered via the REST interface at `http://localhost:8000`.

## Notes

- Optional dependencies are soft-imported; missing libraries will disable related features without crashing the service.
- The gRPC server starts automatically when `A2A_PORT` is set and shuts down cleanly on exit.
- Numeric environment variables are parsed leniently; invalid values fall back to the defaults above.
- See `../requirements.txt` for the full dependency list.

