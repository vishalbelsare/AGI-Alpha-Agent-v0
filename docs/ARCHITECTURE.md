[See docs/DISCLAIMER_SNIPPET.md](../docs/DISCLAIMER_SNIPPET.md)

# Architecture Overview

This document gives a concise overview of the Alpha‑Factory stack.

```mermaid
flowchart TD
    subgraph Orchestrator
        O["BaseOrchestrator"]
        M["AgentManager"]
        S[(REST/gRPC)]
    end
    A["Agents"] -->|heartbeats| M
    O --> M
    O --> S
    M -->|tasks| A
    A -->|memory ops| F[(Memory Fabric)]
    M -->|events| B[(Kafka)]
```

The orchestrator instantiates an `AgentManager` which schedules agent cycles and
monitors their health. Optional REST and gRPC servers expose control endpoints.
Agents persist data through the Memory Fabric and may publish events to Kafka.

All components can run independently in development and are designed to fail
gracefully when optional dependencies (like Kafka or FastAPI) are missing.

## Key Components

- **BaseOrchestrator** – shared helper that starts/stops the servers and
  underlying `AgentManager`.
- **AgentManager** – maintains a collection of `AgentRunner` instances,
  coordinating their execution and heartbeats.
- **Memory Fabric** – pluggable storage combining vector and graph databases.
- **Telemetry** – Prometheus metrics exporting agent cycle latency and errors.

For more details see `docs/DESIGN.md` and the module docstrings within
`alpha_factory_v1/backend`.
