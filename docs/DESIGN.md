# Design Overview

This document outlines the architecture of the Alpha Factory demo included in this repository. It also explains the individual agent roles and the Meta-Agentic Tree Search (MATS) algorithm used for evolutionary optimisation.

## Architecture

The system is composed of a lightweight orchestrator that coordinates a small swarm of agents. Each agent is a micro service with a specific responsibility. The orchestrator exposes both a command line interface and a minimal REST API so the demo can run headless or with a web front end. All communication is performed through simple envelope objects exchanged on an in-memory message bus.

### Orchestrator

The orchestrator manages message routing between agents and persists every
interaction in a ledger. This ledger can be replayed to analyse decision steps or
to visualise the overall run. Agents are invoked sequentially in short cycles so
the system remains deterministic and easy to debug.

The simulation core consists of two modules:

- `forecast.py` implements a basic capability forecast and a thermodynamic disruption trigger.
- `mats.py` implements zero-data evolutionary search used to refine candidate innovations.

Both modules are intentionally small so they can be inspected and extended easily.

## Agent roles

Seven agents are bootstrapped by the orchestrator:

1. **PlanningAgent** – builds a high level execution plan.
2. **ResearchAgent** – gathers background information and assumptions.
3. **StrategyAgent** – decides which sectors or ideas to pursue.
4. **MarketAgent** – evaluates potential economic impact.
5. **CodeGenAgent** – produces runnable code snippets when needed.
6. **SafetyGuardianAgent** – performs lightweight policy and safety checks.
7. **MemoryAgent** – persists ledger events for later replay.

Each agent implements short cycles of work which the orchestrator invokes sequentially. The ledger records every envelope processed so the entire run can be replayed for inspection.

## The MATS algorithm

MATS (Meta-Agentic Tree Search) is an NSGA-II style evolutionary loop that evolves a population of candidate solutions in two objective dimensions. Each individual has a numeric genome and is evaluated by a custom fitness function. Non-dominated sorting and crowding distance ensure that the search explores the trade‑off surface effectively. The resulting Pareto front highlights the best compromises discovered so far.

The demo uses MATS with a toy function `(x^2, y^2)` but the optimiser can be repurposed for arbitrary metrics. Results are visualised either in the Streamlit dashboard or through the REST API.

The helper function `run_evolution` initialises the population and executes the
NSGA‑II loop for a configurable number of generations. The population size,
mutation rate and generation count can be adjusted and a random ``seed`` enables
deterministic runs which is useful for testing and reproducibility.

## Data flow

Messages traverse the orchestrator in discrete cycles. Each cycle begins with the PlanningAgent emitting a high level goal. Subsequent agents enrich this envelope with research, strategy and market data before the CodeGenAgent proposes executable actions. The SafetyGuardianAgent performs a final policy check and, if approved, the MemoryAgent records the action to the ledger. This deterministic loop makes it easy to trace how a decision emerged from the collective agent swarm.

## Interfaces

The system exposes both a command line interface and a REST/WS API. The CLI is suitable for quick local experiments and supports subcommands for running simulations, replaying ledger events and inspecting agent status. The API server wraps the same orchestrator in a FastAPI application. Clients start a simulation via `POST /simulate`, fetch results with `GET /results/{id}` and stream logs through `WS /ws/{id}`. A lightweight web dashboard consumes these endpoints to visualise progress.

## Deployment model

The repository includes container and infrastructure templates for Docker Compose, Helm and Terraform. Each mode deploys the orchestrator together with optional agents and the web UI. Environment variables configured in `.env` control credentials, ports and runtime options. When running in Kubernetes, the Helm chart maps these variables to pod environment settings. The Terraform examples show how to provision equivalent services on AWS Fargate or Google Cloud Run.
