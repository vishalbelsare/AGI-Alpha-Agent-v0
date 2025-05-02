"""
backend.orchestrator
===================================================================
Alpha-Factory v1 👁️✨ — Multi-Agent AGENTIC α-AGI
-------------------------------------------------------------------
One-stop control-tower for every demo, lab, or production rollout.

Key capabilities
----------------
✓  Auto-discovers & drives every agent found in ``backend/agents`` or
   exposed via Python plugin entry-points (``alpha_factory.agents``).

✓  REST + OpenAPI (FastAPI) **AND** bidirectional gRPC (A2A-v0.5) in parallel.

✓  Optional **OpenAI Agents SDK** & **Google ADK** bridges – the moment those
   libs are `pip install`-ed the orchestrator registers tools / heartbeats.

✓  Kafka (experience-replay), Prometheus (/metrics), Health-probe (/healthz),
   TLS / mTLS (out-of-box) & fully offline fall-backs.

✓  Declarative scheduling: cron / RRULE via ``SCHED_SPEC`` *or* classic
   ``CYCLE_SECONDS`` cadence – no code changes needed.

Run it
------
    python -m backend.orchestrator         # local dev
    PORT=8080 METRICS_PORT=9090 python -m backend.orchestrator
    docker compose -f demos/docker-compose.finance.yml up      etc.

All heavy deps are **soft-imports**: if a library (or Kafka) is missing the
feature gracefully degrades instead of crashing – perfect for air-gapped /
edge deployments.

---------------------------------------------------------------------------
Copyright © 2023-2025 Montreal AI.
License: Apache-2.0
"""
from __future__ import annotations

# std-lib ---------------------------------------------------------------------
import asyncio
import datetime as _dt
import importlib
import json
import logging
import os
import signal
import ssl
import sys
import time
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any, Awaitable, Callable, Dict, List, Optional

# -----------------------------------------------------------------------------
# Dynamic / optional third-party imports  (ALL are soft-fail)
# -----------------------------------------------------------------------------
try:
    # OpenAI Agents SDK (2025.04+)
    from openai.agents import AgentRuntime, AgentContext  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    AgentRuntime = None  # type: ignore
    AgentContext = object  # type: ignore

try:
    # Google Agent Development Kit
    import adk  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    adk = None  # type: ignore

try:
    # gRPC / protobuf
    import grpc  # type: ignore
    from concurrent import futures
except ModuleNotFoundError:  # pragma: no cover
    grpc = None  # type: ignore

try:
    # Kafka producer for event / experience stream
    from kafka import KafkaProducer  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    KafkaProducer = None  # type: ignore

try:
    # Prometheus metrics
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST, start_http_server  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    Counter = Histogram = Gauge = generate_latest = CONTENT_TYPE_LATEST = start_http_server = None  # type: ignore

try:
    # FastAPI REST gateway
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import PlainTextResponse
    import uvicorn
except ModuleNotFoundError:  # pragma: no cover
    FastAPI = None  # type: ignore
    HTTPException = PlainTextResponse = uvicorn = None  # type: ignore

# -----------------------------------------------------------------------------
# Local imports (must always exist)
# -----------------------------------------------------------------------------
from backend.agents import get_agent, list_agents  # public helper API

# -----------------------------------------------------------------------------
# Environment / configuration --------------------------------------------------
ENV = os.environ.get
_PORT = int(ENV("PORT", "8000"))
_METRICS_PORT = int(ENV("METRICS_PORT", "0"))
_KAFKA_BROKER = ENV("ALPHA_KAFKA_BROKER")          # host:port or comma-sep
_CYCLE_DEFAULT = int(ENV("ALPHA_CYCLE_SECONDS", "60"))
_A2A_PORT = int(ENV("A2A_PORT", "0"))              # 0 → disabled
_SSL_DISABLE = ENV("INSECURE_DISABLE_TLS", "false").lower() == "true"
_LOG_LEVEL = ENV("LOGLEVEL", "INFO").upper()

logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s.%(msecs)03d %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("alpha_factory.orchestrator")

# -----------------------------------------------------------------------------
# Prometheus metrics  (no-op stubs if lib absent) ------------------------------
# -----------------------------------------------------------------------------
def _noop(*_a, **_kw):  # type: ignore
    class _N:  # pylint: disable=too-few-public-methods
        def labels(self, *_a, **_kw):  # noqa: D401
            return self

        def observe(self, *_a):  # noqa: D401
            pass

        def inc(self, *_a):  # noqa: D401
            pass

        def set(self, *_a):  # noqa: D401
            pass

    return _N()


MET_CYCLE_LAT = Histogram("agent_cycle_latency_ms", "Per-cycle latency", ["agent"]) if Histogram else _noop()
MET_CYCLE_ERR = Counter("agent_cycle_errors_total", "Exceptions per agent", ["agent"]) if Counter else _noop()
MET_AGENT_UP = Gauge("agent_up", "1 if agent thread active", ["agent"]) if Gauge else _noop()

if _METRICS_PORT and start_http_server:
    start_http_server(_METRICS_PORT)
    logger.info("Prometheus metrics exposed at :%d/metrics", _METRICS_PORT)

# -----------------------------------------------------------------------------
# Kafka producer helper (experience replay stream) ----------------------------
# -----------------------------------------------------------------------------
if _KAFKA_BROKER and KafkaProducer:
    _kafka_producer: Any = KafkaProducer(
        bootstrap_servers=_KAFKA_BROKER,
        value_serializer=lambda v: json.dumps(v).encode(),
        linger_ms=50,
    )

    def _publish(topic: str, payload: Dict[str, Any]) -> None:
        _kafka_producer.send(topic, payload)
else:  # fallback: stdout debug
    def _publish(topic: str, payload: Dict[str, Any]) -> None:  # type: ignore
        logger.debug("▶ EVT %-22s %s", topic, json.dumps(payload))

# -----------------------------------------------------------------------------
# Model Context Protocol (MCP) minimalist envelope ----------------------------
# -----------------------------------------------------------------------------
def _mcp_wrap(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mcp_version": "0.1",
        "timestamp": _dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
        "payload": payload,
    }

# -----------------------------------------------------------------------------
# Async helper -----------------------------------------------------------------
async def _maybe_await(fn: Callable[[], Any]) -> Any:
    if asyncio.iscoroutinefunction(fn):
        return await fn()
    return await asyncio.to_thread(fn)

# -----------------------------------------------------------------------------
# OpenAI Agents runtime singleton ---------------------------------------------
class _OAI_Runtime:
    _instance: Optional[AgentRuntime] = None

    @classmethod
    def get(cls) -> Optional[AgentRuntime]:
        if cls._instance is None and AgentRuntime is not None:
            cls._instance = AgentRuntime()
            logger.info("OpenAI Agents SDK detected — tools auto-registered")
        return cls._instance

# -----------------------------------------------------------------------------
# Agent runner (schedule & supervise) ------------------------------------------
class AgentRunner:  # pylint: disable=too-few-public-methods
    """Wrap one agent instance and execute it according to its schedule."""

    def __init__(self, name: str):
        self.name = name
        self.instance = get_agent(name)
        self.period = getattr(self.instance, "CYCLE_SECONDS", _CYCLE_DEFAULT)
        self.spec = getattr(self.instance, "SCHED_SPEC", None)
        self._next_run = time.time()
        self._task: Optional[asyncio.Task] = None

        # auto-register OpenAI Agents tools if SDK present & the agent subclasses AgentContext
        if AgentRuntime and isinstance(self.instance, AgentContext):
            _OAI_Runtime.get().register(self.instance)  # type: ignore[arg-type]

    # -------------------------------------------------------------------------
    def _calc_next(self) -> None:
        """Compute next execution timestamp (cron or fixed period)."""
        now = time.time()
        if self.spec:
            try:
                from croniter import croniter  # type: ignore
                self._next_run = croniter(self.spec, _dt.datetime.fromtimestamp(now)).get_next(float)
                return
            except Exception as exc:  # invalid cron – fall back
                logger.warning("%s: invalid SCHED_SPEC '%s' (%s) – using fixed period", self.name, self.spec, exc)
                self.spec = None
        self._next_run = now + self.period

    # -------------------------------------------------------------------------
    async def step(self) -> None:
        if time.time() < self._next_run:
            return
        self._calc_next()

        async def _one_cycle() -> None:
            t0 = time.time()
            try:
                await _maybe_await(self.instance.run_cycle)
                MET_CYCLE_LAT.labels(self.name).observe((time.time() - t0) * 1e3)
                _publish("exp.stream", _mcp_wrap({"agent": self.name, "latency_ms": (time.time() - t0) * 1e3}))
            except Exception as exc:  # never crash the orchestrator
                MET_CYCLE_ERR.labels(self.name).inc()
                logger.exception("%s.run_cycle crashed: %s", self.name, exc)

        self._task = asyncio.create_task(_one_cycle())

# -----------------------------------------------------------------------------
# gRPC A2A bidirectional stream service ---------------------------------------
async def _serve_a2a(runners: Dict[str, AgentRunner]) -> None:  # noqa: D401
    if not _A2A_PORT or grpc is None:
        return  # disabled or lib missing

    try:
        from backend.proto import a2a_pb2, a2a_pb2_grpc  # generated stubs
    except ModuleNotFoundError:
        logger.warning("grpc enabled but backend.proto stubs missing – A2A disabled")
        return

    class PeerService(a2a_pb2_grpc.PeerServiceServicer):  # type: ignore
        async def Stream(self, request_iterator, context):  # noqa: N802
            async for msg in request_iterator:  # type: ignore[assignment]
                if msg.WhichOneof("payload") == "trigger":
                    tgt = msg.trigger.name
                    if tgt in runners:
                        runners[tgt]._next_run = 0  # prompt immediate exec
                        yield a2a_pb2.StreamReply(ack=a2a_pb2.Ack(id=msg.id))
                elif msg.WhichOneof("payload") == "status":
                    stats = [
                        a2a_pb2.AgentStat(name=n, next_run=int(r._next_run))
                        for n, r in runners.items()
                    ]
                    yield a2a_pb2.StreamReply(status_reply=a2a_pb2.StatusReply(stats=stats))

    # TLS/mTLS setup ----------------------------------------------------------
    creds = None
    if not _SSL_DISABLE:
        # look for mounted certs (server.crt + server.key)
        cert_dir = Path(ENV("TLS_CERT_DIR", "/certs"))
        crt, key = cert_dir / "server.crt", cert_dir / "server.key"
        if crt.exists() and key.exists():
            with crt.open("rb") as c, key.open("rb") as k:
                creds = grpc.ssl_server_credentials(((k.read(), c.read()),))
    server = grpc.aio.server(options=[
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
    ])
    a2a_pb2_grpc.add_PeerServiceServicer_to_server(PeerService(), server)
    bind_addr = f"[::]:{_A2A_PORT}"
    server.add_secure_port(bind_addr, creds) if creds else server.add_insecure_port(bind_addr)
    await server.start()
    logger.info("A2A gRPC server listening on %s (%s)", bind_addr, "plaintext" if creds is None else "TLS")

# -----------------------------------------------------------------------------
# Google ADK node registration -------------------------------------------------
async def _adk_register() -> None:
    if adk is None:
        return
    client = adk.Client()
    await client.register(node_type="orchestrator", metadata={"runtime": "alpha_factory"})
    logger.info("Registered with ADK mesh (node-id: %s)", client.node_id)

# -----------------------------------------------------------------------------
# FastAPI REST gateway ---------------------------------------------------------
_app: Optional[FastAPI] = None
if FastAPI:
    _app = FastAPI(
        title="Alpha-Factory v1 Orchestrator",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
    )


def _build_rest_api(runners: Dict[str, AgentRunner]) -> Optional[FastAPI]:
    if _app is None:
        return None

    @_app.get("/healthz", response_class=PlainTextResponse)
    async def _healthz():  # noqa: D401
        return "ok"

    @_app.get("/agents")
    async def _agents():  # noqa: D401
        return {"agents": list(runners.keys())}

    @_app.post("/agent/{name}/trigger")
    async def _trigger(name: str):  # noqa: D401
        if name not in runners:
            raise HTTPException(404, "Agent not found")
        runners[name]._next_run = 0
        return {"status": "queued"}

    @_app.get("/metrics", response_class=PlainTextResponse)
    async def _metrics():  # noqa: D401
        if generate_latest is None:
            raise HTTPException(503, "prometheus_client not installed")
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return _app


# -----------------------------------------------------------------------------
# Main async supervisor loop ---------------------------------------------------
async def _main() -> None:
    # Discover & instantiate agents -------------------------------------------------
    names = list_agents()
    if not names:
        logger.error("No agents registered – aborting")
        sys.exit(1)

    runners = {n: AgentRunner(n) for n in names}
    logger.info("Bootstrapped %d agent(s): %s", len(runners), ", ".join(runners))

    # Expose REST API (if FastAPI present) -----------------------------------------
    rest_api = _build_rest_api(runners)
    rest_server: Optional[uvicorn.Server] = None
    if rest_api and uvicorn:
        config = uvicorn.Config(rest_api, host="0.0.0.0", port=_PORT, log_level=_LOG_LEVEL.lower())
        rest_server = uvicorn.Server(config)
        asyncio.create_task(rest_server.serve())
        logger.info("REST API available at http://localhost:%d/docs", _PORT)

    # Kick off optional subsystems --------------------------------------------------
    await asyncio.gather(_serve_a2a(runners), _adk_register())

    # Graceful shutdown handling ----------------------------------------------------
    stop_event = asyncio.Event()

    def _graceful_shutdown(*_args):  # noqa: D401
        logger.warning("Shutdown signal received — cleaning up…")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            asyncio.get_running_loop().add_signal_handler(sig, partial(_graceful_shutdown, sig))
        except NotImplementedError:  # Windows
            signal.signal(sig, _graceful_shutdown)  # type: ignore[arg-type]

    # Core scheduling loop ----------------------------------------------------------
    while not stop_event.is_set():
        await asyncio.gather(*(r.step() for r in runners.values()))
        await asyncio.sleep(0.2)  # tight-ish loop; agents self-pace

    # Drain pending tasks -----------------------------------------------------------
    await asyncio.gather(*(r._task for r in runners.values() if r._task), return_exceptions=True)
    if rest_server:
        await rest_server.shutdown()
    logger.info("Orchestrator stopped cleanly")

# -----------------------------------------------------------------------------
# CLI entry-point --------------------------------------------------------------
if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
