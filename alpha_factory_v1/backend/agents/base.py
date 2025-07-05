# SPDX-License-Identifier: Apache-2.0
"""
alpha_factory_v1.backend.agents.base
====================================

AgentBase – canonical contract ALL Alpha-Factory agents must implement.

Key features
------------
✔ Graceful *degradation* – imports of heavy/optional deps are best-effort; the
  file remains import-safe on a tiny python-only system.

✔ First-class *observability* – Prometheus metrics, structured JSON logs,
  Kafka heart-beats; everything is disabled automatically when a dependency
  is absent.

✔ Hassle-free *scheduling* – fixed interval, cron-style (`aiocron`) or event-
  driven (no schedule).  Agents only have to implement `async step()`.

✔ Built-in *service wiring* – properties expose Memory Fabric, world-model,
  LLM helper and message-bus exactly the same way in every agent.

✔ Uniform *shutdown* & error-handling – a single `_safe_step()` wrapper counts
  runs, records latency and prevents rogue exceptions from killing the process.

✔ Tiny *registry* decorator – `@register` auto-adds the class to the global
  `backend.agents.AGENT_REGISTRY`, enabling true plug-and-play discovery.

The class is intentionally framework-agnostic: **no FastAPI, no OpenAI SDK,
no OR-Tools** imports appear here – that belongs in concrete agent modules,
never in the bare contract.

---------------------------------------------------------------------------
For details see the Developer Guide (§2 “Writing a New Agent”) in the repo doc.
---------------------------------------------------------------------------
"""

from __future__ import annotations

# ───────────────────────────────────────────────────────────────────────────────
# ░░░ 1. Std-lib imports only (guaranteed present) ░░░
# ───────────────────────────────────────────────────────────────────────────────
import abc
import asyncio
import contextlib
import datetime as _dt
import json
import logging
import os
import random
import time
import traceback
from typing import Any, List, Mapping, MutableMapping, Optional

# ───────────────────────────────────────────────────────────────────────────────
# ░░░ 2. Optional, best-effort 3rd-party imports ░░░
# ───────────────────────────────────────────────────────────────────────────────
try:  # -- Prometheus metrics
    from backend.agents.registry import Counter, Gauge  # type: ignore
except Exception:  # pragma: no cover
    logging.getLogger(__name__).warning("prometheus_client missing – metrics disabled")
    Counter = Gauge = None  # type: ignore

from backend.agents.registry import register

try:  # -- Kafka producer for heart-beats
    from kafka import KafkaProducer  # type: ignore
    from kafka.errors import KafkaError
except ModuleNotFoundError:  # pragma: no cover
    logging.getLogger(__name__).warning("kafka-python missing – heartbeats disabled")
    KafkaProducer = None  # type: ignore

try:  # -- Cron / RRULE scheduling helper
    import aiocron  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    logging.getLogger(__name__).warning("aiocron not installed – scheduling disabled")
    aiocron = None  # type: ignore

# ───────────────────────────────────────────────────────────────────────────────
# ░░░ 3. Global logger configured once, reused by every agent ░░░
# ───────────────────────────────────────────────────────────────────────────────
_logger = logging.getLogger("alpha_factory.agent")
if not _logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s.%(agent)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    _logger.addHandler(_h)
    _logger.setLevel(os.getenv("AF_AGENT_LOGLEVEL", "INFO").upper())

with contextlib.suppress(ModuleNotFoundError):
    from opentelemetry import trace

tracer = trace.get_tracer(__name__) if "trace" in globals() else None  # type: ignore

# ───────────────────────────────────────────────────────────────────────────────
# ░░░ 4. Internal helper factories ░░░
# ───────────────────────────────────────────────────────────────────────────────
_RUN_COUNTER: Counter | None = None
_ERR_COUNTER: Counter | None = None
_LAT_GAUGE: Gauge | None = None


def _prom_metrics(agent_name: str):
    """Return Prometheus metrics for ``agent_name``.

    Metrics are instantiated lazily on first call and reused across agents to
    avoid duplicate registration errors during testing.
    """
    global _RUN_COUNTER, _ERR_COUNTER, _LAT_GAUGE
    if Counter is None:
        return None, None, None

    if _RUN_COUNTER is None or type(_RUN_COUNTER) is not Counter:
        _RUN_COUNTER = Counter("af_agent_runs_total", "Total step() calls", ["agent"])
        _ERR_COUNTER = Counter("af_agent_errors_total", "Unhandled exceptions", ["agent"])
        _LAT_GAUGE = Gauge("af_agent_latency_seconds", "Step latency", ["agent"])

    assert _ERR_COUNTER is not None and _LAT_GAUGE is not None
    return (
        _RUN_COUNTER.labels(agent_name),
        _ERR_COUNTER.labels(agent_name),
        _LAT_GAUGE.labels(agent_name),
    )


def _kafka_producer() -> Optional[KafkaProducer]:
    """Instantiate a KafkaProducer if ALPHA_KAFKA_BROKER is set *and*
    `kafka-python` is installed; otherwise return None."""
    broker = os.getenv("ALPHA_KAFKA_BROKER")
    if not broker or KafkaProducer is None:
        return None

    try:
        return KafkaProducer(
            bootstrap_servers=[b.strip() for b in broker.split(",") if b.strip()],
            value_serializer=lambda v: (v if isinstance(v, bytes) else json.dumps(v).encode()),
            linger_ms=250,
        )
    except KafkaError:
        _logger.exception("Failed to bootstrap Kafka producer")
        return None


# ───────────────────────────────────────────────────────────────────────────────
# ░░░ 5. AgentBase definition ░░░
# ───────────────────────────────────────────────────────────────────────────────
class AgentBase(abc.ABC):
    # --- Class-level metadata (override in subclasses) ---
    NAME: str = "base"  # snake-case, globally unique
    VERSION: str = "0.1.0"
    CAPABILITIES: List[str] = []  # e.g. ["forecast", "trade", "plan"]
    COMPLIANCE_TAGS: List[str] = []  # e.g. ["GDPR", "HIPAA"]
    REQUIRES_API_KEY: bool = False  # orchestrator will validate if True

    # Scheduling ────────────────────────────────────────────────────────────
    CYCLE_SECONDS: int | None = 60  # fixed-interval; None → use SCHED_SPEC
    SCHED_SPEC: str | None = None  # cron-style, processed by aiocron

    # Runtime-injected by orchestrator ──────────────────────────────────────
    orchestrator: Any = None  # Fabric / bus / world-model / cfg

    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        self._stop_evt: asyncio.Event = asyncio.Event()
        self._metrics_run, self._metrics_err, self._metrics_lat = _prom_metrics(self.NAME)
        self._kafka = _kafka_producer()

    # ════ Life-cycle hooks ════
    async def setup(self) -> None:  # noqa: D401
        """One-time async initialization (DB warm-up, model load…)."""
        return None

    async def init_async(self) -> None:  # pragma: no cover - optional hook
        """Launch background tasks once the event loop is running."""
        return None

    @abc.abstractmethod
    async def step(self) -> None:  # noqa: D401
        """The agent's main unit of work.  MUST be overridden."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    async def run_cycle(self) -> None:  # pragma: no cover - default wrapper
        """Single orchestrator cycle – runs :meth:`step` once."""
        span_cm = tracer.start_as_current_span(f"{self.NAME}.run_cycle") if tracer else contextlib.nullcontext()
        with span_cm:
            await self.step()

    async def teardown(self) -> None:  # noqa: D401
        """Optional async clean-up (closing DB handles etc.)."""
        return None

    # ════ Convenience properties exposed by orchestrator ════
    # Memory Fabric
    @property
    def mem_vector(self):  # -> MemoryVector | None
        return getattr(self.orchestrator, "mem_vector", None)

    @property
    def mem_graph(self):  # -> MemoryGraph | None
        return getattr(self.orchestrator, "mem_graph", None)

    # Latent world-model / planner
    @property
    def world_model(self):
        return getattr(self.orchestrator, "world_model", None)

    # Unified LLM helper
    async def call_llm(self, *args, **kwargs):
        """LLM abstraction – delegates to orchestrator, supports transparent
        fallback to local models when OPENAI_API_KEY is unset."""
        return await self.orchestrator.llm(*args, **kwargs)  # type: ignore[attr-defined]

    # Message-bus helpers
    async def publish(self, topic: str, msg: Mapping[str, Any]):
        await self.orchestrator.publish(topic, msg)  # type: ignore[attr-defined]

    async def subscribe(self, topic: str):
        async for m in self.orchestrator.subscribe(topic):  # type: ignore[attr-defined]
            yield m

    # ════ Internal run-loop (managed by orchestrator) ════
    async def _run(self) -> None:  # pragma: no cover
        log_ctx = {"agent": self.NAME}

        # ── Option A: cron / rrule schedule ────────────────────────────────
        if self.SCHED_SPEC and aiocron:
            _logger.info("Cron schedule [%s] activated", self.SCHED_SPEC, extra=log_ctx)

            async def _cron_wrapper():  # each cron tick triggers exactly ONE step
                await self._safe_step(log_ctx)

            aiocron.crontab(self.SCHED_SPEC)(_cron_wrapper)  # type: ignore[arg-type]
            await self._stop_evt.wait()
            return

        # ── Option B: fixed interval (default) ────────────────────────────
        interval = max(0, self.CYCLE_SECONDS or 0)
        if interval == 0:
            _logger.info("Event-driven mode (no automatic loop)", extra=log_ctx)
            await self._stop_evt.wait()
            return

        jitter = max(0.2, float(os.getenv("AF_LOOP_JITTER_MS", "250"))) / 1000.0
        _logger.info("Loop each %ss (+/- jitter)", interval, extra=log_ctx)

        while not self._stop_evt.is_set():
            await self._safe_step(log_ctx)
            await asyncio.sleep(interval + random.uniform(0, jitter))

    # ────────────────────────────────────────────────────────────────────
    async def _safe_step(self, log_ctx: MutableMapping[str, Any]):  # pragma: no cover
        t0 = time.perf_counter()
        if self._metrics_run:
            self._metrics_run.inc()

        ok = True
        try:
            await self.step()
        except Exception as exc:  # noqa: BLE001 - step() may raise anything
            ok = False
            if self._metrics_err:
                self._metrics_err.inc()
            _logger.error("step() raised: %s", exc, extra=log_ctx)
            _logger.debug("Traceback:\n%s", traceback.format_exc(), extra=log_ctx)

        latency = time.perf_counter() - t0
        if self._metrics_lat:
            self._metrics_lat.set(latency)

        # -- Heart-beat to Kafka (non-blocking) --
        if self._kafka:
            try:
                self._kafka.send(
                    "agent.heartbeat",
                    {
                        "name": self.NAME,
                        "ts": _dt.datetime.utcnow().isoformat(timespec="seconds"),
                        "latency_ms": round(latency * 1000, 3),
                        "ok": ok,
                    },
                )
            except Exception:  # pragma: no cover
                _logger.debug("Kafka heart-beat failed", extra=log_ctx)

    # ════ Public stop() called by orchestrator ════
    async def stop(self):
        """Signal the agent to shut down gracefully."""
        self._stop_evt.set()

    # ------------------------------------------------------------------
    def load_weights(self, path: str) -> None:
        """Load updated model weights from *path*.

        Subclasses may override this to implement hot-swapping of
        learning artefacts.  The default implementation simply stores the
        path for later use.
        """
        self._weights_path = path

    async def skill_test(self, payload: dict) -> dict:
        """Execute a diagnostic skill test.

        Agents may override this method to provide custom behaviour.
        The default implementation returns ``{"ok": True}``.
        """
        return {"ok": True}

    # ────────────────────────────────────────────────────────────────────
    # Pretty representation (helps debugging & logging)
    # ────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.__class__.__name__} name={self.NAME!r} v{self.VERSION}>"


# ───────────────────────────────────────────────────────────────────────────────
