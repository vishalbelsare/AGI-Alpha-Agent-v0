"""
alpha_factory_v1.backend.agents.base
====================================

**AgentBase** – the canonical contract every Alpha-Factory agent abides by.
It offers:

• A rich, opinionated life-cycle (`setup → step()⭮ → teardown`) wrapped in a
  resilient *run-loop* with automatic back-off, Prometheus metrics emission
  and Kafka heartbeat integration (all *optional* – they degrade gracefully).

• First-class access to core platform services that the Orchestrator injects:
      · mem_vector / mem_graph           – Memory Fabric
      · world_model                      – Latent world-model / planner
      · publish()  / subscribe()         – Message-bus (Kafka or in-memory)
      · call_llm()                       – Unified LLM helper (OpenAI *or* local)

• Flexible scheduling:
      · `CYCLE_SECONDS`  – simple fixed-interval loop         (default 60 s)
      · `SCHED_SPEC`     – cron / rrule / human “every 5 m”   (if aiocron found)
      · `event-driven`   – omit both above and call step() yourself via bus/API

The class is intentionally **framework-free** (only stdlib) – every heavy
dependency is *best-effort imported* and silently skipped if unavailable;
your agent remains importable even in a bare Python environment.

---------------------------------------------------------------------------
Design Philosophy
---------------------------------------------------------------------------
1.  *Flawless degradation*   → No missing library should crash import time.
2.  *Observability first*    → Run-loop exposes metrics & heartbeats OOTB.
3.  *Security mindful*       → No dynamic `eval`, no shelling-out, strict attr
    whitelist when loading configs.
4.  *Zero boiler-plate*      → Sub-classes only implement two things:
        - class constants (NAME, …)            - `async def step(self): ...`
"""

from __future__ import annotations

###############################################################################
# Standard library imports – keep lightweight to guarantee availability
###############################################################################
import abc
import asyncio
import datetime as _dt
import json
import logging
import os
import random
import time
import traceback
from typing import Any, Dict, List, Mapping, Optional

###############################################################################
# Optional heavy deps (never hard-fail)
###############################################################################
try:                                            # Prometheus metrics
    from prometheus_client import Counter, Gauge
except ModuleNotFoundError:                     # pragma: no cover
    Counter = Gauge = None                      # type: ignore

try:                                            # Kafka producer for heart-beats
    from kafka import KafkaProducer             # type: ignore
except ModuleNotFoundError:                     # pragma: no cover
    KafkaProducer = None                        # type: ignore

try:                                            # Fancy scheduling (cron / rrule)
    import aiocron                              # type: ignore
except ModuleNotFoundError:                     # pragma: no cover
    aiocron = None                              # type: ignore

###############################################################################
# Logging – a single, consistent format across *every* agent
###############################################################################
_log = logging.getLogger("alpha_factory.agent")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "%(asctime)s  %(levelname)-8s  %(name)s.%(agent)s: %(message)s"
        )
    )
    _log.addHandler(_h)
    _log.setLevel(os.getenv("AF_AGENT_LOGLEVEL", "INFO").upper())


###############################################################################
# Internal helpers
###############################################################################
def _metrics(name: str):
    """Return (run_counter, err_counter, latency_gauge) or (None, None, None)."""
    if Counter is None:
        return None, None, None
    run_c  = Counter("af_agent_runs_total", "Agent run() calls", ["agent"])
    err_c  = Counter("af_agent_errors_total", "Exceptions", ["agent"])
    lat_g  = Gauge("af_agent_latency_seconds", "Step latency", ["agent"])
    return run_c.labels(name), err_c.labels(name), lat_g.labels(name)


def _kafka() -> Optional[KafkaProducer]:        # best-effort init – NON blocking
    broker = os.getenv("ALPHA_KAFKA_BROKER")
    if not broker or KafkaProducer is None:
        return None
    try:
        return KafkaProducer(
            bootstrap_servers=broker.split(","),
            value_serializer=lambda v: v if isinstance(v, bytes) else json.dumps(v).encode(),
            linger_ms=100,
        )
    except Exception:                           # noqa: BLE001
        _log.exception("Kafka bootstrap failed")
        return None


###############################################################################
# Public – every agent must inherit this
###############################################################################
class AgentBase(abc.ABC):
    # --------------------------------------------------------------------- #
    # Static meta – override in subclasses
    # --------------------------------------------------------------------- #
    NAME:              str  = "base"                 # 🛈 snake-case, unique
    VERSION:           str  = "0.1.0"
    CAPABILITIES:      List[str] = []                # e.g. ["forecast", "trade"]
    COMPLIANCE_TAGS:   List[str] = []                # e.g. ["GDPR"]
    REQUIRES_API_KEY:  bool = False                  # True → Orchestrator checks

    # Scheduling ----------------------------------------------------------------
    CYCLE_SECONDS: int | None = 60                   # simple loop (default 60 s)
    SCHED_SPEC:   str | None = None                  # cron "*/5 * * * *" etc.

    # Runtime-injected by Orchestrator ------------------------------------------
    orchestrator: Any = None                         # fabric / bus / world / cfg

    # --------------------------------------------------------------------- #
    # Life-cycle – subclasses rarely need to touch these
    # --------------------------------------------------------------------- #
    def __init__(self) -> None:
        self._stop_evt: asyncio.Event = asyncio.Event()
        self._kafka   : Optional[KafkaProducer] = _kafka()
        self._run_ctr , self._err_ctr , self._lat_g = _metrics(self.NAME)

    # 🎬 Called once by Orchestrator right after __init__
    async def setup(self) -> None:                   # noqa: D401
        """Optional one-time initialization (DB warm-up, model load, …)."""
        return None

    # 🏃 Main body – **must** be implemented by concrete agent
    @abc.abstractmethod
    async def step(self) -> None: ...               # pragma: no cover

    # 🔻 Called once at graceful shutdown
    async def teardown(self) -> None:                # noqa: D401
        """Optional clean-up (flush logs, close DB handles, …)."""
        return None

    # ------------------------------------------------------------------ #
    # ↓ Public helpers every agent can rely on without extra imports
    # ------------------------------------------------------------------ #
    # Memory Fabric
    @property
    def mem_vector(self):      return getattr(self.orchestrator, "mem_vector", None)
    @property
    def mem_graph(self):       return getattr(self.orchestrator, "mem_graph", None)
    # Latent world-model
    @property
    def world_model(self):     return getattr(self.orchestrator, "world_model", None)

    # Unified LLM helper (OpenAI / local)
    async def call_llm(self, **kw):                   # noqa: D401
        """Delegates to orchestrator.llm() – guarantees fallback availability."""
        return await self.orchestrator.llm(**kw)      # type: ignore[attr-defined]

    # Message-bus
    async def publish(self, topic: str, payload: Mapping[str, Any]):          # noqa: D401
        """Emit *payload* on *topic* (Kafka or in-mem depending on deployment)."""
        await self.orchestrator.publish(topic, payload)                       # type: ignore[attr-defined]

    async def subscribe(self, topic: str):
        """Async iterator over messages on *topic*."""
        async for m in self.orchestrator.subscribe(topic):                    # type: ignore[attr-defined]
            yield m

    # ------------------------------------------------------------------ #
    # Internal – run-loop wrapper handling metrics, heart-beats, back-off
    # (Orchestrator schedules this – agents never call ._run() themselves)
    # ------------------------------------------------------------------ #
    async def _run(self) -> None:                     # pragma: no cover
        ctx = {"agent": self.NAME}                    # logger extra

        # Option A – cron / aiocron spec ------------------------------------------------
        if self.SCHED_SPEC and aiocron:
            _log.info("Starting cron scheduler (%s)", self.SCHED_SPEC, extra=ctx)

            async def _cron_wrapper():
                while not self._stop_evt.is_set():
                    await self._safe_step(ctx)
            aiocron.crontab(self.SCHED_SPEC)(_cron_wrapper)                   # type: ignore[arg-type]
            await self._stop_evt.wait()
            return

        # Option B – fixed interval (CYCLE_SECONDS) -------------------------------
        interval = self.CYCLE_SECONDS or 0
        if interval <= 0:   # event-driven – nothing to schedule
            _log.info("No scheduler – event-driven mode", extra=ctx)
            await self._stop_evt.wait()
            return

        _log.info("Loop every %ss", interval, extra=ctx)
        jitter = max(1, int(os.getenv("AF_LOOP_JITTER_MS", "500"))) / 1000.0

        while not self._stop_evt.is_set():
            await self._safe_step(ctx)
            await asyncio.sleep(interval + random.uniform(0, jitter))

    # ------------------------------------------------------------------ #
    async def _safe_step(self, ctx):                  # pragma: no cover
        t0 = time.perf_counter()
        if self._run_ctr: self._run_ctr.inc()

        try:
            await self.step()
            ok = True
        except Exception as exc:                      # noqa: BLE001
            ok = False
            if self._err_ctr: self._err_ctr.inc()
            _log.error("step() raised: %s\n%s", exc, traceback.format_exc(), extra=ctx)

        latency = time.perf_counter() - t0
        if self._lat_g: self._lat_g.set(latency)

        # Heart-beat to Kafka (best-effort)
        if self._kafka:
            try:
                self._kafka.send(
                    "agent.heartbeat",
                    {
                        "name": self.NAME,
                        "ts": _dt.datetime.utcnow().isoformat(),
                        "latency_ms": round(latency*1000, 3),
                        "ok": ok,
                    },
                )
                # no explicit flush – linger_ms handles
            except Exception:                         # noqa: BLE001
                _log.debug("Kafka heartbeat failed", extra=ctx)

    # ------------------------------------------------------------------ #
    # Public stop() – Orchestrator calls this on graceful shutdown
    # ------------------------------------------------------------------ #
    async def stop(self) -> None:                      # noqa: D401
        """Signal run-loop to exit ASAP."""
        self._stop_evt.set()


###############################################################################
# Utility – thin decorator to auto-register agent classes at import time
###############################################################################
def register(cls: type[AgentBase]) -> type[AgentBase]:
    """`@register` → adds the class to backend.agents.AGENT_REGISTRY."""
    # local import to avoid circular dep
    from . import AGENT_REGISTRY                     # type: ignore
    AGENT_REGISTRY[getattr(cls, "NAME", cls.__name__)] = cls
    return cls
