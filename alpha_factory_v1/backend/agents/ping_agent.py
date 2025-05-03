"""
alpha_factory_v1.backend.agents.ping_agent
==========================================

A *zero-risk* heartbeat agent that proves the orchestration loop, metrics stack
and message bus are alive.  It remains **harmlessly optional** – set  
`AF_DISABLE_PING_AGENT=true` to skip instantiation.

Key design notes
----------------
• 100 % dependency-light – runs even if Kafka, Prometheus *or* OpenTelemetry are
  missing.  
• First-class observability – exposes Prom-metrics, OTEL trace-spans and emits a
  Kafka heartbeat event `agent.ping` every cycle.  
• Self-documenting JSON logs – compliant with the unified logging contract used
  across Alpha-Factory.  
• Immutable constants & dataclass-like config so that a non-technical operator
  can adjust the cadence via a single env-var (`AF_PING_INTERVAL`).

Environment
-----------
AF_PING_INTERVAL   Seconds between pings   (int, default **60**)  
AF_DISABLE_PING_AGENT   When “true” the orchestrator ignores this class.

"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from backend.agents.base import AgentBase
from backend.agents import register

# --------------------------------------------------------------------------- #
# Logging – inherits JSON formatter from AgentBase initialiser
# --------------------------------------------------------------------------- #
_log = logging.getLogger("alpha_factory.agent.ping")

# --------------------------------------------------------------------------- #
@register
class PingAgent(AgentBase):
    """
    Tiny but *production-critical* diagnostic agent.

    The class is auto-discovered by `backend.agents.__init__`.
    """

    # ─────────────────────────────────────────────────────────────── constants ──
    NAME: str = "ping"
    VERSION: str = "1.1.0"
    CAPABILITIES: list[str] = ["diagnostics", "observability"]
    COMPLIANCE_TAGS: list[str] = []
    SAFE_TO_REMOVE: bool = True                # lets DevOps disable in prod
    _DEFAULT_INTERVAL: int = 60

    # Read-only setting (env overrides default)
    CYCLE_SECONDS: int = int(
        os.getenv("AF_PING_INTERVAL", str(_DEFAULT_INTERVAL)).strip() or _DEFAULT_INTERVAL
    )

    # ─────────────────────────────────────────────────────────── lifecycle ──
    async def setup(self) -> None:                        # noqa: D401
        """Initialise local metric helpers & sanity-log a welcome line."""
        self._prom_gauge = self._lazy_prometheus_gauge()
        _log.info(
            "PingAgent online – interval=%s s, kafka=%s",
            self.CYCLE_SECONDS,
            bool(self._kafka_producer),  # provided by AgentBase
            extra={"agent": self.NAME},
        )

    async def step(self) -> None:                         # noqa: D401
        """Runs automatically every *CYCLE_SECONDS* (see AgentBase scheduler)."""
        ts = datetime.now(tz=timezone.utc)
        iso_ts = ts.isoformat(timespec="seconds")
        ctx: Mapping[str, Any] = {"agent": self.NAME}

        # 1️⃣  Structured JSON log
        _log.info("ping", extra={**ctx, "ts": iso_ts})

        # 2️⃣  Prometheus metric
        if self._prom_gauge:
            self._prom_gauge.set(ts.timestamp())

        # 3️⃣  OTEL trace span (auto-no-op if OTEL absent)
        with self._tracer.start_as_current_span("ping_cycle") as span:  # type: ignore[attr-defined]
            span.set_attribute("ping.iso_ts", iso_ts)
            span.set_attribute("ping.agent_version", self.VERSION)

        # 4️⃣  Kafka heartbeat  (AgentBase handles None-producer fallback)
        await self.publish(
            topic="agent.ping",
            msg={"ts": iso_ts, "agent": self.NAME, "version": self.VERSION},
        )

    async def teardown(self) -> None:
        _log.info("PingAgent shutting down", extra={"agent": self.NAME})

    # ──────────────────────────────────────────────────────── helpers / private ──
    def _lazy_prometheus_gauge(self):
        try:
            from prometheus_client import Gauge  # pylint: disable=import-error
        except ModuleNotFoundError:
            _log.debug("prometheus_client missing – metric disabled")
            return None
        return Gauge(
            "af_ping_last_epoch",
            "Unix epoch of last successful PingAgent heartbeat",
        )

    # AgentBase already provides:  self._kafka_producer, self.publish(),
    # self._tracer  (OpenTelemetry tracer), scheduler glue, etc.
