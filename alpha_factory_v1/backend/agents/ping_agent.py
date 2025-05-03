"""
alpha_factory_v1.backend.agents.ping_agent
==========================================

A minimal **heartbeat / smoke-test agent**.

Purpose
-------
* Confirms the orchestration loop, metrics, and message-bus are operational.
* Emits a log line, Prometheus gauge update, and Kafka “ping” message each run.
* Safe to disable in production (set env **AF_DISABLE_PING_AGENT=true**).

Environment
-----------
* **AF_PING_INTERVAL** – seconds between pings (int, default 60).
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Any, Mapping

from backend.agents.base import AgentBase
from backend.agents import register

# ───────────────────────────────────────────────────────────────────────────────
# Logger (inherits unified formatting from AgentBase initialiser)
# ───────────────────────────────────────────────────────────────────────────────
_log = logging.getLogger("alpha_factory.agent.ping")

# ───────────────────────────────────────────────────────────────────────────────
@register
class PingAgent(AgentBase):
    """
    The simplest possible Alpha-Factory agent.

    It demonstrates:
    • automatic scheduling (`CYCLE_SECONDS`)
    • usage of AgentBase helpers (`publish`, Prometheus counters, etc.)
    • graceful degradation when Kafka / Prometheus are unavailable
    """

    NAME = "ping"
    VERSION = "1.0.0"
    CAPABILITIES = ["diagnostics"]
    COMPLIANCE_TAGS: list[str] = []
    SAFE_TO_REMOVE: bool = True  # hint for orchestrator / DevOps

    # ── Scheduling (env-override honoured) ────────────────────────────────────
    _DEFAULT_INTERVAL: int = 60  # seconds
    CYCLE_SECONDS: int = int(
        os.getenv("AF_PING_INTERVAL", str(_DEFAULT_INTERVAL)).strip() or _DEFAULT_INTERVAL
    )

    # ════════════════════════════════════════════════════════════════════════
    # Core lifecycle
    # ════════════════════════════════════════════════════════════════════════
    async def setup(self) -> None:
        _log.info("PingAgent initialised; ping every %ss", self.CYCLE_SECONDS, extra={"agent": self.NAME})

    async def step(self) -> None:
        """
        Called automatically by AgentBase every `CYCLE_SECONDS`.

        Actions:
        1. Log current UTC timestamp.
        2. Update Prometheus gauge `af_ping_last_epoch`.
        3. Publish a small heartbeat message to Kafka (topic `agent.ping`).
        """
        now_iso = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        ctx: Mapping[str, Any] = {"agent": self.NAME}

        # 1. Structured log
        _log.info("Ping ⏰ %s", now_iso, extra=ctx)

        # 2. Prometheus metric (created lazily on first call)
        if not hasattr(self, "_prom_gauge"):
            try:
                from prometheus_client import Gauge  # type: ignore
                self._prom_gauge = Gauge("af_ping_last_epoch", "Last successful ping (unix epoch)")
            except ModuleNotFoundError:  # pragma: no cover
                self._prom_gauge = None  # type: ignore
        if getattr(self, "_prom_gauge", None):
            self._prom_gauge.set(datetime.now(tz=timezone.utc).timestamp())

        # 3. Kafka message (safe even if broker absent – AgentBase handles Producer None)
        await self.publish(
            "agent.ping",
            {"ts": now_iso, "agent": self.NAME, "version": self.VERSION},
        )

    async def teardown(self) -> None:
        _log.info("PingAgent shutting down", extra={"agent": self.NAME})
