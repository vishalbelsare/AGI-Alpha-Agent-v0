"""
backend.agents.base
-------------------
Abstract base-class that **all** agents must inherit.
"""

from __future__ import annotations
import asyncio
from typing import Any, Dict


class AgentBase:
    """Minimal contract understood by the Orchestrator."""
    #: human-readable unique name – override in subclasses
    NAME: str = "Base"
    #: seconds to sleep after each `run_cycle` if no SCHED_SPEC provided
    CYCLE_SECONDS: int = 60
    #: optional cron / RRULE schedule string
    SCHED_SPEC: str | None = None

    # orchestrator sets this when instantiating (runtime helpers, mem, bus…)
    orchestrator: Any = None

    # --------------------------------------------------------------------- #
    async def run_cycle(self) -> None:
        """Override with your agent's core logic."""
        raise NotImplementedError

    # --------------------------------------------------------------------- #
    # Convenience helpers available in every agent without extra imports
    # --------------------------------------------------------------------- #
    @property
    def mem_vector(self):
        return self.orchestrator.mem_vector  # type: ignore[attr-defined]

    @property
    def mem_graph(self):
        return self.orchestrator.mem_graph  # type: ignore[attr-defined]

    async def publish(self, topic: str, msg: Dict[str, Any]):
        await self.orchestrator.publish(topic, msg)  # type: ignore[attr-defined]

    async def subscribe(self, topic: str):
        return await self.orchestrator.subscribe(topic)  # type: ignore[attr-defined]
