"""Planning agent."""

from __future__ import annotations

from .base_agent import BaseAgent
from ..utils import messaging
from ..utils.logging import Ledger


class PlanningAgent(BaseAgent):
    """Generate research plans for downstream agents."""

    def __init__(self, bus: messaging.A2ABus, ledger: "Ledger") -> None:
        super().__init__("planning", bus, ledger)

    async def run_cycle(self) -> None:
        """Emit a high level research request."""
        plan = "collect baseline metrics"
        if self.oai_ctx and not self.bus.settings.offline:
            try:  # pragma: no cover - SDK optional
                plan = await self.oai_ctx.run(prompt="plan research task")
            except Exception:
                plan = "collect baseline metrics"
        await self.emit("research", {"plan": plan})

    async def handle(self, env: messaging.Envelope) -> None:
        """Log incoming feedback for future planning."""
        self.ledger.log(env)
