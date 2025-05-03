"""
backend.agents.ping_agent
-------------------------
Heartbeat agent – logs a message every minute; safe to remove in production.
"""
from __future__ import annotations
import logging
from datetime import datetime
from backend.agents.base import AgentBase
from backend.agents import register

logger = logging.getLogger("PingAgent")

@register
class PingAgent(AgentBase):
    NAME = "Ping"
    CYCLE_SECONDS = 60

    async def run_cycle(self):
        logger.info("Ping @ %s", datetime.utcnow().isoformat(timespec="seconds"))
