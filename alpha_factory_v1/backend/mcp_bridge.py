"""
alpha_factory_v1.backend.mcp_bridge
===================================

Ultra-thin helper around Anthropic’s *Model-Context-Protocol* (MCP).

Current capability → **store** chat messages for long-term memory / retrieval.
Future work → fetch condensed context back into prompts (§3.2 of MCP spec).

Enable by exporting, e.g.:

    MCP_ENDPOINT="http://mcp.service.local:8980/v1"

If the variable is unset this module becomes a silent no-op.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, List

import httpx

_ENDPOINT = os.getenv("MCP_ENDPOINT")          # http://host:port/v1
_TIMEOUT = float(os.getenv("MCP_TIMEOUT_SEC", 10))


async def store(messages: List[Dict[str, str]]) -> None:
    """
    Asynchronously push *raw* chat messages to an MCP server.

    The call is fire-and-forget and **never** raises – MCP is non-critical.
    """
    if not _ENDPOINT:
        return

    payload = {"messages": messages, "timestamp": time.time()}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            await client.post(f"{_ENDPOINT}/context", json=payload)
    except Exception:  # noqa: BLE001
        # Best-effort only → log at debug level if the caller configured logging
        import logging

        logging.getLogger("alpha_factory.mcp").debug(
            "MCP push failed – continuing without persistence", exc_info=True
        )


def store_sync(messages: List[Dict[str, str]]) -> None:
    """
    Synchronous convenience wrapper for rare call-sites that are outside
    an event-loop (e.g. CLI utilities or unit tests).
    """
    asyncio.run(store(messages))
