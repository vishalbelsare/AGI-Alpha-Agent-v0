"""
alpha_factory_v1.backend.mcp_bridge
────────────────────────────────────
Tiny abstraction over Anthropic's Model-Context-Protocol.

Right now we only **store** chat messages so the MCP server can build a
long-term memory / retrieval index. Later you can extend this file to also
*retrieve* compressed context (per the MCP white-paper §3.2).

Set env-var ``MCP_ENDPOINT`` to activate, e.g.:

    MCP_ENDPOINT="http://mcp.gpu-cluster.local:8980/v1" \
        uvicorn alpha_factory_v1.backend.orchestrator:app

If unset, all functions become cheap no-ops.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Dict, List

import httpx

_MCP_ENDPOINT = os.getenv("MCP_ENDPOINT")
_TIMEOUT = float(os.getenv("MCP_TIMEOUT_SEC", 10))


async def store(messages: List[Dict[str, str]]) -> None:
    """Fire-and-forget store of raw chat messages to MCP."""
    if not _MCP_ENDPOINT:
        return
    payload = {"messages": messages, "timestamp": time.time()}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            await client.post(f"{_MCP_ENDPOINT}/context", json=payload)
    except Exception:  # noqa: BLE001
        # Silently ignore – MCP is best-effort
        pass


# Convenience sync wrapper (rarely needed)
def store_sync(messages: List[Dict[str, str]]) -> None:
    asyncio.run(store(messages))
