"""
Live‑trace WebSocket hub.

Other backend modules can simply do:
    from backend.trace_ws import hub
    await hub.broadcast({...})

Nothing here depends on FastAPI except the `attach()` helper that plugs the
endpoint into the existing FastAPI instance.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Set

from fastapi import WebSocket, WebSocketDisconnect

_LOG = logging.getLogger(__name__)


class _TraceHub:
    """Very small fan‑out hub (no external deps)."""

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        # Protect the client set – several coroutines may touch it
        self._lock = asyncio.Lock()

    # ───────────────────────── connection mgmt ──────────────────────────
    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)
        _LOG.debug("Trace‑WS client connected (%s)", id(ws))

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)
        _LOG.debug("Trace‑WS client disconnected (%s)", id(ws))

    # ───────────────────────── broadcasting ─────────────────────────────
    async def broadcast(self, payload: Any) -> None:
        """
        Fire‑and‑forget broadcast – slow / broken clients are removed silently.
        """
        async with self._lock:
            clients = list(self._clients)

        for ws in clients:
            try:
                await ws.send_json(payload)
            except Exception:
                await self.disconnect(ws)


# single, import‑able instance
hub = _TraceHub()


def attach(app) -> None:
    """
    Mount `/ws/trace` on the provided FastAPI *app*.

    Call once, right after the FastAPI application is created.
    """

    @app.websocket("/ws/trace")
    async def _trace_socket(ws: WebSocket):  # pylint: disable=unused-variable
        await hub.connect(ws)
        try:
            # keep the connection open – we do not expect messages from the client
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            await hub.disconnect(ws)

