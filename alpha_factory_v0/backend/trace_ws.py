"""
backend/trace_ws.py
───────────────────
Live trace‑graph WebSocket hub.

▪ Provides a single FastAPI‑driven endpoint `/ws/trace`.
▪ Keeps a registry of active WebSocket clients (thread‑safe).
▪ `hub.broadcast({...})` serialises the payload and pushes it to *all*
  connected front‑ends.  Any JSON‑serialisable dict is accepted, but the
  recommended schema is:

    {
        "id": "unique‑node‑id",
        "label": "Human‑friendly label",
        "edges": ["parent‑id", ...]      # optional list of upstream nodes
    }

The module is imported (and `attach(app)` called) by `backend/__init__.py`
*only* when FastAPI is available, so the rest of the codebase still runs
in the pure‑ASGI fallback path.

The design follows OpenAI‑Agents + A2A best‑practice:

* Messages are opaque to the hub; validation is pushed to the front‑end
  and/or emitting agent.  This keeps the hub fast and future‑proof.
* All operations are asyncio‑aware and safe for hundreds of concurrent
  sockets.
* Works with or without an `OPENAI_API_KEY`; no external dependencies
  other than FastAPI/Starlette (already in requirements.txt).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List, Set

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect

__all__ = ["hub", "attach"]


# ────────────────────────── internal data model ───────────────────────────


def _now() -> float:  # small helper for a monotonic-ish timestamp
    return time.time()


# ───────────────────────────── broadcast hub ──────────────────────────────
class _TraceHub:
    """
    Singleton hub responsible for:

    • registering / unregistering WebSocket clients
    • broadcasting JSON payloads to *all* connected clients
    """

    def __init__(self) -> None:
        self._clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    # ‑‑ connection management ‑‑
    async def _register(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._clients.add(ws)

    async def _unregister(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(ws)

    # ‑‑ public API ‑‑
    async def broadcast(self, payload: Dict[str, Any]) -> None:
        """
        Fan‑out *one* JSON payload to every live client.

        The method is resilient: bad or disconnected sockets are dropped
        quietly so that one misbehaving browser cannot DOS the hub.
        """
        # enrich with server‑side timestamp if missing
        payload.setdefault("ts", _now())
        message = json.dumps(payload, ensure_ascii=False).encode()

        # take a snapshot of clients under the lock, then release to keep the
        # critical section short while we await writes.
        async with self._lock:
            clients: List[WebSocket] = list(self._clients)

        for ws in clients:
            try:
                await ws.send_bytes(message)
            except Exception:  # pragma: no cover  – network edge‑cases
                await self._unregister(ws)


# exposed singleton
hub = _TraceHub()


# ────────────────────────────── FastAPI glue ──────────────────────────────
def attach(app: FastAPI, *, route: str = "/ws/trace") -> None:  # noqa: D401
    """
    Plug the trace hub into an existing FastAPI instance.

    Usage (already done in `backend/__init__.py`):

        from backend.trace_ws import attach
        fast_app = FastAPI()
        attach(fast_app)

    Nothing is returned; the function mutates the passed‑in `app` by adding
    a single WebSocket route.
    """
    router = APIRouter()

    @router.websocket(route)
    async def _trace_endpoint(ws: WebSocket) -> None:  # noqa: WPS430
        await hub._register(ws)  # private but OK inside same module
        try:
            while True:
                # We *ignore* any inbound data; pings keep the connection open.
                await ws.receive_text()
        except WebSocketDisconnect:
            await hub._unregister(ws)

    app.include_router(router)
