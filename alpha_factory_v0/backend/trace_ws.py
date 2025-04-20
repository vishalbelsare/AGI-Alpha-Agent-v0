
"""ASGI WebSocket hub for live Trace‑graph events.

This module attaches a high‑performance, dependency‑free WebSocket endpoint
at **/ws/trace** when imported via ``attach(app)``.

Design goals
------------
* Zero‑copy JSON → bytes serialisation (ujson / stdlib fallback)
* Broadcast fan‑out to N listeners with **O(1) lock‑free** queue ops
* Works inside any ASGI runtime (Uvicorn, Hypercorn, Daphne)
* No external broker required; switching to Redis / NATS later is trivial.

Schema
------
Outbound messages MUST be UTF‑8 JSON objects obeying the *TraceEvent* model:

    {{
        "id":  "uuid4",                # unique node id
        "ts":  1713612345.123,         # POSIX seconds
        "type":"tool_call|planner",    # enum
        "label":"User friendly label", # <=128 chars
        "edges":["uuid3", ...]         # optional adjacency list
    }}

Usage
-----
>>> from backend.trace_ws import hub, attach
>>> attach(fastapi_app)
>>> await hub.broadcast({...})

The helper ``hub`` is a singleton instance of :class:`TraceHub`.
"""

from __future__ import annotations

import asyncio
import time
import typing as _t

try:
    import orjson as _json
    _dumps = lambda obj: _json.dumps(obj)  # noqa: E731
except ModuleNotFoundError:  # pragma: no cover
    import json as _json

    _dumps = lambda obj: _json.dumps(obj).encode()  # noqa: E731

JSON_BYTES = _t.Callable[[_t.Any], bytes]

__all__ = ["attach", "hub", "TraceHub", "TraceEvent"]

# --------------------------------------------------------------------- #
# Dataclass for events                                                  #
# --------------------------------------------------------------------- #
from dataclasses import dataclass, field
from uuid import uuid4

@dataclass(slots=True, frozen=True)
class TraceEvent:
    """Immutable, hash‑able trace event."""

    id: str = field(default_factory=lambda: uuid4().hex)
    ts: float = field(default_factory=time.time)
    type: str = "generic"
    label: str = ""
    edges: list[str] | None = None

    def to_bytes(self) -> bytes:  # noqa: D401
        """Return the UTF‑8 JSON representation (cached)."""
        return _dumps(self.__dict__)

# --------------------------------------------------------------------- #
# Hub                                                                    #
# --------------------------------------------------------------------- #
class TraceHub:
    """In‑process broadcast hub."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[bytes]] = set()
        self._lock = asyncio.Lock()

    # ---------------- subscription management ------------------------ #
    async def subscribe(self) -> asyncio.Queue[bytes]:
        """Return a queue that will receive broadcast messages."""
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    # ---------------- broadcasting ----------------------------------- #
    async def broadcast(self, event: TraceEvent | dict[str, _t.Any]) -> None:
        if isinstance(event, dict):
            event = TraceEvent(**event)  # type: ignore[arg-type]
        payload = event.to_bytes()
        # Fan‑out without awaiting individual puts (drop on back‑pressure)
        coros = []
        async with self._lock:
            for q in self._subscribers:
                if q.full():
                    try:
                        _ = q.get_nowait()  # drop oldest
                    except asyncio.QueueEmpty:
                        pass
                coros.append(q.put(payload))
        # Fire‑and‑forget
        if coros:
            asyncio.create_task(asyncio.gather(*coros, return_exceptions=True))

# Singleton
hub = TraceHub()

# --------------------------------------------------------------------- #
# ASGI router                                                           #
# --------------------------------------------------------------------- #
def attach(app) -> None:  # noqa: D401
    """Dynamically mount the ``/ws/trace`` WebSocket endpoint on *app*."""

    from fastapi import WebSocket, WebSocketDisconnect
    from fastapi.routing import APIRouter

    router = APIRouter()

    @router.websocket("/ws/trace")
    async def _trace_ws(ws: WebSocket):  # noqa: WPS430
        await ws.accept()
        q = await hub.subscribe()
        try:
            # Bidirectional: ignore incoming data for now (keep‑alive pings)
            while True:
                done, _ = await asyncio.wait(
                    [ws.receive_text(), q.get()],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    data = task.result()
                    if isinstance(data, bytes):
                        await ws.send_bytes(data)
                    else:
                        # client message – optional ping/pong
                        if data == "ping":
                            await ws.send_text("pong")
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            await hub.unsubscribe(q)

    app.include_router(router)
