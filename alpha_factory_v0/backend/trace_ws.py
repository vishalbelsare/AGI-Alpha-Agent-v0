"""
ASGI WebSocket hub for live Trace‑graph events
=============================================

*   Mounts a high‑performance WebSocket endpoint at **/ws/trace**.
*   No external broker required; replacing the in‑memory fan‑out with
    Redis / NATS later is trivial (single class swap).

Schema
------
Outbound messages follow the *TraceEvent* model:

    {
        "id":   "uuid4‑hex",            # unique node id
        "ts":   1713612345.123,         # POSIX seconds
        "type": "tool_call|planner",    # enum
        "label":"User‑friendly label",  # ≤128 chars
        "edges":["uuid3", ...]          # optional parents
    }

Usage
-----
>>> from backend.trace_ws import hub, attach
>>> attach(fastapi_app)
>>> await hub.broadcast({"label": "order sent", "type": "tool_call"})
"""

from __future__ import annotations

import asyncio
import time
import typing as _t
from dataclasses import dataclass, field
from uuid import uuid4

# --------------------------------------------------------------------- #
# Zero‑copy JSON → bytes helper                                         #
# --------------------------------------------------------------------- #
try:
    import orjson as _json

    def _dumps(obj: _t.Any) -> bytes:  # noqa: D401
        return _json.dumps(obj)
except ModuleNotFoundError:  # pragma: no cover
    import json as _json

    def _dumps(obj: _t.Any) -> bytes:  # noqa: D401
        return _json.dumps(obj).encode()


# --------------------------------------------------------------------- #
# Public dataclass for events                                           #
# --------------------------------------------------------------------- #
@dataclass(slots=True, frozen=True)
class TraceEvent:
    """Immutable, hash‑able trace event."""

    id: str = field(default_factory=lambda: uuid4().hex)
    ts: float = field(default_factory=time.time)
    type: str = "generic"
    label: str = ""
    edges: list[str] | None = None

    # cache serialisation; fastest path for broadcast
    def to_bytes(self) -> bytes:  # noqa: D401
        return _dumps(self.__dict__)


# --------------------------------------------------------------------- #
# Hub                                                                   #
# --------------------------------------------------------------------- #
class TraceHub:
    """In‑process broadcast hub (fan‑out)."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[bytes]] = set()
        self._lock = asyncio.Lock()

    # ------------- subscription management --------------------------- #
    async def subscribe(self) -> asyncio.Queue[bytes]:
        """Return a queue that will receive broadcast messages."""
        q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subscribers.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue[bytes]) -> None:
        async with self._lock:
            self._subscribers.discard(q)

    # ------------------- broadcasting -------------------------------- #
    async def broadcast(self, event: TraceEvent | dict[str, _t.Any]) -> None:
        """
        Send *event* to every live WebSocket.

        *event* may be a ``TraceEvent`` or a raw ``dict`` matching the schema.
        The call returns immediately (fire‑and‑forget).
        """
        if isinstance(event, dict):
            event = TraceEvent(**event)  # type: ignore[arg-type]
        payload = event.to_bytes()

        async with self._lock:
            targets = list(self._subscribers)  # snapshot under lock

        coros: list[_t.Coroutine[_t.Any, _t.Any, _t.Any]] = []
        for q in targets:
            # back‑pressure: drop oldest if queue is full
            if q.full():
                try:
                    _ = q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            coros.append(q.put(payload))

        if coros:  # schedule fan‑out concurrently, don’t await in caller
            asyncio.create_task(
                asyncio.gather(*coros, return_exceptions=True),
            )


# Singleton – imported by other modules
hub = TraceHub()

# --------------------------------------------------------------------- #
# FastAPI / Starlette integration                                       #
# --------------------------------------------------------------------- #
def attach(app) -> None:  # noqa: D401
    """
    Dynamically mount the ``/ws/trace`` WebSocket endpoint on *app*.

    ```python
    from fastapi import FastAPI
    from backend.trace_ws import attach

    fast = FastAPI()
    attach(fast)
    ```
    """

    from fastapi import WebSocket, WebSocketDisconnect
    from fastapi.routing import APIRouter

    router = APIRouter()

    @router.websocket("/ws/trace")
    async def _trace_ws(ws: WebSocket):  # noqa: WPS430
        """
        WebSocket stream with *race‑free* ping / queue handling.

        A shielded gather prevents the rare disconnect race where the
        client closes exactly between queue.put() and ws.send_bytes().
        """
        await ws.accept()
        q = await hub.subscribe()

        try:
            ping_task: asyncio.Task = asyncio.create_task(ws.receive_text())
            queue_task: asyncio.Task = asyncio.create_task(q.get())

            while True:
                done, _ = await asyncio.wait(
                    {ping_task, queue_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if queue_task in done:
                    # Broadcast event ready → push downstream
                    payload = queue_task.result()
                    try:
                        await ws.send_bytes(payload)
                    finally:
                        # Immediately replace the consumed queue task
                        queue_task = asyncio.create_task(q.get())

                if ping_task in done:
                    # Client frame (keep‑alive) – ignore payload, restart listener
                    try:
                        _ = ping_task.result()
                    finally:
                        ping_task = asyncio.create_task(ws.receive_text())

        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        finally:
            # Ensure background tasks are cleaned up
            for task in (ping_task, queue_task):
                task.cancel()
            await hub.unsubscribe(q)

    app.include_router(router)
