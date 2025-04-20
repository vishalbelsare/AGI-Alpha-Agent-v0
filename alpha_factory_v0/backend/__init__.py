"""
Minimal ASGI entry‑point that works *with or without* FastAPI.

• /api/logs  → JSON list with the last 100 log lines
• /ws/trace  → live trace WebSocket (only when FastAPI is available)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

# ───────────────────────── log helpers ────────────────────────────────────
LOG_DIR = Path("/tmp/alphafactory")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _read_logs(max_lines: int = 100) -> List[str]:
    log_files = sorted(LOG_DIR.glob("*.log"))
    if not log_files:
        return []
    return log_files[-1].read_text(errors="replace").splitlines()[-max_lines:]


# ────────────────────── preferred FastAPI branch ──────────────────────────
try:
    from fastapi import FastAPI

    fast_app = FastAPI()

    @fast_app.get("/api/logs")
    async def api_logs() -> List[str]:
        """Return recent log lines (max 100)."""
        return _read_logs()

    # live Trace‑graph socket – optional, but nice to have in dev
    try:
        from .trace_ws import attach as _attach_trace_ws  # noqa: WPS433

        _attach_trace_ws(fast_app)
    except Exception:  # pragma: no cover
        # If anything goes wrong, leave the rest of the API alive.
        pass

    app = fast_app  # what uvicorn / gunicorn import


# ─────────────────── fallback: zero‑dependency ASGI ───────────────────────
except ModuleNotFoundError:  # pragma: no cover
    async def app(scope, receive, send):  # noqa: D401, N802
        """Tiny HTTP‑only ASGI app used when FastAPI is absent."""
        if scope["type"] != "http":
            return

        if scope.get("path") == "/api/logs":
            body = json.dumps(_read_logs()).encode()
            ctype = b"application/json"
        else:
            body = b"Alpha-Factory online"  # must be pure ASCII
            ctype = b"text/plain"

        headers = [(b"content-type", ctype)]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": body})

