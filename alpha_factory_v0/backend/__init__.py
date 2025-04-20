"""
backend/__init__.py
───────────────────
Minimal ASGI entry‑point that works *with or without* FastAPI.

Routes
------
/api/logs   → JSON list with the last 100 log lines.
/ws/trace   → Live trace WebSocket (only when FastAPI is present).
/metrics    → Prometheus metrics (when FastAPI **and** prometheus‑client are present).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

# ────────────────────────── log helpers ────────────────────────────────────
LOG_DIR = Path("/tmp/alphafactory")
LOG_DIR.mkdir(parents=True, exist_ok=True)


def _read_logs(max_lines: int = 100) -> List[str]:
    log_files = sorted(LOG_DIR.glob("*.log"))
    if not log_files:
        return []
    return log_files[-1].read_text(errors="replace").splitlines()[-max_lines:]


# ───────────────────── preferred FastAPI branch ────────────────────────────
try:
    from fastapi import FastAPI

    fast_app = FastAPI(title="Alpha‑Factory API")

    # .—/api/logs───────────────────────────────────────────────────────────.
    @fast_app.get("/api/logs")
    async def api_logs() -> List[str]:
        """Return the most recent (≤100) log lines."""
        return _read_logs()

    # .—/ws/trace (optional)───────────────────────────────────────────────.
    try:
        from .trace_ws import attach as _attach_trace_ws  # noqa: WPS433 (dynamic import)

        _attach_trace_ws(fast_app)
    except Exception:  # pragma: no cover
        # Keep the rest of the API alive if trace‑socket fails to load
        pass

    # .—/metrics (Prometheus)──────────────────────────────────────────────.
    try:
        from .finance_agent import metrics_asgi_app  # noqa: WPS433

        fast_app.mount("/metrics", metrics_asgi_app())
    except Exception:  # pragma: no cover
        # Either prometheus_client missing or FinanceAgent not installed.
        pass

    # Export the ASGI object expected by uvicorn / gunicorn
    app = fast_app

# ──────────────────── fallback: zero‑dependency ASGI ───────────────────────
except ModuleNotFoundError:  # pragma: no cover
    async def app(scope, receive, send):  # noqa: D401, N802 (ASGI signature)
        """Tiny HTTP‑only ASGI app used when FastAPI is not installed."""
        if scope["type"] != "http":  # Only handle plain HTTP requests
            return

        path = scope.get("path", "/")

        if path == "/api/logs":
            body = json.dumps(_read_logs()).encode()
            ctype = b"application/json"
        else:
            # NB: payload must be ASCII for bytes literal → use regular hyphen
            body = b"Alpha-Factory online"
            ctype = b"text/plain"

        headers = [(b"content-type", ctype)]
        await send({"type": "http.response.start", "status": 200, "headers": headers})
        await send({"type": "http.response.body", "body": body})
