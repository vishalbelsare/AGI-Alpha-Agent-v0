"""
Tiny on‑disk trade‑ledger used by FinanceAgent.

It now **streams each fill** to the live Trace‑graph WebSocket
(`/ws/trace`) when FastAPI + `backend.trace_ws` are available.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Dict, List

# Directory is overridable for tests (they patch $ALPHA_DATA_DIR)
DATA_DIR = Path(os.getenv("ALPHA_DATA_DIR", "/tmp/alphafactory"))
DB_PATH = DATA_DIR / "portfolio.jsonl"
DATA_DIR.mkdir(parents=True, exist_ok=True)


class Portfolio:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db_path = db_path
        self._positions: Dict[str, float] = {}

        # ── load existing fills (if any) ──────────────────────────────────
        if db_path.exists():
            for line in db_path.read_text().splitlines():
                rec = json.loads(line)
                self._apply(rec, persist=False)

    # ── public API ────────────────────────────────────────────────────────
    def record_fill(self, symbol: str, qty: float, price: float, side: str) -> None:
        """
        Persist a trade fill and update in‑memory positions.

        Additionally, broadcast the event to the Trace‑graph UI (best‑effort;
        never breaks the main flow if the WebSocket layer is absent).
        """
        fill = {
            "ts": time.time(),
            "symbol": symbol,
            "qty": float(qty),
            "price": float(price),
            "side": side.upper(),
        }

        self._apply(fill)         # update positions + append to disk
        self._broadcast(fill)     # fire‑and‑forget notification

    def position(self, symbol: str) -> float:
        """Current net position (0.0 if the symbol has never traded)."""
        return self._positions.get(symbol, 0.0)

    # ── internal helpers ──────────────────────────────────────────────────
    def _apply(self, fill: dict, *, persist: bool = True) -> None:
        mult = 1 if fill["side"] == "BUY" else -1
        self._positions[fill["symbol"]] = (
            self._positions.get(fill["symbol"], 0.0) + mult * fill["qty"]
        )
        if persist:
            self._append(fill)

    def _append(self, fill: dict) -> None:
        """Append‑only JSONL persistence."""
        with self._db_path.open("a") as fh:
            fh.write(json.dumps(fill) + "\n")

    # ── trace‑graph integration ───────────────────────────────────────────
    def _broadcast(self, fill: dict) -> None:  # pragma: no cover
        """
        Best‑effort broadcast of the fill to any connected Trace‑graph UI.

        Uses a *lazy import* so the portfolio works even if `trace_ws`
        (and its FastAPI dependency) are not installed in production.
        """
        try:
            from backend.trace_ws import hub  # local import avoids cycles
        except ModuleNotFoundError:
            return  # tracing not available

        asyncio.create_task(
            hub.broadcast(
                {
                    "id": f"fill-{int(fill['ts']*1000)}",
                    "label": f"{fill['side']} {fill['qty']} {fill['symbol']} @ {fill['price']}",
                    "data": fill,
                }
            )
        )

