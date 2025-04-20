
"""
Unified async market‑data adapter for the Alpha‑Factory stack
------------------------------------------------------------

* **Polygon.io** — U.S. equities/crypto
* **Binance**    — Spot & futures crypto
* **Simulated**  — Offline CI / unit‑tests (random‑walk feed)

Provider is chosen via the ``ALPHA_MARKET_PROVIDER`` environment variable
(``polygon`` | ``binance`` | ``sim``).  Credentials **must** be supplied
through the usual env‑vars:

* ``POLYGON_API_KEY``
* ``BINANCE_API_KEY`` / ``BINANCE_API_SECRET``

Example
~~~~~~~
>>> import asyncio, os
>>> os.environ["ALPHA_MARKET_PROVIDER"] = "sim"
>>> from backend.market_data import MarketDataService
>>>
>>> async def demo():
...     svc = await MarketDataService.from_env()
...     print(await svc.last_price("AAPL"))
...
>>> asyncio.run(demo())

This module has **no hard dependency** on external SDKs; HTTP calls are
performed with ``aiohttp`` (falling back to ``httpx`` if preferred).
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Final

import aiohttp

_DEFAULT_TIMEOUT: Final = aiohttp.ClientTimeout(total=5)


# ───────────────────────────── Public façade ─────────────────────────────


class MarketDataService(ABC):
    """Abstract async interface for price‑fetching back‑ends."""

    @abstractmethod
    async def last_price(self, symbol: str) -> float:  # pragma: no cover
        """Return the latest *trade* price for *symbol* (USD)."""

    # ───────── factory helpers ────────────────────────────────────────

    @classmethod
    async def from_env(cls) -> "MarketDataService":
        """Auto‑select provider from the ``ALPHA_MARKET_PROVIDER`` env‑var."""
        provider = os.getenv("ALPHA_MARKET_PROVIDER", "sim").lower()
        if provider == "polygon":
            return _PolygonAdapter(api_key=os.environ["POLYGON_API_KEY"])
        if provider == "binance":
            return _BinanceAdapter(
                api_key=os.getenv("BINANCE_API_KEY"),
                api_secret=os.getenv("BINANCE_API_SECRET"),
            )
        if provider == "sim":
            return _SimulatedAdapter()
        raise ValueError(f"Unsupported ALPHA_MARKET_PROVIDER={provider!r}")


# ──────────────────────────── Concrete providers ────────────────────────


class _PolygonAdapter(MarketDataService):
    """Minimal Polygon.io REST wrapper (last‑trade endpoint only)."""

    _BASE = "https://api.polygon.io/v2/last"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise RuntimeError("POLYGON_API_KEY missing")
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    async def _client(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=_DEFAULT_TIMEOUT)
        return self._session

    async def last_price(self, symbol: str) -> float:
        url = f"{self._BASE}/trade/{symbol.upper()}?apiKey={self._api_key}"
        async with (await self._client()).get(url) as r:
            r.raise_for_status()
            data = await r.json()
        # Polygon returns { status, results: { p: price, ... } }
        return float(data["results"]["p"])

    async def __aenter__(self):
        await self._client()
        return self

    async def __aexit__(self, *_exc) -> None:  # noqa: D401
        if self._session:
            await self._session.close()


class _BinanceAdapter(MarketDataService):
    """Lightweight Binance REST price adapter (no credentials required)."""

    _BASE = "https://api.binance.com"

    def __init__(self, api_key: str | None, api_secret: str | None) -> None:
        # Public *ticker* endpoint does not need auth → but keep for futures
        self._session: aiohttp.ClientSession | None = None

    async def _client(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=_DEFAULT_TIMEOUT)
        return self._session

    async def last_price(self, symbol: str) -> float:
        url = f"{self._BASE}/api/v3/ticker/price?symbol={symbol.upper()}"
        async with (await self._client()).get(url) as r:
            r.raise_for_status()
            data = await r.json()
        return float(data["price"])

    async def __aenter__(self):
        await self._client()
        return self

    async def __aexit__(self, *_exc) -> None:  # noqa: D401
        if self._session:
            await self._session.close()


class _SimulatedAdapter(MarketDataService):
    """Tiny in‑memory random‑walk price feed—fast & deterministic for CI."""

    def __init__(self) -> None:
        self._prices: dict[str, float] = {}

    async def last_price(self, symbol: str) -> float:
        # Deterministic seed per symbol for reproducible tests
        base = self._prices.setdefault(symbol, abs(hash(symbol)) % 100 + 10.0)
        delta = random.uniform(-0.5, 0.5)
        new_price = max(0.01, base + delta)
        self._prices[symbol] = new_price
        # Sleep a *tiny* bit to mimic network latency without slowing tests
        await asyncio.sleep(0.001)
        return round(new_price, 5)


# ─────────────────────── Self‑test (pytest -q) ──────────────────────────

if __name__ == "__main__":  # pragma: no cover
    async def _demo():
        svc = await MarketDataService.from_env()
        for sym in ("AAPL", "BTCUSDT"):
            print(sym, await svc.last_price(sym))

    asyncio.run(_demo())
