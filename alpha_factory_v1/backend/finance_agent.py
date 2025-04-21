"""
backend/finance_agent.py
~~~~~~~~~~~~~~~~~~~~~~~~
Vectorised, risk‑aware autonomous finance agent.

* Fetches historical price data via MarketDataService (async).
* Computes multi‑factor alpha signals (numpy / pandas vectorised).
* Executes trades through a pluggable **Broker** adapter
  (auto‑selected via ``ALPHA_BROKER`` env‑var if none is provided).
* Continuously monitors risk: 99 % Cornish‑Fisher VaR + max draw‑down.
* Exposes Prometheus metrics at ``/metrics`` (auto‑mounted by backend.__init__).

Environment
-----------
ALPHA_MARKET_PROVIDER   polygon|binance|sim   (picked up by MarketDataService)
ALPHA_BROKER            alpaca|ibkr|sim       (selected in backend.broker)
ALPHA_MAX_VAR_USD       Hard VaR limit (defaults $50 000)
ALPHA_MAX_DD_PCT        Hard max‑draw‑down limit (defaults 20 %)

This module still **runs without numpy/pandas/scipy**; it falls back
to pure‑Python implementations (slower, but keeps tests green).
"""
from __future__ import annotations

import asyncio
import os
import statistics
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# ── Scientific stack (optional) ───────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    from scipy.special import erfcinv
    _HAS_SCI = True
except ModuleNotFoundError:  # pragma: no cover
    np = pd = erfcinv = None  # type: ignore
    _HAS_SCI = False

# ── Prometheus metrics (optional) ─────────────────────────────────────────
try:
    from prometheus_client import Gauge, make_asgi_app  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    Gauge = make_asgi_app = None  # type: ignore

# ── Local imports ─────────────────────────────────────────────────────────
from .market_data import MarketDataService
from .portfolio import Portfolio

# Broker auto‑selector (ALPHA_BROKER env‑var)
from backend.broker import Broker as _DefaultBroker  # noqa: WPS433  (runtime import)

# ───────────────────────────── configuration ──────────────────────────────
_DEFAULT_VAR_LIMIT = float(os.getenv("ALPHA_MAX_VAR_USD", 50_000))
_DEFAULT_DD_LIMIT = float(os.getenv("ALPHA_MAX_DD_PCT", 20.0)) / 100.0  # ratio
_CONF_LEVEL = 0.99  # one‑tail confidence

# -------------------------------------------------------------------------#
#                           MAIN AGENT CLASS                               #
# -------------------------------------------------------------------------#
class FinanceAgent:
    """Production‑grade, risk‑aware finance agent."""

    def __init__(
        self,
        universe: List[str],
        market: MarketDataService,
        broker: Optional[object] = None,
        portfolio: Portfolio | None = None,
        lookback: int = 252,
        var_limit: float = _DEFAULT_VAR_LIMIT,
        dd_limit: float = _DEFAULT_DD_LIMIT,
    ) -> None:
        self.universe = universe
        self.market = market
        # Auto‑instantiate the correct broker implementation if none supplied
        self.broker = broker or _DefaultBroker()
        self.portfolio = portfolio or Portfolio()
        self.lookback = lookback
        self.var_limit = var_limit
        self.dd_limit = dd_limit
        self.current_var: float = 0.0
        self.current_maxdd: float = 0.0
        self._log("FinanceAgent online.")

        # ─── Prometheus metrics ─────────────────────────────────────────
        if Gauge:
            self.pnl_gauge = Gauge("af_pnl_usd", "Un‑realised P&L", ["symbol"])
            self.var_gauge = Gauge("af_var99_usd", "99 % Cornish‑Fisher VaR (USD)")
            self.dd_gauge = Gauge("af_max_drawdown_pct", "Max draw‑down ratio")

    # ------------------------------------------------------------------ #
    #                       PUBLIC ENTRY‑POINT                           #
    # ------------------------------------------------------------------ #
    async def step(self) -> None:
        """One trading‑loop iteration: update data → risk → decide/act."""
        prices = await self._fetch_prices()
        self._update_risk_metrics(prices)

        if self._breached_limits():
            return  # freeze trading until risk normalises

        signals = self._compute_alpha(prices)
        orders = self._position_sizer(signals)
        await self._execute_orders(orders)

    # ------------------------------------------------------------------ #
    #                       RISK MANAGEMENT                              #
    # ------------------------------------------------------------------ #
    def _update_risk_metrics(
        self,
        hist: "pd.DataFrame | Dict[str, List[float]]",
    ) -> None:
        returns = (
            hist.pct_change().dropna()
            if _HAS_SCI and isinstance(hist, pd.DataFrame)
            else _pct_change_py(hist)
        )

        last_prices = (
            hist.iloc[-1].to_dict()
            if _HAS_SCI and isinstance(hist, pd.DataFrame)
            else {k: v[-1] for k, v in hist.items()}
        )
        weights = self._current_weights(last_prices)
        port_ret_series = (
            (returns * weights).sum(axis=1)
            if _HAS_SCI and isinstance(returns, pd.DataFrame)
            else [sum(r_i * weights[s] for s, r_i in zip(self.universe, row)) for row in returns]
        )

        var_usd = _cornish_fisher_var(
            port_ret_series,
            _CONF_LEVEL,
            self.portfolio.value(last_prices),
        )
        max_dd = _max_drawdown(port_ret_series)

        self.current_var = var_usd
        self.current_maxdd = max_dd

        if Gauge:
            self.var_gauge.set(var_usd)
            self.dd_gauge.set(max_dd)

    def _breached_limits(self) -> bool:
        if self.current_var > self.var_limit:
            self._log(
                f"VAR limit breached: {self.current_var:,.0f} USD > "
                f"{self.var_limit:,.0f}. Trading paused."
            )
            return True
        if self.current_maxdd > self.dd_limit:
            self._log(
                f"Draw‑down limit breached: {self.current_maxdd:.1%} > "
                f"{self.dd_limit:.1%}. Trading paused."
            )
            return True
        return False

    # ------------------------------------------------------------------ #
    #                   ALPHA MODEL & POSITION SIZING                    #
    # ------------------------------------------------------------------ #
    def _compute_alpha(
        self,
        hist: "pd.DataFrame | Dict[str, List[float]]",
    ) -> Dict[str, float]:
        """Return z‑score alpha per symbol."""
        if _HAS_SCI and isinstance(hist, pd.DataFrame):
            mom = hist.iloc[-1] / hist.iloc[0] - 1.0                    # momentum
            vol = hist.pct_change().std() * np.sqrt(252)                # realised vol
            idio = hist.pct_change().apply(lambda col: col - col.mean())
            idio_mom = idio.cumsum().iloc[-1]

            df = pd.DataFrame({"mom": mom, "vol": vol, "idio": idio_mom})
            factors = df.apply(lambda s: (s - s.mean()) / s.std())      # z‑score
            return factors.mean(axis=1).to_dict()

        # ---------- pure‑Python fallback --------------------------------
        return _alpha_py(hist)

    def _position_sizer(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Equal‑risk weight long‑short portfolio."""
        longs = {s for s, sc in scores.items() if sc > 0}
        shorts = {s for s, sc in scores.items() if sc < 0}
        if not longs or not shorts:
            return {}
        target_w = 1.0 / (len(longs) + len(shorts))
        return {s: target_w * (1 if s in longs else -1) for s in scores}

    # ------------------------------------------------------------------ #
    #                           EXECUTION                                #
    # ------------------------------------------------------------------ #
    async def _execute_orders(self, targets: Dict[str, float]) -> None:
        for symbol, tgt_w in targets.items():
            pos_qty = self.portfolio.position(symbol)
            price = await self.market.last_price(symbol)
            port_val = self.portfolio.value(
                await self.market.last_prices(self.universe)
            )
            tgt_dollar = tgt_w * port_val
            tgt_qty = tgt_dollar / price
            delta = tgt_qty - pos_qty
            if abs(delta) * price < 1:  # ignore dust
                continue
            side = "BUY" if delta > 0 else "SELL"
            await self.broker.submit_order(symbol, abs(delta), side)
            self.portfolio.record_fill(symbol, abs(delta), price, side)
            if Gauge:
                self.pnl_gauge.labels(symbol=symbol).set(
                    self.portfolio.unrealised_pnl(symbol, price)
                )

    # ------------------------------------------------------------------ #
    #                         DATA HELPERS                               #
    # ------------------------------------------------------------------ #
    async def _fetch_prices(self):
        end = datetime.utcnow()
        start = end - timedelta(days=self.lookback)
        return await self.market.history(self.universe, start, end)

    def _current_weights(self, last_prices) -> Dict[str, float]:
        port_val = self.portfolio.value(last_prices)
        if port_val == 0:
            return {s: 0.0 for s in self.universe}
        return {
            s: (self.portfolio.position(s) * last_prices[s]) / port_val
            for s in self.universe
        }

    # ------------------------------------------------------------------ #
    #                               UTILS                                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _log(msg: str) -> None:
        print(f"[FinanceAgent] {datetime.utcnow().isoformat()} – {msg}")

# -------------------------------------------------------------------------#
#                     HELPER FUNCTIONS (NumPy fallbacks)                   #
# -------------------------------------------------------------------------#
def _pct_change_py(data: Dict[str, List[float]]):
    """Return list‑of‑lists of percentage changes (pure‑python)."""
    out = {
        s: [(p2 - p1) / p1 for p1, p2 in zip(prices, prices[1:])]
        for s, prices in data.items()
    }
    # transpose into rows
    return list(zip(*[out[s] for s in data]))

def _alpha_py(hist: Dict[str, List[float]]) -> Dict[str, float]:
    alpha: Dict[str, float] = {}
    # rough z‑score normalisation
    for s, prices in hist.items():
        mom = prices[-1] / prices[0] - 1
        daily = [(p2 - p1) / p1 for p1, p2 in zip(prices, prices[1:])]
        vol = statistics.pstdev(daily) * (len(daily) ** 0.5)
        idio = sum(daily) - len(daily) * statistics.mean(daily)
        alpha[s] = (mom - vol + idio) / 3.0
    mu = statistics.mean(alpha.values())
    sigma = statistics.pstdev(alpha.values()) or 1.0
    return {s: (v - mu) / sigma for s, v in alpha.items()}

def _cornish_fisher_var(returns, conf: float, port_val: float) -> float:
    """Cornish‑Fisher VaR; falls back to parametric if SciPy unavailable."""
    if _HAS_SCI and isinstance(returns, pd.Series):
        z = abs(np.sqrt(2) * erfcinv(2 * (1 - conf)))
        s = returns.skew()
        k = returns.kurtosis()
        z_cf = (
            z
            + (1 / 6) * (z**2 - 1) * s
            + (1 / 24) * (z**3 - 3 * z) * k
            - (1 / 36) * (2 * z**3 - 5 * z) * (s**2)
        )
        sigma = returns.std()
        mu = returns.mean()
        return max(0.0, abs(port_val * (mu + z_cf * sigma)))

    vals = list(returns)
    mu, sigma = statistics.mean(vals), statistics.pstdev(vals)
    try:
        from scipy.stats import norm  # type: ignore
        z = norm.ppf(conf)
    except ModuleNotFoundError:
        z = 2.326  # ≈ 99 %
    return abs(port_val * (mu + z * sigma))

def _max_drawdown(returns) -> float:
    if _HAS_SCI and isinstance(returns, pd.Series):
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return abs(dd.min())

    cum, peak, max_dd = 1.0, 1.0, 0.0
    for r in returns:
        cum *= 1 + r
        peak = max(peak, cum)
        max_dd = min(max_dd, (cum - peak) / peak)
    return abs(max_dd)

# -------------------------------------------------------------------------#
#                       METRICS ASGI MOUNT HOOK                            #
# -------------------------------------------------------------------------#
def metrics_asgi_app():
    """Return an ASGI app exposing /metrics (or 404 if prometheus unavailable)."""
    if make_asgi_app:
        return make_asgi_app()  # default registry
    async def _dummy(scope, receive, send):  # pragma: no cover
        await send({"type": "http.response.start", "status": 404, "headers": []})
        await send({"type": "http.response.body", "body": b""})
    return _dummy
