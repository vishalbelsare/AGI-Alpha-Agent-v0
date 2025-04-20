"""
alpha_model.py
---------------
Lightweight price‑action alpha functions.
No ML libs needed; relies on numpy which Torch already brings in.
"""

import numpy as np


def momentum(prices: list[float], lookback: int = 20) -> float:
    """Return percentage momentum over *lookback* periods; >0 is bullish."""
    if len(prices) < lookback + 1:
        return 0.0
    return (prices[-1] - prices[-lookback - 1]) / prices[-lookback - 1]


def sma_crossover(prices: list[float], fast: int = 20, slow: int = 50) -> int:
    """
    Return +1 (bull), ‑1 (bear) or 0 (neutral) if the *fast* SMA crosses the *slow* SMA.
    """
    if len(prices) < slow + 1:
        return 0
    fast_now = np.mean(prices[-fast:])
    slow_now = np.mean(prices[-slow:])
    fast_prev = np.mean(prices[-fast - 1 : -1])
    slow_prev = np.mean(prices[-slow - 1 : -1])

    if fast_prev <= slow_prev and fast_now > slow_now:
        return +1
    if fast_prev >= slow_prev and fast_now < slow_now:
        return -1
    return 0

