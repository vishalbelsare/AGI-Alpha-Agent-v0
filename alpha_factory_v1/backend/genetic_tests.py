
"""Genetic test functions (fitness proxies) for the AI‑GA demo.

Each fitness function must accept a candidate *gene dict* and return a
**scalar float** – higher is better.  They are deliberately lightweight so
that the demo can run on a laptop without network access or long runtimes.

You are encouraged to replace these with domain‑specific benchmarks
(financial P&L, manufacturing throughput, policy‑compliance score, etc.).
"""

import math
import random
from typing import Dict


def _score_temperature(t: float) -> float:
    """A synthetic utility curve preferring ~0.7."""
    # upside‑down parabola centred at 0.7
    return max(0.0, 1.0 - ((t - 0.7) ** 2) / 0.25)


def _score_top_p(p: float) -> float:
    """Prefers values above 0.85 but penalises extremes."""
    return 1.0 - abs(p - 0.9) * 2.0


def _score_max_tokens(n: int) -> float:
    """Sweet‑spot around 128 tokens."""
    return 1.0 - abs(n - 128) / 192.0  # 0 when |n-128| == 192


def toy_fitness(genes: Dict[str, float]) -> float:
    """Deterministic, fast, CI‑friendly fitness function.

    Scores are between 0 and 3, higher is better.
    """
    t_score = _score_temperature(float(genes["temperature"]))
    p_score = _score_top_p(float(genes["top_p"]))
    tok_score = _score_max_tokens(int(genes["max_tokens"]))
    return t_score + p_score + tok_score


def stochastic_fitness(genes: Dict[str, float], noise: float = 0.05) -> float:
    """Same as *toy_fitness* but with Gaussian noise added.

    Useful for testing the GA's robustness under noisy objectives.
    """
    base = toy_fitness(genes)
    return base + random.gauss(0.0, noise)


__all__ = ["toy_fitness", "stochastic_fitness"]
