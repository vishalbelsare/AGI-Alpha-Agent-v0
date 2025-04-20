"""
FinanceAgent – autonomous trading agent
=======================================

Implements a simple observe → think → act loop that trades AAPL using a
momentum + SMA‑crossover signal.  The module re‑exports helper classes so the
tests can access them directly (e.g. `fa.ModelProvider()`).
"""

import logging
import random   # retained for potential stochastic logic

from .agent_base import AgentBase
from . import data_feed, broker, alpha_model, portfolio, risk

from .model_provider import ModelProvider
from .memory import Memory                    # ← updated path
from .governance import Governance

log = logging.getLogger(__name__)


class FinanceAgent(AgentBase):
    """A very small illustrative trading agent."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.prices: list[float] = []
        self.book = portfolio.Portfolio()

    # ── observe ────────────────────────────────────────────
    def observe(self):
        price = data_feed.last_price("AAPL")
        self.prices.append(price)
        if len(self.prices) > 200:
            self.prices.pop(0)

        self.memory.write(self.name, "observation", {"price": price})
        return [{"price": price}]

    # ── think ──────────────────────────────────────────────
    def think(self, obs):
        signals = {
            "mom":   alpha_model.momentum(self.prices),
            "xover": alpha_model.sma_crossover(self.prices),
        }

        if signals["xover"] == +1 and signals["mom"] > 0:
            side = "BUY"
        elif signals["xover"] == -1 and signals["mom"] < 0:
            side = "SELL"
        else:
            return []          # no clear signal → no action

        qty = risk.position_size(obs[-1]["price"])
        idea = {
            "type":     "trade",
            "symbol":   "AAPL",
            "side":     side,
            "qty":      qty,
            "notional": qty * obs[-1]["price"],
            "reason":   signals,
        }
        self.memory.write(self.name, "idea", idea)
        return [idea]

    # ── act ────────────────────────────────────────────────
    def act(self, tasks):
        for t in tasks:
            fill = broker.place_order(
                t["symbol"], t["qty"], t["side"], t["notional"] / t["qty"]
            )
            self.book.record_fill(t["symbol"], t["qty"], fill["price"], t["side"])
            self.memory.write(self.name, "action", {"trade": t, "fill": fill})
            self.log.info("Executed %s", fill)


# ------------------------------------------------------------------ re‑exports
Memory = Memory
ModelProvider = ModelProvider
Governance = Governance

__all__ = [
    "FinanceAgent",
    "risk",
    "Memory",
    "ModelProvider",
    "Governance",
]

