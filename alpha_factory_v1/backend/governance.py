"""
Central safety & policy enforcement.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List

from better_profanity import profanity

# ── legacy re‑export (tests import Memory from this module) ────────────────
from .memory import Memory  # noqa: F401

# ── hard‑coded disallowed content ------------------------------------------
# If a string matches ANY of these patterns → block.
_DANGER_PATTERNS: List[str] = [
    # Explosives
    r"\bbomb recipe\b",
    r"\bhow\s+to\s+(make|build).*?(bomb|explosive)\b",
    # Extremism
    r"\b(extremist|terrorist)\b",
    r"\b(extremist|terrorist).*?propaganda\b",
    r"\bpropaganda.*?(extremist|terrorist)\b",
    # Illicit finance / money‑laundering
    r"\blaunder\b",                          # <-- broader catch‑all
    r"\bmoney laundering\b",
    # Violence
    r"\bkill\b",
]
_DANGER_RE = re.compile("|".join(_DANGER_PATTERNS), re.IGNORECASE)


class Governance:
    """
    • `moderate(text) -> bool`  – content gate. True  = safe.
    • `vet_plans(agent, plans)` – risk gate for numeric limits, etc.
    """

    def __init__(self, memory: Memory) -> None:
        self.log = logging.getLogger("Governance")
        self.memory = memory
        self.trade_limit = float(os.getenv("ALPHA_TRADE_LIMIT", "1000000"))
        profanity.load_censor_words()

    # ── content guard‑rail --------------------------------------------------
    def moderate(self, text: str) -> bool:
        """Return **True** iff *text* is safe to use / display."""
        # 1️⃣ explicit disallowed patterns
        if _DANGER_RE.search(text):
            self.log.warning("Blocked disallowed content: %s", text)
            return False
        # 2️⃣ generic profanity
        return not profanity.contains_profanity(text)

    # ── action / numeric guard‑rail ----------------------------------------
    def vet_plans(self, agent, plans):
        """Strip plans that violate numeric policy (e.g. notional > limit)."""
        vetted = []
        for p in plans:
            if p.get("type") == "trade" and p.get("notional", 0) > self.trade_limit:
                self.log.warning("Blocked high‑notional trade: %s", p)
                self.memory.write(agent.name, "blocked", p)
                continue
            vetted.append(p)
        return vetted


__all__ = ["Governance", "Memory"]

