"""
backend/governance.py
────────────────────────────────────────────────────────────────────────────
Central safety & policy‑enforcement layer for α‑Factory.

Adds to the original implementation:
• OpenTelemetry span around every governed decision
• W3C Verifiable Credential (VC) snapshot signed and pushed to IPFS
• Graceful degradation if `opentelemetry` or `vc` libraries are absent
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from typing import List, Dict, Any

# ─── optional telemetry / VC deps (fail‑soft) ────────────────────────────
try:
    from opentelemetry import trace  # type: ignore
    _tracer = trace.get_tracer("alpha_factory.governance")
except ModuleNotFoundError:  # pragma: no cover
    class _DummySpan:  # pylint: disable=too-few-public-methods
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def set_attribute(self, *_): pass
        def get_span_context(self): return type("ctx", (), {"trace_id": 0})()
    class _DummyTracer:  # pylint: disable=too-few-public-methods
        def start_as_current_span(self, *_a, **_k): return _DummySpan()
    _tracer = _DummyTracer()  # type: ignore

try:
    from vc import Credential, sign  # hypothetical library
    _vc_enabled = True
except ModuleNotFoundError:  # pragma: no cover
    _vc_enabled = False
    class Credential:  # type: ignore
        def __init__(self, **_): pass
    def sign(c):  # type: ignore
        return type("sig", (), {"store_ipfs": lambda self: None})()

from better_profanity import profanity

# legacy re‑export (tests expect it here)
from .memory import Memory  # noqa: F401

# ─── disallowed regex patterns (unchanged) ───────────────────────────────
_DANGER_PATTERNS: List[str] = [
    r"\bbomb recipe\b",
    r"\bhow\s+to\s+(make|build).*?(bomb|explosive)\b",
    r"\b(extremist|terrorist)\b",
    r"\b(extremist|terrorist).*?propaganda\b",
    r"\bpropaganda.*?(extremist|terrorist)\b",
    r"\blaunder\b",
    r"\bmoney laundering\b",
    r"\bkill\b",
]
_DANGER_RE = re.compile("|".join(_DANGER_PATTERNS), re.IGNORECASE)

# ─── telemetry helper ────────────────────────────────────────────────────
@contextmanager
def decision_span(name: str, **attrs):
    """
    Context‑manager that opens an OTEL span *and* writes a signed VC when
    the guarded action completes.  No‑ops when deps are missing.
    """
    with _tracer.start_as_current_span(name, attributes=attrs) as span:
        yield span
        if _vc_enabled:
            vc = Credential(
                id=name,
                attrs=attrs | {"trace_id": span.get_span_context().trace_id}
            )
            sign(vc).store_ipfs()          # fire‑and‑forget


# ─── Governance class (original + telemetry hooks) ───────────────────────
class Governance:
    """
    Content & numeric guard‑rails + audit telemetry.

    • `moderate(text) -> bool`  – content gate
    • `vet_plans(agent, plans)` – risk gate for numeric limits, etc.
    """

    def __init__(self, memory: Memory) -> None:
        self.log = logging.getLogger("Governance")
        self.memory = memory
        self.trade_limit = float(os.getenv("ALPHA_TRADE_LIMIT", "1000000"))
        profanity.load_censor_words()

    # ── content moderation ───────────────────────────────────────────────
    def moderate(self, text: str) -> bool:
        """Return **True** iff *text* is safe to use / display."""
        with decision_span("governance.moderate", length=len(text)):
            if _DANGER_RE.search(text):
                self.log.warning("Blocked disallowed content: %s", text)
                return False
            return not profanity.contains_profanity(text)

    # ── numeric / action guard‑rail ──────────────────────────────────────
    def vet_plans(self, agent, plans: List[Dict[str, Any]]):
        """Strip plans that violate numeric policy and return the vetted list."""
        vetted: List[Dict[str, Any]] = []
        for p in plans:
            if p.get("type") == "trade" and p.get("notional", 0) > self.trade_limit:
                with decision_span("governance.block_trade",
                                   agent=agent.name,
                                   notional=p.get("notional", 0)):
                    self.log.warning("Blocked high‑notional trade: %s", p)
                    self.memory.write(agent.name, "blocked", p)
                continue
            vetted.append(p)
        return vetted


__all__ = ["Governance", "Memory", "decision_span"]
