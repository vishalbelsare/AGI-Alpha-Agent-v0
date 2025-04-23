
"""
reward_backends · Alpha‑Factory v1 👁️✨
──────────────────────────────────────────
Pluggable reward‑function framework inspired by the “Era of Experience”
grounded‑reward pillar.

• Any *.py file in this package that defines a callable named
  `reward(state, action, result) -> float` is auto‑discovered.

• Public API
    list_rewards()               → tuple[str, …]
    reward_signal(name, s, a, r) → float
    blend(signals, weights=None) → float
    refresh()                    → rescan package

Design notes
────────────
▸ Zero runtime dependencies beyond the Python ≥ 3.9 std‑lib  
▸ Hot‑reload friendly – call refresh() to pick up new files  
▸ Fault‑tolerant – bad back‑ends are logged & quarantined  
▸ Thread‑safe read path via MappingProxyType
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import types
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Dict, Tuple

# ──────────────────────────── internal registry ──────────────────────────────
_pkg_path = Path(__file__).parent
_registry: Dict[str, Callable] = {}
_frozen: MappingProxyType | None = None

# ─────────────────────────── helper functions ────────────────────────────────
def _qualname(mod: types.ModuleType) -> str:
    return f"{mod.__name__}.reward"

def _scan_package() -> None:
    """Import every reward backend & register its reward() callable."""
    global _registry, _frozen
    _registry.clear()

    for info in pkgutil.iter_modules([_pkg_path]):
        if info.ispkg or info.name.startswith("_"):
            continue
        mod_name = f"{__name__}.{info.name}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            print(f"[reward_backends] ⚠ failed loading {mod_name}: {exc}")
            continue

        fn = getattr(mod, "reward", None)
        if not callable(fn):
            print(f"[reward_backends] ⤬ {mod_name} has no callable reward()")
            continue
        if len(inspect.signature(fn).parameters) != 3:
            print(
                f"[reward_backends] ⤬ {_qualname(mod)} invalid signature "
                "(expected (state, action, result))"
            )
            continue

        _registry[info.name] = fn

    _frozen = MappingProxyType(_registry.copy())
    joined = ", ".join(_registry) or "none"
    print(f"[reward_backends] ✓ registered: {joined}")

# ───────────────────────────── public API ────────────────────────────────────
def list_rewards() -> Tuple[str, ...]:
    """Return an immutable tuple of available reward back‑end names."""
    return tuple(_frozen or ())

def reward_signal(name: str, state, action, result) -> float:
    """
    Execute a single back‑end by *name*.

    Parameters
    ----------
    name   : str  – registered back‑end name
    state  : Any  – environment / agent state snapshot
    action : Any  – action the agent just took
    result : Any  – observation / env outcome

    Raises
    ------
    KeyError – if *name* is unknown
    """
    fn = (_frozen or {}).get(name)
    if fn is None:
        raise KeyError(f"Unknown reward back‑end: {name!r}")
    return float(fn(state, action, result))

def blend(signals: Dict[str, float],
          weights: Dict[str, float] | None = None) -> float:
    """
    Weighted blend of pre‑computed reward signals.

    Parameters
    ----------
    signals : mapping name → value
    weights : mapping name → weight (default = equal)

    Returns
    -------
    float – aggregated reward
    """
    if not signals:
        return 0.0
    if weights is None:
        weights = {k: 1.0 for k in signals}

    total_w = sum(weights.get(k, 0.0) for k in signals) or 1.0
    return sum(signals[k] * weights.get(k, 0.0) for k in signals) / total_w

def refresh() -> None:
    """Force a rescan (hot‑reload during iterative development)."""
    _scan_package()

# ───────────────────────────── bootstrap ─────────────────────────────────────
_scan_package()
