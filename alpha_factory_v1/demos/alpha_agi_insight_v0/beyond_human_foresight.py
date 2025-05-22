#!/usr/bin/env python3
"""Launch the Beyond Human Foresight variant of the α‑AGI Insight demo.

This thin wrapper prints a short banner then delegates to
:mod:`official_demo_production`, inheriting automatic environment
verification, optional OpenAI Agents SDK integration and graceful
offline mode.  The behaviour mirrors ``alpha-agi-insight-production``
while providing a flashier startup message.
"""
from __future__ import annotations

from typing import List

if __package__ is None:  # pragma: no cover - allow direct execution
    import pathlib
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
    __package__ = "alpha_factory_v1.demos.alpha_agi_insight_v0"

from .official_demo_production import main as _main


def main(argv: List[str] | None = None) -> None:
    """Entry point for the Beyond Human Foresight demo."""
    banner = "\N{MILITARY MEDAL} \N{GREEK SMALL LETTER ALPHA}\N{HYPHEN-MINUS}AGI Insight \N{EYE}\N{SPARKLES} — Beyond Human Foresight"
    print(banner)
    _main(argv)


if __name__ == "__main__":  # pragma: no cover
    main()
