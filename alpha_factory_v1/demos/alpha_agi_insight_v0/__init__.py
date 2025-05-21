"""α‑AGI Insight demo package."""

# ``main`` gives callers the convenient entrypoint from ``__main__``.
from .__main__ import main  # re-export
from . import openai_agents_bridge

__all__ = ["main", "openai_agents_bridge"]
