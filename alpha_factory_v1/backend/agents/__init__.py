"""backend.agents
===================================================================
Alpha‑Factory v1 👁️✨ — Multi‑Agent AGENTIC α‑AGI
-------------------------------------------------------------------
Domain‑agent registry and dynamic discovery layer
===================================================================
This module boots *all* first‑party and third‑party Agent subclasses so the
orchestrator can iterate over them without hard‑coding import paths.

Key design goals
----------------
* **Self‑discovery & antifragility** – scans local package, then optional
  entry‑points (``alpha_factory.agents``) to pull in plugins. Failures are
  caught and logged; partial availability never crashes the runtime.
* **Reg–ready transparency** – every registered agent exposes structured
  ``capabilities`` and ``compliance_tags`` so governance hooks can filter or
  audit behaviour (GDPR, SOX, MiFID, FDA 21 CFR §11, etc.).
* **Offline resilience** – agents that *require* an OpenAI API key can declare
  ``REQUIRES_API_KEY = True``; if the key is absent we skip registration and
  fall back to dummy stubs so edge devices or air‑gapped regulators can still
  launch the stack.
* **Non‑technical deployment** – no code edits needed; drop‑in Python wheels or
  extra ``*_agent.py`` files are auto‑loaded at start‑up.

This file is intentionally **self‑contained** (no external libs beyond the
standard library) so that the entire Alpha‑Factory can cold‑start on a vanilla
Python >=3.10 image with: ``pip install -r backend/requirements.txt && python -m
backend.orchestrator``.
"""
from __future__ import annotations

import importlib
import inspect
import logging
import os
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Type

try:
    # Optional – for plugin discovery via PEP 621 entry‑points
    import importlib.metadata as importlib_metadata  # noqa: F401
except ModuleNotFoundError:  # Python <3.8 back‑compat (unlikely)
    import importlib_metadata  # type: ignore

# ---------------------------------------------------------------------------
# Logging setup (inherits root handlers in production, safe in notebooks)
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Runtime feature flags
# ---------------------------------------------------------------------------
_OPENAI_READY = bool(os.getenv("OPENAI_API_KEY"))
_DISABLED = {n.strip().lower() for n in os.getenv("DISABLED_AGENTS", "").split(",") if n.strip()}

# ---------------------------------------------------------------------------
# Metadata structure for each agent
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentMetadata:
    name: str
    cls: Type
    version: str = "0.1.0"
    capabilities: List[str] = field(default_factory=list)
    compliance_tags: List[str] = field(default_factory=list)
    requires_api_key: bool = False

    def instantiate(self, **kwargs):  # convenience wrapper
        return self.cls(**kwargs)  # type: ignore[arg-type]

    # Pretty‑print helper (for CLI / dashboards)
    def as_dict(self):
        return {
            "name": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "compliance": self.compliance_tags,
            "requires_api_key": self.requires_api_key,
        }

# ---------------------------------------------------------------------------
# Global in‑memory registry
# ---------------------------------------------------------------------------
AGENT_REGISTRY: Dict[str, AgentMetadata] = {}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _should_register(meta: AgentMetadata) -> bool:
    """Policy gate applied before final registration."""
    if meta.name in _DISABLED:
        logger.info("Agent '%s' disabled via DISABLED_AGENTS env‑var", meta.name)
        return False
    if meta.requires_api_key and not _OPENAI_READY:
        logger.warning("Skipping '%s' – OPENAI_API_KEY absent", meta.name)
        return False
    return True


def _import_agent(module_name: str) -> Optional[AgentMetadata]:
    """Import module and return first subclass of AgentBase with metadata."""
    try:
        mod: ModuleType = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001, broad catch ok here
        logger.exception("Failed to import %s: %s", module_name, exc)
        return None

    # Delayed import to avoid circular deps if agent_base imports registry
    from backend.agent_base import AgentBase  # pylint: disable=import‑error

    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, AgentBase) and obj is not AgentBase:
            meta = AgentMetadata(
                name=getattr(obj, "NAME", obj.__name__.replace("Agent", "").lower()),
                cls=obj,
                version=getattr(obj, "__version__", "0.1.0"),
                capabilities=getattr(obj, "CAPABILITIES", []),
                compliance_tags=getattr(obj, "COMPLIANCE_TAGS", []),
                requires_api_key=getattr(obj, "REQUIRES_API_KEY", False),
            )
            return meta
    logger.warning("No Agent subclass found in %s", module_name)
    return None


def _discover_local_agents() -> None:
    """Scan the current package for ``*_agent.py`` modules."""
    pkg_root = Path(__file__).parent
    prefix = f"{__name__}."
    for _, name, is_pkg in pkgutil.iter_modules([pkg_root.as_posix()]):
        if is_pkg or not name.endswith("_agent"):
            continue
        module_name = prefix + name
        meta = _import_agent(module_name)
        if meta and _should_register(meta):
            AGENT_REGISTRY[meta.name] = meta
            logger.info("Registered local agent: %s (%s)", meta.name, meta.version)


def _discover_plugin_agents() -> None:  # noqa: C901 (keep simple, low cyclomatic)
    """Discover third‑party agents via entry‑points (optional)."""
    try:
        eps = importlib_metadata.entry_points(group="alpha_factory.agents")  # type: ignore[attr‑defined]
    except Exception:  # noqa: BLE001
        return  # entry‑point machinery unavailable (older Python)

    for ep in eps:
        try:
            cls = ep.load()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to load entry‑point '%s': %s", ep.name, exc)
            continue

        # Late import to avoid cost if no plugins
        from backend.agent_base import AgentBase  # pylint: disable=import‑error

        if not inspect.isclass(cls) or not issubclass(cls, AgentBase):
            logger.warning("Entry‑point '%s' is not an Agent subclass", ep.name)
            continue

        meta = AgentMetadata(
            name=getattr(cls, "NAME", ep.name),
            cls=cls,
            version=getattr(cls, "__version__", "0.1.0"),
            capabilities=getattr(cls, "CAPABILITIES", []),
            compliance_tags=getattr(cls, "COMPLIANCE_TAGS", []),
            requires_api_key=getattr(cls, "REQUIRES_API_KEY", False),
        )
        if _should_register(meta):
            AGENT_REGISTRY[meta.name] = meta
            logger.info("Registered plugin agent: %s (%s)", meta.name, meta.version)


# ---------------------------------------------------------------------------
# Public API (used by orchestrator, CLI, and tests)
# ---------------------------------------------------------------------------

def list_agents(as_dict: bool = False):
    """Return sorted agent list (optionally as metadata dicts)."""
    metas = sorted(AGENT_REGISTRY.values(), key=lambda m: m.name)
    return [m.as_dict() if as_dict else m.name for m in metas]


def get_agent(name: str):
    """Instantiate an agent by registry key."""
    meta = AGENT_REGISTRY[name]
    return meta.instantiate()


def register_agent(meta: AgentMetadata, *, overwrite: bool = False) -> None:
    """Runtime registration hook (used by unit tests or hot‑reload).

    Example::
        from backend.agents import AgentMetadata, register_agent
        from my_pkg.my_agent import MyAgent
        register_agent(AgentMetadata(name="my", cls=MyAgent))
    """
    if not overwrite and meta.name in AGENT_REGISTRY:
        raise ValueError(f"Agent '{meta.name}' already registered; pass overwrite=True to replace.")
    if _should_register(meta):
        AGENT_REGISTRY[meta.name] = meta
        logger.info("Runtime‑registered agent: %s", meta.name)


# ---------------------------------------------------------------------------
# One‑time discovery at import time
# ---------------------------------------------------------------------------
_discover_local_agents()
_discover_plugin_agents()

# ---------------------------------------------------------------------------
# Sentinel log line for ops grep‑ability
# ---------------------------------------------------------------------------
logger.info("Alpha‑Factory Agent Registry initialised with %d agents → %s", len(AGENT_REGISTRY), list_agents())
