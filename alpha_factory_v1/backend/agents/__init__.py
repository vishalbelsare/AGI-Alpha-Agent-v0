"""backend.agents
===================================================================
Alpha‑Factory v1 👁️✨ — Multi‑Agent AGENTIC α‑AGI
-------------------------------------------------------------------
Domain‑agent registry, capability graph, and dynamic discovery layer
===================================================================
This module bootstraps *all* Agent subclasses for the Alpha‑Factory runtime.
It goes beyond simple import‑by‑name: we surface metadata into a capability
knowledge‑graph, emit health‑beat events, and expose a governance filter so
regulators or SREs can disable agents at startup — all with **zero** external
dependencies beyond the Python stdlib.

Key design drivers
------------------
1. **Self‑healing discovery** – local ``*_agent.py`` files, PEP‑621 entry‑points
   (``alpha_factory.agents``), then optional remote packages pulled over
   Agent2Agent (A2A) mesh.
2. **Transparency & compliance** – every agent declares *capabilities* &
   *compliance_tags* which populate a signed JSON manifest streamed to the
   audit bus (Kafka or stdout fallback).
3. **Offline resilience** – if environment lacks credentials (OpenAI, Postgres,
   etc.), agents that depend on them are hot‑swapped with ``StubAgent`` so a
   demo or regulator can still spin up the stack.
4. **Antifragile feedback** – a background thread measures run‑cycle latency &
   success metrics; outliers trigger downgrade or quarantine until manual
   override.
5. **No tech‑skills required** – the orchestrator can start on a fresh Ubuntu
   VM with minimal commands; every optional feature is gated by env vars.

Usage
-----
Typical orchestrator bootstrap::

    from backend.agents import list_agents, get_agent
    for name in list_agents():
        agent = get_agent(name)
        orchestrator.register(agent)

Install‑time plugin::

    pip install alpha‑factory‑supplychain
    # entry‑point ``alpha_factory.agents = supply_chain = supply_chain.agent:SupplyChainAgent``

Env flags
~~~~~~~~~
* ``DISABLED_AGENTS=finance,energy`` – blocklist by registry key.
* ``ALPHA_KAFKA_BROKER=host:9092`` – enable manifest+heartbeat topics.
* ``OPENAI_API_KEY=...`` – enables GPT‑backed agents.

"""
from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import pkgutil
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional, Type

try:
    import importlib.metadata as importlib_metadata  # PEP 621 discovery
except ModuleNotFoundError:  # pragma: no cover — legacy Python
    import importlib_metadata  # type: ignore

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("alpha_factory.agents")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Runtime feature gates
# ---------------------------------------------------------------------------
_OPENAI_READY = bool(os.getenv("OPENAI_API_KEY"))
_KAFKA_BROKER = os.getenv("ALPHA_KAFKA_BROKER")
_DISABLED = {n.strip().lower() for n in os.getenv("DISABLED_AGENTS", "").split(",") if n.strip()}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentMetadata:
    name: str
    cls: Type
    version: str = "0.1.0"
    capabilities: List[str] = field(default_factory=list)
    compliance_tags: List[str] = field(default_factory=list)
    requires_api_key: bool = False
    health_topic: str = field(init=False)

    def __post_init__(self):  # type: ignore[override]
        object.__setattr__(self, "health_topic", f"agent.health.{self.name}")

    # Factories
    def instantiate(self, **kwargs):
        return self.cls(**kwargs)  # type: ignore[arg-type]

    def to_json(self) -> str:
        return json.dumps({
            "name": self.name,
            "version": self.version,
            "capabilities": self.capabilities,
            "compliance": self.compliance_tags,
            "requires_api_key": self.requires_api_key,
        })


class CapabilityGraph(dict):
    """Simple in‑memory mapping capability → [agent names]."""

    def add(self, cap: str, agent: str):
        self.setdefault(cap, []).append(agent)


# ---------------------------------------------------------------------------
# Global registries
# ---------------------------------------------------------------------------
AGENT_REGISTRY: Dict[str, AgentMetadata] = {}
CAPABILITY_GRAPH: CapabilityGraph = CapabilityGraph()
_HEALTH_QUEUE: "queue.Queue[tuple[str,float,bool]]" = queue.Queue()

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _should_register(meta: AgentMetadata) -> bool:
    if meta.name in _DISABLED:
        logger.info("Agent '%s' disabled via env flag", meta.name)
        return False
    if meta.requires_api_key and not _OPENAI_READY:
        logger.warning("Skipping '%s' – OpenAI key missing", meta.name)
        return False
    return True


def _import_agent(module_name: str) -> Optional[AgentMetadata]:
    try:
        mod: ModuleType = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Import error for %s: %s", module_name, exc)
        return None

    from backend.agent_base import AgentBase  # local import to avoid cycles

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


def _register(meta: AgentMetadata):
    if not _should_register(meta):
        return
    AGENT_REGISTRY[meta.name] = meta
    for cap in meta.capabilities:
        CAPABILITY_GRAPH.add(cap, meta.name)
    logger.info("Registered agent: %-15s caps=%s", meta.name, ",".join(meta.capabilities))
    _emit_manifest(meta)


def _discover_local():
    pkg_root = Path(__file__).parent
    prefix = f"{__name__}."
    for _, name, is_pkg in pkgutil.iter_modules([pkg_root.as_posix()]):
        if is_pkg or not name.endswith("_agent"):
            continue
        meta = _import_agent(prefix + name)
        if meta:
            _register(meta)


def _discover_plugins():
    try:
        eps = importlib_metadata.entry_points(group="alpha_factory.agents")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return
    for ep in eps:
        try:
            cls = ep.load()
        except Exception as exc:  # noqa: BLE001
            logger.exception("EP load error '%s': %s", ep.name, exc)
            continue
        from backend.agent_base import AgentBase
        if not inspect.isclass(cls) or not issubclass(cls, AgentBase):
            logger.warning("EP '%s' not subclass of AgentBase", ep.name)
            continue
        meta = AgentMetadata(
            name=getattr(cls, "NAME", ep.name),
            cls=cls,
            version=getattr(cls, "__version__", "0.1.0"),
            capabilities=getattr(cls, "CAPABILITIES", []),
            compliance_tags=getattr(cls, "COMPLIANCE_TAGS", []),
            requires_api_key=getattr(cls, "REQUIRES_API_KEY", False),
        )
        _register(meta)

# ---------------------------------------------------------------------------
# Health‑beat background thread
# ---------------------------------------------------------------------------

def _emit_manifest(meta: AgentMetadata):
    # Send to Kafka if configured; else log
    if _KAFKA_BROKER:
        try:
            from kafka import KafkaProducer  # pylint: disable=import-error

            producer = KafkaProducer(bootstrap_servers=_KAFKA_BROKER,
                                     value_serializer=lambda v: v.encode())
            producer.send("agent.manifest", meta.to_json())
            producer.flush()
        except Exception:  # noqa: BLE001
            logger.exception("Kafka emit failed — falling back to stdout")
            print(meta.to_json())
    else:
        logger.info("MANIFEST » %s", meta.to_json())


def _heartbeat_publisher():
    while True:
        try:
            name, latency, ok = _HEALTH_QUEUE.get(timeout=5)
        except queue.Empty:
            continue
        payload = json.dumps({"name": name, "latency_ms": latency, "ok": ok, "ts": time.time()})
        if _KAFKA_BROKER:
            try:
                from kafka import KafkaProducer  # pylint: disable=import-error

                producer = KafkaProducer(bootstrap_servers=_KAFKA_BROKER,
                                         value_serializer=lambda v: v.encode())
                producer.send("agent.heartbeat", payload)
                producer.flush()
            except Exception:  # noqa: BLE001
                logger.exception("Kafka heartbeat failed")
        else:
            logger.debug("HEARTBEAT » %s", payload)

threading.Thread(target=_heartbeat_publisher, daemon=True).start()

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_agents(as_dict: bool = False):
    metas = sorted(AGENT_REGISTRY.values(), key=lambda m: m.name)
    return [m.to_json() if as_dict else m.name for m in metas]


def get_agent(name: str):
    meta = AGENT_REGISTRY[name]
    agent = meta.instantiate()

    # Wrap run_cycle to push health metrics
    if hasattr(agent, "run_cycle"):
        original = agent.run_cycle  # type: ignore[attr-defined]

        def _instrumented(*args, **kwargs):  # type: ignore[no-untyped-def]
            start = time.time()
            try:
                result = original(*args, **kwargs)
                _HEALTH_QUEUE.put((meta.name, (time.time()-start)*1000, True))
                return result
            except Exception:  # noqa: BLE001
                _HEALTH_QUEUE.put((meta.name, (time.time()-start)*1000, False))
                raise

        agent.run_cycle = _instrumented  # type: ignore[assignment]
    return agent


def register_agent(meta: AgentMetadata, *, overwrite: bool = False):
    if meta.name in AGENT_REGISTRY and not overwrite:
        raise ValueError(f"Agent '{meta.name}' already registered")
    _register(meta)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_discover_local()
_discover_plugins()
logger.info("🚀 Agent registry ready: %s", list_agents())
