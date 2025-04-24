from __future__ import annotations
"""
backend.agents
==============

Alpha-Factory v1 👁️✨ — Multi-Agent AGENTIC α-AGI
-------------------------------------------------
*Dynamic agent discovery, registry, governance & health.*

This module is the **single source of truth** for every domain-agent that
participates in Alpha-Factory.  It safely loads local `*_agent.py` files,
PEP-621 entry-points, *and* (optionally) wheels streamed via Google ADK’s
Agent2Agent mesh – all while remaining 100 % importable on a fresh
Python standard-library-only environment.
"""
##############################################################################
#                              std-lib imports                               #
##############################################################################
import importlib
import importlib.util
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
from typing import Dict, List, Optional, Type, Union

##############################################################################
#                optional heavy deps (never hard-fail at import)             #
##############################################################################
try:
    import importlib.metadata as importlib_metadata  # ≥ Py 3.10
except ModuleNotFoundError:                           # pragma: no cover
    import importlib_metadata                         # type: ignore

try:                                                  # Kafka telemetry
    from kafka import KafkaProducer                   # type: ignore
except ModuleNotFoundError:                           # pragma: no cover
    KafkaProducer = None                              # type: ignore

try:                                                  # Prometheus counter
    from prometheus_client import Counter             # type: ignore
except ModuleNotFoundError:                           # pragma: no cover
    Counter = None                                    # type: ignore

try:                                                  # ADK remote wheel pull-in
    import adk                                        # type: ignore
except ModuleNotFoundError:                           # pragma: no cover
    adk = None                                        # type: ignore

##############################################################################
#                              configuration                                 #
##############################################################################
_OPENAI_READY   = bool(os.getenv("OPENAI_API_KEY"))
_KAFKA_BROKER   = os.getenv("ALPHA_KAFKA_BROKER")
_DISABLED       = {x.strip().lower()
                   for x in os.getenv("DISABLED_AGENTS", "").split(",")
                   if x.strip()}
_ERR_THRESHOLD  = int(os.getenv("AGENT_ERR_THRESHOLD", 3))
_HOT_DIR        = Path(os.getenv("AGENT_HOT_DIR", "")).expanduser()
_HEARTBEAT_INT  = int(os.getenv("AGENT_HEARTBEAT_SEC", 10))

##############################################################################
#                                  logging                                   #
##############################################################################
logger = logging.getLogger("alpha_factory.agents")
logger.setLevel(logging.INFO)

##############################################################################
#                               public types                                 #
##############################################################################
@dataclass(frozen=True)
class AgentMetadata:
    name:            str
    cls:             Type            # reference to Agent class (or Stub)
    version:         str  = "0.1.0"
    capabilities:    List[str] = field(default_factory=list)
    compliance_tags: List[str] = field(default_factory=list)
    requires_api_key: bool = False
    err_count:       int  = 0

    # serialisation helpers -------------------------------------------------
    def as_dict(self) -> Dict:
        return {
            "name":         self.name,
            "version":      self.version,
            "capabilities": self.capabilities,
            "compliance":   self.compliance_tags,
            "requires_api_key": self.requires_api_key,
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), separators=(",", ":"))

    # instantiation helper --------------------------------------------------
    def instantiate(self, **kwargs):
        return self.cls(**kwargs)     # type: ignore[arg-type]


class CapabilityGraph(Dict[str, List[str]]):
    """capability ➜ [ agent names ]"""
    def add(self, capability: str, agent_name: str):
        self.setdefault(capability, []).append(agent_name)

##############################################################################
#                              stub fallback                                 #
##############################################################################
class StubAgent:                               # pragma: no cover
    """Lightweight inert replacement for unavailable agents."""
    NAME             = "stub"
    CAPABILITIES     = []
    COMPLIANCE_TAGS  = []
    REQUIRES_API_KEY = False
    async def run_cycle(self):                 # noqa: D401
        await asyncio.sleep(3600)  # never does anything

##############################################################################
#                           internal state                                   #
##############################################################################
AGENT_REGISTRY:    Dict[str, AgentMetadata] = {}
CAPABILITY_GRAPH:  CapabilityGraph          = CapabilityGraph()
_HEALTH_Q:         "queue.Queue[tuple[str,float,bool]]" = queue.Queue()

if Counter:
    _err_counter = Counter("af_agent_exceptions_total",
                           "Exceptions raised by agents",
                           ["agent"])

##############################################################################
#                     lightweight helper functions                           #
##############################################################################
def _kafka_producer():
    if not _KAFKA_BROKER or KafkaProducer is None:
        return None
    return KafkaProducer(bootstrap_servers=_KAFKA_BROKER,
                         value_serializer=lambda v: v.encode() if isinstance(v, str) else v)

_PRODUCER = _kafka_producer()

def _emit_kafka(topic: str, payload: str):
    if _PRODUCER:
        try:
            _PRODUCER.send(topic, payload)
            _PRODUCER.flush()
        except Exception:                      # noqa: BLE001
            logger.exception("Kafka emit failed (topic=%s)", topic)

def _should_register(meta: AgentMetadata) -> bool:
    if meta.name.lower() in _DISABLED:
        logger.info("Agent %s disabled via env", meta.name)
        return False
    if meta.requires_api_key and not _OPENAI_READY:
        logger.warning("Skipping %s (requires OpenAI key)", meta.name)
        return False
    return True

##############################################################################
#                  discovery – local, entry-points, ADK                      #
##############################################################################
from backend.agent_base import AgentBase      # local import to avoid cycles

def _register(meta: AgentMetadata, *, overwrite: bool = False):
    if not _should_register(meta):
        return
    if meta.name in AGENT_REGISTRY and not overwrite:
        logger.error("Duplicate agent name '%s'", meta.name)
        return

    AGENT_REGISTRY[meta.name] = meta
    for cap in meta.capabilities:
        CAPABILITY_GRAPH.add(cap, meta.name)

    logger.info("✓ agent %-18s caps=%s", meta.name, ",".join(meta.capabilities))
    _emit_kafka("agent.manifest", meta.to_json())


def _inspect_module(mod: ModuleType) -> Optional[AgentMetadata]:
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, AgentBase) and obj is not AgentBase:
            return AgentMetadata(
                name=getattr(obj, "NAME", obj.__name__.replace("Agent", "").lower()),
                cls=obj,
                version=getattr(obj, "__version__", "0.1.0"),
                capabilities=list(getattr(obj, "CAPABILITIES", [])),
                compliance_tags=list(getattr(obj, "COMPLIANCE_TAGS", [])),
                requires_api_key=getattr(obj, "REQUIRES_API_KEY", False),
            )
    return None


def _discover_local():
    pkg_root = Path(__file__).parent
    prefix   = f"{__name__}."
    for _, mod_name, is_pkg in pkgutil.iter_modules([str(pkg_root)]):
        if is_pkg or not mod_name.endswith("_agent"):
            continue
        try:
            meta = _inspect_module(importlib.import_module(prefix + mod_name))
            if meta:
                _register(meta)
        except Exception:                      # noqa: BLE001
            logger.exception("Import error %s", mod_name)


def _discover_entrypoints():
    try:
        eps = importlib_metadata.entry_points(group="alpha_factory.agents")  # type: ignore[arg-type]
    except Exception:                              # noqa: BLE001
        return
    for ep in eps:
        try:
            obj = ep.load()
        except Exception:                          # noqa: BLE001
            logger.exception("EP %s load failed", ep.name)
            continue
        if inspect.isclass(obj) and issubclass(obj, AgentBase):
            _register(AgentMetadata(
                name=getattr(obj, "NAME", ep.name),
                cls=obj,
                version=getattr(obj, "__version__", "0.1.0"),
                capabilities=list(getattr(obj, "CAPABILITIES", [])),
                compliance_tags=list(getattr(obj, "COMPLIANCE_TAGS", [])),
                requires_api_key=getattr(obj, "REQUIRES_API_KEY", False),
            ))

def _discover_hot_dir():
    if not _HOT_DIR.is_dir():
        return
    for wheel in _HOT_DIR.glob("*.whl"):
        mod_name = wheel.stem.replace("-", "_")
        if mod_name in AGENT_REGISTRY:
            continue
        spec = importlib.util.spec_from_file_location(mod_name, wheel)
        if not spec:
            continue
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)          # type: ignore[attr-defined]
            meta = _inspect_module(mod)
            if meta:
                _register(meta)
        except Exception:                          # noqa: BLE001
            logger.exception("Hot-dir load failed for %s", wheel.name)

def _discover_adk():
    if adk is None or not os.getenv("ADK_MESH"):
        return
    try:
        client = adk.Client()
        pkgs   = client.list_remote_packages()
        for pkg in pkgs:
            if pkg.name in AGENT_REGISTRY:
                continue
            wheel_path = client.download_package(pkg.name)
            _HOT_DIR.mkdir(parents=True, exist_ok=True)
            dest = _HOT_DIR / wheel_path.name
            dest.write_bytes(wheel_path.read_bytes())
            logger.info("Pulled %s from ADK mesh", pkg.name)
        _discover_hot_dir()
    except Exception:                              # noqa: BLE001
        logger.exception("ADK discovery failed")

##############################################################################
#                       health & quarantine loop                             #
##############################################################################
def _health_loop():
    while True:
        try:
            name, latency_ms, ok = _HEALTH_Q.get(timeout=_HEARTBEAT_INT)
        except queue.Empty:
            continue

        meta = AGENT_REGISTRY.get(name)
        if meta and not ok:
            if Counter:
                _err_counter.labels(agent=name).inc()
            # bump error counter (mutable via object.__setattr__)
            object.__setattr__(meta, "err_count", meta.err_count + 1)
            if meta.err_count >= _ERR_THRESHOLD:
                logger.error("Quarantining agent '%s' after %d errors ⛔",
                             name, meta.err_count)
                # replace with stub to keep capability graph consistent
                stub_meta = AgentMetadata(
                    name=meta.name,
                    cls=StubAgent,
                    version=meta.version + "+stub",
                    capabilities=meta.capabilities,
                    compliance_tags=meta.compliance_tags,
                )
                _register(stub_meta, overwrite=True)

        payload = json.dumps({"name": name,
                              "latency_ms": latency_ms,
                              "ok": ok,
                              "ts": time.time()})
        _emit_kafka("agent.heartbeat", payload)

threading.Thread(target=_health_loop,
                 daemon=True, name="agent-health").start()

##############################################################################
#                          public convenience API                            #
##############################################################################
def list_agents(detail: bool = False):
    metas = sorted(AGENT_REGISTRY.values(), key=lambda m: m.name)
    return [m.as_dict() if detail else m.name for m in metas]

def capability_agents(capability: str):
    """Return agents that expose the given capability."""
    return CAPABILITY_GRAPH.get(capability, []).copy()

def get_agent(name: str, **kwargs):
    meta  = AGENT_REGISTRY[name]
    agent = meta.instantiate(**kwargs)

    # health instrumentation wrapper ----------------------------------
    if hasattr(agent, "run_cycle") and inspect.iscoroutinefunction(agent.run_cycle):
        orig = agent.run_cycle

        async def _wrapped(*a, **kw):            # type: ignore[no-untyped-def]
            t0  = time.perf_counter()
            ok  = True
            try:
                return await orig(*a, **kw)      # type: ignore[misc]
            except Exception:                    # noqa: BLE001
                ok = False
                raise
            finally:
                _HEALTH_Q.put((meta.name, (time.perf_counter()-t0)*1000, ok))

        agent.run_cycle = _wrapped               # type: ignore[assignment]

    return agent

def register_agent(meta: AgentMetadata, *, overwrite: bool = False):
    """Public hook for dynamically created agents (e.g. from notebooks)."""
    _register(meta, overwrite=overwrite)

##############################################################################
#                           initial discovery pass                           #
##############################################################################
_discover_local()
_discover_entrypoints()
_discover_hot_dir()
_discover_adk()

logger.info("🚀 Agent registry ready – %d agents, %d capabilities",
            len(AGENT_REGISTRY), len(CAPABILITY_GRAPH))
