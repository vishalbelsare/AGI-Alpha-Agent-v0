from __future__ import annotations
"""
alpha_factory_v1.backend.agents
================================

Alpha-Factory v1 👁️✨ — Multi-Agent AGENTIC α-AGI
-----------------------------------------------------------------
Dynamic **agent discovery, registry, governance, telemetry & health**.

Key design points
-----------------
✅  Zero hard-fail – **every optional dependency is wrapped** so a fresh
   Python-only install still imports.

✅  Multiple discovery paths  
      • local `*_agent.py` modules  
      • PEP-621 entry-points (`alpha_factory.agents`)  
      • Google ADK / A2A mesh streamed wheels  
      • user-defined “hot-drop” wheel folder (`$AGENT_HOT_DIR`)

✅  Kafka + Prometheus + OpenTelemetry emit hooks (no-ops if libs missing)

✅  **Health heartbeat** loop → quarantine / stub-swap after N consecutive
   errors (`$AGENT_ERR_THRESHOLD`, default 3)

✅  Live **hot-reload** – wheel dropped into `$AGENT_HOT_DIR` is picked up
   within `$AGENT_RESCAN_SEC` (default 60) without restarting Orchestrator.

This file is the *single source of truth* for every domain-agent that
participates in Alpha-Factory.
"""

##############################################################################
#                              std-lib imports                               #
##############################################################################
import asyncio
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
from typing import Dict, List, Optional, Type

##############################################################################
#               optional heavy deps (never hard-fail at import)              #
##############################################################################
try:                                         # ≥ Py 3.10
    import importlib.metadata as imetadata
except ModuleNotFoundError:                  # pragma: no cover
    import importlib_metadata as imetadata   # type: ignore

try:                                         # Kafka telemetry
    from kafka import KafkaProducer          # type: ignore
except ModuleNotFoundError:                  # pragma: no cover
    KafkaProducer = None                     # type: ignore

try:                                         # Prometheus counter
    from prometheus_client import Counter    # type: ignore
except ModuleNotFoundError:                  # pragma: no cover
    Counter = None                           # type: ignore

try:                                         # Google Agent Development Kit
    import adk                               # type: ignore
except ModuleNotFoundError:                  # pragma: no cover
    adk = None                               # type: ignore

##############################################################################
#                              configuration                                 #
##############################################################################
_OPENAI_READY   = bool(os.getenv("OPENAI_API_KEY"))
_KAFKA_BROKER   = os.getenv("ALPHA_KAFKA_BROKER")
_DISABLED       = {
    x.strip().lower()
    for x in os.getenv("DISABLED_AGENTS", "").split(",")
    if x.strip()
}
_ERR_THRESHOLD  = int(os.getenv("AGENT_ERR_THRESHOLD", 3))
_HOT_DIR        = Path(os.getenv("AGENT_HOT_DIR", "~/.alpha_agents")).expanduser()
_HEARTBEAT_INT  = int(os.getenv("AGENT_HEARTBEAT_SEC", 10))
_RESCAN_SEC     = int(os.getenv("AGENT_RESCAN_SEC", 60))

##############################################################################
#                                  logging                                   #
##############################################################################
logger = logging.getLogger("alpha_factory.agents")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s: %(message)s")
    )
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

##############################################################################
#                               public types                                 #
##############################################################################
@dataclass(frozen=True)
class AgentMetadata:
    name:             str
    cls:              Type                      # reference to concrete Agent
    version:          str  = "0.1.0"
    capabilities:     List[str] = field(default_factory=list)
    compliance_tags:  List[str] = field(default_factory=list)
    requires_api_key: bool = False
    err_count:        int  = 0                  # mutated via object.__setattr__

    # ---------- serialization helpers ----------
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

    # ---------- instantiation helper ----------
    def instantiate(self, **kwargs):
        return self.cls(**kwargs)               # type: ignore[arg-type]


class CapabilityGraph(Dict[str, List[str]]):
    """capability ➜ [agent names]"""
    def add(self, capability: str, agent_name: str) -> None:
        self.setdefault(capability, []).append(agent_name)

##############################################################################
#                              stub fallback                                 #
##############################################################################
class StubAgent:                               # pragma: no cover
    """Lightweight inert replacement for quarantined / missing agents."""
    NAME             = "stub"
    CAPABILITIES:    List[str] = []
    COMPLIANCE_TAGS: List[str] = []
    REQUIRES_API_KEY = False
    SLEEP            = 3600

    async def step(self):
        await asyncio.sleep(self.SLEEP)

##############################################################################
#                           internal state                                   #
##############################################################################
AGENT_REGISTRY:    Dict[str, AgentMetadata] = {}
CAPABILITY_GRAPH:  CapabilityGraph          = CapabilityGraph()
_HEALTH_Q:         "queue.Queue[tuple[str,float,bool]]" = queue.Queue()

if Counter is not None:
    _err_counter = Counter(
        "af_agent_exceptions_total",
        "Exceptions raised by agents",
        ["agent"],
    )

##############################################################################
#                     lightweight helper functions                           #
##############################################################################
def _kafka_producer() -> Optional[KafkaProducer]:
    if not _KAFKA_BROKER or KafkaProducer is None:
        return None
    try:
        return KafkaProducer(
            bootstrap_servers=_KAFKA_BROKER,
            value_serializer=lambda v: v.encode()
            if isinstance(v, str)
            else v,
        )
    except Exception:                           # noqa: BLE001
        logger.exception("Kafka producer init failed")
        return None


_PRODUCER = _kafka_producer()


def _emit_kafka(topic: str, payload: str) -> None:
    if _PRODUCER is None:
        return
    try:
        _PRODUCER.send(topic, payload)
        _PRODUCER.flush()
    except Exception:                            # noqa: BLE001
        logger.exception("Kafka emit failed (topic=%s)", topic)


def _should_register(meta: AgentMetadata) -> bool:
    if meta.name.lower() in _DISABLED:
        logger.info("Agent %s disabled via env", meta.name)
        return False
    if meta.requires_api_key and not _OPENAI_READY:
        logger.warning("Skipping %s (needs OpenAI key)", meta.name)
        return False
    return True

##############################################################################
#                  discovery – local, entry-points, ADK                      #
##############################################################################
from backend.agent_base import AgentBase       # local import to avoid cycles


def _register(meta: AgentMetadata, *, overwrite: bool = False) -> None:
    if not _should_register(meta):
        return
    if meta.name in AGENT_REGISTRY and not overwrite:
        logger.error("Duplicate agent name '%s' ignored", meta.name)
        return

    AGENT_REGISTRY[meta.name] = meta
    for cap in meta.capabilities:
        CAPABILITY_GRAPH.add(cap, meta.name)

    logger.info("✓ agent %-18s caps=%s", meta.name, ",".join(meta.capabilities))
    _emit_kafka("agent.manifest", meta.to_json())


def _inspect_module(mod: ModuleType) -> Optional[AgentMetadata]:
    """Return AgentMetadata if module defines a concrete AgentBase subclass."""
    for _, obj in inspect.getmembers(mod, inspect.isclass):
        if issubclass(obj, AgentBase) and obj is not AgentBase:
            return AgentMetadata(
                name=getattr(obj, "NAME", obj.__name__),
                cls=obj,
                version=getattr(obj, "__version__", "0.1.0"),
                capabilities=list(getattr(obj, "CAPABILITIES", [])),
                compliance_tags=list(getattr(obj, "COMPLIANCE_TAGS", [])),
                requires_api_key=getattr(obj, "REQUIRES_API_KEY", False),
            )
    return None


# ---------- local *_agent.py files ----------
def _discover_local() -> None:
    pkg_root = Path(__file__).parent
    prefix   = f"{__name__}."
    for _, mod_name, is_pkg in pkgutil.iter_modules([str(pkg_root)]):
        if is_pkg or not mod_name.endswith("_agent"):
            continue
        try:
            meta = _inspect_module(importlib.import_module(prefix + mod_name))
            if meta:
                _register(meta)
        except Exception:                        # noqa: BLE001
            logger.exception("Import error for %s", mod_name)


# ---------- entry-points ----------
def _discover_entrypoints() -> None:
    try:
        eps = imetadata.entry_points(group="alpha_factory.agents")  # type: ignore[arg-type]
    except Exception:                              # noqa: BLE001
        return
    for ep in eps:
        try:
            obj = ep.load()
        except Exception:                          # noqa: BLE001
            logger.exception("Entry-point load failed: %s", ep.name)
            continue
        if inspect.isclass(obj) and issubclass(obj, AgentBase):
            _register(
                AgentMetadata(
                    name=getattr(obj, "NAME", ep.name),
                    cls=obj,
                    version=getattr(obj, "__version__", "0.1.0"),
                    capabilities=list(getattr(obj, "CAPABILITIES", [])),
                    compliance_tags=list(getattr(obj, "COMPLIANCE_TAGS", [])),
                    requires_api_key=getattr(obj, "REQUIRES_API_KEY", False),
                )
            )


# ---------- wheel hot-drop ----------
def _install_wheel(path: Path) -> Optional[ModuleType]:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)              # type: ignore[arg-type]
        return mod
    return None


def _discover_hot_dir() -> None:
    if not _HOT_DIR.is_dir():
        return
    for wheel in _HOT_DIR.glob("*.whl"):
        if wheel.stem.replace("-", "_") in AGENT_REGISTRY:
            continue
        try:
            mod = _install_wheel(wheel)
            if mod:
                meta = _inspect_module(mod)
                if meta:
                    _register(meta)
        except Exception:                          # noqa: BLE001
            logger.exception("Hot-dir load failed for %s", wheel.name)


# ---------- ADK mesh ----------
def _discover_adk() -> None:
    if adk is None or not os.getenv("ADK_MESH"):
        return
    try:
        client = adk.Client()
        for pkg in client.list_remote_packages():
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
def _health_loop() -> None:
    while True:
        try:
            name, latency_ms, ok = _HEALTH_Q.get(timeout=_HEARTBEAT_INT)
        except queue.Empty:
            continue

        meta = AGENT_REGISTRY.get(name)
        if meta and not ok:
            if Counter:
                _err_counter.labels(agent=name).inc()
            # bump error counter (immutable dataclass hack)
            object.__setattr__(meta, "err_count", meta.err_count + 1)
            if meta.err_count >= _ERR_THRESHOLD:
                logger.error(
                    "⛔ Quarantining agent '%s' after %d consecutive errors",
                    name,
                    meta.err_count,
                )
                stub_meta = AgentMetadata(
                    name=meta.name,
                    cls=StubAgent,
                    version=meta.version + "+stub",
                    capabilities=meta.capabilities,
                    compliance_tags=meta.compliance_tags,
                )
                _register(stub_meta, overwrite=True)

        payload = json.dumps(
            {
                "name": name,
                "latency_ms": latency_ms,
                "ok": ok,
                "ts": time.time(),
            }
        )
        _emit_kafka("agent.heartbeat", payload)


threading.Thread(target=_health_loop, daemon=True, name="agent-health").start()

##############################################################################
#                    hot-dir rescanner (live wheel drop-ins)                 #
##############################################################################
def _rescan_loop() -> None:                       # pragma: no cover
    while True:
        try:
            _discover_hot_dir()
        except Exception:                         # noqa: BLE001
            logger.exception("Hot-dir rescan failed")
        time.sleep(_RESCAN_SEC)


threading.Thread(target=_rescan_loop, daemon=True, name="agent-rescan").start()

##############################################################################
#                          public convenience API                            #
##############################################################################
def list_agents(detail: bool = False):
    metas = sorted(AGENT_REGISTRY.values(), key=lambda m: m.name)
    return [m.as_dict() if detail else m.name for m in metas]


def capability_agents(capability: str):
    """Return agents exposing *capability*."""
    return CAPABILITY_GRAPH.get(capability, []).copy()


def get_agent(name: str, **kwargs):
    """Instantiate agent by name and wrap run-cycle with heartbeat probe."""
    meta  = AGENT_REGISTRY[name]
    agent = meta.instantiate(**kwargs)

    if hasattr(agent, "step") and inspect.iscoroutinefunction(agent.step):
        orig = agent.step

        async def _wrapped(*a, **kw):             # type: ignore[no-untyped-def]
            t0 = time.perf_counter()
            ok = True
            try:
                return await orig(*a, **kw)       # type: ignore[misc]
            except Exception:                     # noqa: BLE001
                ok = False
                raise
            finally:
                _HEALTH_Q.put((meta.name, (time.perf_counter()-t0)*1000, ok))

        agent.step = _wrapped                     # type: ignore[assignment]

    return agent


def register_agent(meta: AgentMetadata, *, overwrite: bool = False):
    """Public hook: dynamically created agents can self-register at runtime."""
    _register(meta, overwrite=overwrite)

##############################################################################
#                           initial discovery pass                           #
##############################################################################
_discover_local()
_discover_entrypoints()
_discover_hot_dir()
_discover_adk()

logger.info(
    "🚀 Agent registry ready – %3d agents, %3d distinct capabilities",
    len(AGENT_REGISTRY),
    len(CAPABILITY_GRAPH),
)
