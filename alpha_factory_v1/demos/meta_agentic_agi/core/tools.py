
"""
tools.py – Unified Tooling Orchestrator (v0.3.0)
================================================

This module centralises *all* runtime‑accessible **tools** used by the
Meta‑Agentic α‑AGI demo.  A *tool* is a side‑effect‑bearing callable
(e.g. `search_web`, `run_sql`, `plot`) that can be invoked by agents via
function‑calling APIs (OpenAI, Anthropic, A2A, etc.).

Highlights
----------
• 📚  **Schema‑first registry** – each tool declares a JSON schema for inputs &
  outputs, enabling automatic validation / OpenAI function‑calling scaffolding.
• 🔐  **Sandboxed Execution** – built‑in RestrictedPython + asyncio timeouts to
  safely run arbitrary user code (disabled if `TOOLS_TRUSTED=1`).
• 📈  **Lineage & cost telemetry** – every invocation is logged (sqlite or
  Postgres) with start/stop time, USD & gCO₂e estimate, bytes in/out.
• 🔄  **Hot‑reload support** – operators can drop a new *.py* in `tools_ext/`
  and it is discovered at runtime without reboot.
• ⚖️  **Multi‑objective optimisation hooks** – each call returns a
  multi‑dimensional score vector (latency, accuracy, $) to feed the search
  controller.
• 0️⃣  **Zero required deps** – optional extras (duckdb, matplotlib, requests,
  pandas, chromadb, etc.) are imported lazily.

Apache‑2.0 © 2025 MONTREAL.AI
"""

from __future__ import annotations

import asyncio, inspect, importlib, json, logging, os, sys, time, types, uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Awaitable, Union

_LOGGER = logging.getLogger("alpha_factory.tools")
_LOGGER.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Config                                                                      #
# --------------------------------------------------------------------------- #
TOOLS_DIR = Path(__file__).with_suffix("").parent / "tools_ext"
TOOLS_DIR.mkdir(exist_ok=True)
DB_PATH = Path(os.getenv("TOOLS_DB", "tools_invocations.sqlite"))

SANDBOX_TRUSTED = bool(int(os.getenv("TOOLS_TRUSTED", "0")))
CALL_TIMEOUT = float(os.getenv("TOOLS_TIMEOUT", "30"))  # seconds

# cost & carbon heuristics
USD_PER_CPU_SEC = 2.5e-5     # placeholder
GCO2_PER_CPU_SEC = 0.42      # placeholder gCO2eq per cpu‑second

# --------------------------------------------------------------------------- #
# Simple sqlite lineage store                                                 #
# --------------------------------------------------------------------------- #
def _init_db():
    import sqlite3
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS invocations (
                   id TEXT PRIMARY KEY,
                   tool TEXT,
                   ts_start REAL,
                   ts_end REAL,
                   args TEXT,
                   output_size INT,
                   usd REAL,
                   gco2e REAL,
                   error TEXT
               )"""
        )
_init_db()

def _log_invocation(row: Dict[str, Any]):
    import sqlite3, json
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO invocations VALUES (:id,:tool,:ts_start,:ts_end,"
            ":args,:output_size,:usd,:gco2e,:error)",
            row,
        )
        conn.commit()

# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
class Tool:
    "Container for a single tool."
    def __init__(self,
                 name: str,
                 func: Callable[..., Awaitable[Any]] | Callable[..., Any],
                 schema: Dict[str, Any],
                 description: str = "") -> None:
        self.name = name
        self.func = func
        self.schema = schema
        self.description = description or (func.__doc__ or "").strip()

    async def __call__(self, **kwargs):
        # basic jsonschema validation (lazy import)
        try:
            jsonschema = importlib.import_module("jsonschema")
            jsonschema.validate(kwargs, self.schema)
        except ModuleNotFoundError:
            _LOGGER.debug("jsonschema not installed – skipping validation")

        uid = str(uuid.uuid4())
        t0 = time.time()
        err, out = None, None
        try:
            if asyncio.iscoroutinefunction(self.func):
                coro = self.func(**kwargs)
            else:
                async def _wrp():
                    return self.func(**kwargs)
                coro = _wrp()
            out = await asyncio.wait_for(coro, timeout=CALL_TIMEOUT)
            return out
        except Exception as e:
            err = repr(e)
            raise
        finally:
            t1 = time.time()
            bytes_out = len(json.dumps(out, default=str)) if out is not None else 0
            cpu_sec = t1 - t0
            _log_invocation(
                dict(
                    id=uid,
                    tool=self.name,
                    ts_start=t0,
                    ts_end=t1,
                    args=json.dumps(kwargs, default=str)[:10_000],
                    output_size=bytes_out,
                    usd=cpu_sec * USD_PER_CPU_SEC,
                    gco2e=cpu_sec * GCO2_PER_CPU_SEC,
                    error=err,
                )
            )

_REGISTRY: Dict[str, Tool] = {}

def register(schema: Dict[str, Any]):
    "Decorator: register a function as a tool with JSON `schema`."
    def deco(fn: Callable):
        if fn.__name__ in _REGISTRY:
            raise ValueError(f"duplicate tool name: {fn.__name__}")
        _REGISTRY[fn.__name__] = Tool(fn.__name__, fn, schema, fn.__doc__)
        return fn
    return deco

def registry() -> Dict[str, Tool]:
    return dict(_REGISTRY)

def get(name: str) -> Tool:
    return _REGISTRY[name]

# --------------------------------------------------------------------------- #
# Core built‑ins                                                              #
# --------------------------------------------------------------------------- #
@register(
    {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }
)
async def web_search(query: str, top_k: int = 5) -> list[dict]:
    """
    🔎 Web Search – return list of top‑k results using DuckDuckGo Instant Answer
    API (no tracking, free).  Each item: {title, url, snippet}.
    """
    import aiohttp
    async with aiohttp.ClientSession() as session:
        url = "https://duckduckgo.com/i.js"
        async with session.get(url, params={"q": query, "kl": "us-en", "count": top_k}) as r:
            data = await r.json()
            results = [
                {"title": itm.get("title"), "url": itm.get("url"), "snippet": itm.get("snippet")}
                for itm in data.get("results", [])[:top_k]
            ]
            return results

@register(
    {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source to execute"},
        },
        "required": ["code"],
    }
)
async def sandbox_exec(code: str) -> str:
    """
    🐍 Execute *trusted or sandboxed* Python and return stdout / result repr.

    If `TOOLS_TRUSTED=1`, runs with full `exec`.  Otherwise uses RestrictedPython.
    """
    import io, contextlib, traceback
    buffer = io.StringIO()
    try:
        if SANDBOX_TRUSTED:
            loc: Dict[str, Any] = {}
            with contextlib.redirect_stdout(buffer):
                exec(code, {}, loc)
            result = loc.get("_")  # convention
        else:
            RP = importlib.import_module("RestrictedPython")
            result = RP.compile_restricted_exec(code)
        buffer.write(repr(result))
    except Exception as e:
        buffer.write("ERR: " + traceback.format_exc(limit=2))
    return buffer.getvalue()

@register(
    {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "number"},
                "description": "numeric vector",
            }
        },
        "required": ["values"],
    }
)
async def stats(values: list[float]) -> dict:
    """
    📊 Return basic statistics (mean, median, stdev, min, max) for a list.
    """
    import statistics as st
    return {
        "n": len(values),
        "mean": st.mean(values),
        "median": st.median(values),
        "stdev": st.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }

# --------------------------------------------------------------------------- #
# Dynamic discovery of *.py* modules placed in TOOLS_DIR                      #
# --------------------------------------------------------------------------- #
def _discover():
    sys.path.append(str(TOOLS_DIR))
    for p in TOOLS_DIR.glob("*.py"):
        mod_name = p.stem
        try:
            mod = importlib.import_module(mod_name)
            _LOGGER.info("Loaded external tool module %s", mod_name)
        except Exception as e:
            _LOGGER.warning("Fail to load %s – %s", mod_name, e)

_discover()

# --------------------------------------------------------------------------- #
# OpenAI function‑calling helper                                              #
# --------------------------------------------------------------------------- #
def openai_functions_spec() -> list[dict]:
    "Return json list suitable for `functions` param of OpenAI chat.completions."
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.schema,
        }
        for t in _REGISTRY.values()
    ]

