"""
╭──────────────────────────────────────────────────────────────────────────────╮
│  Alpha-Factory v1 👁️✨ — Multi-Agent AGENTIC α-AGI • World-Model Demo        │
│  ░░  “Outlearn · Outthink · Outdesign · Outstrategize · Outexecute”  ░░      │
│                                                                              │
│  Single-import convenience wrapper around `alpha_asi_world_model_demo.py`.   │
│  Runs completely offline; transparently augments itself with GPT/Claude etc. │
│  if API keys are present.                                                    │
╰──────────────────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import importlib, os, threading
from types import ModuleType
from typing import Final, List

# re-export orchestrator & FastAPI app
from .alpha_asi_world_model_demo import Orchestrator, app, _main as _demo_cli  # noqa: F401

__all__: Final[List[str]] = ["Orchestrator", "run_headless", "run_ui", "app", "__version__", "Agent_DOC"]
__version__: Final[str]   = "1.0.1"

# ──────────────────────────────────────────────────────────────────────────────
#  Human-readable agent showcase
# ──────────────────────────────────────────────────────────────────────────────
Agent_DOC = """
Integrated Alpha-Factory agents (stubbed if source class missing):

• PlanningAgent        — decomposes objectives into actionable plans  
• ResearchAgent        — literature/data scouting → distilled insights  
• StrategyAgent        — converts insights into competitive moves  
• MarketAnalysisAgent  — assesses financial / market impact  
• SafetyAgent          — continuous risk monitor; halts on anomaly

These map to the Alpha-Pipeline:  Detect → Research → Strategise → Execute → Safeguard.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  Convenience helpers
# ──────────────────────────────────────────────────────────────────────────────
def _lazy_uvicorn() -> ModuleType: return importlib.import_module("uvicorn")

def run_headless(steps: int = 50_000) -> Orchestrator:
    """Run the orchestrator loop without the web server (useful for tests)."""
    orch = Orchestrator()
    threading.Thread(target=orch.loop, kwargs={"steps":steps}, daemon=True).start()
    return orch

def run_ui(host: str = "127.0.0.1", port: int = 7860, reload: bool=False, log_level: str="info") -> None:
    """Spin up the FastAPI UI."""
    _lazy_uvicorn().run(
        "alpha_factory_v1.demos.alpha_asi_world_model.alpha_asi_world_model_demo:app",
        host=host, port=port, reload=reload, log_level=log_level,
    )

# informative banner (printed once)
if os.getenv("ALPHA_ASI_SILENT","0") != "1":
    print(f"\n💡  Alpha-ASI demo ready — v{__version__} • "
          "call `alpha_asi_world_model.run_ui()` or see `help(alpha_asi_world_model)`.\n")

# allow `python -m alpha_asi_world_model`
def _module_cli(): _demo_cli()
if __name__ == "__main__": _module_cli()
