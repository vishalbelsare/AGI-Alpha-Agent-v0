
"""
backend/agent_factory.py
────────────────────────
Central factory helpers that create **ready-for-production** agents for
Alpha-Factory v1 👁️✨.

✨  Design goals
•  Zero-config demos – runs *with or without* an OPENAI_API_KEY and even
   if the OpenAI Agents SDK itself is missing.
•  Hardened defaults – only safe, audited tools are enabled unless the user
   explicitly opts-in to local code execution.
•  Single source of truth – every domain-specific agent (FinanceAgent,
   SupplyChainAgent, …) should import *one* function from here so the whole
   stack stays consistent.
•  Graceful degradation – when something is unavailable we fall back to a
   lightweight stub instead of crashing the orchestrator.

Typical usage
=============
```python
from backend.agent_factory import build_core_agent

sentinel = build_core_agent(
    name="Macro-Sentinel",
    instructions="Monitor global news and hedge the portfolio.",
)
print(sentinel.run("What is headline risk right now?"))
```
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence

# ╭──────────────────────────────────────────────────────────────────────╮
# │ 1 ▸ Attempt to import the OpenAI Agents SDK                          │
# ╰──────────────────────────────────────────────────────────────────────╯
try:
    agents_sdk = importlib.import_module("agents")  # noqa: F401
    from agents import (  # type: ignore
        Agent,
        FileSearchTool,
        WebSearchTool,
        ComputerTool,
        PythonTool,
        ModelSettings,
        RunContextWrapper,
    )

    SDK_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    # ▶ The machine does not have the SDK – we create *tiny* stubs so that
    #   the rest of Alpha-Factory continues to import & run.
    SDK_AVAILABLE = False

    class _StubTool:  # noqa: D401
        """Fallback that simply reports unavailability."""

        def __init__(self, *_, **__):
            self.name = self.__class__.__name__

        def __call__(self, *_a, **_kw):  # noqa: D401
            return f"[{self.name} unavailable – install `openai-agents`]"

    class FileSearchTool(_StubTool):  # type: ignore
        pass

    class WebSearchTool(_StubTool):  # type: ignore
        pass

    class ComputerTool(_StubTool):  # type: ignore
        pass

    class PythonTool(_StubTool):  # type: ignore
        pass

    class ModelSettings:  # type: ignore
        def __init__(self, **__): ...

    class Agent:  # type: ignore
        """Minimal stand-in so demos never crash."""

        def __init__(
            self,
            name: str,
            instructions: str,
            model: str,
            model_settings: 'ModelSettings' | None = None,
            tools: Sequence[Any] | None = None,
        ):
            self.name = name
            self.instructions = instructions
            self.model = model
            self._tools = list(tools or [])

        def run(self, prompt: str, *_, **__) -> str:  # noqa: D401
            return f"[{self.name} -- stub agent] echo: {prompt}"

        chat_stream = run  # SDK compatibility


# ╭──────────────────────────────────────────────────────────────────────╮
# │ 2 ▸ Alpha-Factory internal tools                                    │
# ╰──────────────────────────────────────────────────────────────────────╯
try:
    # Always present because we ship it in backend/tools/
    from .tools.local_pytest import run_pytest
except Exception as exc:  # pragma: no cover
    # Should never happen, but keep startup resilient.
    def run_pytest(*_, **__) -> str:  # type: ignore
        return f"[local_pytest failed to load: {exc}]"


# ╭──────────────────────────────────────────────────────────────────────╮
# │ 3 ▸ Helper – pick a sensible default model                          │
# ╰──────────────────────────────────────────────────────────────────────╯
def _auto_select_model() -> str:
    """
    Decide which model string to pass to the agent constructor.

    Preference order
    ----------------
    1. ``OPENAI_MODEL`` env override
    2. If an OpenAI API key exists → ``gpt-4o-mini``
    3. If ``LLAMA_CPP_MODEL`` path provided → ``local-llama3-8b-q4``
    4. Fallback stub id → ``local-sbert``
    """
    override = os.getenv("OPENAI_MODEL")
    if override:
        return override

    if os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"

    if os.getenv("LLAMA_CPP_MODEL"):
        return "local-llama3-8b-q4"

    return "local-sbert"


# ╭──────────────────────────────────────────────────────────────────────╮
# │ 4 ▸ Canonical default tool-chain                                    │
# ╰──────────────────────────────────────────────────────────────────────╯
DEFAULT_TOOLS: List[Any] = [
    FileSearchTool(max_num_results=5),
    WebSearchTool(),
    run_pytest,
]

# High-risk code-execution tools are enabled *only* when the user
# explicitly allows them or when they are safely executed in OpenAI’s
# remote sandbox infrastructure.
ALLOW_LOCAL_CODE = os.getenv("ALPHAFAC_ALLOW_LOCAL_CODE") == "1"
if SDK_AVAILABLE and os.getenv("OPENAI_API_KEY"):
    DEFAULT_TOOLS.append(ComputerTool())
    if ALLOW_LOCAL_CODE:
        # PythonTool still runs locally – keep it behind an extra flag.
        DEFAULT_TOOLS.append(PythonTool())


# ╭──────────────────────────────────────────────────────────────────────╮
# │ 5 ▸ Public factory helpers                                          │
# ╰──────────────────────────────────────────────────────────────────────╯
def build_core_agent(
    *,
    name: str,
    instructions: str,
    extra_tools: Optional[Sequence[Any]] = None,
    model: Optional[str] = None,
    temperature: float = 0.30,
) -> Agent:
    """
    Create a fully configured **Agent** (or stub) ready for orchestration.
    """
    toolset: List[Any] = list(DEFAULT_TOOLS)
    if extra_tools:
        toolset.extend(extra_tools)

    selected_model = model or _auto_select_model()

    return Agent(
        name=name,
        instructions=instructions,
        model=selected_model,
        model_settings=ModelSettings(temperature=temperature),
        tools=toolset,
    )


def save_agent_manifest(agent: Agent, path: str | Path) -> None:
    """Persist an agent manifest (JSON) for auditing or sharing."""
    manifest = {
        "name": getattr(agent, "name", ""),
        "instructions": getattr(agent, "instructions", ""),
        "model": getattr(agent, "model", ""),
        "tools": [getattr(t, "name", str(t)) for t in getattr(agent, "_tools", [])],
    }
    Path(path).expanduser().write_text(json.dumps(manifest, indent=2))


# Backwards-compat helper – old notebooks call `build_agent(…)`
build_agent = build_core_agent

__all__ = [
    "build_core_agent",
    "build_agent",
    "save_agent_manifest",
    "DEFAULT_TOOLS",
    *(
        ["Agent", "FileSearchTool", "WebSearchTool", "ComputerTool", "PythonTool"]
        if SDK_AVAILABLE
        else []
    ),
]
