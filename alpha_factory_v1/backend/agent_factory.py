"""
Factory helpers that return pre-configured OpenAI Agents for the
Alpha-Factory demos.  Import from elsewhere with:

    from backend.agent_factory import build_core_agent
"""

import os
from typing import List, Optional

from agents import (
    Agent,
    FileSearchTool,
    WebSearchTool,
    ComputerTool,
    ModelSettings,
    Runner,
)

from .tools.local_pytest import run_pytest

# ----------------------------------------------------------------------
#  Default tool-chain (auto-adjusts to API-key presence)
# ----------------------------------------------------------------------
DEFAULT_TOOLS: List = [
    WebSearchTool(),
    FileSearchTool(max_num_results=5),
    run_pytest,                  # local FunctionTool – always available
]

if os.getenv("OPENAI_API_KEY"):
    # ComputerTool runs in OpenAI's remote sandbox
    DEFAULT_TOOLS.append(ComputerTool())


def build_core_agent(
    *,
    name: str,
    instructions: str,
    extra_tools: Optional[List] = None,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
) -> Agent:
    """Return a production-ready Agent instance."""
    tools = DEFAULT_TOOLS.copy()
    if extra_tools:
        tools.extend(extra_tools)

    return Agent(
        name=name,
        instructions=instructions,
        model=model,
        model_settings=ModelSettings(temperature=temperature),
        tools=tools,
    )

