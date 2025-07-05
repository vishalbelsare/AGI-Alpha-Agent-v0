# SPDX-License-Identifier: Apache-2.0
"""Minimal stub for openai_agents.

Provides basic classes so demos import without the real SDK."""

__version__ = "0.0.0"


class Agent:
    pass


class AgentRuntime:
    def __init__(self, *args, **kwargs):
        pass

    def register(self, *args, **kwargs):
        pass


class OpenAIAgent:
    def __init__(self, *args, **kwargs):
        pass

    async def __call__(self, text: str) -> str:  # pragma: no cover - demo stub
        return "ok"


def Tool(*_args, **_kwargs):
    def decorator(func):
        return func

    return decorator
