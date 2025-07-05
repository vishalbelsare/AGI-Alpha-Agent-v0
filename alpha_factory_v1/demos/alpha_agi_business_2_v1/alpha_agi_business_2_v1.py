#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# NOTE: This demo is a research prototype. References to "AGI" or "superintelligence" describe aspirational goals and do not indicate the presence of real general intelligence. Use at your own risk. Nothing herein constitutes financial advice.
"""Alpha‑AGI Business v2 demo.

Boots the Alpha‑Factory orchestrator with two minimal agents. The demo
operates fully offline but automatically upgrades to cloud LLM providers
when ``OPENAI_API_KEY`` is detected.
"""
from __future__ import annotations

import argparse
import logging
import os

from alpha_factory_v1.backend import orchestrator
from alpha_factory_v1.backend.agents import (
    AgentBase,
    AgentMetadata,
    register_agent,
)


class PlanningAgent(AgentBase):
    """Toy planner agent emitting a single planning message."""

    NAME = "planning"
    CAPABILITIES = ["plan"]
    __slots__ = ()

    async def step(self) -> None:
        """Publish a mock optimal plan."""
        await self.publish("alpha.plan", {"plan": "explore_market"})


class ResearchAgent(AgentBase):
    """Toy researcher agent publishing summarised findings."""

    NAME = "research"
    CAPABILITIES = ["research"]
    CYCLE_SECONDS = 90
    __slots__ = ()

    async def step(self) -> None:
        """Publish a mock research summary."""
        await self.publish("alpha.research", {"summary": "market stable"})


class LLMCommentAgent(AgentBase):
    """Optional LLM-powered commentary agent."""

    NAME = "llm_comment"
    CAPABILITIES = ["insight"]
    REQUIRES_API_KEY = False
    __slots__ = ()

    async def step(self) -> None:
        """Post a short market insight using OpenAI Agents if available."""
        insight = "LLM unavailable"
        try:
            from openai_agents import OpenAIAgent

            agent = OpenAIAgent(
                model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=None if os.getenv("OPENAI_API_KEY") else "http://ollama:11434/v1",
            )
            insight = await agent("One sentence on today's market outlook")
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("LLM fallback: %s", exc)
        await self.publish("alpha.insight", {"insight": insight})


def register_demo_agents() -> None:
    """Register built-in demo agents with the framework."""

    register_agent(
        AgentMetadata(
            name=PlanningAgent.NAME,
            cls=PlanningAgent,
            version="1.0.0",
            capabilities=PlanningAgent.CAPABILITIES,
        )
    )

    register_agent(
        AgentMetadata(
            name=LLMCommentAgent.NAME,
            cls=LLMCommentAgent,
            version="1.0.0",
            capabilities=LLMCommentAgent.CAPABILITIES,
            requires_api_key=LLMCommentAgent.REQUIRES_API_KEY,
        )
    )

    register_agent(
        AgentMetadata(
            name=ResearchAgent.NAME,
            cls=ResearchAgent,
            version="1.0.0",
            capabilities=ResearchAgent.CAPABILITIES,
        )
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Return parsed CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the α‑AGI Business demo")
    parser.add_argument(
        "--loglevel",
        default=os.getenv("LOGLEVEL", "INFO"),
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Launch the orchestrator with demo agents registered."""

    args = _parse_args(argv)
    logging.basicConfig(
        level=args.loglevel.upper(),
        format="%(asctime)s %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    register_demo_agents()

    try:
        orchestrator.Orchestrator().run_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
