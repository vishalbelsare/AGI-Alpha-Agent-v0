import json
import logging
import random
from typing import List, Dict, Any

from .agent_base import AgentBase


class PlannerAgent(AgentBase):
    """
    LLM‑driven scheduler that decides which domain agent should run next.

    • Uses ModelProvider.complete() to ask an LLM for a JSON decision.
    • Falls back to a random agent if no LLM backend or the response is invalid.
    • Writes a “dispatch” record to memory, then triggers the chosen agent.
    """

    def __init__(self, *args, domain_agents, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_agents = list(domain_agents)
        self.log = logging.getLogger("PlannerAgent")

    # ──────────────────────────────────────────
    # Planner loop
    # ──────────────────────────────────────────
    def observe(self) -> List[Dict[str, Any]]:
        # No external observations yet (could ingest metrics later)
        return []

    def think(self, _obs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        agent_names = [a.name for a in self.domain_agents]
        prompt = (
            "You are the Planner.\n"
            f"Available agents: {', '.join(agent_names)}.\n"
            "Decide which agent to trigger next and why. "
            'Respond in JSON: {"agent":"<name>","reason":"<short explanation>"}'
        )

        # Ask the LLM; handle any backend / JSON issues gracefully
        try:
            resp = self.model.complete(prompt)
            data = json.loads(resp)
            if data.get("agent") not in agent_names:
                raise ValueError("unknown agent")
        except Exception as err:
            self.log.warning("Planner fallback due to %s", err)
            data = {
                "agent": random.choice(agent_names),
                "reason": "fallback (no LLM or invalid response)",
            }

        return [data]

    def act(self, tasks: List[Dict[str, Any]]) -> None:
        target_name = tasks[0]["agent"]
        target = next((a for a in self.domain_agents if a.name == target_name), None)
        if not target:
            self.log.warning("Planner chose unknown agent %s", target_name)
            return

        # Log the dispatch for auditability
        self.memory.write(self.name, "dispatch", tasks[0])

        # Run the selected agent’s cycle (synchronously; could be threaded)
        target.run_cycle()

