# backend/orchestrator.py
import logging
import threading
import time

from .memory import Memory
from .model_provider import ModelProvider
from .governance import Governance

# domain specialists
from .finance_agent import FinanceAgent
from .biotech_agent import BiotechAgent
from .policy_agent import PolicyAgent
from .manufacturing_agent import ManufacturingAgent

# planner
from .planner_agent import PlannerAgent


class Orchestrator:
    """
    Spins up all domain agents, hands control to the Planner, and
    keeps looping forever.  Only the Planner is scheduled directly;
    it decides which specialist to run each cycle.
    """

    def __init__(self, cycle_seconds: int = 30):
        logging.basicConfig(level=logging.INFO)

        # shared services
        self.memory = Memory()
        self.model = ModelProvider()
        self.gov = Governance(self.memory)

        # domain agents
        self.fin = FinanceAgent("FinanceAgent", self.model, self.memory, self.gov)
        self.bio = BiotechAgent("BiotechAgent", self.model, self.memory, self.gov)
        self.pol = PolicyAgent("PolicyAgent", self.model, self.memory, self.gov)
        self.mfg = ManufacturingAgent("ManufacturingAgent", self.model, self.memory, self.gov)

        # planner orchestrates them
        self.planner = PlannerAgent(
            "PlannerAgent",
            self.model,
            self.memory,
            self.gov,
            domain_agents=[self.fin, self.bio, self.pol, self.mfg],
        )

        self.agents = [self.planner]
        self.interval = cycle_seconds

        logging.getLogger("Orchestrator").info(
            "Initialized agents: %s", [a.name for a in self.planner.domain_agents]
        )

    # ────────────────────────────────────────────────────────────────
    def run_forever(self):
        while True:
            for agent in self.agents:
                threading.Thread(target=agent.run_cycle, daemon=True).start()
            time.sleep(self.interval)


if __name__ == "__main__":
    Orchestrator().run_forever()

