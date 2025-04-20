# backend/agent_base.py
import abc
import datetime
import logging
import uuid
from typing import Any, Dict, List

from .tracer import Tracer

class AgentBase(abc.ABC):
    """
    Shared skeleton for every domain agent:

        • observe() → collect data
        • think()   → propose tasks / ideas
        • act()     → execute vetted tasks

    Each phase is traced and persisted so evaluation harnesses can
    replay or diff behaviour across versions.
    """

    # ────────────────────────────────────────────────────────────────
    def __init__(self, name: str, model, memory, gov):
        self.id = str(uuid.uuid4())
        self.name = name
        self.model = model
        self.memory = memory
        self.gov = gov

        self.log = logging.getLogger(name)
        self.tracer = Tracer(self.memory)

    # ───────── framework hooks (must be implemented) ───────────────
    @abc.abstractmethod
    def observe(self) -> List[Dict[str, Any]]:
        ...

    @abc.abstractmethod
    def think(self, observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ...

    @abc.abstractmethod
    def act(self, tasks: List[Dict[str, Any]]) -> None:
        ...

    # ───────── single life‑cycle ───────────────────────────────────
    def run_cycle(self) -> None:
        ts = datetime.datetime.utcnow().isoformat()
        self.log.info("%s cycle start", self.name)

        try:
            observations = self.observe()
            self.tracer.record(self.name, "observe", observations)

            ideas = self.think(observations)
            self.tracer.record(self.name, "think", ideas)

            vetted = self.gov.vet_plans(self, ideas)
            self.tracer.record(self.name, "vet", vetted)

            self.act(vetted)
            self.tracer.record(self.name, "act", vetted)

        except Exception as err:
            self.log.exception("Cycle error: %s", err)
            self.memory.write(
                self.name, "error", {"msg": str(err), "ts": ts}
            )

        self.log.info("%s cycle end", self.name)

