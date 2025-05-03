# SPDX-License-Identifier: Apache-2.0
"""
alpha_factory_v1.backend.world_model
====================================

Latent World-Model & Planner (v1.0.0, 2025-05-02)
──────────────────────────────────────────────────
▸ 5×5 Grid-World reference env  (used by demo #6)  
▸ MuZero-general planner  (soft-import, depth-limited)  
▸ LLM-based counter-factual simulator fallback (GPT-4 / local-LLM)  
▸ Safety-Guard vetoes risky trajectories (configurable)  
▸ Meta-Learner monitors agent KPI stream  → proposes new agents / hyper-params  

All heavy deps are **optional**.  Everything degrades to in-proc heuristics so
the Alpha-Factory never crashes, even on a Raspberry Pi, air-gapped, or without
an OpenAI API key.
"""
from __future__ import annotations

# ───────────────────────────── std-lib ───────────────────────────────
import contextlib
import importlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ────────────────────────── soft-imports ─────────────────────────────
with contextlib.suppress(ModuleNotFoundError):
    # MuZero-general (https://github.com/werner-duvaud/muzero-general)
    from muzero import muzero  # type: ignore

with contextlib.suppress(ModuleNotFoundError):
    # OpenAI LLM for language-simulation fallback
    import openai  # type: ignore[attr-defined]

with contextlib.suppress(ModuleNotFoundError):
    # Google ADK telemetry (optional)
    import adk  # type: ignore

# ─────────────────────── configuration ──────────────────────────────
ENV = os.getenv
DEV_MODE = ENV("DEV_MODE", "false").lower() == "true"
GRID_SIZE = int(ENV("WM_GRID_SIZE", "5"))              # NxN grid
MCTS_SIMS = int(ENV("WM_MCTS_SIMULATIONS", "50"))      # MuZero search budget
LLM_MODEL = ENV("WM_LLM_MODEL", "gpt-4o-mini")         # override if needed
RISK_THRESHOLD = float(ENV("WM_RISK_THRESHOLD", "-50"))  # min cumulative reward

# Safety-Guard rule-set (expandable via ENV JSON if desired)
_DEFAULT_RULES = {
    "max_position": 0.25,   # e.g. Finance: max 25 % of NAV in one asset
    "max_overtime": 12,     # e.g. Manufacturing: max 12 h/day
}
RULES = json.loads(ENV("WM_SAFETY_RULES", json.dumps(_DEFAULT_RULES)))

# ───────────────── Grid-World reference env ─────────────────────────
@dataclass
class GridState:
    x: int = 0
    y: int = 0
    goal: Tuple[int, int] = (GRID_SIZE - 1, GRID_SIZE - 1)
    reward: float = 0.0
    done: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return dict(x=self.x, y=self.y, goal=self.goal, reward=self.reward, done=self.done)


class GridWorldEnv:  # minimal OpenAI-Gym-like API
    ACTIONS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
    def __init__(self, size: int = GRID_SIZE) -> None:
        self.size = size
        self.state = GridState()

    # gym-style helpers ------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.state = GridState()
        return self.state.to_dict()

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if action not in self.ACTIONS or self.state.done:
            return self.state.to_dict(), 0.0, self.state.done, {}
        dx, dy = self.ACTIONS[action]
        self.state.x = max(0, min(self.size - 1, self.state.x + dx))
        self.state.y = max(0, min(self.size - 1, self.state.y + dy))
        self.state.reward -= 1  # time penalty
        if (self.state.x, self.state.y) == self.state.goal:
            self.state.reward += 100
            self.state.done = True
        return self.state.to_dict(), self.state.reward, self.state.done, {}

# ─────────────── Safety-Guard  (rule-based for now) ─────────────────
class SafetyGuard:
    def __init__(self, rules: Dict[str, Any]) -> None:
        self.rules = rules

    def veto(self, rewards: List[float]) -> bool:
        if sum(rewards) < RISK_THRESHOLD:
            return True
        # Extend with domain-specific rules (position limits, overtime, etc.)
        return False

# ───────────── MuZero / MCTS planner wrapper (soft) ─────────────────
class MuZeroPlanner:
    def __init__(self) -> None:
        self.enabled = 'muzero' in globals()
        if self.enabled:
            # configure a minimal MuZero model for GridWorld
            conf = Path(__file__).with_name("muzero_grid_config.py")
            if not conf.exists():
                conf.write_text(_GEN_MUZERO_CFG)
            self._mz = muzero.MuZero(config=str(conf))  # type: ignore[arg-type]

    def plan(self, env: GridWorldEnv, sims: int = MCTS_SIMS) -> Tuple[str, List[str]]:
        if not self.enabled:
            # fallback: heuristic (move towards goal greedily)
            gx, gy = env.state.goal
            dx = "RIGHT" if env.state.x < gx else "LEFT"
            dy = "DOWN" if env.state.y < gy else "UP"
            return (dx if random.random() < 0.5 else dy), []
        trajectory, _ = self._mz.plan(env, sims)  # type: ignore[attr-defined]
        return trajectory[0], trajectory  # first action + full plan

# ─────────────── LLM-based counter-factual simulator ────────────────
class LLMSim:
    def __init__(self) -> None:
        self.enabled = 'openai' in globals() and bool(ENV("OPENAI_API_KEY"))
        if self.enabled:
            openai.api_key = ENV("OPENAI_API_KEY")

    def predict(self, prompt: str) -> str:
        if not self.enabled:
            return "LLM unavailable – using heuristic outcome."
        resp = openai.ChatCompletion.create(  # type: ignore[attr-defined]
            model=LLM_MODEL,
            temperature=0.3,
            max_tokens=64,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["choices"][0]["message"]["content"]  # type: ignore[index]

# ─────────────── Meta-Learner (AI-GA Autogenesis) ───────────────────
class MetaLearner:
    """Monitors KPI stream; proposes new agents / hyper-params as Markdown."""

    def __init__(self) -> None:
        self.last_proposal_ts = 0.0
        self.interval_sec = int(ENV("META_INTERVAL_SEC", "3600"))

    def maybe_propose(self, kpis: Dict[str, float]) -> Optional[str]:
        now = time.time()
        if now - self.last_proposal_ts < self.interval_sec:
            return None
        self.last_proposal_ts = now

        worst_agent = min(kpis, key=kpis.get)  # simplistic
        hypothesis = (
            f"Agent **{worst_agent}** under-performs (score {round(kpis[worst_agent],2)}).  \n"
            f"→ _Proposal_: spawn `ResearchAgent` with `domain={worst_agent}` to explore "
            "fringe strategies using evolutionary search.\n"
            "- Increase exploration-rate ε to 0.4 for next 1 h.\n"
            "- Log novel trajectories to Kafka topic `exp.fringe`."
        )
        Path("NewAgentProposal.md").write_text(hypothesis)
        return hypothesis

# ──────────────────── World-Model service (singleton) ───────────────
class WorldModel:
    def __init__(self) -> None:
        self.env = GridWorldEnv()
        self.planner = MuZeroPlanner()
        self.llm = LLMSim()
        self.guard = SafetyGuard(RULES)
        self.meta = MetaLearner()

    # ---------------- high-level API ---------------------------------
    def reset_env(self) -> Dict[str, Any]:
        return self.env.reset()

    def simulate(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        """Lightweight forward model (heuristic or LLM)."""
        # fast path: Grid-World internal simulation
        if {"x", "y", "goal"}.issubset(state) and "dir" in action:
            self.env.state = GridState(**state)
            ns, r, done, _ = self.env.step(action["dir"])
            return dict(state=ns, reward=r, done=done)

        # generic fallback → ask LLM
        prompt = (
            f"Current state JSON:\n{json.dumps(state)}\n\n"
            f"Action JSON:\n{json.dumps(action)}\n\n"
            "Predict the **next state** (JSON) and **numerical reward**. "
            "Keep keys identical; include `done` boolean."
        )
        reply = self.llm.predict(prompt)
        with contextlib.suppress(json.JSONDecodeError):
            return json.loads(reply)
        # heuristic if LLM failed
        return dict(state=state, reward=0.0, done=False, note="heuristic-noop")

    def plan(self, agent: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return best immediate action + imagined trajectory."""
        self.env.state = GridState(**{k: state[k] for k in ("x", "y", "goal")})
        a, traj = self.planner.plan(self.env)
        rewards = [-1] * len(traj)  # pessimistic est.
        if self.guard.veto(rewards):
            return {"action": None, "reason": "safety_guard_veto", "trajectory": traj}
        return {"action": {"dir": a}, "trajectory": traj}

    # ---------------- meta-learning hook -----------------------------
    def ingest_kpis(self, kpis: Dict[str, float]) -> Optional[str]:
        """Feed KPI dict {agent: score}.  Returns Markdown proposal if any."""
        return self.meta.maybe_propose(kpis)

# ─────────────────── global singleton export ────────────────────────
wm = WorldModel()

# ─── minimal MuZero config auto-generated on first import (5×5 grid) ─
_GEN_MUZERO_CFG = r"""
###  Auto-generated minimal MuZero config for 5×5 Grid-World. ###
###  Just enough to run shallow MCTS planning -- not for full RL training. ###
game_name = "gridworld"
observation_shape = (1, 5, 5)
action_space = ["UP", "DOWN", "LEFT", "RIGHT"]
players = [0]
stacked_observations = 0
train_steps = 0
num_actors = 0
max_moves = 50
temperature_threshold = 15    # low exploration (planning, not training)
selfplay_on_gpu = False
"""
