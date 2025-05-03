# SPDX-License-Identifier: Apache-2.0
"""
OMNI-Factory · Smart-City Resilience Demo
═════════════════════════════════════════
A complete open-ended task-generation / multi-agent execution loop that plugs
directly into Alpha-Factory v1.

Key modules
───────────
 • TaskGenerator…… Foundation-model or rule-based scenario synthesis
 • Interestingness… MoI post-filter that prevents stale / redundant tasks
 • SmartCityEnv……  Fast, headless city “digital-twin” (traffic + grid)
 • SuccessEvaluator  Universal success / reward / CityCoin ledger
 • Orchestrator shim Bridges generated tasks to existing Alpha agents

Offline-first:   runs with *no* API key.  
Online-enhanced:  set `OPENAI_API_KEY` for GPT-4o-mini powered variety.

Launch from repo root
─────────────────────
    python -m alpha_factory_v1.demos.omni_factory_demo
"""
from __future__ import annotations

import asyncio
import contextlib
import dataclasses as _dc
import json
import os
import random
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ───── Soft-deps ──────────────────────────────────────────────────────────────
with contextlib.suppress(ModuleNotFoundError):
    import openai                           # type: ignore
with contextlib.suppress(ModuleNotFoundError):
    import numpy as np                      # type: ignore
with contextlib.suppress(ModuleNotFoundError):
    from gymnasium import spaces, Env       # type: ignore

# ultra-light fallback so demo works without gymnasium
if "spaces" not in globals():
    class _Dummy:                 # pylint: disable=too-few-public-methods
        def __init__(self, *_, **__): pass
    class spaces:                                       # type: ignore
        Discrete = Box = _Dummy
    class Env:                                          # type: ignore
        pass

# ───── Alpha-Factory back-plane (already in repo) ────────────────────────────
from alpha_factory_v1.backend.orchestrator import Orchestrator
from alpha_factory_v1.backend.world_model import wm

# ───── Configuration via env vars ────────────────────────────────────────────
LLM_MODEL            = os.getenv("OMNI_LLM_MODEL", "gpt-4o-mini")
TEMPERATURE          = float(os.getenv("OMNI_TEMPERATURE", "0.7"))
TOKENS_PER_TASK      = int(os.getenv("OMNI_TOKENS_PER_TASK", "50"))
LEDGER_PATH          = Path(os.getenv("OMNI_LEDGER", "./omni_ledger.sqlite"))
SEED                 = int(os.getenv("OMNI_SEED", "0"))
random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Task generation layer (OMNI-EPIC style)                        :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}
# ──────────────────────────────────────────────────────────────────────────────
_FALLBACK_SCENARIOS = [
    "A flash-flood closes two arterial bridges at 17:30 commuter peak.",
    "Cyber attack cripples downtown traffic-light timing network.",
    "Heatwave drives record HVAC demand; rolling brown-outs imminent.",
    "Unexpected protest blocks central business district during lunch."
]

def _llm_call(prompt: str) -> str:
    """Return a one-sentence scenario description (LLM or fallback)."""
    if "openai" not in globals() or not os.getenv("OPENAI_API_KEY"):
        return random.choice(_FALLBACK_SCENARIOS)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    resp = openai.ChatCompletion.create(
        model=LLM_MODEL, temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()    # type: ignore[attr-defined]

class TaskGenerator:
    """Produces *novel yet learnable* smart-city disruption scenarios."""
    _history: List[str]
    def __init__(self) -> None:
        self._history = []

    # ---------- public API ---------------------------------------------------
    def next(self) -> str:
        prompt = (
            "You are OMNI-EPIC. Propose ONE new smart-city resilience scenario "
            "unlike the following recent tasks:\n"
            f"{json.dumps(self._history[-6:], indent=2)}\n"
            "Return ONE concise sentence."
        )
        scenario = _llm_call(prompt)
        self._history.append(scenario)
        return scenario

    # ---------- MoI post-filter ---------------------------------------------
    def interesting(self, scenario: str) -> bool:
        """Very lightweight novelty check versus last N tasks."""
        return scenario not in self._history[-10:]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Minimal yet expressive smart-city environment
# ──────────────────────────────────────────────────────────────────────────────
@_dc.dataclass(slots=True)
class CityState:
    power_ok: float     # 0–1 fraction of grid functional
    traffic_ok: float   # 0–1 inverse congestion
    minute:     int

class SmartCityEnv(Env):
    """
    Gym-style continuous control environment (fast headless python).
    Observation  … [power_ok, traffic_ok, time/240]
    Action       … 2-vector: repair allocation, traffic-ops budget
    Reward       … weighted service availability (×100 at success)
    Episode done … ≥95 % service or after 4 simulated hours
    """
    metadata: Dict = {}
    action_space  = spaces.Box(low=0.0, high=1.0, shape=(2,))
    observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,))

    def __init__(self):
        super().__init__()
        self.rng = random.Random(SEED)
        self._scenario = "<un-set>"
        self.state = CityState(.5, .5, 0)

    # gymnasium v0.29 signature
    def reset(self, *, seed=None, options=None):
        del seed, options
        self._scenario = options.get("scenario") if options else "cold-start"
        self.state = CityState(
            self.rng.uniform(.3, .9),
            self.rng.uniform(.3, .9),
            0
        )
        return self._obs(), {}

    def step(self, action: Tuple[float, float]):
        repair, traffic = map(float, action)
        s = self.state
        s.power_ok   = min(1.0, s.power_ok   + 0.65 * repair)
        s.traffic_ok = min(1.0, s.traffic_ok + 0.55 * traffic)
        s.minute    += 1
        reward = 50*(s.power_ok + s.traffic_ok)             # 0-100
        done   = reward > 95 or s.minute >= 240
        truncated = False
        return self._obs(), reward, done, truncated, {}

    # ---------- helpers ------------------------------------------------------
    def _obs(self):
        s = self.state
        return [s.power_ok, s.traffic_ok, s.minute/240.0]

    # used by SuccessEvaluator
    def solved(self) -> bool:
        return (self.state.power_ok + self.state.traffic_ok) / 2 > .95


# ──────────────────────────────────────────────────────────────────────────────
# 3. CityCoin ledger (lightweight sqlite, append-only)
# ──────────────────────────────────────────────────────────────────────────────
_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger(
    ts        REAL,
    scenario  TEXT,
    tokens    INT,
    avg_reward REAL
);
"""

def _ledger_init() -> None:
    conn = sqlite3.connect(LEDGER_PATH)
    with conn:
        conn.executescript(_LEDGER_SCHEMA)
    conn.close()

def mint(scenario: str, avg_reward: float) -> int:
    tokens = int(avg_reward * TOKENS_PER_TASK)
    conn = sqlite3.connect(LEDGER_PATH)
    with conn:
        conn.execute("INSERT INTO ledger VALUES (?,?,?,?)",
                     (time.time(), scenario, tokens, avg_reward))
    return tokens


# ──────────────────────────────────────────────────────────────────────────────
# 4. Success evaluation layer
# ──────────────────────────────────────────────────────────────────────────────
class SuccessEvaluator:
    """Universal success check – uses env.solved() then archives."""
    def check(self, env: SmartCityEnv) -> bool:
        return env.solved()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Asynchronous main loop: Task → Orchestrator → Env
# ──────────────────────────────────────────────────────────────────────────────
async def open_ended_loop():
    orch   = Orchestrator(dev_mode=True)     # zero-config stub
    env    = SmartCityEnv()
    tgen   = TaskGenerator()
    evaler = SuccessEvaluator()

    while True:
        # 5-a  generate + filter scenario
        scenario = tgen.next()
        if not tgen.interesting(scenario):
            continue                        # skip stale prompt

        # 5-b  initialise environment with scenario narrative
        obs, _ = env.reset(options={"scenario": scenario})

        # 5-c  let the Alpha-Factory planner pick + call agent policy
        cumulative = 0.0
        for step in range(240):             # ≤4 hours sim
            plan = wm.plan("smart_city", {"obs": obs, "scenario": scenario})
            act_id = plan.get("action", {}).get("id", 0)
            # heuristic mapping → 2-float control vector
            action = (0.3 + 0.7*(act_id%3==0), 0.3 + 0.7*(act_id%3==1))
            obs, rew, done, _, _ = env.step(action)
            cumulative += rew
            if done: break

        avg_reward = cumulative / (step+1)
        solved = evaler.check(env)

        if solved:
            tokens = mint(scenario, avg_reward)
            print(f"✅ Solved ‘{scenario}’ in {step+1} m, "
                  f"avg reward {avg_reward:.1f} → {tokens} CityCoins.")
        else:
            print(f"✖️ Failed ‘{scenario}’ after {step+1} m "
                  f"(avg reward {avg_reward:.1f}).")

        await asyncio.sleep(0.05)           # cooperative pause


# ──────────────────────────────────────────────────────────────────────────────
# 6. CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("╭──────────────────────────────────────────────────────────╮")
    print("│   OMNI-Factory • Smart-City Resilience Open-End Loop     │")
    print("╰──────────────────────────────────────────────────────────╯")
    print("Ledger file:", LEDGER_PATH.resolve())
    _ledger_init()
    try:
        asyncio.run(open_ended_loop())
    except KeyboardInterrupt:
        print("\nShutdown requested – goodbye!")

if __name__ == "__main__":
    main()
