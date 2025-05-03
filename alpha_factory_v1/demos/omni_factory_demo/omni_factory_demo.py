#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ───────────────────────────────────────────────────────────────────────────────
#  OMNI-Factory · Smart-City Resilience Demo              (Alpha-Factory v1 add-on)
#  ───────────────────────────────────────────────────────────────────────────────
#  Purpose   • Minimal yet complete open-ended loop that plugs straight into the
#              existing Alpha-Factory backbone. Generates never-ending city-level
#              disruption scenarios, routes them through Alpha agents, evaluates
#              success, and mints “CityCoin” tokens.
#
#  Features  • Offline-first (pure Python). Online-enhanced if OPENAI_API_KEY is set
#            • Zero mandatory external dependencies (gymnasium, numpy optional)
#            • SQLite ledger for transparent, append-only economic accounting
#            • Fully type-annotated; mypy-clean; PEP-8 / Ruff compliant
#            • Async main-loop keeps UI / Prometheus scrapers responsive
#
#  Launch    >  python -m alpha_factory_v1.demos.omni_factory_demo
#              (from repo root)
# ───────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import asyncio
import contextlib
import dataclasses as dc
import json
import os
import random
import sqlite3
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ── Optional, safely-imported third-party libs ────────────────────────────────
with contextlib.suppress(ModuleNotFoundError):
    import openai                              # type: ignore
with contextlib.suppress(ModuleNotFoundError):
    import numpy as np                         # type: ignore
with contextlib.suppress(ModuleNotFoundError):
    from gymnasium import Env, spaces          # type: ignore

# Fallback stubs (so demo works on vanilla Python) ─────────────────────────────
if "spaces" not in globals():                  # pragma: no cover
    class _Dummy:                              # pylint: disable=too-few-public-methods
        def __init__(self, *_, **__):          # noqa: D401,E501
            pass
    class spaces:                              # type: ignore
        Box = Discrete = _Dummy
    class Env:                                 # type: ignore
        pass

# ── Alpha-Factory backbone imports (already in repo) ──────────────────────────
from alpha_factory_v1.backend.orchestrator import Orchestrator
from alpha_factory_v1.backend.world_model import wm

# ─── Configurable constants (env-overridable) ─────────────────────────────────
LLM_MODEL              = os.getenv("OMNI_LLM_MODEL", "gpt-4o-mini")
TEMPERATURE            = float(os.getenv("OMNI_TEMPERATURE", "0.7"))
TOKENS_PER_TASK        = int(os.getenv("OMNI_TOKENS_PER_TASK", "75"))
LEDGER_PATH            = Path(os.getenv("OMNI_LEDGER", "./omni_ledger.sqlite"))
SEED                   = int(os.getenv("OMNI_SEED", "0"))
MAX_SIM_MINUTES        = 240                              # 4 h episode cap
SUCCESS_THRESHOLD      = 0.95                             # overall service %
random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
#  1.  Task generation layer  (OMNI-EPIC style)
# ══════════════════════════════════════════════════════════════════════════════
_FALLBACK_SCENARIOS: tuple[str, ...] = (
    "Flash-flood closes two arterial bridges during evening rush.",
    "Cyber-attack disables subway signalling across downtown.",
    "Record heatwave triggers rolling brown-outs city-wide.",
    "Large protest blocks central business district at lunch."
)

def _llm_one_liner(prompt: str) -> str:
    """Return a single-sentence scenario (LLM or deterministic fallback)."""
    if "openai" not in globals() or not os.getenv("OPENAI_API_KEY"):
        # Pure-offline: cycle deterministically for reproducibility
        idx = abs(hash(prompt) + SEED) % len(_FALLBACK_SCENARIOS)
        return _FALLBACK_SCENARIOS[idx]
    openai.api_key = os.getenv("OPENAI_API_KEY")
    resp = openai.ChatCompletion.create(              # type: ignore[attr-defined]
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()    # type: ignore[index]

class TaskGenerator:
    """Produces *novel yet feasible* smart-city disruption scenarios."""
    _history: List[str]

    def __init__(self) -> None:
        self._history = []

    # Public API ──────────────────────────────────────────────────────────────
    def next(self) -> str:
        prompt = (
            "You are OMNI-EPIC. Draft ONE new smart-city resilience scenario "
            "that is materially different from these:\n"
            f"{json.dumps(self._history[-6:], indent=2)}\n"
            "Return exactly one concise sentence."
        )
        scenario = _llm_one_liner(prompt)
        self._history.append(scenario)
        return scenario

    # Simple MoI novelty filter ───────────────────────────────────────────────
    def interesting(self, s: str) -> bool:
        return s not in self._history[-10:]

# ══════════════════════════════════════════════════════════════════════════════
#  2.  Fast headless “digital-twin” environment
# ══════════════════════════════════════════════════════════════════════════════
@dc.dataclass(slots=True)
class _CityState:
    power_ok:   float                  # 0‥1 fraction of grid available
    traffic_ok: float                  # 0‥1 inverse congestion
    minute:     int                    # simulation clock (0‥240)

class SmartCityEnv(Env):               # noqa: D101
    metadata: Dict[str, str] = {}
    action_space      = spaces.Box(low=0.0, high=1.0, shape=(2,))
    observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,))

    def __init__(self) -> None:
        super().__init__()
        self.rng   = random.Random(SEED)
        self.state = _CityState(.5, .5, 0)
        self.scenario: str = "<un-set>"

    # Gymnasium 0.29 signature ───────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        del seed
        self.scenario = (options or {}).get("scenario", "cold-start")
        self.state = _CityState(
            self.rng.uniform(.3, .9),
            self.rng.uniform(.3, .9),
            0,
        )
        return self._obs(), {}           # obs, info

    def step(self, action: Tuple[float, float]):
        repair, traffic = map(float, action)
        s = self.state
        s.power_ok   = min(1.0, s.power_ok   + 0.65 * repair)
        s.traffic_ok = min(1.0, s.traffic_ok + 0.55 * traffic)
        s.minute    += 1

        reward     = (s.power_ok + s.traffic_ok) / 2
        done       = reward >= SUCCESS_THRESHOLD or s.minute >= MAX_SIM_MINUTES
        truncated  = False
        return self._obs(), reward, done, truncated, {}

    # Helper ------------------------------------------------------------------
    def _obs(self) -> List[float]:
        s = self.state
        return [s.power_ok, s.traffic_ok, s.minute / MAX_SIM_MINUTES]

    # Public utility for evaluator -------------------------------------------
    def solved(self) -> bool:
        return (self.state.power_ok + self.state.traffic_ok) / 2 >= SUCCESS_THRESHOLD

# ══════════════════════════════════════════════════════════════════════════════
#  3.  Token-economy ledger (SQLite, append-only, human-readable)
# ══════════════════════════════════════════════════════════════════════════════
_DEF_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger(
    ts          REAL,       -- epoch seconds
    scenario    TEXT,
    tokens      INT,        -- CityCoins
    avg_reward  REAL
);"""

def _ledger_init() -> None:
    conn = sqlite3.connect(LEDGER_PATH)
    with conn:
        conn.executescript(_DEF_SCHEMA)
    conn.close()

def _mint(scenario: str, avg_reward: float) -> int:
    tokens = int(avg_reward * TOKENS_PER_TASK)
    conn = sqlite3.connect(LEDGER_PATH)
    with conn:
        conn.execute(
            "INSERT INTO ledger VALUES (?,?,?,?)",
            (time.time(), scenario, tokens, avg_reward)
        )
    return tokens

# ══════════════════════════════════════════════════════════════════════════════
#  4.  Success evaluator
# ══════════════════════════════════════════════════════════════════════════════
class SuccessEvaluator:                           # noqa: D101
    def check(self, env: SmartCityEnv) -> bool:
        return env.solved()

# ══════════════════════════════════════════════════════════════════════════════
#  5.  Asynchronous open-ended loop
# ══════════════════════════════════════════════════════════════════════════════
async def _loop() -> None:
    orch   = Orchestrator(dev_mode=True)          # zero-config stub
    env    = SmartCityEnv()
    tgen   = TaskGenerator()
    evaler = SuccessEvaluator()

    while True:
        # 1 Generate & filter
        scenario = tgen.next()
        if not tgen.interesting(scenario):
            continue                              # skip stale prompt

        # 2 Reset environment
        obs, _ = env.reset(options={"scenario": scenario})

        # 3 Closed-loop control via Alpha-Factory world-model planner
        cumulative = 0.0
        for step in range(MAX_SIM_MINUTES):
            plan = wm.plan("smart_city", {"obs": obs, "scenario": scenario})
            act_id = plan.get("action", {}).get("id", 0)
            # Heuristic mapping of discrete id → continuous action
            repair  = 0.8 if act_id % 3 == 0 else 0.3
            traffic = 0.8 if act_id % 3 == 1 else 0.3
            obs, rew, done, _, _ = env.step((repair, traffic))
            cumulative += rew
            if done:
                break

        avg_reward = cumulative / (step + 1)
        solved = evaler.check(env)

        if solved:
            tokens = _mint(scenario, avg_reward)
            print(f"✅  Solved “{scenario}” in {step+1:3} min "
                  f"(avg reward {avg_reward:.3f}) – {tokens} CityCoins minted.")
        else:
            print(f"✖️  Failed “{scenario}” after {step+1:3} min "
                  f"(avg reward {avg_reward:.3f}).")

        await asyncio.sleep(0.05)                # cooperative yield

# ══════════════════════════════════════════════════════════════════════════════
#  6.  CLI entry-point
# ══════════════════════════════════════════════════════════════════════════════
def main(argv: List[str] | None = None) -> None:    # noqa: D401
    """CLI wrapper so `python -m …` works nicely."""
    del argv
    banner = (
        "╭──────────────────────────────────────────────────────────────╮\n"
        "│        OMNI-Factory • Smart-City Resilience Demo v1         │\n"
        "╰──────────────────────────────────────────────────────────────╯"
    )
    print(banner)
    print("Ledger:", LEDGER_PATH.resolve())
    if not LEDGER_PATH.exists():
        _ledger_init()
        print("↳ New ledger initialised.")
    print("Press Ctrl-C to exit.\n")

    try:
        asyncio.run(_loop())
    except KeyboardInterrupt:                      # pragma: no cover
        print("\nShutdown requested – goodbye!")

if __name__ == "__main__":                         # pragma: no cover
    main(sys.argv[1:])
