# SPDX-License-Identifier: Apache-2.0
"""
OMNI-Factory · Smart-City Resilience Demo  (Alpha-Factory v1 add-on)
════════════════════════════════════════════════════════════════════
Runs a minimal yet complete open-ended loop:
    1. TaskGenerator → produces a city disruption scenario
    2. Orchestrator shim  → routes tasks to existing Alpha agents
    3. SmartCityEnv       → fast python env (traffic + power grid toy model)
    4. Evaluator          → success / reward / CityCoin minting
    5. Loop forever       → archive to `./omni_ledger.sqlite`

Works fully offline; if OPENAI_API_KEY is set it upgrades TaskGenerator
to GPT-4o-mini for richer, never-repeating scenarios.

Launch:  python -m alpha_factory_v1.demos.omni_factory_demo
"""
from __future__ import annotations
import asyncio, contextlib, json, os, random, sqlite3, time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

# ── Soft-imports ─────────────────────────────────────────────────────
with contextlib.suppress(ModuleNotFoundError):
    import openai                     # type: ignore
with contextlib.suppress(ModuleNotFoundError):
    from gymnasium import spaces      # type: ignore
else:
    class spaces:                     # tiny fallback
        class Discrete(int): pass
        class Box(list): pass

# ── Glue existing Alpha-Factory backbone ────────────────────────────
from alpha_factory_v1.backend.orchestrator import Orchestrator  # re-use, no changes
from alpha_factory_v1.backend.world_model import wm             # planning fallback

# ── Global settings (env overrides) ─────────────────────────────────
LLM_MODEL   = os.getenv("OMNI_LLM_MODEL", "gpt-4o-mini")
SEED        = int(os.getenv("OMNI_SEED", "0"))
TOK_PER_EVT = int(os.getenv("OMNI_TOKENS_PER_EVENT", "10"))
LEDGER_FILE = Path(os.getenv("OMNI_LEDGER", "omni_ledger.sqlite"))

random.seed(SEED)

# ────────────────────────────────────────────────────────────────────
# 1. Task Generator
# ────────────────────────────────────────────────────────────────────
_SCENARIO_TEMPLATES = [
    "City-wide power outage during a record heatwave.",
    "Flash-flood cuts off two major arterial roads.",
    "Cyber-attack cripples the subway signalling network.",
    "Sudden protest blocks downtown core during rush hour.",
]

def _llm(prompt: str) -> str:
    if "openai" not in globals() or not os.getenv("OPENAI_API_KEY"):
        return random.choice(_SCENARIO_TEMPLATES)
    openai.api_key = os.getenv("OPENAI_API_KEY")
    resp = openai.ChatCompletion.create(
        model=LLM_MODEL, temperature=0.7, max_tokens=120,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp["choices"][0]["message"]["content"]  # type: ignore[index]

class TaskGenerator:
    """Open-ended scenario creator (LLM-backed if key present)."""
    def next_task(self, history: List[str]) -> str:
        prompt = (
            "Generate a NEW smart-city disruption scenario unlike these:\n"
            f"{json.dumps(history[-5:], indent=2)}\n"
            "Respond with ONE sentence describing the event."
        )
        return _llm(prompt).strip()

# ────────────────────────────────────────────────────────────────────
# 2. Smart-City toy environment
# ────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class CityObs:
    power_ok: float   # 0…1  fraction of grid capacity online
    traffic_ok: float # 0…1  inverse congestion metric
    time: int         # discrete minute

class SmartCityEnv:
    """
    Very light environment: state is tuple (power_ok, traffic_ok, t).
    Agents act by allocating repair units or traffic control budget.
    """
    action_space = spaces.Box([0, 0], [1, 1])  # [repair_frac, traffic_budget]
    observation_space = spaces.Box([0, 0, 0], [1, 1, 1440])

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self.reset("initial boot")

    def reset(self, scenario: str) -> CityObs:
        self.scenario = scenario
        self.state = CityObs(power_ok=self.rng.uniform(.4, .9),
                             traffic_ok=self.rng.uniform(.4, .9),
                             time=0)
        return self.state

    def step(self, action: Tuple[float, float]) -> Tuple[CityObs, float, bool, Dict]:
        repair, traffic = action
        # simple deterministic dynamics
        self.state.power_ok   = min(1.0, self.state.power_ok + 0.6*repair)
        self.state.traffic_ok = min(1.0, self.state.traffic_ok + 0.5*traffic)
        self.state.time      += 1
        reward = 0.5*self.state.power_ok + 0.5*self.state.traffic_ok
        done   = reward > 0.95 or self.state.time >= 240  # 4 h max
        return self.state, reward, done, {}

# ────────────────────────────────────────────────────────────────────
# 3. Ledger (token economy)
# ────────────────────────────────────────────────────────────────────
def _init_ledger() -> None:
    conn = sqlite3.connect(LEDGER_FILE)
    with conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS ledger(
            ts REAL, scenario TEXT, tokens INT, reward REAL
        )""")
    conn.close()

def mint_tokens(scenario: str, reward: float) -> int:
    tokens = int(reward*100 * TOK_PER_EVT)
    conn = sqlite3.connect(LEDGER_FILE)
    with conn:
        conn.execute("INSERT INTO ledger VALUES (?,?,?,?)",
                     (time.time(), scenario, tokens, reward))
    return tokens

# ────────────────────────────────────────────────────────────────────
# 4. Main loop (async so UI remains responsive)
# ────────────────────────────────────────────────────────────────────
async def run_loop() -> None:
    orchestrator = Orchestrator(dev_mode=True)  # ← no extra config needed
    tgen   = TaskGenerator()
    env    = SmartCityEnv(seed=SEED)
    history: List[str] = []

    while True:
        scenario = tgen.next_task(history)
        history.append(scenario)
        obs = env.reset(scenario)

        # very small planning call using existing wm planner
        step = 0
        cum_r = 0.0
        while True:
            plan = wm.plan("omni_demo", obs.__dict__)  # uses MuZero wrapper if avail.
            act_id = plan.get("action", {}).get("id", 0)
            # map discrete id→continuous action for toy env
            repair = [1,0,0,0][act_id%4] if act_id<4 else 0.3
            traffic= [0,1,0,0][act_id%4] if act_id<4 else 0.3
            obs, r, done, _ = env.step((repair, traffic))
            step += 1
            cum_r += r
            if done: break

        tokens = mint_tokens(scenario, cum_r/step)
        print(f"✔ Completed '{scenario}' in {step} m, "
              f"avg reward {cum_r/step:.3f} → minted {tokens} CityCoins.")
        # brief pause so Prometheus scrape & UI updates have time
        await asyncio.sleep(0.1)

# ────────────────────────────────────────────────────────────────────
# 5. Entry-point
# ────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== OMNI-Factory · Smart-City Resilience Demo ===")
    print("Ctrl-C to exit.  Ledger at:", LEDGER_FILE.resolve())
    _init_ledger()
    try:
        asyncio.run(run_loop())
    except KeyboardInterrupt:
        print("\nShutdown requested by user – goodbye.")

if __name__ == "__main__":
    main()
