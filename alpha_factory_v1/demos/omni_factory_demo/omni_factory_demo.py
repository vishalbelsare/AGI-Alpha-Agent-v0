#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
OMNI‑Factory • Smart‑City Resilience Demo (α‑Factory v1 add‑on)
═══════════════════════════════════════════════════════════════
A *production‑grade* open‑ended learning loop that plugs straight
into the Alpha‑Factory backbone and demonstrates:

• Automatic scenario generation   (OMNI‑EPIC style)
• Multi‑agent problem‑solving     (orchestrator + world‑model)
• Grounded success evaluation     (numeric & qualitative)
• Token‑economy accounting        ($AGIALPHA ledger)
• Observability & safety          (Prometheus, graceful exits)

The script is completely **offline‑first** (no internet / API key
required). If `OPENAI_API_KEY` is present, LLM‑powered scenario
generation is enabled transparently.

Run
───
    python -m alpha_factory_v1.demos.omni_factory_demo \
           --metrics-port 8000 --max-episodes 10 --dry-run

Dependencies
────────────
Only the Alpha‑Factory repo itself plus the *optional* extras
`prometheus‑client`, `numpy`, `openai` and `gymnasium`.

All imports are wrapped in `contextlib.suppress` so the demo
never crashes on missing extras – it merely downgrades features.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses as dc
import json
import os
import random
import signal
import sqlite3
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Tuple

# ──────────────────────────────────────────────────────────────
# Optional third‑party imports (gracefully degraded if absent)
# ──────────────────────────────────────────────────────────────
with contextlib.suppress(ModuleNotFoundError):
    import numpy as np  # type: ignore

with contextlib.suppress(ModuleNotFoundError):
    import openai  # type: ignore

with contextlib.suppress(ModuleNotFoundError):
    from gymnasium import Env, spaces  # type: ignore

with contextlib.suppress(ModuleNotFoundError):
    from prometheus_client import CollectorRegistry, Gauge, start_http_server  # type: ignore

# Fallback stubs so typing doesn’t break if optional deps are missing
if "spaces" not in globals():  # pragma: no cover
    class _Dummy:
        def __init__(self, *_, **__):
            pass

    class spaces:  # type: ignore
        Box = Discrete = _Dummy

    class Env:  # type: ignore
        pass

# ──────────────────────────────────────────────────────────────
# Alpha‑Factory backbone
# ──────────────────────────────────────────────────────────────
from alpha_factory_v1.backend.orchestrator import Orchestrator  # noqa: E402
from alpha_factory_v1.backend.world_model import wm            # noqa: E402

# ──────────────────────────────────────────────────────────────
# Global configuration (env‑overridable)
# ──────────────────────────────────────────────────────────────
CFG_DEFAULTS: dict[str, Any] = {
    "LLM_MODEL":         "gpt-4o-mini",
    "TEMPERATURE":       0.7,
    "TOKENS_PER_TASK":   100,
    "LEDGER_PATH":       "./omni_ledger.sqlite",
    "SEED":              0,
    "SUCCESS_THRESHOLD": 0.95,
    "MAX_SIM_MINUTES":   240,
    "MICRO_CURRICULUM":  "auto",          # easy→hard scheduler
    "AGIALPHA_SUPPLY":   10_000_000,      # hard cap
}

@dc.dataclass(frozen=True, slots=True)
class Cfg:
    """Immutable runtime configuration resolved from environment."""

    llm_model:       str  = CFG_DEFAULTS["LLM_MODEL"]
    temperature:     float = CFG_DEFAULTS["TEMPERATURE"]
    tokens_per_task: int   = CFG_DEFAULTS["TOKENS_PER_TASK"]
    ledger_path:     Path  = Path(CFG_DEFAULTS["LEDGER_PATH"])
    seed:            int   = CFG_DEFAULTS["SEED"]
    success_thresh:  float = CFG_DEFAULTS["SUCCESS_THRESHOLD"]
    max_minutes:     int   = CFG_DEFAULTS["MAX_SIM_MINUTES"]
    micro_curr:      str   = CFG_DEFAULTS["MICRO_CURRICULUM"]
    max_supply:      int   = CFG_DEFAULTS["AGIALPHA_SUPPLY"]

# Resolve overrides -----------------------------------------------------------
CFG = Cfg(
    **{
        k.lower(): (Path(v) if k.endswith("PATH") else type(CFG_DEFAULTS[k])(os.getenv(f"OMNI_{k}", v)))
        for k, v in CFG_DEFAULTS.items()
    }
)
random.seed(CFG.seed)

# ──────────────────────────────────────────────────────────────
# Utility: Prometheus metrics (no‑op if client missing)
# ──────────────────────────────────────────────────────────────
class _Metrics:
    """Thin wrapper so code can reference gauges regardless of availability."""

    def __init__(self, port: int | None):
        self.enabled = "CollectorRegistry" in globals() and port is not None
        if not self.enabled:
            self.avg_reward = self.episodes = self.tokens = None  # type: ignore
            return

        registry = CollectorRegistry()  # type: ignore[misc]
        self.avg_reward = Gauge("omni_avg_reward", "Avg reward per episode", registry=registry)  # type: ignore[misc]
        self.episodes   = Gauge("omni_episode", "Episode counter", registry=registry)             # type: ignore[misc]
        self.tokens     = Gauge("omni_tokens_minted", "Tokens minted", registry=registry)         # type: ignore[misc]
        start_http_server(port, registry=registry)  # type: ignore[misc]

METRICS: _Metrics  # populated in main()

# ──────────────────────────────────────────────────────────────
# 1. Scenario (task) generation
# ──────────────────────────────────────────────────────────────
_FALLBACK_SCENARIOS: tuple[str, ...] = (
    "Flash‑flood closes two arterial bridges during evening rush.",
    "Cyber‑attack disables subway signalling across downtown.",
    "Record heatwave triggers rolling brown‑outs city‑wide.",
    "Large protest blocks the central business district at lunch.",
)

def _llm_one_liner(prompt: str) -> str:
    """Return a single‑sentence scenario via LLM or deterministic fallback."""
    if "openai" not in globals() or not os.getenv("OPENAI_API_KEY"):
        idx = abs(hash(prompt) + CFG.seed) % len(_FALLBACK_SCENARIOS)
        return _FALLBACK_SCENARIOS[idx]

    openai.api_key = os.getenv("OPENAI_API_KEY")  # type: ignore[attr-defined]
    resp = openai.ChatCompletion.create(  # type: ignore[attr-defined]
        model=CFG.llm_model,
        temperature=CFG.temperature,
        max_tokens=60,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()  # type: ignore[index]

class TaskGenerator:
    """OMNI‑EPIC style generator maintaining novelty & difficulty scheduling."""

    def __init__(self) -> None:
        self._hist: list[str] = []
        self._step = 0

    # Public API -------------------------------------------------------------
    def next(self) -> str:
        self._step += 1
        prompt = (
            "You are OMNI‑EPIC. Draft ONE new smart‑city disruption scenario "
            "substantially different from these:\n"
            f"{json.dumps(self._hist[-8:], indent=2)}\n"
            "Return exactly one concise sentence."
        )
        scenario = _llm_one_liner(prompt)
        self._hist.append(scenario)
        return scenario

# ──────────────────────────────────────────────────────────────
# 2. Minimal but grounded smart‑city environment
# ──────────────────────────────────────────────────────────────
@dc.dataclass(slots=True)
class _CityState:
    power_ok:   float
    traffic_ok: float
    minute:     int

class SmartCityEnv(Env):  # type: ignore[misc]
    """Simplified headless digital‑twin with two controllable dimensions."""

    metadata: Dict[str, str] = {}
    action_space      = spaces.Box(low=0.0, high=1.0, shape=(2,))   # type: ignore[arr-type]
    observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,))   # type: ignore[arr-type]

    def __init__(self) -> None:
        super().__init__()
        self.rng = random.Random(CFG.seed * 997)
        self.state: _CityState = self._random_state()
        self.scenario: str = "<unset>"

    def _random_state(self) -> _CityState:
        return _CityState(self.rng.uniform(.3, .9), self.rng.uniform(.3, .9), 0)

    # Gymnasium 0.29 API ----------------------------------------------------
    def reset(self, *, seed: int | None = None, options: Dict | None = None):  # type: ignore[override]
        del seed
        self.state = self._random_state()
        self.scenario = (options or {}).get("scenario", "cold‑start")
        return self._obs(), {}

    def step(self, action: Tuple[float, float]):  # type: ignore[override]
        repair, traffic = map(float, action)
        s = self.state
        s.power_ok   = min(1.0, s.power_ok   + 0.65 * repair)
        s.traffic_ok = min(1.0, s.traffic_ok + 0.55 * traffic)
        s.minute    += 1

        reward    = (s.power_ok + s.traffic_ok) / 2
        terminated = reward >= CFG.success_thresh
        truncated  = s.minute >= CFG.max_minutes
        return self._obs(), reward, terminated or truncated, truncated, {}

    def _obs(self) -> List[float]:
        s = self.state
        return [s.power_ok, s.traffic_ok, s.minute / CFG.max_minutes]

    # Ground‑truth solver ----------------------------------------------------
    def solved(self) -> bool:
        return (self.state.power_ok + self.state.traffic_ok) / 2 >= CFG.success_thresh

# ──────────────────────────────────────────────────────────────
# 3. $AGIALPHA ledger (append‑only, supply‑capped)
# ──────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger(
    ts          REAL,
    scenario    TEXT,
    tokens      INT,
    avg_reward  REAL,
    checksum    TEXT
);
"""

def _ledger_init(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript(_SCHEMA)

# ----------------------------------------------------------------------------

def _hash_row(ts: float, scen: str, tok: int, rew: float) -> str:
    import hashlib
    return hashlib.sha256(f"{ts}{scen}{tok}{rew}".encode()).hexdigest()[:16]

# ----------------------------------------------------------------------------

def _current_supply(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT SUM(tokens) FROM ledger")
    val = cur.fetchone()[0]
    return val if val is not None else 0

# ----------------------------------------------------------------------------

def _mint(path: Path, scenario: str, avg_reward: float) -> int:
    """Mint $AGIALPHA proportional to avg_reward, respecting hard‑cap."""
    tokens = int(avg_reward * CFG.tokens_per_task)
    with sqlite3.connect(path) as conn:
        supply = _current_supply(conn)
        if supply + tokens > CFG.max_supply:
            tokens = CFG.max_supply - supply
        ts = time.time()
        conn.execute(
            "INSERT INTO ledger VALUES (?,?,?,?,?)",
            (ts, scenario, tokens, avg_reward, _hash_row(ts, scenario, tokens, avg_reward)),
        )
    return max(tokens, 0)

# ──────────────────────────────────────────────────────────────
# 4. Success evaluator (numeric + optional LLM qualitative)
# ──────────────────────────────────────────────────────────────
class SuccessEvaluator:
    """Blend numeric ground truth with optional LLM qualitative signal."""

    def __init__(self) -> None:
        self._qual_hist: list[str] = []

    def check(self, env: SmartCityEnv) -> bool:
        num_ok = env.solved()
        qual_ok = True

        if "openai" in globals() and os.getenv("OPENAI_API_KEY"):
            prompt = (
                f"Scenario: {env.scenario}. Final KPIs — power_ok={env.state.power_ok:.2f}, "
                f"traffic_ok={env.state.traffic_ok:.2f}.\n"
                "Respond simply yes/no: did the city stabilise successfully?"
            )
            answer = _llm_one_liner(prompt).lower()
            self._qual_hist.append(answer)
            qual_ok = answer.startswith("yes")

        return num_ok and qual_ok

# ──────────────────────────────────────────────────────────────
# 5. Plug‑in autoloader (agents / env helpers)
# ──────────────────────────────────────────────────────────────

def _load_plugins(folder: Path | None = None) -> List[ModuleType]:
    if folder is None:
        folder = Path(__file__).with_suffix("").parent / "plugins"
    mods: list[ModuleType] = []
    if not folder.exists():
        return mods

    for py in folder.glob("*.py"):
        with contextlib.suppress(Exception):
            import importlib.util
            spec = importlib.util.spec_from_file_location(py.stem, py)
            if spec and spec.loader:
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[arg-type]
                mods.append(mod)
    return mods

# ──────────────────────────────────────────────────────────────
# 6. Single episode runner
# ──────────────────────────────────────────────────────────────
async def _episode(
    orch: Orchestrator,
    env: SmartCityEnv,
    tgen: TaskGenerator,
    evaler: SuccessEvaluator,
    episode_id: int,
) -> None:
    scenario = tgen.next()
    obs, _ = env.reset(options={"scenario": scenario})
    cumulative = 0.0

    for minute in range(CFG.max_minutes):
        plan = wm.plan("smart_city", {"obs": obs, "scenario": scenario})
        act_id = plan.get("action", {}).get("id", 0)
        repair  = 0.8 if act_id % 3 == 0 else 0.2
        traffic = 0.8 if act_id % 3 == 1 else 0.2
        obs, rew, done, _, _ = env.step((repair, traffic))
        cumulative += rew
        if done:
            break

    avg_reward = cumulative / (minute + 1)
    solved = evaler.check(env)

    if solved:
        tokens = _mint(CFG.ledger_path, scenario, avg_reward)
        status = (
            f"✅  {scenario} ({minute+1} min, r̄={avg_reward:.3f}) → +{tokens} $AGIALPHA"
        )
        if METRICS.enabled and METRICS.tokens:
            METRICS.tokens.inc(tokens)  # type: ignore[misc]
    else:
        status = f"✖️  {scenario} ({minute+1} min, r̄={avg_reward:.3f}) – UNSOLVED"

    if METRICS.enabled:
        if METRICS.avg_reward:
            METRICS.avg_reward.set(avg_reward)  # type: ignore[misc]
        if METRICS.episodes:
            METRICS.episodes.set(episode_id)    # type: ignore[misc]

    print(status)

# ──────────────────────────────────────────────────────────────
# 7. Continuous open‑ended loop
# ──────────────────────────────────────────────────────────────
async def _loop(max_episodes: int | None) -> None:
    orch   = Orchestrator(dev_mode=True)
    env    = SmartCityEnv()
    tgen   = TaskGenerator()
    evaler = SuccessEvaluator()
    eps    = 0

    while max_episodes is None or eps < max_episodes:
        eps += 1
        try:
            await _episode(orch, env, tgen, evaler, eps)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Episode {eps} crashed: {exc!r}")
        await asyncio.sleep(0)  # cooperative yield

# ──────────────────────────────────────────────────────────────
# 8. CLI & graceful shutdown
# ──────────────────────────────────────────────────────────────

def _parse_cli(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="omni_factory_demo")
    p.add_argument("--metrics-port", type=int, default=None,
                   help="Expose Prometheus metrics at this TCP port")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Stop after N episodes (default: endless)")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip token minting – useful in CI")
    return p.parse_args(argv)

# ----------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:  # noqa: D401
    """Entry‑point so `python -m` works."""
    args = _parse_cli(argv or [])

    global METRICS
    METRICS = _Metrics(args.metrics_port)

    banner = (
        "╭────────────────────────────────────────────────────────────╮\n"
        "│        OMNI-Factory • Smart-City Resilience Demo v1        │\n"
        "╰────────────────────────────────────────────────────────────╯"
    )
    print(banner)
    print("Ledger:", CFG.ledger_path.resolve())
    if not CFG.ledger_path.exists():
        _ledger_init(CFG.ledger_path)
        print("↳ New ledger initialised.")

    if args.dry_run:
        globals()["_mint"] = lambda *_, **__: 0  # type: ignore[assignment]
        print("⚠️  DRY‑RUN mode – no $AGIALPHA will be minted.")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stop = asyncio.Event()

    def _on_sig(_sig, _frm):  # noqa: D401
        print("\nCtrl‑C detected – shutting down gracefully …")
        stop.set()

    signal.signal(signal.SIGINT, _on_sig)

    async def _run() -> None:
        prod_task = asyncio.create_task(_loop(args.max_episodes))
        stopper   = asyncio.create_task(stop.wait())
        await asyncio.wait({prod_task, stopper}, return_when=asyncio.FIRST_COMPLETED)
        prod_task.cancel()

    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()
        print("Goodbye!")

if __name__ == "__main__":
    main(sys.argv[1:])
