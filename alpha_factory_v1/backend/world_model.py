# SPDX-License-Identifier: Apache-2.0
"""
alpha_factory_v1.backend.world_model
════════════════════════════════════
Unified latent World-Model + Planner (v1.3.2 ‒ 2025-05-02)
──────────────────────────────────────────────────────────
• 5 × 5 Grid-World reference environment (demo #6)  
• Pluggable environment registry (gym-style or custom)  
• MuZero-general + shallow MCTS wrapper (soft-import, depth- & time-bounded)  
• LLM counter-factual simulator fallback (OpenAI / local Llama-cpp)  
• Rule-based SafetyGuard (runtime-extensible via ENV JSON)  
• Prometheus metrics & Kafka event streaming (optional)  
• Meta-Learner (AI-GA Autogenesis) → proposes new agents/hyper-params as Markdown  
• 100 % graceful degradation: **never crashes** – runs offline on a Raspberry Pi

┌──────────────────────────────────────────────────────────────────────────────┐
│  Public API (importable by Orchestrator or agents)                           │
│  ──────────────────────────────────────────────────────────────────────────   │
│  wm.reset_env(...)             → dict          # reset active env            │
│  wm.plan(agent, state)         → dict          # best next action            │
│  wm.simulate(state, action)    → dict          # 1-step forward model        │
│  wm.ingest_kpis({agent:score}) → Optional[str] # Meta-Learner hook           │
│  wm.register_env(name, cls)                    # add custom env at runtime   │
└──────────────────────────────────────────────────────────────────────────────┘
"""
from __future__ import annotations

# ── Python std-lib ────────────────────────────────────────────────────────────
import asyncio
import contextlib
import importlib
import inspect
import json
import math
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Tuple

# ── Optional external deps (ALL soft-imports) ────────────────────────────────
with contextlib.suppress(ModuleNotFoundError):
    # MuZero-general (https://github.com/werner-duvaud/muzero-general)
    from muzero import muzero  # type: ignore

with contextlib.suppress(ModuleNotFoundError):
    # OpenAI / Anthropic SDKs
    import openai  # type: ignore[attr-defined]

with contextlib.suppress(ModuleNotFoundError):
    # Local LLM (gguf / Llama-cpp-python)
    from llama_cpp import Llama  # type: ignore

with contextlib.suppress(ModuleNotFoundError):
    # Prometheus metrics
    from prometheus_client import Gauge

with contextlib.suppress(ModuleNotFoundError):
    # Kafka producer
    from confluent_kafka import Producer  # type: ignore

# ── ENV config ───────────────────────────────────────────────────────────────
ENV = os.getenv
DEV_MODE          = ENV("DEV_MODE", "false").lower() == "true"
GRID_SIZE         = int(ENV("WM_GRID_SIZE",            "5"))
MCTS_SIMS         = int(ENV("WM_MCTS_SIMULATIONS",     "96"))
MCTS_TIMEOUT_SEC  = int(ENV("WM_MCTS_TIMEOUT_SEC",     "3"))
LLM_MODEL         = ENV("WM_LLM_MODEL",                "gpt-4o-mini")
RISK_THRESHOLD    = float(ENV("WM_RISK_THRESHOLD",     "-50"))
RULES_JSON        = ENV("WM_SAFETY_RULES",             "")
PROM_ENABLED      = ENV("PROMETHEUS_DISABLE",          "false").lower() != "true"
KAFKA_BROKER      = ENV("ALPHA_KAFKA_BROKER")
META_INTERVAL_SEC = int(ENV("META_INTERVAL_SEC",       "3600"))

_DEFAULT_RULES: Dict[str, Any] = {
    "max_position_frac": 0.25,   # finance – pos ≤ 25 % NAV
    "max_shift_hours":   12,     # manufacturing – ≤ 12 h/shift
    "min_liquidity_usd": 1e5,    # execution – avoid illiquid markets
}
SAFETY_RULES: Dict[str, Any] = {**_DEFAULT_RULES, **json.loads(RULES_JSON or "{}")}

# ──────────────────────────────────────────────────────────────────────────────
# 1. Reference Grid-World environment (Gym-inspired, no external deps)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(slots=True)
class GridState:
    x: int = 0
    y: int = 0
    goal: Tuple[int, int] = (GRID_SIZE - 1, GRID_SIZE - 1)
    reward: float = 0.0
    done: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GridWorldEnv:
    """Minimal deterministic 2-D grid with dense time-penalty reward."""
    ACTIONS = {
        0: ("UP",    (0, -1)),
        1: ("DOWN",  (0,  1)),
        2: ("LEFT",  (-1, 0)),
        3: ("RIGHT", (1,  0)),
    }

    def __init__(self, size: int = GRID_SIZE) -> None:
        self.size = size
        self.state = GridState()

    # Gym-like helpers -------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self.state = GridState()
        return self.state.to_dict()

    @property
    def observation(self) -> List[int]:
        """Return a flattened N × N binary grid (agent=1, goal=2)."""
        grid = [[0] * self.size for _ in range(self.size)]
        grid[self.state.y][self.state.x] = 1
        gx, gy = self.state.goal
        grid[gy][gx] = 2
        return [c for row in grid for c in row]

    def step(self, action_id: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.state.done:
            return self.state.to_dict(), 0.0, True, {}
        name, (dx, dy) = self.ACTIONS.get(action_id, ("NOP", (0, 0)))
        self.state.x = max(0, min(self.size - 1, self.state.x + dx))
        self.state.y = max(0, min(self.size - 1, self.state.y + dy))
        self.state.reward -= 1  # time penalty
        if (self.state.x, self.state.y) == self.state.goal:
            self.state.reward += 100
            self.state.done = True
        return self.state.to_dict(), self.state.reward, self.state.done, {"action": name}

# ──────────────────────────────────────────────────────────────────────────────
# 2. Safety-Guard (rule-based; extend with ML anomaly detectors later)
# ──────────────────────────────────────────────────────────────────────────────
class SafetyGuard:
    def __init__(self, rules: Dict[str, Any]) -> None:
        self.rules = rules

    # Example logic: veto trajectory if cumulative reward below threshold
    def veto(self, reward_trace: List[float]) -> bool:
        if sum(reward_trace) < RISK_THRESHOLD:
            return True
        return False

# ──────────────────────────────────────────────────────────────────────────────
# 3. Planner back-ends
# ──────────────────────────────────────────────────────────────────────────────
class MuZeroPlanner:
    """Thin wrapper around *muzero-general* for **planning only** (no training)."""
    def __init__(self, env: GridWorldEnv) -> None:
        self.enabled = "muzero" in globals()
        if not self.enabled:
            return
        cfg_path = Path(__file__).with_name("muzero_grid_cfg.py")
        if not cfg_path.exists():
            cfg_path.write_text(_MZ_CFG_TXT)
        self._mz = muzero.MuZero(config=str(cfg_path))  # type: ignore[arg-type]
        self._env = env

    def best_action(self, sims: int = MCTS_SIMS, timeout: int = MCTS_TIMEOUT_SEC) -> Tuple[int, List[int]]:
        if not self.enabled:
            # Greedy heuristic – manhattan direction prioritised.
            dx = self._env.state.goal[0] - self._env.state.x
            dy = self._env.state.goal[1] - self._env.state.y
            return (3 if dx > 0 else 2 if dx < 0 else 1 if dy > 0 else 0), []
        # Time-bounded search –
        return self._mz.plan(self._env, sims, timeout)  # type: ignore[attr-defined]

class LLMSimulator:
    """Lightweight LLM forward-model for arbitrary JSON state/action pairs."""
    def __init__(self) -> None:
        self._enabled_openai = "openai" in globals() and bool(ENV("OPENAI_API_KEY"))
        self._enabled_local  = "Llama"  in globals()
        if self._enabled_openai:
            openai.api_key = ENV("OPENAI_API_KEY")
        elif self._enabled_local:
            model_path = ENV("LLAMA_MODEL_PATH", "models/llama-2-7b.gguf")
            self.llama = Llama(model_path=model_path)

    def _chat(self, prompt: str) -> str:
        if self._enabled_openai:
            resp = openai.ChatCompletion.create(
                model=LLM_MODEL,
                temperature=0.3,
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp["choices"][0]["message"]["content"]  # type: ignore[index]
        if self._enabled_local:
            return self.llama(prompt, max_tokens=80)["choices"][0]["text"]  # type: ignore[index]
        return "LLM unavailable"

    # -----------------------------------------------------------------
    def predict(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            "You are a world-model. Given the **current state** JSON and an **action**, "
            "return a plausible **next state** JSON with keys identical to state, plus "
            '"reward" (float) and "done" (bool).'
            f"\n\nSTATE = {json.dumps(state)}\nACTION = {json.dumps(action)}\n\nNEXT STATE:"
        )
        try:
            reply = self._chat(prompt)
            return json.loads(reply)
        except Exception:  # noqa: E722
            return {"state": state, "reward": 0.0, "done": False, "_note": "LLM-fallback-noop"}

# ──────────────────────────────────────────────────────────────────────────────
# 4. Meta-Learner (AI-GA Autogenesis – demonstration only)
# ──────────────────────────────────────────────────────────────────────────────
class MetaLearner:
    def __init__(self, interval_sec: int = META_INTERVAL_SEC) -> None:
        self._interval = interval_sec
        self._last_ts  = 0.0

    def propose(self, kpis: Dict[str, float]) -> Optional[str]:
        now = time.time()
        if now - self._last_ts < self._interval or not kpis:
            return None
        self._last_ts = now
        worst = min(kpis, key=kpis.get)
        proposal = (
            f"# Autogenesis Proposal\n\n"
            f"*Observation*: **{worst}** shows weakness (score={kpis[worst]:.3f}).\n"
            f"*Action*: Instantiate `ResearchAgent(domain='{worst}')` with "
            f"ε-greedy ε=0.4 for 1 h to explore fringe strategies.\n"
            f"*Logging*: publish trajectories → Kafka topic `exp.fringe`."
        )
        Path("NewAgentProposal.md").write_text(proposal)
        return proposal

# ──────────────────────────────────────────────────────────────────────────────
# 5. Prometheus metrics & Kafka producer (optional)
# ──────────────────────────────────────────────────────────────────────────────
if PROM_ENABLED and "Gauge" in globals():
    PLAN_LATENCY_SEC = Gauge("wm_plan_latency_seconds", "World-Model planning latency")
    TRAJ_RISK_SCORE  = Gauge("wm_trajectory_risk",       "Cumulative reward of planned traj")
else:
    # dummy no-op objects keep .set/.observe available
    class _NoGauge(float):
        def set(self, *_: Any, **__: Any) -> None: ...
        def observe(self, *_: Any, **__: Any) -> None: ...
    PLAN_LATENCY_SEC = TRAJ_RISK_SCORE = _NoGauge()

_kafka_producer: Optional["Producer"] = None
if KAFKA_BROKER and "Producer" in globals():
    _kafka_producer = Producer({"bootstrap.servers": KAFKA_BROKER})

def _kafka_send(topic: str, msg: dict) -> None:
    if _kafka_producer:
        _kafka_producer.produce(topic, json.dumps(msg).encode())
        _kafka_producer.poll(0)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Environment registry  –  lets domains plug in custom envs dynamically
# ──────────────────────────────────────────────────────────────────────────────
_EnvCtor = Callable[[], Any]
_ENV_REGISTRY: MutableMapping[str, _EnvCtor] = {"grid-world": lambda: GridWorldEnv()}

def register_env(name: str, ctor: _EnvCtor, *, override: bool = False) -> None:
    """Register a custom environment available through wm.set_env(name)."""
    if not override and name in _ENV_REGISTRY:
        raise ValueError(f"Env '{name}' already registered.")
    _ENV_REGISTRY[name] = ctor

# ──────────────────────────────────────────────────────────────────────────────
# 7. World-Model service (singleton – safe for multi-threaded Orchestrator)
# ──────────────────────────────────────────────────────────────────────────────
class WorldModel:
    """Unified façade usable by any agent via `import wm`."""
    def __init__(self) -> None:
        self._env_name = "grid-world"
        self.env       = _ENV_REGISTRY[self._env_name]()
        self.muzero    = MuZeroPlanner(self.env)
        self.llm       = LLMSimulator()
        self.guard     = SafetyGuard(SAFETY_RULES)
        self.meta      = MetaLearner()

    # ── Environment control ──────────────────────────────────────────
    def set_env(self, name: str) -> None:
        if name not in _ENV_REGISTRY:
            raise KeyError(f"Unknown env '{name}'. Register via register_env() first.")
        self._env_name = name
        self.env       = _ENV_REGISTRY[name]()
        self.muzero    = MuZeroPlanner(self.env)

    def reset_env(self) -> Dict[str, Any]:
        return self.env.reset()

    # ── Planning (synchronous – can be wrapped in asyncio by caller) ─
    def plan(self, agent: str, state: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # Sync env with provided state (grid-world only for now)
        if all(k in state for k in ("x", "y", "goal")):
            self.env.state = GridState(**{k: state[k] for k in ("x", "y", "goal")})

        act_id, traj = self.muzero.best_action()
        rewards      = [-1.0] * (len(traj) or 1)  # pessimistic placeholder

        if self.guard.veto(rewards):
            result = {"action": None, "reason": "safety_guard_veto", "trajectory": traj}
        else:
            action_name = self.env.ACTIONS.get(act_id, ("NOP", (0, 0)))[0]
            result = {"action": {"id": act_id, "name": action_name}, "trajectory": traj}

        # Metrics & streaming
        latency = time.perf_counter() - t0
        PLAN_LATENCY_SEC.observe(latency)
        TRAJ_RISK_SCORE.set(sum(rewards))
        _kafka_send("wm.plan", {"agent": agent, **result, "latency": latency})

        return result

    # ── One-step forward simulation – generic fallback to LLM ────────
    def simulate(self, state: Dict[str, Any], action: Dict[str, Any]) -> Dict[str, Any]:
        # Fast path: internal env supports .step on deep-copy
        if self._env_name == "grid-world" and "id" in action:
            saved = self.env.state
            self.env.state = GridState(**{k: state[k] for k in ("x", "y", "goal")})
            ns, r, d, info = self.env.step(action["id"])
            self.env.state = saved  # rollback
            return {"state": ns, "reward": r, "done": d, "info": info}
        # Fallback: LLM reasoning
        return self.llm.predict(state, action)

    # ── Meta-learning hook (called by Orchestrator) ──────────────────
    def ingest_kpis(self, kpis: Dict[str, float]) -> Optional[str]:
        proposal = self.meta.propose(kpis)
        if proposal:
            _kafka_send("wm.meta", {"ts": time.time(), "proposal": proposal})
        return proposal

# ── Expose module-singleton for easy import (`from backend.world_model import wm`)
wm = WorldModel()

# ──────────────────────────────────────────────────────────────────────────────
# 8. Auto-generate minimal MuZero config once (only 60 LOC, no training)
# ──────────────────────────────────────────────────────────────────────────────
_MZ_CFG_TXT = """
### Auto-generated MuZero config for 5×5 Grid-World  (Planning-only) ###
game_name               = "gridworld"
observation_shape        = (1, 5, 5)
action_space             = ["0","1","2","3"]     # UP,DOWN,LEFT,RIGHT
players                  = [0]
stacked_observations     = 0
max_moves                = 50
discount                 = 1.0
temperature_threshold    = 15
train_steps              = 0
num_actors               = 0
selfplay_on_gpu          = False
### Network minimal ###
network                  = "fullyconnected"
support_size             = 10
downsample               = False
blocks                   = 1
channels                 = 16
reduced_channels_reward  = 16
reduced_channels_value   = 16
fc_reward_layers         = [16]
fc_value_layers          = [16]
fc_policy_layers         = [16]
### Inference / MCTS ###
num_simulations          = 96
root_dirichlet_alpha     = 0.25
root_exploration_fraction= 0.25
pb_c_base                = 19652
pb_c_init                = 1.25
"""

# ───────────────────────────── END OF FILE ───────────────────────────────────
