# alpha_factory_v1/demos/aiga_meta_evolution/curriculum_env.py
# © 2025 MONTREAL.AI  MIT License
"""
CurriculumEnv
─────────────
Dynamic Gymnasium environment that **mutates its own geometry** whenever the
agent “masters” the current task, embodying Pillar 3 of Clune’s AI‑GA vision.

Levels                Genome
──────────────────────────────────────────────────────────────────────────────
0  Line‑follow   ┐     layout_id : 0‑3   (which curriculum stage)
1  Zig‑zag       ┤──►  noise      : float (wall density)
2  Gap‑cross     ┤     max_steps  : int   (episode length)
3  Maze‑nav      ┘

Mastery threshold = reach goal in < 50 % of `max_steps` five episodes in a row.
When mastered, `layout_id += 1` and we inject fresh noise + longer maze,
returning a **new environment genome** (so MetaEvolver can archive it).
"""
from __future__ import annotations
import random, numpy as np, gymnasium as gym
from gymnasium import spaces
import dataclasses as dc

Grid = np.ndarray
Vec  = tuple[int,int]
RNG  = np.random.Generator

# ─────────────────────────────── Genome ───────────────────────────────────────
@dc.dataclass
class EnvGenome:
    layout_id: int = 0      # 0‑3
    noise: float = 0.05     # probability of wall per cell
    max_steps: int = 200

    def mutate(self)->"EnvGenome":
        return EnvGenome(
            layout_id=min(self.layout_id+1, 3),
            noise=min(0.25, self.noise + 0.05),
            max_steps=int(self.max_steps * 1.2)
        )

# ───────────────────────────── Environment ───────────────────────────────────
class CurriculumEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, genome:EnvGenome|None=None, size:int=10, seed:int|None=None):
        self.rng:RNG = np.random.default_rng(seed)
        self.genome  = genome or EnvGenome()
        self.size    = size
        self.stage   = self.genome.layout_id
        # spaces ----------------------------------------------------------------
        self.action_space      = spaces.Discrete(4)  # 0U 1R 2D 3L
        low, high = np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # mastery tracking
        self._success_streak = 0

    # ---------------------------- core API ------------------------------------
    def _gen_map(self)->tuple[Grid,Vec,Vec]:
        grid = np.zeros((self.size, self.size), dtype=np.int8)
        # sprinkle walls
        walls = self.rng.random(grid.shape) < self.genome.noise
        grid[walls] = 1
        # carve level‑specific pattern
        if self.stage == 0:          # Line
            grid[:, self.size//2] = 0
        elif self.stage == 1:        # Zig‑zag
            for i in range(self.size):
                grid[i, (i//2)%self.size] = 0
        elif self.stage == 2:        # Gap
            grid[self.size//2, :] = 1
            gap = self.rng.integers(1, self.size-1)
            grid[self.size//2, gap] = 0
        else:                        # Maze (nav)
            for i in range(1,self.size-1,2):
                grid[i,1:-1] = 1
                gap = self.rng.integers(1,self.size-1)
                grid[i,gap] = 0
        # pick start/goal on free cells
        free = np.argwhere(grid==0)
        start = tuple(free[self.rng.integers(len(free))])
        goal  = tuple(free[self.rng.integers(len(free))])
        while goal == start:
            goal = tuple(free[self.rng.integers(len(free))])
        return grid, start, goal

    def reset(self, *, seed=None, options=None):
        self.steps = 0
        self.grid, self.agent, self.goal = self._gen_map()
        return self._obs(), {}

    # -------------------------------------------------------------------------
    def step(self, action:int):
        self.steps += 1
        y,x = self.agent
        if   action==0: ny,nx = y-1,x
        elif action==1: ny,nx = y,x+1
        elif action==2: ny,nx = y+1,x
        else:           ny,nx = y,x-1
        if 0<=ny<self.size and 0<=nx<self.size and self.grid[ny,nx]==0:
            self.agent = (ny,nx)
        done = self.agent == self.goal or self.steps>=self.genome.max_steps
        reward = 1.0 if self.agent==self.goal else 0.0
        # mastery check
        if done and reward>0 and self.steps < self.genome.max_steps*0.5:
            self._success_streak += 1
        else:
            self._success_streak = 0
        if self._success_streak>=5 and self.stage<3:   # mutate env
            self.genome = self.genome.mutate()
            self.stage  = self.genome.layout_id
            self._success_streak = 0
        truncated = self.steps>=self.genome.max_steps
        return self._obs(), reward, done, truncated, {}

    # -------------------------------------------------------------------------
    def _obs(self):
        ay,ax = self.agent
        gy,gx = self.goal
        return np.array([ay/self.size, ax/self.size, gy/self.size, gx/self.size],dtype=np.float32)
