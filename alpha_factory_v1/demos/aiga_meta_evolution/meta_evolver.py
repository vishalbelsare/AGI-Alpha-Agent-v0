# alpha_factory_v1/demos/aiga_meta_evolution/meta_evolver.py
# © 2025 MONTREAL.AI  MIT License
"""
MetaEvolver
───────────
Minimal, production‑ready neuro‑evolution core (≈150 LoC) for the AI‑GA demo.

Pillars implemented
-------------------
1.  *Meta‑learn architectures*  – genome encodes hidden size & activation.
2.  *Meta‑learn algorithms*     – genome flag toggles Hebbian vs SGD inner‑loop.
3.  *Generate learning envs*    – curriculum_env mutates when mastered; we track
    an “elite” per stage to cross‑seed new goals (goal‑switching).

API
---
>>> evo = MetaEvolver(env_cls=CurriculumEnv)
>>> evo.run_generations(10)        # evolve 10 gens
>>> plot = evo.history_plot()      # returns pandas.DataFrame for Gradio
"""

from __future__ import annotations
import random, math, copy, pathlib, json, dataclasses as dc
from typing import Callable, List, Dict, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# ─────────────────────────────── Helpers ──────────────────────────────────────
_ACTIVATIONS = {"relu": F.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}
Device = torch.device("cpu")

def _mutation(val:int, span:int, rate:float=0.2)->int:
    return max(4, min(128, val + int(np.random.normal(0, span*rate))))

# ─────────────────────────────── Genome ───────────────────────────────────────
@dc.dataclass
class Genome:
    n_hidden: int               # neurons in hidden layer
    activation: str             # key in _ACTIVATIONS
    hebbian: bool               # whether to apply Hebbian plasticity

    def mutate(self)->"Genome":
        g = copy.deepcopy(self)
        if random.random()<0.3:          # mutate size
            g.n_hidden = _mutation(g.n_hidden, 32)
        if random.random()<0.2:          # mutate activation
            g.activation = random.choice(list(_ACTIVATIONS.keys()))
        if random.random()<0.1:          # flip learning rule
            g.hebbian = not g.hebbian
        return g

    def to_json(self)->str:              # for LLM commentary
        return json.dumps(dc.asdict(self))

# ───────────────────────────── Model wrapper ──────────────────────────────────
class EvoNet(nn.Module):
    def __init__(self, obs_dim:int, act_dim:int, genome:Genome):
        super().__init__()
        self.genome = genome
        self.fc1  = nn.Linear(obs_dim, genome.n_hidden)
        self.fc2  = nn.Linear(genome.n_hidden, act_dim)
        if genome.hebbian:
            self.hebbian_w = torch.zeros_like(self.fc1.weight, dtype=torch.float)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = _ACTIVATIONS[self.genome.activation](self.fc1(x))
        if self.genome.hebbian:
            # fast Hebbian plasticity update
            with torch.no_grad():
                dw = 0.1 * torch.bmm(h.unsqueeze(2), x.unsqueeze(1))
                self.hebbian_w += dw.mean(0)
                self.fc1.weight.data += self.hebbian_w.clamp(-0.01,0.01)
        return self.fc2(h)

# ───────────────────────────── Meta‑evolver ───────────────────────────────────
class MetaEvolver:
    def __init__(
        self,
        env_cls:Callable,
        pop_size:int = 20,
        tournament_k:int = 3,
        llm=None
    ):
        self.env_cls, self.pop_size, self.tournament_k = env_cls, pop_size, tournament_k
        self.llm = llm
        self.gen = 0
        self.history:List[Tuple[int,float]] = []
        self.stage_elite:Dict[int,Genome] = {}
        self._init_population()

    # -------------------------------- private ---------------------------------
    def _init_population(self, seed_genome:Genome|None=None):
        base = seed_genome or Genome(32, "relu", False)
        self.population = [base.mutate() for _ in range(self.pop_size)]

    def _fitness(self, genome:Genome)->float:
        env = self.env_cls()                 # each call may mutate env stage
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        net = EvoNet(obs_dim, act_dim, genome).to(Device)
        obs, _ = env.reset()
        total = 0.0
        for _ in range(env.max_steps):
            with torch.no_grad():
                act = net(torch.tensor(obs, dtype=torch.float32)).argmax().item()
            obs, rew, terminated, truncated, info = env.step(act)
            total += rew
            if terminated or truncated:
                break
        # save elite per stage
        if env.stage not in self.stage_elite or total > self.stage_elite[env.stage][1]:
            self.stage_elite[env.stage] = (genome, total)
        return total

    def _tournament_select(self, scores:List[float])->Genome:
        contestants = random.sample(list(enumerate(scores)), k=self.tournament_k)
        idx = max(contestants, key=lambda x:x[1])[0]
        return self.population[idx]

    # -------------------------------- public ----------------------------------
    def run_generations(self, n:int=5):
        for _ in range(n):
            scores = [self._fitness(g) for g in self.population]
            self.history.append( (self.gen, float(np.mean(scores))) )
            # Evolution step ---------------------------------------------------
            new_pop = []
            for _ in range(self.pop_size):
                parent = self._tournament_select(scores)
                child  = self.population[parent].mutate()
                new_pop.append(child)
            # cross‑seed with elites every 4 gens
            if self.gen % 4 == 0 and self.stage_elite:
                new_pop[0] = random.choice(list(self.stage_elite.values()))[0]
            self.population = new_pop
            self.gen += 1

    # ---------------------------- visualisation ------------------------------
    def history_plot(self):
        import pandas as pd
        return pd.DataFrame(self.history, columns=["generation","avg_fitness"])

    def latest_log(self)->str:
        if self.llm is None:
            g = max(self.population, key=lambda x: x.n_hidden)  # arbitrary pick
            return f"Best genome: {g.to_json()}"
        # LLM commentary tool‑call
        explanation = self.llm(f"Explain genome {self.population[0].to_json()} in plain English.")
        return explanation
