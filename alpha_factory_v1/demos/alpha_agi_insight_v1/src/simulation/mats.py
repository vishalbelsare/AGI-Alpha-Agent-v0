"""NSGA-II style evolutionary optimiser."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Tuple


@dataclass(slots=True)
class Individual:
    genome: List[float]
    fitness: Tuple[float, float] | None = None
    crowd: float = 0.0
    rank: int = 0


Population = List[Individual]


def evaluate(pop: Population, fn: Callable[[List[float]], Tuple[float, float]]) -> None:
    for ind in pop:
        ind.fitness = fn(ind.genome)


def _crowding(pop: Population) -> None:
    if not pop or pop[0].fitness is None:
        return
    m = len(pop[0].fitness)
    for ind in pop:
        ind.crowd = 0.0
    for i in range(m):
        pop.sort(key=lambda x: x.fitness[i])  # type: ignore[index]
        pop[0].crowd = pop[-1].crowd = float("inf")
        fmin = pop[0].fitness[i]
        fmax = pop[-1].fitness[i]
        span = fmax - fmin or 1.0
        for j in range(1, len(pop) - 1):
            prev_f = pop[j - 1].fitness[i]
            next_f = pop[j + 1].fitness[i]
            pop[j].crowd += (next_f - prev_f) / span


def _non_dominated_sort(pop: Population) -> List[Population]:
    fronts: List[Population] = []
    S = {id(ind): [] for ind in pop}
    n = {id(ind): 0 for ind in pop}
    for p in pop:
        for q in pop:
            if p is q:
                continue
            if all(pf <= qf for pf, qf in zip(p.fitness, q.fitness)):  # type: ignore[arg-type]
                if any(pf < qf for pf, qf in zip(p.fitness, q.fitness)):  # type: ignore[arg-type]
                    S[id(p)].append(q)
            elif all(qf <= pf for pf, qf in zip(p.fitness, q.fitness)):  # type: ignore[arg-type]
                if any(qf < pf for pf, qf in zip(p.fitness, q.fitness)):  # type: ignore[arg-type]
                    n[id(p)] += 1
        if n[id(p)] == 0:
            p.rank = 0
            if not fronts:
                fronts.append([])
            fronts[0].append(p)
    i = 0
    while i < len(fronts):
        nxt: Population = []
        for p in fronts[i]:
            for q in S[id(p)]:
                n[id(q)] -= 1
                if n[id(q)] == 0:
                    q.rank = i + 1
                    nxt.append(q)
        if nxt:
            fronts.append(nxt)
        i += 1
    return fronts


def nsga2_step(pop: Population, fn: Callable[[List[float]], Tuple[float, float]], mu: int = 20) -> Population:
    evaluate(pop, fn)
    offspring: Population = []
    while len(offspring) < mu:
        a, b = random.sample(pop, 2)
        cut = random.randint(1, len(a.genome) - 1)
        child_genome = a.genome[:cut] + b.genome[cut:]
        if random.random() < 0.1:
            idx = random.randrange(len(child_genome))
            child_genome[idx] += random.uniform(-1, 1)
        offspring.append(Individual(child_genome))
    evaluate(offspring, fn)
    union = pop + offspring
    fronts = _non_dominated_sort(union)
    new_pop: Population = []
    for front in fronts:
        _crowding(front)
        front.sort(key=lambda x: (-x.rank, -x.crowd))
        for ind in front:
            if len(new_pop) < mu:
                new_pop.append(ind)
    return new_pop


def run_evolution(
    fn: Callable[[List[float]], Tuple[float, float]],
    genome_length: int,
    *,
    population_size: int = 20,
    mutation_rate: float = 0.1,
    generations: int = 10,
    seed: int | None = None,
) -> Population:
    """Execute a complete NSGA-II evolutionary run.

    Args:
        fn: Function evaluating an individual's genome.
        genome_length: Number of float genes per individual.
        population_size: Number of individuals preserved each generation.
        mutation_rate: Probability of mutating a gene during crossover.
        generations: Number of NSGA-II steps to perform.
        seed: Optional random seed for deterministic behaviour.

    Returns:
        The final population after ``generations`` steps.
    """

    rng = random.Random(seed)
    pop = [Individual([rng.uniform(-1, 1) for _ in range(genome_length)]) for _ in range(population_size)]

    def _step(population: Population) -> Population:
        evaluate(population, fn)
        offspring: Population = []
        while len(offspring) < population_size:
            a, b = rng.sample(population, 2)
            cut = rng.randint(1, genome_length - 1)
            child_genome = a.genome[:cut] + b.genome[cut:]
            if rng.random() < mutation_rate:
                idx = rng.randrange(genome_length)
                child_genome[idx] += rng.uniform(-1, 1)
            offspring.append(Individual(child_genome))
        evaluate(offspring, fn)
        union = population + offspring
        fronts = _non_dominated_sort(union)
        new_pop: Population = []
        for front in fronts:
            _crowding(front)
            front.sort(key=lambda x: (-x.rank, -x.crowd))
            for ind in front:
                if len(new_pop) < population_size:
                    new_pop.append(ind)
        return new_pop

    for _ in range(generations):
        pop = _step(pop)

    return pop
