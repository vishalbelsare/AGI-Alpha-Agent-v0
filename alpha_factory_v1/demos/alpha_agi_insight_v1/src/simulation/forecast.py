"""Thermodynamic trigger forecasting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

from .sector import Sector
from . import mats


@dataclass(slots=True)
class ForecastPoint:
    year: int
    capability: float
    affected: List[Sector]


@dataclass(slots=True)
class TrajectoryPoint:
    """Detailed snapshot for a single year."""

    year: int
    capability: float
    sectors: List[Sector]


def logistic_curve(t: float, k: float = 1.0, x0: float = 0.0) -> float:
    return 1.0 / (1.0 + math.exp(-k * (t - x0)))


def linear_curve(t: float) -> float:
    return max(0.0, min(1.0, t))


def exponential_curve(t: float, k: float = 3.0) -> float:
    scale = math.exp(k) - 1.0
    return min(1.0, (math.exp(k * t) - 1.0) / scale)


def capability_growth(t: float, curve: str = "logistic") -> float:
    if curve == "linear":
        return linear_curve(t)
    if curve == "exponential":
        return exponential_curve(t)
    return logistic_curve(10.0 * t)


def free_energy(sector: Sector, capability: float) -> float:
    return sector.energy - capability * sector.entropy


def thermodynamic_trigger(sector: Sector, capability: float) -> bool:
    return free_energy(sector, capability) < 0


def _innovation_gain(pop_size: int = 6, generations: int = 1) -> float:
    """Return a small gain from a short MATS run."""

    def fn(genome: list[float]) -> tuple[float, float]:
        x, y = genome
        return x**2, y**2

    pop = mats.run_evolution(fn, 2, population_size=pop_size, generations=generations, seed=42)
    best = min(pop, key=lambda ind: sum(ind.fitness or (0.0, 0.0)))
    return 0.1 / (1.0 + sum(best.fitness or (0.0, 0.0)))


def forecast_disruptions(
    sectors: Iterable[Sector],
    horizon: int,
    curve: str = "logistic",
    *,
    pop_size: int = 6,
    generations: int = 1,
) -> List[TrajectoryPoint]:
    """Simulate sector trajectories and disruption events."""

    secs = list(sectors)
    results: List[TrajectoryPoint] = []
    for year in range(1, horizon + 1):
        t = year / horizon
        cap = capability_growth(t, curve)
        affected: List[Sector] = []
        for sec in secs:
            if not sec.disrupted:
                sec.energy *= 1.0 + sec.growth
                if thermodynamic_trigger(sec, cap):
                    sec.disrupted = True
                    sec.energy += _innovation_gain(pop_size, generations)
                    affected.append(sec)
        snapshot = [Sector(s.name, s.energy, s.entropy, s.growth, s.disrupted) for s in secs]
        results.append(TrajectoryPoint(year, cap, snapshot))
    return results


def simulate_years(sectors: Iterable[Sector], horizon: int) -> List[ForecastPoint]:
    results: List[ForecastPoint] = []
    traj = forecast_disruptions(sectors, horizon)
    for point in traj:
        affected = [s for s in point.sectors if s.disrupted]
        results.append(ForecastPoint(point.year, point.capability, affected))
    return results
