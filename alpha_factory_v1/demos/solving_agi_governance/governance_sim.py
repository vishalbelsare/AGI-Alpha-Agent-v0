#!/usr/bin/env python3
"""Simple Monte-Carlo simulation illustrating cooperative equilibrium.

Each agent plays a discounted iterated Prisoner's Dilemma with token staking.
Cooperation becomes dominant when the discount factor ``delta`` is sufficiently
large (δ ≥ 0.8 as suggested by the README).
"""
from __future__ import annotations

import argparse
import random

# Payoff matrix for a standard Prisoner's Dilemma (T > R > P > S)
R, T, P, S = 3.0, 5.0, 1.0, 0.0


def run_sim(agents: int, rounds: int, delta: float, stake: float) -> float:
    """Return the mean cooperation probability after ``rounds`` iterations."""
    probs = [random.random() for _ in range(agents)]
    for _ in range(rounds):
        avg_p = sum(probs) / agents
        coop_payoff = R * avg_p + S * (1 - avg_p)
        defect_payoff = T * avg_p + P * (1 - avg_p) - stake
        for i, p in enumerate(probs):
            avg_payoff = p * coop_payoff + (1 - p) * defect_payoff
            p += delta * p * (coop_payoff - avg_payoff)
            probs[i] = max(0.0, min(1.0, p))
    return sum(probs) / agents


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="AGIALPHA governance Monte-Carlo demo")
    ap.add_argument("-N", "--agents", type=int, default=100, help="number of agents")
    ap.add_argument("-r", "--rounds", type=int, default=1000, help="simulation rounds")
    ap.add_argument("--delta", type=float, default=0.8, help="discount factor δ")
    ap.add_argument("--stake", type=float, default=2.5, help="stake penalty")
    args = ap.parse_args(argv)

    coop = run_sim(args.agents, args.rounds, args.delta, args.stake)
    print(f"mean cooperation ≈ {coop:.3f}")


if __name__ == "__main__":
    main()
