# SPDX-License-Identifier: Apache-2.0

from alpha_factory_v1.core.evaluators import (
    LogicCritic,
    FeasibilityCritic,
    load_logic_examples,
)

DATA = load_logic_examples()


def test_logic_scores_monotonic() -> None:
    critic = LogicCritic(DATA, seed=1)
    scores = [critic.score(item) for item in DATA]
    assert scores == sorted(scores)


def test_feasibility_scores_monotonic() -> None:
    critic = FeasibilityCritic(DATA, seed=1)
    scores = [critic.score(item) for item in DATA]
    assert scores == sorted(scores)
