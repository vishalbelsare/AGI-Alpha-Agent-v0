# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from alpha_factory_v1.core.critics import DualCriticService
from alpha_factory_v1.core.analysis.chaos_monkey import ChaosMonkey


def test_adversarial_cases_detected() -> None:
    service = DualCriticService(["The sky is blue."])
    monkey = ChaosMonkey(service)
    fraction = monkey.detected_fraction("The sky is blue.", "The sky is blue.")
    assert fraction >= 0.8
