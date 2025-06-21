# SPDX-License-Identifier: Apache-2.0
"""Ensure reloading the world model demo does not duplicate bus callbacks."""

from __future__ import annotations

import importlib
import sys

MODULE = "alpha_factory_v1.demos.alpha_asi_world_model.alpha_asi_world_model_demo"


def _load():
    if MODULE in sys.modules:
        return importlib.reload(sys.modules[MODULE])
    return importlib.import_module(MODULE)


def test_module_reload_no_duplicate_callbacks(monkeypatch) -> None:
    monkeypatch.setenv("NO_LLM", "1")
    monkeypatch.setenv("ALPHA_ASI_SILENT", "1")
    monkeypatch.setenv("ALPHA_ASI_MAX_STEPS", "1")

    mod = _load()
    counts = {k: len(v) for k, v in mod.A2ABus._subs.items()}

    for agent in mod.AGENTS.values():
        agent.close()

    mod = importlib.reload(mod)
    counts2 = {k: len(v) for k, v in mod.A2ABus._subs.items()}

    assert counts2 == counts
