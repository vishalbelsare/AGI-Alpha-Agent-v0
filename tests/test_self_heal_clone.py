# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
import subprocess
import importlib
from pathlib import Path

import pytest

pytest.importorskip("gradio")
from alpha_factory_v1.demos.self_healing_repo import agent_selfheal_entrypoint as entrypoint


def test_clone_sample_repo_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "repo"
    monkeypatch.setenv("CLONE_DIR", str(target))
    importlib.reload(entrypoint)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))
    entrypoint.clone_sample_repo()
    assert (target / "calc.py").exists()
