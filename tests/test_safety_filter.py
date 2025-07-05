# SPDX-License-Identifier: Apache-2.0
"""Tests for self-edit safety filters."""

from pathlib import Path
import random

from alpha_factory_v1.core.self_edit.safety import is_patch_safe


FIXTURES = Path(__file__).parent / "fixtures"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_blocks_malicious_patch() -> None:
    diff = _read("malicious_patch.diff")
    assert not is_patch_safe(diff)


def test_allows_safe_patch() -> None:
    diff = _read("safe_patch.diff")
    assert not is_patch_safe(diff)


from alpha_factory_v1.core.simulation import SelfRewriteOperator  # noqa: E402


def test_rewrite_blocks_malicious() -> None:
    op = SelfRewriteOperator(steps=1, rng=random.Random(0))
    code = "import os\nos.system('rm -rf /')"
    result = op(code)
    assert "os.system" in result
