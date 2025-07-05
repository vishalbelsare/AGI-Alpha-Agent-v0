# SPDX-License-Identifier: Apache-2.0
import shutil
import subprocess
from pathlib import Path

import pytest

if not shutil.which("docker"):
    pytest.skip("docker not available", allow_module_level=True)

COMPOSE_FILE = Path(__file__).resolve().parents[1] / "infrastructure" / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose_stack() -> None:
    subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "up",
            "-d",
            "agents",
        ],
        check=True,
    )
    try:
        yield
    finally:
        subprocess.run(
            [
                "docker",
                "compose",
                "-f",
                str(COMPOSE_FILE),
                "down",
                "-v",
            ],
            check=False,
        )


def test_agents_no_outbound_network(compose_stack: None) -> None:
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "exec",
            "-T",
            "agents",
            "python",
            "-c",
            "import urllib.request,sys; urllib.request.urlopen('https://example.com')",
        ],
        capture_output=True,
    )
    assert result.returncode != 0
