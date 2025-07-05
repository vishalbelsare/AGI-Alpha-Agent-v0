# SPDX-License-Identifier: Apache-2.0
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import requests

if not shutil.which("docker"):
    pytest.skip("docker not available", allow_module_level=True)

COMPOSE_FILE = Path(__file__).resolve().parents[1] / "infrastructure" / "docker-compose.yml"


def _wait(url: str, timeout: int = 60) -> bool:
    for _ in range(timeout):
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


@pytest.fixture(scope="module")
def compose_stack() -> None:
    subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d"], check=True)
    try:
        yield
    finally:
        subprocess.run(["docker", "compose", "-f", str(COMPOSE_FILE), "down", "-v"], check=False)


def test_compose_health(compose_stack: None) -> None:
    assert _wait("http://localhost:8000/healthz"), "/healthz endpoint not healthy"
    assert _wait("http://localhost:8000/readiness"), "/readiness endpoint not healthy"
