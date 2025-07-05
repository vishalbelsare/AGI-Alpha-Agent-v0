# SPDX-License-Identifier: Apache-2.0
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

websockets = pytest.importorskip("websockets.sync.client")

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

os.environ.setdefault("API_TOKEN", "test-token")
os.environ.setdefault("API_RATE_LIMIT", "1000")

ROOT = Path(__file__).resolve().parents[1]
STUB_DIR = ROOT / "tests" / "resources"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_simulate_curve_subprocess() -> None:
    port = _free_port()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{STUB_DIR}:{ROOT}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        "-m",
        "src.interface.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(cmd, env=env)
    url = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer test-token"}
    try:
        for _ in range(100):
            try:
                r = httpx.get(f"{url}/runs", headers=headers)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.1)
        else:
            proc.terminate()
            raise AssertionError("server did not start")
        r = httpx.post(
            f"{url}/simulate",
            json={
                "horizon": 1,
                "num_sectors": 2,
                "pop_size": 2,
                "generations": 1,
                "mut_rate": 0.1,
                "xover_rate": 0.5,
                "curve": "linear",
                "energy": 1.0,
                "entropy": 1.0,
            },
            headers=headers,
        )
        assert r.status_code == 200
        sim_id = r.json()["id"]
        for _ in range(400):
            r = httpx.get(f"{url}/results/{sim_id}", headers=headers)
            if r.status_code == 200:
                results = r.json()
                break
            time.sleep(0.05)
        assert r.status_code == 200

        r_latest = httpx.get(f"{url}/results", headers=headers)
        assert r_latest.status_code == 200
        assert r_latest.json()["id"] == sim_id

        r_pop = httpx.get(f"{url}/population/{sim_id}", headers=headers)
        assert r_pop.status_code == 200
        pop = r_pop.json()
        assert pop["id"] == sim_id
        assert pop["population"] == results["population"]
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def _start_server(port: int, env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable,
        "-m",
        "src.interface.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    env = env or os.environ.copy()
    env["PYTHONPATH"] = f"{STUB_DIR}:{ROOT}:{env.get('PYTHONPATH', '')}"
    return subprocess.Popen(cmd, env=env)


def _start_demo_server(port: int, env: dict[str, str] | None = None) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable,
        "-m",
        "alpha_factory_v1.demos.alpha_agi_insight_v1.src.interface.api_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    env = env or os.environ.copy()
    env["PYTHONPATH"] = f"{STUB_DIR}:{ROOT}:{env.get('PYTHONPATH', '')}"
    return subprocess.Popen(cmd, env=env)


def _wait_running(url: str, headers: dict[str, str]) -> None:
    for _ in range(100):
        try:
            r = httpx.get(f"{url}/runs", headers=headers)
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.1)
    raise AssertionError("server did not start")


def test_results_requires_auth() -> None:
    port = _free_port()
    proc = _start_server(port)
    url = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer test-token"}
    try:
        _wait_running(url, headers)
        r = httpx.get(f"{url}/results")
        assert r.status_code == 403
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_rate_limit_exceeded() -> None:
    port = _free_port()
    env = os.environ.copy()
    env["API_RATE_LIMIT"] = "3"
    proc = _start_server(port, env)
    url = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer test-token"}
    try:
        _wait_running(url, headers)
        assert httpx.get(f"{url}/runs", headers=headers).status_code == 200
        assert httpx.get(f"{url}/runs", headers=headers).status_code == 200
        r = httpx.get(f"{url}/runs", headers=headers)
        assert r.status_code == 429
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_results_requires_auth_cors() -> None:
    port = _free_port()
    env = os.environ.copy()
    env["API_CORS_ORIGINS"] = "http://example.com"
    proc = _start_server(port, env)
    url = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer test-token"}
    try:
        _wait_running(url, headers)
        r = httpx.get(f"{url}/results", headers={"Origin": "http://example.com"})
        assert r.status_code == 403
        assert r.headers.get("access-control-allow-origin") == "http://example.com"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_problem_json_subprocess() -> None:
    port = _free_port()
    proc = _start_server(port)
    url = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer test-token"}
    try:
        _wait_running(url, headers)
        r = httpx.get(f"{url}/results/missing", headers=headers)
        assert r.status_code == 404
        assert r.headers.get("content-type", "").startswith("application/problem+json")
        data = r.json()
        assert data.get("type") == "about:blank"
        assert data.get("status") == 404
        assert "title" in data
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_ws_progress_token_param() -> None:
    port = _free_port()
    proc = _start_server(port)
    url = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer test-token"}
    try:
        _wait_running(url, headers)
        ws_url = f"ws://127.0.0.1:{port}/ws/progress?token=test-token"
        with websockets.connect(ws_url):
            pass
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_insight_endpoint_subprocess() -> None:
    port = _free_port()
    proc = _start_server(port)
    url = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer test-token"}
    try:
        _wait_running(url, headers)
        r = httpx.post(
            f"{url}/simulate",
            json={
                "horizon": 1,
                "num_sectors": 2,
                "pop_size": 2,
                "generations": 1,
                "mut_rate": 0.1,
                "xover_rate": 0.5,
                "curve": "linear",
                "energy": 1.0,
                "entropy": 1.0,
            },
            headers=headers,
        )
        assert r.status_code == 200
        sim_id = r.json()["id"]
        for _ in range(400):
            r = httpx.get(f"{url}/results/{sim_id}", headers=headers)
            if r.status_code == 200:
                results = r.json()
                break
            time.sleep(0.05)
        assert r.status_code == 200

        r_insight = httpx.post(
            f"{url}/insight",
            json={"ids": [sim_id]},
            headers=headers,
        )
        assert r_insight.status_code == 200
        assert r_insight.json()["forecast"] == results["forecast"]
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_invalid_sim_request_returns_422() -> None:
    port = _free_port()
    proc = _start_demo_server(port)
    url = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer test-token"}
    try:
        _wait_running(url, headers)
        r = httpx.post(
            f"{url}/simulate",
            json={
                "horizon": 0,
                "pop_size": 2,
                "generations": 1,
                "mut_rate": 0.1,
                "xover_rate": 0.5,
                "curve": "linear",
            },
            headers=headers,
        )
        assert r.status_code == 422
    finally:
        proc.terminate()
        proc.wait(timeout=5)
