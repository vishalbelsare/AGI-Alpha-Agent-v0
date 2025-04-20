"""
a2a_client.py
--------------
Tiny helper to call remote agents via Google’s Agent‑to‑Agent (A2A) protocol.
The local Planner posts a Task envelope to /rpc on the remote host;
the remote returns an immediate Ack {task_id, status}.
We then poll /artifact/<task_id> until the Artifact is ready.
"""

import json, time, logging, requests, uuid, os
from typing import Dict, Any

log = logging.getLogger("A2AClient")
TIMEOUT = int(os.getenv("A2A_TIMEOUT", "10"))      # seconds
POLL_INT = float(os.getenv("A2A_POLL_SEC", "1.0"))

def _rpc_url(host: str) -> str:        return f"http://{host}/rpc"
def _artifact_url(host: str, tid: str) -> str: return f"http://{host}/artifact/{tid}"


def send_task(host: str, agent: str, payload: Dict[str, Any]) -> str:
    """Return task_id on success, raise on error."""
    env = {
        "task_id": str(uuid.uuid4()),
        "agent": agent,
        "payload": payload,
        "timestamp": int(time.time()),
    }
    r = requests.post(_rpc_url(host), json=env, timeout=TIMEOUT)
    r.raise_for_status()
    tid = r.json().get("task_id") or env["task_id"]
    return tid


def await_artifact(host: str, task_id: str) -> Dict[str, Any]:
    """Poll remote for up to TIMEOUT seconds, return artifact dict."""
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        r = requests.get(_artifact_url(host, task_id), timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
        time.sleep(POLL_INT)
    raise TimeoutError(f"Artifact for {task_id} not ready on {host}")

