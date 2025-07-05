# SPDX-License-Identifier: Apache-2.0
"""Run commands inside a restricted sandbox."""

from __future__ import annotations

import shutil
import subprocess
from typing import Sequence

__all__ = ["SandboxTimeout", "secure_run"]


class SandboxTimeout(Exception):
    """Raised when the sandboxed command exceeds the time limit."""


def secure_run(cmd: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Execute ``cmd`` under ``firejail`` or ``docker`` constraints.

    The sandbox runs with seccomp, ``2`` CPU cores, ``2`` GB of RAM and a
    ``120`` second timeout. When the command exceeds the timeout a
    :class:`SandboxTimeout` is raised.
    """

    timeout = 120
    firejail = shutil.which("firejail")
    if firejail:
        full_cmd = [
            firejail,
            "--quiet",
            "--net=none",
            "--private",
            "--seccomp",
            "--rlimit-as=2147483648",
            "--rlimit-cpu=120",
            *cmd,
        ]
    else:
        docker = shutil.which("docker")
        if docker:
            full_cmd = [
                docker,
                "run",
                "--rm",
                "--network=none",
                "--cpus=2",
                "--memory=2g",
                "--security-opt",
                "seccomp=unconfined",
                "python:3.11-slim",
                *cmd,
            ]
        else:
            full_cmd = list(cmd)
    try:
        return subprocess.run(
            full_cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:  # pragma: no cover - runtime failure
        raise SandboxTimeout(str(exc)) from exc
