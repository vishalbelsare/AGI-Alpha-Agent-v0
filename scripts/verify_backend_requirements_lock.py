#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Ensure backend requirements-lock.txt matches requirements.txt."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
import tempfile


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    req_txt = repo_root / "alpha_factory_v1" / "backend" / "requirements.txt"
    lock_file = repo_root / "alpha_factory_v1" / "backend" / "requirements-lock.txt"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "requirements.lock"
        pip_compile = shutil.which("pip-compile")
        if pip_compile:
            cmd = [pip_compile]
        else:
            cmd = [sys.executable, "-m", "piptools", "compile"]
        wheelhouse = os.getenv("WHEELHOUSE")
        cmd += ["--quiet"]
        if wheelhouse:
            cmd += ["--no-index", "--find-links", wheelhouse]
        cmd += ["--generate-hashes", str(req_txt), "-o", str(out_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        if result.returncode != 0:
            return result.returncode
        if out_path.read_bytes() != lock_file.read_bytes():
            extra = ""
            if wheelhouse:
                extra = f"--no-index --find-links {wheelhouse} "
            msg = (
                "alpha_factory_v1/backend/requirements-lock.txt is outdated. Run 'pip-compile "
                f"{extra}--quiet --generate-hashes alpha_factory_v1/backend/requirements.txt'\n"
            )
            sys.stderr.write(msg)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
