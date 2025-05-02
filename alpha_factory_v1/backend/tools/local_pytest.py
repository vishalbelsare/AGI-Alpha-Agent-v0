"""
FunctionTool: safely run pytest in a jailed subprocess.

Used by the self-healing demo even when no OPENAI_API_KEY is present.
"""

import resource
import subprocess
from pathlib import Path
from typing import Dict

from agents import function_tool, RunContextWrapper

CPU_SECS = 5        # hard kill after 5 CPU-seconds
MEM_MB  = 256       # 256 MB RAM cap


@function_tool(name_override="run_pytest")
def run_pytest(ctx: RunContextWrapper[Dict], path: str = ".") -> Dict:
    """Execute pytest inside a tight OS sandbox."""
    def _limit():
        resource.setrlimit(resource.RLIMIT_CPU, (CPU_SECS, CPU_SECS + 1))
        resource.setrlimit(resource.RLIMIT_AS, (MEM_MB * 1024 * 1024, -1))

    repo = Path(path).resolve()
    proc = subprocess.run(
        ["pytest", "-q", str(repo)],
        capture_output=True,
        text=True,
        preexec_fn=_limit,
    )
    return {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
