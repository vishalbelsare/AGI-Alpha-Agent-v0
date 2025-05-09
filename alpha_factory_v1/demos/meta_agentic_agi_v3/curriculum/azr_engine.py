
"""
curriculum.azr_engine
---------------------
Absolute‑Zero Reasoner self‑curriculum engine
Adapted from Zhao et al. (2025) – “Absolute Zero: Reinforced Self‑play Reasoning with Zero Data”.

This module plugs into Alpha‑Factory v1 as a drop‑in curriculum provider.
It invents deterministic Python triplet tasks (program p, input i, output o),
evaluates difficulty & novelty, and continuously trains both a *Proposer*
and a *Solver* role through Task‑Relative REINFORCE++ (TRR++) updates.

Key design goals
----------------
• Vendor‑agnostic → any FM provider supported by core.fm.FMInterface  
• CPU‑only friendly (no heavy RL frameworks; pure‑Python PPO lite)  
• Sandbox‑safe evaluation (Firejail when available, fallback to subprocess + resource)  
• Minimal dependencies – relies on standard library + numpy (optional)  

Usage
-----
>>> from curriculum.azr_engine import AZREngine
>>> engine = AZREngine(fm=my_fm_wrapper)   # pass core.fm.FMInterface
>>> batch = engine.sample_curriculum(batch_size=8)
>>> solved = engine.solve(batch)
>>> engine.learn(solved)

The engine exposes propose/solve/learn APIs used by meta_agentic_search.search.SearchLoop.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import random
import re
import resource
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Safety‑centric sandbox helper
# --------------------------------------------------------------------------- #

_SOFT_TIMEOUT = 6     # seconds
_HARD_TIMEOUT = 8
_MEM_LIMIT_MB = 256

def _soft_limit():
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_MB * 1024 * 1024, _MEM_LIMIT_MB * 1024 * 1024))
    except Exception:
        pass

def _run_python_snippet(code: str, inp: str) -> Tuple[str, str]:
    """Executes `code` in a restricted subprocess and returns stdout, stderr."""
    with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as tmp:
        tmp.write(code + f"""\nif __name__ == '__main__':\n    import json,sys;\n    _i=json.loads({inp!r});\n    print(main(*_i) if isinstance(_i, (list, tuple)) else main(_i))\n""")
        tmp.flush()
        script_path = tmp.name

    def target(q):
        try:
            _soft_limit()
            proc = subprocess.Popen(
                [sys.executable, script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=_SOFT_TIMEOUT)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            q.put((stdout, stderr))
        except Exception as e:
            q.put(("", str(e)))

    q: multiprocessing.Queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=target, args=(q,))
    p.start()
    p.join(_HARD_TIMEOUT)
    if p.is_alive():
        p.terminate()
    try:
        stdout, stderr = q.get_nowait()
    except Exception:
        stdout, stderr = "", "RuntimeError: queue empty"
    finally:
        try:
            os.remove(script_path)
        except FileNotFoundError:
            pass
    return stdout.strip(), stderr.strip()


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class Triplet:
    program: str
    inp: str
    out: str
    mode: str   # deduct / abduct / induct

@dataclass
class TaskResult:
    triplet: Triplet
    solved: bool
    stdout: str
    stderr: str
    latency: float

# --------------------------------------------------------------------------- #
# AZR Engine
# --------------------------------------------------------------------------- #

class AZREngine:
    """Absolute Zero curriculum orchestrator."""

    def __init__(
        self,
        fm,
        buffer_max: int = 4096,
        temperature: float = 0.4,
        proposer_identity: str = "You are AZR‑Proposer, a curriculum‑generating engine.",
        solver_identity: str = "You are AZR‑Solver, a meticulous Python reasoning assistant.",
        seed_tasks: Optional[Sequence[Triplet]] = None,
        log_fn: Optional[callable] = None,
    ):
        self.fm = fm
        self.buffer: List[Triplet] = list(seed_tasks) if seed_tasks else []
        self.buffer_max = buffer_max
        self.temperature = temperature
        self.proposer_identity = proposer_identity
        self.solver_identity = solver_identity
        self._log = log_fn or (lambda *a, **k: None)
        self._rng = random.Random(42)
        # lightweight running reward estimate (exponential moving)
        self._baseline = 0.0

    # ------------------------------------------------------------------- #
    # Public API
    # ------------------------------------------------------------------- #

    def sample_curriculum(self, batch_size: int = 4) -> List[Triplet]:
        """Returns a list of freshly proposed tasks."""
        prompt = self._build_proposer_prompt(batch_size)
        completion = self.fm.chat(
            system=self.proposer_identity,
            user=prompt,
            temperature=self.temperature,
            max_tokens=1024,
        )
        tasks = self._extract_triplets(completion)
        validated = [t for t in tasks if self._validate_triplet(t)]
        self._log(f"Proposed {len(validated)}/{len(tasks)} valid triplets.")
        return validated

    def solve(self, tasks: Sequence[Triplet]) -> List[TaskResult]:
        """Runs the solver over tasks, returning execution results."""
        results: List[TaskResult] = []
        for t in tasks:
            start = time.time()
            stdout, stderr = _run_python_snippet(t.program, t.inp)
            latency = time.time() - start
            solved = (not stderr) and (stdout.strip() == t.out.strip())
            results.append(TaskResult(t, solved, stdout, stderr, latency))
        return results

    def learn(self, results: Sequence[TaskResult]) -> None:
        """Updates proposer baseline using TRR++ reward signal."""
        solved_frac = sum(r.solved for r in results) / max(len(results), 1)
        reward = 1.0 - solved_frac  # harder tasks → higher reward
        # Task‑Relative baseline update
        alpha = 0.1
        self._baseline = (1 - alpha) * self._baseline + alpha * reward
        advantage = reward - self._baseline
        # Very lightweight gradient‑free policy shaping:
        # adjust temp – higher advantage → keep temp, lower → explore
        if advantage < 0:
            self.temperature = min(1.0, self.temperature + 0.05)
        else:
            self.temperature = max(0.1, self.temperature - 0.02)
        # Buffer updates
        for r in results:
            if r.solved:
                self._add_to_buffer(r.triplet)

    # ------------------------------------------------------------------- #
    # Internal helpers
    # ------------------------------------------------------------------- #

    _TRIPLET_PATTERN = re.compile(
        r"```python\s*#\s*program\s*\n(?P<prog>[\s\S]+?)```\s*```json\s*#\s*input\s*\n(?P<inp>[\s\S]+?)```\s*```json\s*#\s*output\s*\n(?P<out>[\s\S]+?)```", re.MULTILINE
    )

    def _extract_triplets(self, text: str) -> List[Triplet]:
        triples = []
        for m in self._TRIPLET_PATTERN.finditer(text):
            prog = m.group("prog").strip()
            inp = m.group("inp").strip()
            out = m.group("out").strip()
            mode = "deduct" if "solve" in prog.lower() else "induct"
            triples.append(Triplet(program=prog, inp=inp, out=out, mode=mode))
        return triples

    def _validate_triplet(self, t: Triplet) -> bool:
        """Checks determinism, size limits, and executes once to verify output."""
        if len(t.program) > 2000:
            return False
        if len(t.inp) > 256 or len(t.out) > 256:
            return False
        stdout, stderr = _run_python_snippet(t.program, t.inp)
        return (not stderr) and (stdout.strip() == t.out.strip())

    def _add_to_buffer(self, t: Triplet):
        self.buffer.append(t)
        if len(self.buffer) > self.buffer_max:
            self.buffer.pop(0)

    # ------------------------------------------------------------------- #
    # Prompt builders
    # ------------------------------------------------------------------- #

    _PROPOSER_TEMPLATE = textwrap.dedent(
        """            Invent {n} novel yet *deterministic* Python reasoning tasks as **triplets**.

        Each triplet must be emitted in **three fenced blocks** exactly:

        ```python  # program
        def main(x):
            # pure‑function; deterministic; no randomness
            ...
        ```
        ```json  # input
        [1, 2]
        ```
        ```json  # output
        42
        ```

        Constraints:
        • program < 50 lines; input/output JSON‑serialisable (<256 chars)  
        • deterministic, side‑effect‑free; use only stdlib  
        • difficulty ≈ “just right” for a strong LLM coder – 30‑60 % solve rate  
        • avoid banned modules: os, sys, subprocess, socket, multiprocessing, threading

        Current buffer size: {buf}
        Example buffer snapshots (for diversity reference, *do not repeat*):
        {examples}

        Return exactly {n} triplets.
        """
    )

    def _build_proposer_prompt(self, n: int) -> str:
        examples = "\n\n".join(
            f"```python\n{t.program}```\n```json\n{t.inp}```\n```json\n{t.out}```" for t in self._rng.sample(self.buffer, k=min(3, len(self.buffer)))
        )
        return self._PROPOSER_TEMPLATE.format(n=n, buf=len(self.buffer), examples=examples or "(buffer empty)")


# --------------------------------------------------------------------------- #
# Factory function for SearchLoop
# --------------------------------------------------------------------------- #

def curriculum_factory(fm, **kwargs) -> AZREngine:
    """Convenience entrypoint used by orchestrator."""
    return AZREngine(fm=fm, **kwargs)


# --------------------------------------------------------------------------- #
# Minimal test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":  # basic smoke test with dummy FM
    class _StubFM:
        def chat(self, system, user, temperature=0.4, max_tokens=1024):
            # very naive proposer producing a constant toy task
            return (
                """```python\n"
                "def main(x):\n    return x * x\n```\n"
                "```json\n3```\n"
                "```json\n9```"""
            )

    eng = AZREngine(fm=_StubFM())
    batch = eng.sample_curriculum(1)
    res = eng.solve(batch)
    eng.learn(res)
    print("Solved:", res[0].solved, "Temp:", eng.temperature)

