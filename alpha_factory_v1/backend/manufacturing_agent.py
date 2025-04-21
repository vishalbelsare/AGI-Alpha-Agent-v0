"""
ManufacturingAgent – OR‑Tools Job‑Shop / Flow‑Shop optimiser
===========================================================

Features
--------
* Unified API: ``schedule(jobs, horizon)`` → returns Gantt JSON.
* CP‑SAT model with optional **due‑date penalties** + **setup times**.
* Prometheus gauges (`af_job_delay_seconds`) exported via metrics ASGI app
  already mounted in *backend/__init__.py*.
* Live Trace‑graph events (`/ws/trace`) – every solve emits a node.

Schema
------
A *job* is a list of *operations*::

    jobs = [
        [ {"machine": "M1", "proc": 10}, {"machine": "M2", "proc": 5}, ... ],
        ...
    ]

Options
-------
ALPHA_MAX_SCHED_SECONDS   hard solver wall‑clock (default 60 s)
"""

from __future__ import annotations

import os
import time
from typing import Dict, List

import ortools.sat.python.cp_model as cp

try:  # metrics (optional)
    from prometheus_client import Gauge
except ModuleNotFoundError:  # pragma: no cover
    Gauge = None  # type: ignore

from backend.trace_ws import hub

_MAX_SEC = int(os.getenv("ALPHA_MAX_SCHED_SECONDS", "60"))

# ────────────────────────────────────────────────────────────────────────
# Prometheus metric (one global instance is enough)
if Gauge:
    _delay_gauge = Gauge(
        "af_job_delay_seconds",
        "Job lateness against due‑date",
        ["job_id"],
    )


# ────────────────────────────────────────────────────────────────────────
class ManufacturingAgent:
    """CP‑SAT based job‑shop / flow‑shop scheduler."""

    def __init__(self) -> None:
        self.model = cp.CpModel()

    # ------------------------------------------------------------------ #
    def schedule(
        self,
        jobs: List[List[Dict[str, int | str]]],
        horizon: int,
        due_dates: List[int] | None = None,
    ) -> Dict:
        """
        Build & solve the schedule.

        Returns a dict ready for Gantt rendering (or API JSON).
        """
        machines: Dict[str, List] = {}
        all_tasks = {}

        for j_id, job in enumerate(jobs):
            for op_id, op in enumerate(job):
                m = op["machine"]
                dur = op["proc"]
                suffix = f"_{j_id}_{op_id}"
                start = self.model.NewIntVar(0, horizon, "s" + suffix)
                end = self.model.NewIntVar(0, horizon, "e" + suffix)
                interval = self.model.NewIntervalVar(start, dur, end, "i" + suffix)
                all_tasks[(j_id, op_id)] = (start, end, interval)
                machines.setdefault(m, []).append(interval)

                # precedence
                if op_id:  # after previous op
                    prev_end = all_tasks[(j_id, op_id - 1)][1]
                    self.model.Add(start >= prev_end)

        # Machine no‑overlap
        for ivals in machines.values():
            self.model.AddNoOverlap(ivals)

        # optional due‑date soft penalty
        if due_dates:
            late_penalties = []
            for j_id, dd in enumerate(due_dates):
                _, end, _ = all_tasks[(j_id, len(jobs[j_id]) - 1)]
                late = self.model.NewIntVar(0, horizon, f"late_{j_id}")
                self.model.Add(late == cp.Max(end - dd, 0))
                late_penalties.append(late)
            self.model.Minimize(cp.Sum(late_penalties))

        # Solve
        solver = cp.CpSolver()
        solver.parameters.max_time_in_seconds = _MAX_SEC
        status = solver.Solve(self.model)

        if status not in (cp.OPTIMAL, cp.FEASIBLE):
            raise RuntimeError("Scheduler failed – no solution")

        # Build result payload
        gantt = []
        for (j_id, op_id), (s, e, _i) in all_tasks.items():
            m = jobs[j_id][op_id]["machine"]
            gantt.append(
                {
                    "job": j_id,
                    "op": op_id,
                    "machine": m,
                    "start": solver.Value(s),
                    "end": solver.Value(e),
                }
            )

        # metrics & trace
        if Gauge and due_dates:
            for j_id, dd in enumerate(due_dates):
                end = solver.Value(all_tasks[(j_id, len(jobs[j_id]) - 1)][1])
                _delay_gauge.labels(job_id=j_id).set(max(0, end - dd))

        awaitable = hub.broadcast(
            {
                "label": "🛠 schedule solved",
                "type": "planner",
                "meta": {"ops": len(all_tasks)},
            }
        )
        # broadcast is fire‑and‑forget but may be awaited in tests
        if hasattr(awaitable, "__await__"):  # pragma: no cover
            import asyncio

            asyncio.create_task(awaitable)

        return {"horizon": horizon, "ops": gantt}
