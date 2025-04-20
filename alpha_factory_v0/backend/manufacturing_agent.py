import logging, random
from typing import List, Dict
from .agent_base import AgentBase
from ortools.sat.python import cp_model


class ManufacturingAgent(AgentBase):
    """
    Optimises a toy 3‑machine job‑shop each cycle to minimise makespan.
    """

    def _sample_jobs(self) -> List[List[int]]:
        # job[i] is list of processing times for machine 0,1,2
        return [[random.randint(1, 5) for _ in range(3)] for _ in range(4)]  # 4 jobs

    def observe(self) -> List[Dict]:
        jobs = self._sample_jobs()
        self.memory.write(self.name, "observation", {"jobs": jobs})
        return [{"jobs": jobs}]

    def think(self, obs):
        jobs = obs[-1]["jobs"]
        model = cp_model.CpModel()
        horizon = sum(sum(job) for job in jobs)

        starts, ends, intervals = {}, {}, {}
        machine_to_intervals = {m: [] for m in range(3)}

        for j, job in enumerate(jobs):
            previous_end = None
            for m, duration in enumerate(job):
                suffix = f"_{j}_{m}"
                start = model.NewIntVar(0, horizon, "s" + suffix)
                end = model.NewIntVar(0, horizon, "e" + suffix)
                inter = model.NewIntervalVar(start, duration, end, "i" + suffix)
                starts[(j, m)] = start
                ends[(j, m)] = end
                intervals[(j, m)] = inter
                machine_to_intervals[m].append(inter)
                if previous_end is not None:
                    model.Add(start >= previous_end)
                previous_end = end

        for m in range(3):
            model.AddNoOverlap(machine_to_intervals[m])

        makespan = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(
            makespan, [ends[(j, 2)] for j in range(len(jobs))]
        )
        model.Minimize(makespan)

        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 1
        status = solver.Solve(model)

        plan = {
            "objective": solver.Value(makespan) if status == 4 else None,
            "status": status,
        }
        self.memory.write(self.name, "idea", plan)
        return [plan]

    def act(self, tasks):
        for t in tasks:
            self.memory.write(self.name, "action", t)
            self.log.info("Optimal makespan = %s", t["objective"])

