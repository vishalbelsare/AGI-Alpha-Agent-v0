"""
Tracer captures agent spans and writes them to memory for inspection /
regression diffs. It plugs into PlannerAgent and domain agents via
`Tracer.record(agent, phase, payload)`.
"""
import datetime, logging

log = logging.getLogger("Tracer")

class Tracer:
    def __init__(self, memory):
        self.mem = memory

    def record(self, agent_name: str, phase: str, payload):
        span = {
            "ts": datetime.datetime.utcnow().isoformat(),
            "phase": phase,
            "payload": payload,
        }
        self.mem.write(agent_name, f"trace:{phase}", span)
        log.debug("Trace %s %s", agent_name, phase)

