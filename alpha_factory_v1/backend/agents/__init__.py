from importlib import import_module

AGENT_REGISTRY = {}
for _name in [
    "finance",
    "policy",
    "manufacturing",
    "biotech",
    "supply_chain",
    "energy",
    "climate_risk",
    "cyber_threat",
]:
    mod = import_module(f"backend.agents.{_name}_agent")
    AGENT_REGISTRY[_name] = getattr(mod, f"{_name.title().replace('_', '')}Agent")

