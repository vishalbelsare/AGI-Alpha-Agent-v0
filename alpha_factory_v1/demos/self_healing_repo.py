"""
Demo -- Self-Healing Repository
Run:  python -m alpha_factory_v1.demos.self_healing_repo
"""

from agents import Runner
from alpha_factory_v1.backend.agent_factory import build_core_agent

bug_fixer = build_core_agent(
    name="Bug-Fixer",
    instructions=(
        "You are an elite software engineer. "
        "When tests fail you edit code, rerun pytest, and repeat "
        "until all tests pass."
    ),
)

if __name__ == "__main__":
    result = Runner.run_sync(
        bug_fixer,
        "Our CI is red.  Make the tests green, commit the diff.",
        max_turns=6,
    )
    print(result.final_output)
