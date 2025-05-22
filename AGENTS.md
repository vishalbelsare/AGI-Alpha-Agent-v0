# AGENTS.md

These guidelines apply to all contributors and automated agents.

## Coding style
- Target Python 3.11 or newer and use type hints.
- Indent with 4 spaces and keep lines under 120 characters.
- Write concise Google style docstrings for all public modules, classes and functions.

## Workflow
1. Run `.codex/setup.sh` to configure the environment.
2. Execute `python check_env.py --auto-install`.
3. Run `pytest -q`. If tests fail, note the cause in the pull request description.
