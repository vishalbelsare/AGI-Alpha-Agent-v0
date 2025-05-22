# AGENTS.md

These instructions guide both human contributors and automated agents working in this repository.

## Coding style
- Target Python 3.11 or newer and use type hints.
- Indent with 4 spaces and keep lines under 120 characters.
- Write concise Google style docstrings for all public modules, classes and functions.

## Workflow
1. Run `python check_env.py --auto-install` to ensure optional packages are available.
2. Execute `pytest -q` and confirm the suite completes. If tests fail, mention why in your pull request.
