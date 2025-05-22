# Contributor Guide

This repository hosts the Alpha-Factory v1 package and demos. These rules apply to all contributors and automated agents.

## Development Environment
- Run `./codex/setup.sh` to install minimal dependencies. Set `WHEELHOUSE=/path/to/wheels` for offline installs.
- After setup run `python check_env.py --auto-install` to verify and auto-install any missing packages.
- Execute `pytest -q` and ensure all tests pass. If a test fails explain the reason in the PR description.

## Coding Style
- Target **Python 3.11** or newer and provide type hints.
- Use 4 spaces per indentation level and keep lines under 120 characters.
- Write concise Google style docstrings for public modules, classes and functions.

## Pull Requests
- Keep commits focused and descriptive.
- Ensure the working tree is clean (`git status` shows no changes).
- Summarize your changes and test results in the PR body.

