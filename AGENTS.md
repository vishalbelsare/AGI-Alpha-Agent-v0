# Contributor Guide

This repository contains the Alpha-Factory v1 package and demos.
The instructions below apply to all contributors and automated agents.

## Development Environment
- Create and activate a **Python&nbsp;3.11+** virtual environment before running the setup script.
- Run `./codex/setup.sh` to install the project in editable mode along with minimal runtime dependencies.
- When offline, run `WHEELHOUSE=/path/to/wheels ./codex/setup.sh`. The same path can be passed to `check_env.py --wheelhouse`.
- After setup, validate with `python check_env.py --auto-install`.
This installs any missing optional packages from the wheelhouse if provided.
- Execute `pytest -q` and ensure the entire suite passes. If failures remain, document them in the PR description.

## Coding Style
- Use **Python&nbsp;3.11** or newer and include type hints for public APIs.
- Indent with 4 spaces and keep lines under 120 characters.
- Provide concise [Google style](https://google.github.io/styleguide/pyguide.html#381-docstrings) docstrings
for modules, classes and functions.

## Pull Requests
- Keep commits focused and descriptive. Use meaningful commit messages.
- Ensure `git status` shows a clean working tree before committing.
- Summarize your changes and test results in the PR body.
- Title PRs using `[alpha_factory] <Title>`.
