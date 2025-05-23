# Contributor Guide

This repository contains the Alpha-Factory v1 package and demos.
The instructions below apply to all contributors and automated agents.

## Development Environment
- Create and activate a **Python&nbsp;3.11.x** virtual environment before running the setup script.
- Run `./codex/setup.sh` to install the project in editable mode along with minimal runtime dependencies.
- When offline, run `WHEELHOUSE=/path/to/wheels ./codex/setup.sh`. Pass the same
  path to `check_env.py --wheelhouse` and set `AUTO_INSTALL_MISSING=1` to allow
  automatic installation of missing packages. Example:
  `WHEELHOUSE=/media/wheels python check_env.py --auto-install --wheelhouse /media/wheels`
- After setup, validate with `python check_env.py --auto-install`.
  This installs any missing optional packages from the wheelhouse if provided.
- Execute `pytest -q` (or `python -m alpha_factory_v1.scripts.run_tests`) and ensure the entire suite passes. If failures remain, document them in the PR description.
- Run `python alpha_factory_v1/scripts/preflight.py` to confirm the Python version and required tools are available.
- Before the first launch, run `bash quickstart.sh --preflight` to check
  Docker availability, git, and required packages. After this
  verification, run `./quickstart.sh` to launch the project. The script
  creates the virtual environment and installs required dependencies
  automatically. See the [5‑Minute Quick‑Start](README.md#6-5-minute-quick-start)
  section in the README for more details.
- On Windows or systems without Bash, run
  `python alpha_factory_v1/quickstart.py --preflight`.
- Copy `alpha_factory_v1/.env.sample` to `.env` and add secrets such as
  `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.
- **Never commit** `.env` or other secrets. See
  [`alpha_factory_v1/scripts/README.md`](alpha_factory_v1/scripts/README.md)
  for additional guidance.

## Coding Style
- Use **Python&nbsp;3.11.x** and include type hints for public APIs.
- Indent with 4 spaces and keep lines under 120 characters.
- Provide concise [Google style](https://google.github.io/styleguide/pyguide.html#381-docstrings) docstrings
for modules, classes and functions.
- Format code with `black` (line length 120) and run `ruff` or `flake8` for linting, if available.
- `pyproject.toml` contains the configuration for `black`, `ruff` and `flake8`.
  Adjust lint settings there if needed.
- Ensure code is formatted before committing.
- Run `ruff` or `flake8` and `mypy --strict` before committing to enforce
  consistent style and type safety.
- Run `mypy --config-file mypy.ini .` (or `pyright`) with a **strict** configuration. The
  `mypy.ini` configuration file is located at the repository root.
- Install pre‑commit and set up the git hook:
  1. `pip install pre-commit`
  2. `pre-commit install`
  3. Run `pre-commit run --all-files` once to populate the cache.
  4. Run `pre-commit run --files <paths>` before committing.
    CI will reject commits that fail these checks.
  - The configuration runs `black`, `ruff`, `flake8` and `mypy` using
    `mypy.ini`.
  - After setup, run `pre-commit run --all-files` once and
    `pre-commit run --files <paths>` before every commit.

## Pull Requests
- Keep commits focused and descriptive. Use meaningful commit messages.
- Ensure `git status` shows a clean working tree before committing.
- Remove stray build artifacts with `git clean -fd` if needed.
- Run `python check_env.py --auto-install` and `pytest -q` before committing. \
  Document any remaining test failures in the PR description.
- Summarize your changes and test results in the PR body.
- Title PRs using `[alpha_factory] <Title>`.
- All contributions are licensed under Apache 2.0.

### PR Message Guidelines
- Keep the subject line concise and under 72 characters.
- Optionally include a short body explaining the rationale.
- Consider using a simplified Conventional Commits prefix such as
  `feat:`, `fix:` or `docs:` to ease changelog generation.
