# Contributor Guide

This repository contains the Alpha-Factory v1 package and demos.
The instructions below apply to all contributors and automated agents.

**Quick links**

- [Development Environment](#development-environment)
- [Coding Style](#coding-style)
- [Pull Requests](#pull-requests)
- [Troubleshooting](#troubleshooting)

## Development Environment
- Create and activate a Python 3.11 or 3.12 (**Python ≥3.11 and <3.13**) virtual
  environment before running the setup script. On Linux or macOS:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -U pip
  ```
  On Windows PowerShell:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  pip install -U pip
  ```
- The script `alpha_factory_v1/scripts/preflight.py` enforces this requirement.
- Run `./codex/setup.sh` to install the project in editable mode along with minimal runtime dependencies.
- When offline, first build a local wheelhouse containing all requirements:
  ```bash
  mkdir /media/wheels
  pip wheel -r requirements.txt -r requirements-dev.txt -w /media/wheels
  ```
  Set `WHEELHOUSE=/media/wheels` for both `./codex/setup.sh` and
  `python check_env.py --auto-install` so they use the same wheelhouse.
  Always include `AUTO_INSTALL_MISSING=1` when running `check_env.py` offline.
  Example:
  ```bash
  WHEELHOUSE=/media/wheels ./codex/setup.sh
  AUTO_INSTALL_MISSING=1 WHEELHOUSE=/media/wheels \
    python check_env.py --auto-install
  ```
- After setup, validate with `python check_env.py --auto-install`.
  This installs any missing optional packages from the wheelhouse if provided.
  - When `WHEELHOUSE` is set, run
    `python check_env.py --auto-install --wheelhouse <path>` so optional packages
    install correctly offline.
- The unit tests rely on `fastapi` and `opentelemetry-api`. Install them via
  `requirements-dev.txt` or ensure `check_env.py` reports no missing packages
  before running `pytest`.
- Execute `pytest -q` (or `python -m alpha_factory_v1.scripts.run_tests`) and ensure the entire suite passes. If failures remain, document them in the PR description.
- Run `python alpha_factory_v1/scripts/preflight.py` to confirm the Python version and required tools are available.
- Before the first launch, run `bash quickstart.sh --preflight` to check
  Docker availability, git, and required packages. After this
  verification, run `./quickstart.sh` to launch the project. The script
  creates the virtual environment and installs required dependencies
  automatically. See the [5‑Minute Quick‑Start](README.md#6-5-minute-quick-start)
  section in the README for more details.
- For a one-step build and launch, run
  `alpha_factory_v1/scripts/install_alpha_factory_pro.sh --deploy`.
  See [`alpha_factory_v1/scripts/README.md`](alpha_factory_v1/scripts/README.md)
  for additional options.
- On Windows or systems without Bash, run
  `python alpha_factory_v1/quickstart.py --preflight`.
- Copy `alpha_factory_v1/.env.sample` to `.env` and add secrets such as
  `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.
- **Never commit** `.env` or other secrets. See
  [`alpha_factory_v1/scripts/README.md`](alpha_factory_v1/scripts/README.md)
  for additional guidance.
- Verify `.env` is ignored by running `git status` (it should appear untracked). The repository's `.gitignore` already includes `.env`.

## Coding Style
- Use Python 3.11 or 3.12 (**Python ≥3.11 and <3.13**) and include type hints for public APIs.
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
  3. Run `pre-commit run --all-files` right away to ensure formatting and lint
     checks succeed and to populate the cache.
  4. Run `pre-commit run --files <paths>` before committing.
    CI will reject commits that fail these checks.
  - The configuration runs `black`, `ruff`, `flake8` and `mypy` using
    `mypy.ini`.
  - After setup, run `pre-commit run --all-files` once right away to ensure
    formatting and lint checks succeed, then run `pre-commit run --files
    <paths>` before every commit.

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

### Troubleshooting
- If the stack fails to start, verify Docker and Docker Compose are running.
- Setup errors usually mean Python is older than 3.11. Upgrade to Python 3.11 or newer.
- Missing optional packages can cause test failures; run `python check_env.py --auto-install`.

For detailed troubleshooting steps, see [`alpha_factory_v1/scripts/README.md`](alpha_factory_v1/scripts/README.md).
