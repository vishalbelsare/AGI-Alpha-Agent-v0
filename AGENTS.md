# Contributor Guide

This repository contains the Alpha-Factory v1 package and demos.
The instructions below apply to all contributors and automated agents.

**Quick links**

- [Development Environment](#development-environment)
- [Coding Style](#coding-style)
- [Pull Requests](#pull-requests)
- [Troubleshooting](#troubleshooting)
- [Code of Conduct](CODE_OF_CONDUCT.md)
- [Security Policy](SECURITY.md)
- [Wheel Signing](#wheel-signing)

All contributors must follow the [Code of Conduct](CODE_OF_CONDUCT.md).
Please report security vulnerabilities as described in our [Security Policy](SECURITY.md).
## Prerequisites
- Python 3.11 or 3.12 (**Python ≥3.11 and <3.13**)
- Docker and Docker Compose (Compose ≥2.5)
- Git
- Run `python alpha_factory_v1/scripts/preflight.py` to validate these tools.

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
- From the repository root, run `./codex/setup.sh` to install the project in editable mode along with minimal runtime dependencies. This ensures all relative paths resolve correctly.
 - When offline, build a local wheelhouse for **both** `requirements.txt` and
   `requirements-dev.txt`:
   ```bash
   mkdir /media/wheels
   pip wheel -r requirements.txt -r requirements-dev.txt -w /media/wheels
   ```
   Set `WHEELHOUSE=/media/wheels` and `AUTO_INSTALL_MISSING=1` when running
   `./codex/setup.sh` and `python check_env.py --auto-install` so they install
   from the wheelhouse. Example:
   ```bash
   WHEELHOUSE=/media/wheels AUTO_INSTALL_MISSING=1 ./codex/setup.sh
   WHEELHOUSE=/media/wheels AUTO_INSTALL_MISSING=1 \
     python check_env.py --auto-install
   ```
   - Run `pip check` to verify package integrity. If problems occur, rerun `python check_env.py --auto-install --wheelhouse <path>` to reinstall.

   See [`alpha_factory_v1/scripts/README.md`](alpha_factory_v1/scripts/README.md)
   for additional offline tips.
- After setup, validate with `python check_env.py --auto-install`.
  This installs any missing optional packages from the wheelhouse if provided.
  - When `WHEELHOUSE` is set, run
    `python check_env.py --auto-install --wheelhouse <path>` so optional packages
    install correctly offline.
- The unit tests rely on `fastapi` and `opentelemetry-api`. Install them via
  `requirements-dev.txt` or ensure `check_env.py` reports no missing packages
  before running `pytest`.
- If the project is installed without `./codex/setup.sh`, run
  `pip install -r requirements-dev.txt` to obtain `fastapi` and
  `opentelemetry-api`.
- Execute `pytest -q` (or `python -m alpha_factory_v1.scripts.run_tests`) and ensure the entire suite passes. If failures remain, document them in the PR description.
- Test environment variables (see [`alpha_factory_v1/tests/README.md`](alpha_factory_v1/tests/README.md) for details):
  - `AF_MEMORY_DIR` – temporary memory path.
  - `PYTEST_CPU_SOFT_SEC` – CPU time limit.
  - `PYTEST_MEM_MB` – memory cap in MB.
  - `PYTEST_NET_OFF` – disable network access.
  - The sandbox runner `python -m alpha_factory_v1.backend.tools.local_pytest` enforces these limits.
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
  `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. Replace `NEO4J_PASSWORD=REPLACE_ME`
  with a strong secret— the orchestrator fails to start if this variable is
  missing or left at the default.
- **Never commit** `.env` or other secrets. See
  [`alpha_factory_v1/scripts/README.md`](alpha_factory_v1/scripts/README.md)
  for additional guidance.
- Verify `.env` is ignored by running `git status` (it should appear untracked). The repository's `.gitignore` already includes `.env`.
- Store secrets in environment variables or Docker secrets instead of code to keep them out of version control.
- Set `AF_TRACING=true` to enable tracing (default) or `false` to disable it. See
  [`alpha_factory_v1/backend/tracer.py`](alpha_factory_v1/backend/tracer.py).

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
  3. `pre-commit run --all-files` once after installation
  4. `pre-commit run --files <paths>` before each commit
  5. Re-run `pre-commit run --all-files` when dependencies or configs change.
  - If `requirements.txt`, `requirements-dev.txt` or lint configs change, run \
    `pre-commit run --all-files` before committing.
  - CI will reject commits that fail these checks.
  - The configuration runs `black`, `ruff`, `flake8` and `mypy` using
    `mypy.ini`.

## Pull Requests
- Keep commits focused and descriptive. Use meaningful commit messages.
- Ensure `git status` shows a clean working tree before committing.
- Remove stray build artifacts with `git clean -fd` if needed.
- Run `python check_env.py --auto-install` and `pytest -q` before committing. \
  Document any remaining test failures in the PR description.
- Summarize your changes and test results in the PR body.
- Title PRs using `[alpha_factory] <Title>`.
- All contributions are licensed under Apache 2.0.
- Use the issue templates under `.github/ISSUE_TEMPLATE` for bug reports and feature requests.
- Follow the [pull request template](.github/pull_request_template.md) and fill in
  all sections to confirm linting, type checks and tests pass.

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

### Wheel Signing
All agent wheels must be signed with the project's ED25519 key before they are
loaded from `$AGENT_HOT_DIR`.

Generate the signing key once and capture the base64 public key:

```bash
openssl genpkey -algorithm ed25519 -out agent_signing.key
openssl pkey -in agent_signing.key -pubout -outform DER | base64 -w0
```

Store the public key in the `AGENT_WHEEL_PUBKEY` environment variable so
`alpha_factory_v1/backend/agents/__init__.py` can verify signatures.

Generate `<wheel>.whl.sig` with:

```bash
openssl dgst -sha512 -binary <wheel>.whl |
  openssl pkeyutl -sign -inkey agent_signing.key |
  base64 -w0 > <wheel>.whl.sig
```

Commit the signature file and add the base64 value to `_WHEEL_SIGS` in
`alpha_factory_v1/backend/agents/__init__.py`. Wheels without a valid signature
are ignored at runtime.

### Verify the wheel
Verify that `<wheel>.whl.sig` matches the wheel:

```bash
openssl dgst -sha512 -binary <wheel>.whl |
  openssl pkeyutl -verify -pubin -inkey "$AGENT_WHEEL_PUBKEY" -sigfile <wheel>.whl.sig
```

The orchestrator validates signatures against `_WHEEL_PUBKEY` in `alpha_factory_v1/backend/agents/__init__.py`.
