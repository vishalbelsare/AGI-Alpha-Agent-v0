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

Confirm installed versions:
```bash
python --version
docker --version
docker compose version
git --version
```
Python must report 3.11 or 3.12 and Docker Compose must be at least 2.5.

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
- When offline, build a local wheelhouse using **the same Python version** as your
  virtual environment (Python 3.11 or 3.12). Create a directory and build wheels
  for **both** requirement files:
  ```bash
  mkdir -p /media/wheels
  pip wheel -r requirements.txt -w /media/wheels
  pip wheel -r requirements-dev.txt -w /media/wheels
  ```
  Set `WHEELHOUSE=/media/wheels` and `AUTO_INSTALL_MISSING=1` so
  `./codex/setup.sh` and `python check_env.py --auto-install --wheelhouse /media/wheels`
  install from the wheelhouse. Example:
  ```bash
  WHEELHOUSE=/media/wheels AUTO_INSTALL_MISSING=1 ./codex/setup.sh
  WHEELHOUSE=/media/wheels AUTO_INSTALL_MISSING=1 \
    python check_env.py --auto-install --wheelhouse /media/wheels
  ```
  - Run `pip check` to verify package integrity. If problems occur, rerun
    `python check_env.py --auto-install --wheelhouse /media/wheels` to reinstall.

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
- When running tests directly from the repository without installation, set `PYTHONPATH` as described in [`tests/README.md`](tests/README.md): `export PYTHONPATH=$(pwd)`.
- For coverage metrics, run `pytest --cov` and aim for at least **80%** coverage.
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

### Key Environment Variables
Store secrets in environment variables or Docker secrets instead of code to keep
them out of version control. The table below highlights the most common settings
(see [`alpha_factory_v1/.env.sample`](alpha_factory_v1/.env.sample) for the full
template).

| Variable | Purpose | Default/Example |
|----------|---------|-----------------|
| `AF_TRACING` | Enable or disable tracing | `true` |
| `AF_MEMORY_DIR` | Working memory directory for tests and runtime | `/tmp/alphafactory` |
| `AGENT_WHEEL_PUBKEY` | Base64 ED25519 key verifying agent wheel signatures | _(none)_ |
| `AGENT_ERR_THRESHOLD` | Consecutive errors before quarantine | `3` |
| `AGENT_HOT_DIR` | Directory for hot-loaded agent wheels | `~/.alpha_agents` |
| `AGENT_HEARTBEAT_SEC` | Heartbeat interval in seconds | `10` |
| `AGENT_RESCAN_SEC` | Interval between wheel scans | `60` |
| `OPENAI_API_KEY` | OpenAI credential (blank uses local Ollama Φ‑2) | _(empty)_ |
| `OPENAI_ORG_ID` | OpenAI organization ID | _(none)_ |
| `ANTHROPIC_API_KEY` | Anthropic credential | _(none)_ |
| `MISTRAL_API_KEY` | Mistral credential | _(none)_ |
| `TOGETHER_API_KEY` | Together credential | _(none)_ |
| `GOOGLE_API_KEY` | Optional speech/vision credential | _(none)_ |
| `POLYGON_API_KEY` | Polygon market data API | _(none)_ |
| `ALPACA_KEY_ID` | Alpaca trading API key | _(none)_ |
| `ALPACA_SECRET_KEY` | Alpaca trading API secret | _(none)_ |
| `BINANCE_API_KEY` | Binance trading API key | _(none)_ |
| `BINANCE_API_SECRET` | Binance trading API secret | _(none)_ |
| `IBKR_CLIENT_ID` | Interactive Brokers client ID | _(none)_ |
| `IBKR_CLIENT_SECRET` | Interactive Brokers client secret | _(none)_ |
| `FRED_API_KEY` | FRED economic data key | _(none)_ |
| `NEWSAPI_KEY` | News API key | _(none)_ |
| `NEO4J_URI` | Neo4j database URI | `bolt://neo4j:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Database password required by the orchestrator | `REPLACE_ME` |
| `LLM_PROVIDER` | LLM provider to use | `openai` |
| `MODEL_NAME` | Default model name | `gpt-4-turbo` |
| `PORT` | REST API port | `8000` |
| `PROM_PORT` | Prometheus exporter port | `9090` |
| `TRACE_WS_PORT` | Trace graph WebSocket port | `8088` |
| `AGENTS_RUNTIME_PORT` | OpenAI Agents runtime port | `5001` |
| `BUSINESS_HOST` | Base orchestrator URL for bridges | `http://localhost:8000` |
| `LOGLEVEL` | Logging level | `INFO` |
| `API_TOKEN` | Bearer token for the demo API | `REPLACE_ME_TOKEN` |
| `API_RATE_LIMIT` | Requests per minute per IP | `60` |
| `ALPHA_KAFKA_BROKER` | Kafka broker address | _(none)_ |
| `ALPHA_DATA_DIR` | Base data directory | `/data` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP export endpoint | `http://tempo:4317` |
| `VC_SIGNING_KEY_PATH` | Path to signing key | `/run/secrets/ed25519_private.key` |
| `K8S_CPU_LIMIT` | Kubernetes CPU limit | `500m` |
| `K8S_MEM_LIMIT` | Kubernetes memory limit | `1Gi` |
| `TZ` | Time zone | `America/Toronto` |
| `AF_LLM_CACHE_SIZE` | Maximum in-memory LLM cache entries | `1024` |
| `AF_PING_INTERVAL` | Ping frequency in seconds (min 5) | `60` |
| `AF_DISABLE_PING_AGENT` | Set to `true` to disable the ping agent | _(none)_ |

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
  - Re-run `pre-commit run --all-files` whenever `requirements.txt`, `requirements-dev.txt`,
    `.pre-commit-config.yaml`, `pyproject.toml`, `mypy.ini`, or other lint configs change.
    CI enforces these checks.
  - The configuration runs `black`, `ruff`, `flake8` and `mypy` using
    `mypy.ini`.

## Pull Requests
- Keep commits focused and descriptive. Use meaningful commit messages.
- Ensure `git status` shows a clean working tree before committing.
- Remove stray build artifacts with `git clean -fd` if needed.
- Run `python check_env.py --auto-install` and `pytest -q` before committing. \
  Document any remaining test failures in the PR description.
- See [tests/README.md](tests/README.md) for details on running the suite locally,
  including setting `PYTHONPATH` or installing in editable mode.
- Summarize your changes and test results in the PR body.
- Title PRs using `[alpha_factory] <Title>`.
- All contributions are licensed under Apache 2.0 by default.
- Some files retain existing MIT license headers; keep whatever license a file
  already declares when editing it.
- Add an Apache 2.0 header to new source files unless another license is
  explicitly stated.
  ```
  # SPDX-License-Identifier: Apache-2.0
  ```
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
- Setup errors usually mean Python is older than 3.11. Use Python 3.11 or 3.12 (>=3.11,<3.13).
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

Store `agent_signing.key` **outside** the repository and never commit it. The
root `.gitignore` now ignores `*.key` so automated agents and human
contributors avoid including the private key in pull requests.

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
