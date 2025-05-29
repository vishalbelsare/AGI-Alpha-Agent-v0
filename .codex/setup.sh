#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

# Resolve repository root and default wheelhouse
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${WHEELHOUSE:-}" && -d "${REPO_ROOT}/wheels" ]]; then
  WHEELHOUSE="${REPO_ROOT}/wheels"
fi

# Support offline installation via WHEELHOUSE
# Set COLAB_INSTALL=1 to install the Colab dependencies lock file.
wheel_opts=()
if [[ -n "${WHEELHOUSE:-}" ]]; then
  wheel_opts+=(--no-index --find-links "$WHEELHOUSE")
fi

# Abort when no wheelhouse is available and the network is unreachable
if [[ -z "${WHEELHOUSE:-}" ]]; then
  if ! curl -sSf https://pypi.org/simple/ -o /dev/null 2>&1; then
    cat <<'EOF'
ERROR: No network access and no wheelhouse found.
Create one with:
  mkdir -p wheels
  pip wheel -r requirements.lock -w wheels
  pip wheel -r requirements-dev.txt -w wheels
Then re-run ./codex/setup.sh
EOF
    exit 1
  fi
fi

# Upgrade pip and core build tools
$PYTHON -m pip install --quiet "${wheel_opts[@]}" --upgrade pip setuptools wheel

# Install pip-compile (pip-tools) early so hooks can verify lock files
$PYTHON -m pip install --quiet "${wheel_opts[@]}" pip-tools

# Install package in editable mode
$PYTHON -m pip install --quiet "${wheel_opts[@]}" -e .

# Install additional dependencies when running in Google Colab
if [[ "${COLAB_INSTALL:-0}" == "1" ]]; then
  # COLAB_INSTALL=1 installs packages from the Colab lock file, supporting the
  # same offline mode as WHEELHOUSE.
  $PYTHON -m pip install --quiet "${wheel_opts[@]}" \
    -r alpha_factory_v1/requirements-colab.lock
elif [[ "${FULL_INSTALL:-0}" == "1" ]]; then
  # When FULL_INSTALL=1, use the deterministic lock file for reproducible
  # offline installs.
  $PYTHON -m pip install --quiet "${wheel_opts[@]}" -r requirements.lock
else
  # Minimal runtime/test dependencies
  packages=(
    pytest
    prometheus_client
    openai
    openai-agents
    google-adk
    anthropic
    fastapi
    opentelemetry-api
    httpx
    uvicorn
  )
  $PYTHON -m pip install --quiet "${wheel_opts[@]}" "${packages[@]}"
fi

# Validate environment and install any remaining deps
check_env_opts=()
if [[ -n "${WHEELHOUSE:-}" ]]; then
  check_env_opts+=(--wheelhouse "$WHEELHOUSE")
fi
$PYTHON check_env.py --auto-install "${check_env_opts[@]}"

