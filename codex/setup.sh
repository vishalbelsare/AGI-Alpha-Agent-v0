#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

# Support offline installation via WHEELHOUSE
wheel_opts=()
if [[ -n "${WHEELHOUSE:-}" ]]; then
  wheel_opts+=(--no-index --find-links "$WHEELHOUSE")
fi

# Upgrade pip and core build tools
$PYTHON -m pip install --quiet "${wheel_opts[@]}" --upgrade pip setuptools wheel

# Install pip-compile (pip-tools) early so hooks can verify lock files
$PYTHON -m pip install --quiet "${wheel_opts[@]}" pip-tools

# Install package in editable mode
$PYTHON -m pip install --quiet "${wheel_opts[@]}" -e .

# When FULL_INSTALL=1, install the fully pinned dependencies from the
# deterministic lock file. This path also supports offline installs via a
# wheelhouse. Otherwise install a minimal set of runtime packages for fast
# setup in networked environments.
if [[ "${FULL_INSTALL:-0}" == "1" ]]; then
  $PYTHON -m pip install --quiet "${wheel_opts[@]}" -r requirements.lock
else
  packages=(
    pytest
    prometheus_client
    openai
    openai-agents
    google-adk
    anthropic
    fastapi
    cryptography
    opentelemetry-api
    opentelemetry-sdk
    httpx
    uvicorn
    grpcio
    requests
    pydantic-settings
  )
  if [[ "${DEV_INSTALL:-0}" == "1" ]]; then
    packages+=(
      pytest-benchmark
      pytest-httpx
      hypothesis
      grpcio-tools
      grpcio
      requests
      pydantic-settings
    )
  fi
  $PYTHON -m pip install --quiet "${wheel_opts[@]}" "${packages[@]}"
fi

# Validate environment and install any remaining deps
check_env_opts=()
if [[ -n "${WHEELHOUSE:-}" ]]; then
  check_env_opts+=(--wheelhouse "$WHEELHOUSE")
fi
$PYTHON check_env.py --auto-install "${check_env_opts[@]}"

