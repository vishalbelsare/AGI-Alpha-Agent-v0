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

# Install package in editable mode
$PYTHON -m pip install --quiet "${wheel_opts[@]}" -e .

# Minimal runtime/test dependencies
packages=(
  pytest
  prometheus_client
  openai
  openai_agents
  google_adk
  anthropic
)
$PYTHON -m pip install --quiet "${wheel_opts[@]}" "${packages[@]}"

# Validate environment and install any remaining deps
check_env_opts=()
if [[ -n "${WHEELHOUSE:-}" ]]; then
  check_env_opts+=(--wheelhouse "$WHEELHOUSE")
fi
$PYTHON check_env.py --auto-install "${check_env_opts[@]}"

