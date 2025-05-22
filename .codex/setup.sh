#!/usr/bin/env bash
# Setup script for Codex development environments.
set -euo pipefail

# Keep tooling modern.
python -m pip install --upgrade pip setuptools wheel

# Core runtime requirements for tests and demos.
python -m pip install --prefer-binary 'pydantic<2' pytest openai anthropic \
    prometheus_client

# Install the project in editable mode.
python -m pip install -e .

# Validate optional extras. Failures do not stop the build.
python check_env.py --auto-install || true
