#!/usr/bin/env bash
# Minimal setup script for Codex development environments.
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install 'pydantic<2' pytest openai anthropic prometheus_client

# Attempt to install optional extras needed for some demos.
python check_env.py --auto-install || true

# Install the package in editable mode.
python -m pip install -e .
