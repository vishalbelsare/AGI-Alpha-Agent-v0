#!/usr/bin/env bash
# Minimal setup script for Codex development environments.
set -euo pipefail

# Optional upgrade of packaging tools. Failures are ignored to allow offline use.
python -m pip install --upgrade pip setuptools wheel >/tmp/pip-tools.log || true

# Core runtime requirements. Install from $WHEELHOUSE when provided.
REQS=(pytest prometheus_client openai openai-agents google-adk anthropic)
CMD=(python -m pip install --prefer-binary)
if [[ -n "${WHEELHOUSE:-}" ]]; then
    CMD+=(--no-index --find-links "$WHEELHOUSE")
fi
"${CMD[@]}" "${REQS[@]}" >/tmp/pip-install.log || true

# Install the project in editable mode.
python -m pip install -e . >/tmp/pip-project.log

# Validate runtime dependencies. Errors are non-fatal to support air-gapped setups.
python check_env.py --auto-install || true

echo "Environment setup complete."
