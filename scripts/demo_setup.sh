#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Simple demo setup helper
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install -U pip
fi
"$VENV_DIR/bin/pip" install -r requirements.txt -r requirements-dev.txt
"$VENV_DIR/bin/python" check_env.py --auto-install
echo "Demo environment ready. Activate via 'source $VENV_DIR/bin/activate'"
