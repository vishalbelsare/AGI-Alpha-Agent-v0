#!/usr/bin/env bash
# Alpha-Factory Quickstart Launcher
# Professional production-ready script to bootstrap and run Alpha-Factory v1
set -euo pipefail

header() {
  echo "============================================"
  echo "         Alpha-Factory Quickstart 🚀"
  echo "============================================"
}

header

# check python version
python3 - <<'PY'
import sys
req = (3,9)
if sys.version_info < req:
    sys.exit(f"Python {req[0]}.{req[1]}+ required")
PY

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "→ Creating virtual environment"
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install -U pip
  "$VENV_DIR/bin/pip" install -r requirements.txt
fi

source "$VENV_DIR/bin/activate"

# run preflight checks
python scripts/preflight.py

# launch orchestrator
exec python -m run "$@"
