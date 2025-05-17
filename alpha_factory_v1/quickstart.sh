#!/usr/bin/env bash
# Alpha-Factory Quickstart Launcher
# Professional production-ready script to bootstrap and run Alpha-Factory v1
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [--skip-preflight] [orchestrator args...]

Bootstraps and launches Alpha-Factory in an isolated virtual environment.
Pass any additional arguments directly to the orchestrator.
EOF
}

SKIP_PREFLIGHT=0
if [[ ${1:-} == "--help" ]]; then
  usage; exit 0
elif [[ ${1:-} == "--skip-preflight" ]]; then
  SKIP_PREFLIGHT=1
  shift
fi

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
  REQ="requirements.lock"
  [ -f "$REQ" ] || REQ="requirements.txt"
  "$VENV_DIR/bin/pip" install -r "$REQ"
fi

source "$VENV_DIR/bin/activate"

# run preflight checks unless skipped
if [[ $SKIP_PREFLIGHT -eq 0 ]]; then
  python scripts/preflight.py && echo "✅ Preflight passed"
fi

echo "Starting Orchestrator..."
# launch orchestrator
exec python -m run "$@"
