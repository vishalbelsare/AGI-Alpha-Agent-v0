#!/usr/bin/env bash
# Alpha-Factory Quickstart Launcher
# Professional production-ready script to bootstrap and run Alpha-Factory v1
set -Eeuo pipefail
trap 'echo -e "\n\u274c Error on line $LINENO" >&2' ERR

# log pip output for troubleshooting
PIP_LOG="pip-install.log"

# always operate from repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

usage() {
  cat <<EOF
Usage: $0 [--preflight] [--skip-preflight] [orchestrator args...]

Bootstraps and launches Alpha-Factory in an isolated Python virtual environment.
  --preflight        Run environment checks and exit
  --skip-preflight   Skip automatic preflight checks before launching
  Pip install output logs to $PIP_LOG and failures abort the script.
Any other options are passed directly to the orchestrator.
EOF
}

SKIP_PREFLIGHT=0
PRECHECK_ONLY=0
ORCH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --help|-h) usage; exit 0 ;;
    --preflight) PRECHECK_ONLY=1; shift ;;
    --skip-preflight) SKIP_PREFLIGHT=1; shift ;;
    *) ORCH_ARGS+=("$1"); shift ;;
  esac
done

header() {
  echo "============================================"
  echo "         Alpha-Factory Quickstart 🚀"
  echo "============================================"
}


header

# check python version
python3 - <<'PY'
import sys
req = (3, 11)
max_py = (3, 12)
if sys.version_info < req or sys.version_info >= max_py:
    sys.exit(f"Python {req[0]}.{req[1]}+ and <{max_py[0]}.{max_py[1]} required")
PY

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "→ Creating virtual environment"
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install -U pip >"$PIP_LOG" 2>&1 || { echo 'pip install failed'; exit 1; }
  REQ="alpha_factory_v1/requirements.lock"
  [ -f "$REQ" ] || REQ="alpha_factory_v1/requirements.txt"
  "$VENV_DIR/bin/pip" install -r "$REQ" >>"$PIP_LOG" 2>&1 || { echo 'pip install failed'; exit 1; }
fi

source "$VENV_DIR/bin/activate"

# run preflight checks unless skipped
if [[ $PRECHECK_ONLY -eq 1 ]]; then
  python alpha_factory_v1/scripts/preflight.py
  exit 0
fi

if [[ $SKIP_PREFLIGHT -eq 0 ]]; then
  python alpha_factory_v1/scripts/preflight.py && echo "✅ Preflight passed"
fi

echo "Starting Orchestrator..."
# launch orchestrator
exec python -m alpha_factory_v1.run "${ORCH_ARGS[@]}"
