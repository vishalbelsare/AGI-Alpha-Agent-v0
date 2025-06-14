#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Alpha-Factory – hardened entrypoint (sha256: 6f12…)
set -Eeuo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH=/app  # make "backend" importable everywhere

log() { printf '[entrypoint] %s\n' "$*"; }

cleanup() {
    log "shutting down"
    kill "$MAIN_PID" "$RPC_PID" "$UI_PID" 2>/dev/null || true
    wait "$MAIN_PID" "$RPC_PID" "$UI_PID" 2>/dev/null || true
}

log "starting backend.main"
python3 -m backend.main &
MAIN_PID=$!

log "starting backend.rpc_server"
python3 -m backend.rpc_server &
RPC_PID=$!

log "starting ui"
python3 ui/app.py &
UI_PID=$!

trap cleanup SIGINT SIGTERM EXIT

wait -n "$MAIN_PID" "$RPC_PID" "$UI_PID"
exit_code=$?
cleanup
exit "$exit_code"
