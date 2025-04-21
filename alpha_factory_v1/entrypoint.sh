#!/usr/bin/env bash
# Alpha‑Factory ‑ hardened entry‑point  (sha256: 6f12…)
set -euo pipefail

export PYTHONUNBUFFERED=1
export PYTHONPATH=/app          # make "backend" importable everywhere

# 1️⃣  orchestrator (background)
python3 -m backend.main &

# 2️⃣  A2A RPC FastAPI server on :8000 (background)
python3 -m backend.rpc_server &

# 3️⃣  Flask UI on :3000  → PID‑1 keeps container healthy
exec python3 ui/app.py          # ui/app.py already binds to 0.0.0.0:3000

