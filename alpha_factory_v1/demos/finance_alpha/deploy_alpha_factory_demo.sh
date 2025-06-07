#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# demos/deploy_alpha_factory_demo.sh
# ─────────────────────────────────────────────────────────────────────────
# One‑command demo for Alpha‑Factory v1.
#
#   • Pulls the signed CPU‑slim image (offline‑safe Φ‑2 fallback)
#   • Launches the container with a momentum‑pair strategy (BTC / GLD)
#   • Prints FinanceAgent Positions & P&L via REST
#   • Points the user to the live trace‑graph UI
#
# Requirements: docker 24+, curl, jq              (Python optional)
# Optional env:  STRATEGY   finance alpha (default=btc_gld)
#                PORT_API   host port (default=8000)
#                IMAGE_TAG  override container tag
# ------------------------------------------------------------------------
set -euo pipefail

STRATEGY="${STRATEGY:-btc_gld}"
PORT_API="${PORT_API:-8000}"
IMAGE_TAG="${IMAGE_TAG:-cpu-slim-latest}"
IMAGE="ghcr.io/montrealai/alphafactory_pro:${IMAGE_TAG}"
CONTAINER="af_demo_${STRATEGY}_${PORT_API}"

banner() { printf "\033[1;36m%s\033[0m\n" "$*"; }

# ── sanity checks ───────────────────────────────────────────────────────
for cmd in docker curl jq; do
  command -v "$cmd" >/dev/null || { echo "❌  $cmd not found"; exit 1; }
done
# Check whether the API port is free. Prefer a tiny Python probe if available
if command -v python >/dev/null; then
  if ! python - <<'EOF_P' "$PORT_API"
import socket, sys
s = socket.socket()
result = s.connect_ex(('localhost', int(sys.argv[1])))
sys.exit(0 if result else 1)
EOF_P
  then
    echo "❌  Port $PORT_API is already in use"; exit 1;
  fi
else
  # Fallback to legacy lsof/ss approach when Python is missing
  if command -v lsof >/dev/null; then
    if lsof -i ":$PORT_API" >/dev/null 2>&1; then
      echo "❌  Port $PORT_API is already in use"; exit 1;
    fi
  elif command -v ss >/dev/null; then
    if ss -ltn | grep -q ":$PORT_API " ; then
      echo "❌  Port $PORT_API is already in use"; exit 1;
    fi
  else
    echo "⚠  Unable to verify if port $PORT_API is free (missing python, lsof/ss)" >&2
  fi
fi

# ── pull image if missing ───────────────────────────────────────────────
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  banner "📦  Pulling Alpha‑Factory image ($IMAGE)…"
  docker pull "$IMAGE"
fi

# ── start container ─────────────────────────────────────────────────────
banner "🚀  Starting Alpha‑Factory  (strategy: $STRATEGY)"
CID=$(docker run -d --rm --name "$CONTAINER" \
        -e FINANCE_STRATEGY="$STRATEGY" \
        -e TRACE_WS_PORT=8088 \
        -p "${PORT_API}:8000" \
        -p 8088:8088 "$IMAGE")
trap 'docker stop "$CID" >/dev/null' EXIT

# ── wait for API health endpoint ────────────────────────────────────────
HEALTH="http://localhost:${PORT_API}/health"
printf "⏳  Waiting for API"
ready=false
for _ in {1..60}; do
  if curl -sf "$HEALTH" >/dev/null; then
    ready=true
    break
  fi
  printf "."; sleep 1
done
if $ready; then
  echo " ready!"
else
  echo " failed" >&2
  docker logs "$CID" || true
  exit 1
fi

# ── query positions & P&L ───────────────────────────────────────────────
banner "📈  Finance Positions"
curl -s "http://localhost:${PORT_API}/api/finance/positions" | jq .

banner "💰  Finance P&L"
curl -s "http://localhost:${PORT_API}/api/finance/pnl" | jq .

# ── final instructions ─────────────────────────────────────────────────
cat <<EOF

🎉  Demo complete!
• Trace‑graph UI :  http://localhost:8088
• API docs       :  http://localhost:${PORT_API}/docs

Press Ctrl‑C to stop the container when you're finished.
EOF

banner "⚠  Research demo only. Not financial advice."

# keep running so user can browse UI
while sleep 3600; do :; done
