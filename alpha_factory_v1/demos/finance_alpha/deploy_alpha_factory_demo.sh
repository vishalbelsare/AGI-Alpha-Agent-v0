#!/usr/bin/env bash
# demos/deploy_alpha_factory_demo.sh
# ─────────────────────────────────────────────────────────────────────────
# One‑command demo for Alpha‑Factory v1.
#
#   • Pulls the signed CPU‑slim image (offline‑safe Φ‑2 fallback)
#   • Launches the container with a momentum‑pair strategy (BTC / GLD)
#   • Prints FinanceAgent Positions & P&L via REST
#   • Points the user to the live trace‑graph UI
#
# Requirements: docker 24+, curl, jq              (no Python needed)
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
if lsof -i ":$PORT_API" >/dev/null 2>&1; then
  echo "❌  Port $PORT_API is already in use"; exit 1;
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
        -p "${PORT_API}:8000" "$IMAGE")
trap 'docker stop "$CID" >/dev/null' EXIT

# ── wait for API health endpoint ────────────────────────────────────────
HEALTH="http://localhost:${PORT_API}/health"
printf "⏳  Waiting for API"
for _ in {1..60}; do
  if curl -sf "$HEALTH" >/dev/null; then break; fi
  printf "."; sleep 1
done
echo " ready!"

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

# keep running so user can browse UI
while sleep 3600; do :; done
