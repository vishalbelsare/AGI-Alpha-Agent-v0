#!/usr/bin/env bash
# demos/deploy_alpha_factory_demo.sh
# ───────────────────────────────────────────────────────────────────────
# Purpose: 1) auto‑select the best momentum pair alpha (BTC/GLD for now)
#          2) spin up Alpha‑Factory with that strategy
#          3) prove it works by printing positions & P&L
#
# Prereqs: docker, curl, jq             (no Python needed)
# Optional: yq (nicer YAML edit; sed fallback provided)
# ----------------------------------------------------------------------

set -euo pipefail

ALPHA="btc_gld"               # future: plug analyser to pick dynamically
BRANCH="main"
REPO="MontrealAI/AGI-Alpha-Agent-v0"
PROJECT="af_demo"
PORT_API=8000
IMG="ghcr.io/montrealai/alphafactory_pro:cpu-slim-latest"

echo "🚀  Alpha‑Factory demo – strategy: $ALPHA"

#────────────────────────────────────────────────────────────────────────
# 1. pull latest slim image (≈ 200 MB) if not present
#────────────────────────────────────────────────────────────────────────
if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "→ pulling container…"; docker pull "$IMG"
fi

#────────────────────────────────────────────────────────────────────────
# 2. start container in background
#────────────────────────────────────────────────────────────────────────
CID=$(docker run -d \
      -p $PORT_API:8000 \
      -e FINANCE_STRATEGY="$ALPHA" \
      --name "$PROJECT" --rm "$IMG")

trap 'docker stop $CID >/dev/null' EXIT

#────────────────────────────────────────────────────────────────────────
# 3. wait for API
#────────────────────────────────────────────────────────────────────────
printf "⏳  Waiting for API"; until curl -sf http://localhost:$PORT_API/health; do
  printf "."; sleep 1; done; echo " ready!"

#────────────────────────────────────────────────────────────────────────
# 4. show positions & P&L
#────────────────────────────────────────────────────────────────────────
echo -e "\n📈  Finance positions:"
curl -s http://localhost:$PORT_API/api/finance/positions | jq .

echo -e "\n💰  Finance P&L:"
curl -s http://localhost:$PORT_API/api/finance/pnl | jq .

#────────────────────────────────────────────────────────────────────────
# 5. final banner
#────────────────────────────────────────────────────────────────────────
cat <<EOF

🎉  Demo complete!
Open the live trace‑graph UI at:  http://localhost:8088
(API docs at: http://localhost:$PORT_API/docs)

The container will stop when you exit this script (Ctrl‑C).
EOF
