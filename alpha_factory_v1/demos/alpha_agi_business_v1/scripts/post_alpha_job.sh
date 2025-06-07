#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# scripts/post_alpha_job.sh - queue a demo job via the orchestrator REST API
# Usage: ./scripts/post_alpha_job.sh examples/job_copper_spread.json
# Requirements: curl, jq

set -euo pipefail

JOB_FILE="${1:-examples/job_copper_spread.json}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"

for cmd in curl jq; do
  command -v "$cmd" >/dev/null || { echo "❌  $cmd not found"; exit 1; }
done

JOB_JSON=$(cat "$JOB_FILE")
AGENT=$(jq -r '.agent' "$JOB_FILE")

if [[ -z "$AGENT" || "$AGENT" == "null" ]]; then
  echo "❌  'agent' field missing in $JOB_FILE"; exit 1
fi

echo "🚀  Posting alpha job '$JOB_FILE' ..."
curl -fsS -X POST \
  -H 'Content-Type: application/json' \
  -d "$JOB_JSON" \
  "http://${HOST}:${PORT}/agent/${AGENT}/trigger"
echo

echo "✔  Done."
