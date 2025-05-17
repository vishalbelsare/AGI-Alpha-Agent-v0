#!/usr/bin/env bash
# scripts/post_job.sh - queue a demo job via the orchestrator REST API
# Usage: ./scripts/post_job.sh examples/sample_job.json
# Requirements: curl, jq

set -euo pipefail

JOB_FILE="${1:-examples/sample_job.json}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8000}"

for cmd in curl jq; do
  command -v "$cmd" >/dev/null || { echo "❌  $cmd not found"; exit 1; }
done

AGENT=$(jq -r '.agent' "$JOB_FILE")
if [[ -z "$AGENT" || "$AGENT" == "null" ]]; then
  echo "❌  'agent' field missing in $JOB_FILE"; exit 1
fi

echo "🚀  Queuing job for agent '$AGENT' ..."
curl -fsS -X POST "http://${HOST}:${PORT}/agent/${AGENT}/trigger"
echo

echo "✔  Done."
