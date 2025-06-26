#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# This script builds the Insight demo then serves the generated site locally.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Build the Insight browser bundle and docs
./scripts/build_insight_docs.sh

# Serve the site on localhost:8000
python -m http.server --directory site 8000 &
SERVER_PID=$!
trap 'kill $SERVER_PID' EXIT

# Run a quick headless check to ensure the PWA loads offline
sleep 2
if ! python scripts/verify_insight_offline.py; then
    echo "Offline check failed" >&2
    kill $SERVER_PID
    exit 1
fi

printf '\nServing Insight demo at http://localhost:8000/ (press Ctrl+C to stop)\n'
wait $SERVER_PID

