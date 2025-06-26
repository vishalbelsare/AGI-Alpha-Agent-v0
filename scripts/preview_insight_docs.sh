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
printf '\nServing Insight demo at http://localhost:8000/ (press Ctrl+C to stop)\n'
python -m http.server --directory site 8000

