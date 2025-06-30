#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# This wrapper orchestrates the full α-AGI Insight deployment sprint.
# It performs environment checks, builds the demo, verifies offline
# functionality and publishes the site to GitHub Pages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

"$SCRIPT_DIR/deploy_insight_full.sh"
