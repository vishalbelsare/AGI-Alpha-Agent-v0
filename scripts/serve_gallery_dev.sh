#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# This script is a conceptual research prototype.
# Live-reload the demo gallery using MkDocs.
# Rebuilds demo docs before launching the development server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BROWSER_DIR="alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1"

# Basic environment checks
python alpha_factory_v1/scripts/preflight.py
node "$BROWSER_DIR/build/version_check.js"

# Rebuild browser bundle and docs
npm --prefix "$BROWSER_DIR" run fetch-assets
npm --prefix "$BROWSER_DIR" ci
"$SCRIPT_DIR/build_insight_docs.sh"
python scripts/generate_demo_docs.py
python scripts/generate_gallery_html.py

# Launch MkDocs with live reload
mkdocs serve --dev-addr 0.0.0.0:8000
