#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Orchestrate the Edge-of-Knowledge demo gallery deployment.
# This conceptual research script validates the environment,
# verifies demo packages, builds the site and publishes it to GitHub Pages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Ensure Python, Node and optional packages are present
python alpha_factory_v1/scripts/preflight.py
node alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1/build/version_check.js
python scripts/check_python_deps.py
python check_env.py --auto-install

# Validate demo directories for basic quality
python -m alpha_factory_v1.demos.validate_demos

# Build and deploy the full gallery
"$SCRIPT_DIR/deploy_gallery_pages.sh"
