#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# This script is a conceptual research prototype.
# Deploy the Insight demo to GitHub Pages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BROWSER_DIR="alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1"

node "$BROWSER_DIR/build/version_check.js"

npm --prefix "$BROWSER_DIR" run fetch-assets
npm --prefix "$BROWSER_DIR" ci

"$SCRIPT_DIR/publish_insight_pages.sh"

remote=$(git config --get remote.origin.url)
repo_path=${remote#*github.com[:/]}
repo_path=${repo_path%.git}
org="${repo_path%%/*}"
repo="${repo_path##*/}"
url="https://${org}.github.io/${repo}/alpha_agi_insight_v1/"

echo "Insight demo deployed to $url"
