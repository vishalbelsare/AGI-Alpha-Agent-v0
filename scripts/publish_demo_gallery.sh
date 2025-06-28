#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Consolidated helper to build and deploy the full Alpha-Factory demo gallery.
# Performs extensive environment checks, rebuilds all assets and publishes the
# MkDocs site to GitHub Pages. Designed for one-command usage by non-technical
# users.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BROWSER_DIR="alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1"

# Verify Python, Node and optional dependencies
python alpha_factory_v1/scripts/preflight.py
python scripts/check_python_deps.py
python check_env.py --auto-install
node "$BROWSER_DIR/build/version_check.js"

# Ensure documentation embeds the project disclaimer and demos are valid
python scripts/verify_disclaimer_snippet.py
python -m alpha_factory_v1.demos.validate_demos

# Build the Insight browser demo and refresh documentation
"$SCRIPT_DIR/build_insight_docs.sh"

# Compile the MkDocs site in strict mode so warnings cause failure
mkdocs build --strict
python scripts/verify_workbox_hash.py site/alpha_agi_insight_v1

# Optional offline smoke test
if python - "import importlib,sys;sys.exit(0 if importlib.util.find_spec('playwright') else 1)"; then
  python -m http.server --directory site 8000 &
  SERVER_PID=$!
  trap 'kill $SERVER_PID' EXIT
  sleep 2
  python scripts/verify_insight_offline.py
  kill $SERVER_PID
  trap - EXIT
else
  echo "Playwright not found; skipping offline check" >&2
fi

# Deploy to GitHub Pages
mkdocs gh-deploy --force

remote=$(git config --get remote.origin.url)
repo_path=${remote#*github.com[:/]}
repo_path=${repo_path%.git}
org="${repo_path%%/*}"
repo="${repo_path##*/}"
url="https://${org}.github.io/${repo}/"

echo "Demo gallery deployed to $url"
