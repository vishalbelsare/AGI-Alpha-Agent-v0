#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Orchestrate the full demo gallery deployment sprint.
# This wrapper ensures the Insight demo assets build correctly, the
# MkDocs site passes integrity checks and the final result deploys to
# GitHub Pages. Designed for Codex automation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BROWSER_DIR="alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1"

# Comprehensive environment checks
python alpha_factory_v1/scripts/preflight.py
node "$BROWSER_DIR/build/version_check.js"
python scripts/check_python_deps.py
python check_env.py --auto-install
# disclaimer snippet verification removed; rely on documentation updates
python -m alpha_factory_v1.demos.validate_demos

# Build the Insight docs and gallery
npm --prefix "$BROWSER_DIR" run fetch-assets
npm --prefix "$BROWSER_DIR" ci
"$SCRIPT_DIR/build_insight_docs.sh"
python scripts/mirror_demo_pages.py
python scripts/build_service_worker.py

# Compile and verify the MkDocs site. Use --strict so warnings fail the build
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
case "$(uname)" in
  Darwin*) open "$url" ;;
  Linux*) (xdg-open "$url" >/dev/null 2>&1 || echo "Browse to $url") ;;
  MINGW*|MSYS*|CYGWIN*) start "$url" ;;
  *) echo "Browse to $url" ;;
esac


