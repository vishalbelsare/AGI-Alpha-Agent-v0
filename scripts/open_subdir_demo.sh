#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Open a specific demo page from the subdirectory gallery on GitHub Pages.
set -euo pipefail

usage() { echo "Usage: $0 <demo_name>" >&2; exit 1; }
[[ $# -eq 1 ]] || usage
DEMO="$1"

if [[ -n "${AF_GALLERY_URL:-}" ]]; then
  url="${AF_GALLERY_URL%/}/alpha_factory_v1/demos/${DEMO}/index.html"
else
  remote=$(git config --get remote.origin.url)
  repo_path=${remote#*github.com[:/]}
  repo_path=${repo_path%.git}
  org="${repo_path%%/*}"
  repo="${repo_path##*/}"
  url="https://${org}.github.io/${repo}/alpha_factory_v1/demos/${DEMO}/index.html"
fi

check_remote() {
  local status
  status=$(curl -fsIL "$url" 2>/dev/null | head -n1 | awk '{print $2}') || return 1
  [[ "$status" == "200" ]]
}

if check_remote; then
  echo "Opening $url"
  case "$(uname)" in
    Darwin*) open "$url" ;;
    Linux*) (xdg-open "$url" >/dev/null 2>&1 || echo "Browse to $url") ;;
    MINGW*|MSYS*|CYGWIN*) start "$url" ;;
    *) echo "Browse to $url" ;;
  esac
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
local_page="$REPO_ROOT/site/alpha_factory_v1/demos/${DEMO}/index.html"
if [[ ! -f "$local_page" ]]; then
  echo "Local page missing. Building gallery..." >&2
  "$REPO_ROOT/scripts/build_gallery_site.sh" || {
    echo "Failed to build the gallery" >&2
    exit 1
  }
fi

if [[ -f "$local_page" ]]; then
  echo "Remote page unavailable. Opening local copy at $local_page" >&2
  case "$(uname)" in
    Darwin*) open "$local_page" ;;
    Linux*) (xdg-open "$local_page" >/dev/null 2>&1 || echo "Browse to $local_page") ;;
    MINGW*|MSYS*|CYGWIN*) start "$local_page" ;;
    *) echo "Browse to $local_page" ;;
  esac
else
  echo "Demo $DEMO not found. Build the gallery with ./scripts/build_gallery_site.sh" >&2
  exit 1
fi
