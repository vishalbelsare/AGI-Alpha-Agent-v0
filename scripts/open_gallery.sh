#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Open the Alpha-Factory demo gallery on GitHub Pages.
set -euo pipefail

remote=$(git config --get remote.origin.url)
repo_path=${remote#*github.com[:/]}
repo_path=${repo_path%.git}
org="${repo_path%%/*}"
repo="${repo_path##*/}"
url="https://${org}.github.io/${repo}/gallery.html"

echo "Opening $url"
case "$(uname)" in
  Darwin*) open "$url" ;;
  Linux*) (xdg-open "$url" >/dev/null 2>&1 || echo "Browse to $url") ;;
  MINGW*|MSYS*|CYGWIN*) start "$url" ;;
  *) echo "Browse to $url" ;;
esac
