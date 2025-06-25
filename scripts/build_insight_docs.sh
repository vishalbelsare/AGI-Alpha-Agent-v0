#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# This script is a conceptual research prototype.
# Build the Insight demo and documentation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BROWSER_DIR="alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1"
DOCS_DIR="docs/alpha_agi_insight_v1"

usage() {
    cat <<USAGE
Usage: $0

Build the Insight browser bundle, refresh $DOCS_DIR
and generate the mkdocs site.
USAGE
}

if [[ "${1:-}" =~ ^(-h|--help)$ ]]; then
    usage
    exit 0
fi

# Ensure the correct Node.js version before running npm
if ! node "$BROWSER_DIR/build/version_check.js"; then
    echo "ERROR: Node.js 20+ is required to build the Insight docs." >&2
    exit 1
fi

# Fetch WASM assets then install Node dependencies and build the browser bundle
npm --prefix "$BROWSER_DIR" run fetch-assets
npm --prefix "$BROWSER_DIR" ci
npm --prefix "$BROWSER_DIR" run build:dist

# Refresh docs directory with the new bundle while preserving the README
README_TEMP=""
if [[ -f "$DOCS_DIR/README.md" ]]; then
    README_TEMP="$(mktemp)"
    cp "$DOCS_DIR/README.md" "$README_TEMP"
fi
rm -rf "$DOCS_DIR"
mkdir -p "$DOCS_DIR"
unzip -q -o "$BROWSER_DIR/insight_browser.zip" -d "$DOCS_DIR"
if [[ -n "$README_TEMP" ]]; then
    mv "$README_TEMP" "$DOCS_DIR/README.md"
fi

# Build the MkDocs site
mkdocs build



