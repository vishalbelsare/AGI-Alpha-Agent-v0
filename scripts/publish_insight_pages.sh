#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# This script builds the Insight demo and publishes the docs site to GitHub Pages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

./scripts/build_insight_docs.sh

# Deploy using mkdocs gh-deploy which relies on ghp-import under the hood
mkdocs gh-deploy --force
