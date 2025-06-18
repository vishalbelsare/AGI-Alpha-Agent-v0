#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build a local wheelhouse for offline installation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p wheels
pip wheel -r requirements.lock -w wheels
pip wheel -r requirements-dev.txt -w wheels

