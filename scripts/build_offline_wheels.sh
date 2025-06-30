#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# This script is a conceptual research prototype.
# Collect wheels for offline installation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p wheels

if [[ "${SMOKE_ONLY:-0}" == "1" ]]; then
    pip wheel numpy pyyaml pandas prometheus_client -w wheels
    exit 0
fi

pip wheel -r requirements.lock -w wheels
pip wheel -r requirements-dev.txt -w wheels
pip wheel -r requirements-demo.lock -w wheels

# Build wheels for each demo if a lock file exists
while IFS= read -r -d '' req_file; do
    pip wheel -r "$req_file" -w wheels
done < <(find alpha_factory_v1/demos -name requirements.lock -print0 | sort -z)
