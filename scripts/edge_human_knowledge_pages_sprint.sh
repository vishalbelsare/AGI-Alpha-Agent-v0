#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Thin wrapper orchestrating the Edge-of-Human-Knowledge Pages Sprint.
# Executes edge_of_knowledge_sprint.sh to build and deploy the demo gallery.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$SCRIPT_DIR/edge_of_knowledge_sprint.sh" "$@"
