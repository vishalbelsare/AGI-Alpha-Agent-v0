#!/usr/bin/env bash
# Alpha-Factory v1 □  Production-ready launcher for the α-ASI World-Model demo
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python -m alpha_factory_v1.demos.alpha_asi_world_model.alpha_asi_world_model_demo --demo "$@"
