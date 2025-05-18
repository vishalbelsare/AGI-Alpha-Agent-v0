#!/usr/bin/env bash
# Alpha-Factory v1 □  Production-ready launcher for the world-model demo
set -Eeuo pipefail

# Resolve script path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Force offline mode when no API key is supplied
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  export NO_LLM=1
fi

python -m alpha_factory_v1.demos.alpha_asi_world_model.alpha_asi_world_model_demo --demo "$@"
