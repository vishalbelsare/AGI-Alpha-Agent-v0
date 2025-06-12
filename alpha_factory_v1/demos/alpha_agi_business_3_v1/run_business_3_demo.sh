#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Quick launcher for alpha_agi_business_3_v1
set -euo pipefail

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
root_dir="$(git -C "$script_dir" rev-parse --show-toplevel 2>/dev/null || echo "${script_dir%/*/*/*}")"

cd "$root_dir"

command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required"; exit 1; }

image="alpha_business_v3:latest"

docker build -t "$image" -f alpha_factory_v1/demos/alpha_agi_business_3_v1/Dockerfile .
docker run --rm -it -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  "$image" python -m alpha_factory_v1.demos.alpha_agi_business_3_v1
