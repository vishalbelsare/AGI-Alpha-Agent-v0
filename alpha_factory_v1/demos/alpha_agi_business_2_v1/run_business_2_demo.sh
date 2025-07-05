#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# NOTE: This demo is a research prototype. References to "AGI" or "superintelligence" describe aspirational goals and do not indicate the presence of real general intelligence. Use at your own risk. Nothing herein constitutes financial advice.
# Quick launcher for alpha_agi_business_2_v1
set -euo pipefail

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
root_dir="${script_dir%/*/*}"

cd "$root_dir"

command -v docker >/dev/null 2>&1 || { echo "❌ Docker is required"; exit 1; }

image="alpha_business_v2:latest"

docker build -t "$image" -f alpha_factory_v1/Dockerfile .
docker run --rm -it -p 7860:7860 -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  "$image" python -m alpha_factory_v1.demos.alpha_agi_business_2_v1

