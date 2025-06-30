#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# This repository is a conceptual research prototype. References to "AGI" and
# "superintelligence" describe aspirational goals and do not indicate the
# presence of a real general intelligence. Use at your own risk. Nothing herein
# constitutes financial advice. MontrealAI and the maintainers accept no
# liability for losses incurred from using this software.
# See docs/DISCLAIMER_SNIPPET.md

set -Eeuo pipefail
trap 'echo -e "\n\u274c Error on line $LINENO" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

cat "$SCRIPT_DIR/docs/DISCLAIMER_SNIPPET.md"

IMAGE="alpha-factory-quickstart"

docker build -t "$IMAGE" docker/quickstart

docker run --rm -it \
  -v "$SCRIPT_DIR/.env:/app/.env" \
  -p 8000:8000 "$IMAGE"
