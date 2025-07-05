#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# build_macro_wheelhouse.sh -- build wheels for the Macro-Sentinel demo
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/.." && pwd)"
WHEEL_DIR=${1:-$ROOT/../wheels}
mkdir -p "$WHEEL_DIR"
pip wheel -r "$ROOT/demos/macro_sentinel/requirements.txt" -w "$WHEEL_DIR"
echo "Wheels written to $WHEEL_DIR"

