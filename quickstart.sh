#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# See docs/DISCLAIMER_SNIPPET.md
# Wrapper script for alpha_factory_v1/quickstart.sh
# Provides a friendly top-level entry point.

set -Eeuo pipefail
trap 'echo -e "\n\u274c Error on line $LINENO" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -n "${WHEELHOUSE:-}" ]; then
    python check_env.py --auto-install --wheelhouse "$WHEELHOUSE"
else
    python check_env.py --auto-install
fi

exec "$SCRIPT_DIR/alpha_factory_v1/quickstart.sh" "$@"

