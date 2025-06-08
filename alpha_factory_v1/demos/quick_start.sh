#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Wrapper to maintain backwards compatibility with README instructions
# Delegates to the main quickstart script located one directory up
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../quickstart.sh" "$@"
