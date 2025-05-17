#!/usr/bin/env bash
# Wrapper to maintain backwards compatibility with README instructions
# Delegates to the main quickstart script located one directory up
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/../quickstart.sh" "$@"
