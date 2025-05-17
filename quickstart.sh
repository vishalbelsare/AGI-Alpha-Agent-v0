#!/usr/bin/env bash
# Wrapper script for alpha_factory_v1/quickstart.sh
# Provides a friendly top-level entry point.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/alpha_factory_v1/quickstart.sh" "$@"

