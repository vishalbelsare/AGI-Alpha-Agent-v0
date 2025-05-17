#!/usr/bin/env bash
# one_click_install.sh -- Instant Alpha-Factory deploy
# Usage: ./one_click_install.sh [installer flags]
# Runs preflight checks then invokes install_alpha_factory_pro.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

python3 preflight.py
chmod +x install_alpha_factory_pro.sh

exec ./install_alpha_factory_pro.sh --bootstrap --deploy "$@"
