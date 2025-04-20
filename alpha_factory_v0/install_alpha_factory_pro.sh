#!/usr/bin/env bash
# install_alpha_factory_pro.sh
#
# Build the Alpha‑Factory *Pro* Docker image with optional modules.
# ──────────────────────────────────────────────────────────────────
# Usage examples
#   ./install_alpha_factory_pro.sh                  # ⟶ prompt for choices
#   ./install_alpha_factory_pro.sh --all            # ⟶ everything
#   ./install_alpha_factory_pro.sh --no-ui --trace  # ⟶ headless + trace ws
#
# Flags (can be combined)
#   --all        include every optional module
#   --ui         include the React / D3 front‑end (default: if tty)
#   --trace      include live trace‑graph WebSocket
#   --tests      copy tests & dev tools into the image
#   --no-ui      exclude the UI (useful for servers)
#   --no-cache   pass --no-cache to docker build
#   -h|--help    show this help
set -euo pipefail

# ─── defaults ─────────────────────────────────────────────────────────────
tty=${CI:-}          # a bit of CI detection
want_ui=${tty:-1}    # guess “yes” if interactive; overridable by flags
want_trace=0
want_tests=0
nocache_arg=""
tags="alphafactory_pro:latest"

# ─── cli parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --all)      want_ui=1; want_trace=1; want_tests=1 ;;
    --ui)       want_ui=1 ;;
    --trace)    want_trace=1 ;;
    --tests)    want_tests=1 ;;
    --no-ui)    want_ui=0 ;;
    --no-cache) nocache_arg="--no-cache" ;;
    -h|--help)  grep -E '^#( |$)' "$0" | sed 's/^# ?//' ; exit 0 ;;
    *)          echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
  shift
done

# ─── show summary ────────────────────────────────────────────────────────
echo "╭─────────────────────────────────────────────────────────────"
printf "│ Building %s  \n" "$tags"
printf "│  • UI:     %s\n" "$([[ $want_ui    == 1 ]] && echo ENABLED || echo disabled)"
printf "│  • Trace:  %s\n" "$([[ $want_trace == 1 ]] && echo ENABLED || echo disabled)"
printf "│  • Tests:  %s\n" "$([[ $want_tests == 1 ]] && echo ENABLED || echo disabled)"
echo "╰─────────────────────────────────────────────────────────────"

# ─── build args ─────────────────────────────────────────────────────────
build_args=(
  --build-arg "ENABLE_UI=$want_ui"
  --build-arg "ENABLE_TRACE=$want_trace"
  --build-arg "INCLUDE_TESTS=$want_tests"
)
[[ -n $nocache_arg ]] && build_args+=("$nocache_arg")

# ─── docker build ───────────────────────────────────────────────────────
docker build "${build_args[@]}" -t "$tags" .

echo "✅ Image built: $tags"
echo "   Run it with:"
echo "     docker run --rm -p 33000:3000 $tags"

