#!/usr/bin/env bash
###############################################################################
#  run_macro_demo.sh — Macro-Sentinel • Alpha-Factory v1 👁️✨
#
#  Turn-key production launcher for the **Macro-Sentinel** multi-agent stack.
#  --------------------------------------------------------------------------
#  ▸ Validates host: Docker ≥ 24, Compose plug-in, outbound HTTPS
#  ▸ Creates ./config.env with sane defaults on first run
#  ▸ Downloads/refreshes offline CSV snapshots (Fed speeches, yields, flows)
#  ▸ Auto-detects NVIDIA runtime → enables `gpu` profile
#  ▸ `--live` flag starts real-time collectors (FRED, Etherscan, X/Twitter)
#  ▸ `--reset` stops & purges any previous stack before fresh start
#  ▸ Deterministic image tags, pre-pulls cache layers
#  ▸ Health-gates the orchestrator on /healthz (40 × 2 s)
#  ▸ Prints helper commands (logs, stop, purge) on success
#
#  Usage:
#      ./run_macro_demo.sh [--live] [--reset] [--help]
#
#  Profiles combined automatically:
#      gpu        → CUDA build
#      offline    → no OPENAI_API_KEY in env
#      live-feed  → --live flag
###############################################################################
set -Eeuo pipefail
shopt -s inherit_errexit

# ────────────────────────── helpers ──────────────────────────
say()  { printf '\033[1;36m▶ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m🚨 %s\033[0m\n' "$*" >&2; exit 1; }
need() { command -v "$1" &>/dev/null || die "$1 is required"; }
has_gpu() { docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; }

health_wait() {
  local url=$1 tries=$2
  for ((i=0;i<tries;i++)); do
    curl -fs "$url" &>/dev/null && return 0
    sleep 2
  done
  die "Health-check failed ($url)"
}

usage() {
  cat <<EOF
Macro-Sentinel launcher
Usage: $(basename "$0") [--live] [--reset] [--help]

  --live    Enable live macro collectors (requires API keys in config.env)
  --reset   Stop & purge containers + volumes before new start
  --help    Show this message
EOF
}

# ────────────────────────── CLI flags ─────────────────────────
LIVE=0 RESET=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --live)  LIVE=1 ;;
    --reset) RESET=1 ;;
    --help)  usage; exit 0 ;;
    *) die "Unknown flag $1 (see --help)" ;;
  esac; shift
done

# ──────────────────────── path constants ──────────────────────
demo_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
root_dir="${demo_dir%/*/*}"
compose_file="$demo_dir/docker-compose.macro.yml"
env_file="$demo_dir/config.env"
offline_dir="$demo_dir/offline_samples"
cd "$root_dir"

# ──────────────────────── prerequisites ───────────────────────
need docker
docker compose version &>/dev/null || die "Docker Compose plug-in missing"
curl -fsSL https://google.com &>/dev/null || warn "No outbound HTTPS — live mode may fail"

# Optional reset
if (( RESET )); then
  say "Purging previous stack"
  docker compose -p alpha_macro -f "$compose_file" down -v --remove-orphans || true
fi

# ──────────────────────── config.env init ─────────────────────
if [[ ! -f $env_file ]]; then
  say "Creating default config.env"
  cat >"$env_file"<<'EOF'
# ==== Macro-Sentinel configuration ====
OPENAI_API_KEY=
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.15

# PostgreSQL (TimescaleDB)
PG_PASSWORD=alpha

# Optional live collectors
FRED_API_KEY=
ETHERSCAN_API_KEY=
TW_BEARER_TOKEN=
ALPHA_FACTORY_ENABLE_ADK=0
EOF
fi

# ──────────────────────── offline data ────────────────────────
say "Syncing offline CSV snapshots"
mkdir -p "$offline_dir"
declare -A SRC=(
  [fed_speeches.csv]="https://raw.githubusercontent.com/MontrealAI/demo-assets/main/fed_speeches.csv"
  [yield_curve.csv]="https://raw.githubusercontent.com/MontrealAI/demo-assets/main/yield_curve.csv"
  [stable_flows.csv]="https://raw.githubusercontent.com/MontrealAI/demo-assets/main/stable_flows.csv"
  [cme_settles.csv]="https://raw.githubusercontent.com/MontrealAI/demo-assets/main/cme_settles.csv"
)
for f in "${!SRC[@]}"; do
  curl -fsSL "${SRC[$f]}" -o "$offline_dir/$f"
done

# ──────────────────────── compose profiles ────────────────────
profiles=()
has_gpu && profiles+=(gpu)
[[ -z "${OPENAI_API_KEY:-}" ]] && profiles+=(offline)
(( LIVE )) && profiles+=(live-feed)
export LIVE_FEED=${LIVE}
profile_arg=""
[[ ${#profiles[@]} -gt 0 ]] && profile_arg="--profile $(IFS=,;echo "${profiles[*]}")"

# ───────────────────────── Docker build ───────────────────────
say "🚢 Building images (profiles: ${profiles[*]:-none})"
docker compose -f "$compose_file" $profile_arg pull --quiet || true
docker compose -f "$compose_file" $profile_arg build --pull

# ───────────────────────── stack up ───────────────────────────
say "🔄 Starting containers"
docker compose --project-name alpha_macro -f "$compose_file" $profile_arg up -d

# ─────────────────────── health gate & trap ───────────────────
trap 'docker compose -p alpha_macro stop; exit 0' INT
say "⏳ Waiting for orchestrator health"
health_wait "http://localhost:7864/healthz" 40

# ───────────────────────── success banner ─────────────────────
printf '\n\033[1;32m🎉 Dashboard → http://localhost:7864\033[0m\n'
echo   "📊 Grafana   → http://localhost:3001 (admin / alpha)"
echo   "📜 Logs      → docker compose -p alpha_macro logs -f"
echo   "🛑 Stop      → docker compose -p alpha_macro down"
echo   "🧹 Purge     → docker compose -p alpha_macro down -v --remove-orphans"
