#!/usr/bin/env bash
###############################################################################
#  run_macro_demo.sh – Macro‑Sentinel • Alpha‑Factory v1 👁️✨
#
#  Production‑grade launcher with:
#    • Host prerequisite validation (Docker ≥24, Compose plugin, network)
#    • Auto‑creation of ./config.env with sane defaults
#    • Offline sample‑data sync (Fed speeches, yield curve, stable‑flow)
#    • GPU autodetect → build profile gpu
#    • Live data collectors profile  (--live flag)  → FRED + Twitter streams
#    • Deterministic image tags / layer pulls
#    • Health‑gate on orchestrator  /__live  endpoint
#    • Graceful teardown helpers
#
#  Inspired by agentic‑trading patterns and OpenAI Agents SDK §6.
###############################################################################
set -Eeuo pipefail

###############################  helpers  #####################################
say()  { printf '\033[1;36m▶ %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[1;31m🚨 %s\033[0m\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1 || die "$1 is required"; }
has_gpu() { docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'; }

health_wait() {
  local port=$1 max=$2
  for ((i=0;i<max;i++)); do
    if curl -s "http://localhost:${port}/__live" | grep -q OK; then return 0; fi
    sleep 2
  done
  die "Service on port ${port} failed health check"
}

################################ paths #########################################
demo_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
root_dir="${demo_dir%/*/*}"
compose_file="$demo_dir/docker-compose.macro.yml"
env_file="$demo_dir/config.env"
offline_dir="$demo_dir/offline_samples"

cd "$root_dir"

################################ flags #########################################
PROFILE_LIVE=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --live) PROFILE_LIVE=1 ;;
    *) die "Unknown flag: $1" ;;
  esac
  shift
done

################################ prereqs #######################################
need docker
docker compose version >/dev/null 2>&1 || die "Docker Compose plugin missing"

ver=$(docker version --format '{{.Server.Version}}')
[[ "${ver%%.*}" -ge 24 ]] || warn "Docker $ver < 24 may slow multi‑stage builds"

################################ config.env ####################################
if [[ ! -f "$env_file" ]]; then
  say "Creating default config.env"
  cat > "$env_file" <<EOF
# Macro‑Sentinel env
OPENAI_API_KEY=
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.15
PG_PASSWORD=alpha
FRED_API_KEY=
TW_BEARER_TOKEN=
EOF
fi

################################ offline data ##################################
say "Syncing offline macro snapshots"
mkdir -p "$offline_dir"
declare -A urls=(
  [fed_speeches.csv]=https://raw.githubusercontent.com/MontrealAI/demo-assets/main/fed_speeches.csv
  [yield_curve.csv]=https://raw.githubusercontent.com/MontrealAI/demo-assets/main/yield_curve.csv
  [stable_flows.csv]=https://raw.githubusercontent.com/MontrealAI/demo-assets/main/stable_flows.csv
)
for f in "${!urls[@]}"; do
  curl -sfL "${urls[$f]}" -o "$offline_dir/$f"
done

################################ profiles ######################################
profiles=()
has_gpu && profiles+=(gpu)
[[ -z "${OPENAI_API_KEY:-}" ]] && profiles+=(offline)
(( PROFILE_LIVE )) && profiles+=(live-feed)
profile_arg=""
[[ ${#profiles[@]} -gt 0 ]] && profile_arg="--profile $(IFS=,; echo "${profiles[*]}")"

################################ build & up ####################################
say "🚢 Building images…"
docker compose -f "$compose_file" $profile_arg pull --quiet || true
docker compose -f "$compose_file" $profile_arg build --pull

say "🔄 Starting stack…"
docker compose --project-name alpha_macro -f "$compose_file" $profile_arg up -d

################################ health gate ###################################
say "⏳ Waiting for orchestrator health"
health_wait 7864 40

################################ success #######################################
printf '\n\033[1;32m🎉 Dashboard → http://localhost:7864\033[0m\n'
echo "📊 Grafana   → http://localhost:3001 (admin/alpha)"
echo "📜 Logs      → docker compose -p alpha_macro logs -f"
echo "🛑 Stop      → docker compose -p alpha_macro down"
echo "🧹 Purge     → docker compose -p alpha_macro down -v --remove-orphans"
