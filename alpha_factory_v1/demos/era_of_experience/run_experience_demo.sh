#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
###############################################################################
#  run_experience_demo.sh – Era-of-Experience • Alpha-Factory v1 👁️✨
#
#  Zero-to-dashboard launcher that:
#    • Verifies host requirements (Docker ≥ 24, Compose plugin, outbound net)
#    • Creates ./config.env with sane defaults on first run
#    • Syncs optional offline sample-experience CSVs
#    • Detects NVIDIA runtime → enables   --profile gpu
#    • Enables fully-offline mode when OPENAI_API_KEY is absent
#    • Supports   --live   flag ⇒ starts real-time sensor collectors profile
#    • Health-gates the Gradio orchestrator on  /__live
#    • Prints helper commands for logs / teardown
#
#  Safe for non-technical users: *copy-paste & watch the magic*.
###############################################################################
set -Eeuo pipefail

################################### helpers ###################################
say()  { printf '\033[1;36m▶ %s\033[0m\n'  "$*"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n'  "$*" >&2; }
die()  { printf '\033[1;31m🚨 %s\033[0m\n' "$*" >&2; exit 1; }

need() { command -v "$1" &>/dev/null || die "$1 is required"; }
has_gpu() {
  docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"' 2>/dev/null
}

health_wait() {
  local port=$1 max=$2
  for ((i=0;i<max;i++)); do
    if curl -sf "http://localhost:${port}/__live" | grep -q OK; then return 0; fi
    sleep 2
  done
  die "Service on port ${port} failed health check"
}

################################### paths #####################################
demo_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
root_dir="${demo_dir%/*/*}"                    # → …/alpha_factory_v1
compose_file="$demo_dir/docker-compose.experience.yml"
env_file="$demo_dir/config.env"
sample_dir="${SAMPLE_DATA_DIR:-$demo_dir/offline_samples}"
sample_dir="$(realpath -m "$sample_dir")"
offline_dir="$sample_dir"
export SAMPLE_DATA_DIR="$sample_dir"

cd "$root_dir"                                # required for build context

################################### flags #####################################
PROFILE_LIVE=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --live) PROFILE_LIVE=1 ;;
    -h|--help)
      cat <<EOF
Usage: ./run_experience_demo.sh [--live]

--live   Start real-time collectors (wearables-sim, RSS feeds, etc.)
Place pre-downloaded CSVs in $SAMPLE_DATA_DIR (defaults to ./offline_samples/) for air-gapped runs.
Set SKIP_ENV_CHECK=1 to bypass Python package checks.
EOF
      exit 0 ;;
    *) die "Unknown flag: $1" ;;
  esac
  shift
done

################################# prereqs #####################################
need docker
docker compose version &>/dev/null || die "Docker Compose plugin missing"
ver=$(docker version --format '{{.Server.Version}}')
[[ "${ver%%.*}" -ge 24 ]] || warn "Docker $ver < 24 may slow multi-stage builds"
need curl

################################ env check ####################################
if [[ "${SKIP_ENV_CHECK:-0}" != "1" ]]; then
  if command -v python3 &>/dev/null && [[ -f ../check_env.py ]]; then
    say "Checking host Python packages"
    python3 ../check_env.py --auto-install || warn "Environment check failed"
  fi
fi

################################ config.env ###################################
if [[ ! -f "$env_file" ]]; then
  say "Creating default config.env"
  cat > "$env_file" <<'EOF'
# Era-of-Experience env - edit as you wish
OPENAI_API_KEY=
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.2
FRED_API_KEY=
WEARABLES_API_KEY=
PG_PASSWORD=alpha
LIVE_FEED=0
EOF
fi

############################## offline samples ################################
say "Syncing offline experience snapshots in $offline_dir"
mkdir -p "$offline_dir"
declare -A urls=(
  [wearable_daily.csv]=https://raw.githubusercontent.com/MontrealAI/demo-assets/main/wearable_daily.csv
  [edu_progress.csv]  =https://raw.githubusercontent.com/MontrealAI/demo-assets/main/edu_progress.csv
)
for f in "${!urls[@]}"; do
  if [[ -f "$offline_dir/$f" ]]; then
    say "Local file detected: $offline_dir/$f – skipping download"
    continue  # local file already present
  fi
  if ! curl -sfL "${urls[$f]}" -o "$offline_dir/$f"; then
    warn "Failed downloading $f – using empty placeholder"
    : > "$offline_dir/$f"
  fi
done

################################# profiles ####################################
profiles=()
has_gpu && profiles+=(gpu)
[[ -z "${OPENAI_API_KEY:-}" ]] && profiles+=(offline)
(( PROFILE_LIVE )) && profiles+=(live-feed)
export LIVE_FEED=${PROFILE_LIVE}
profile_arg=""
[[ ${#profiles[@]} -gt 0 ]] && profile_arg="--profile $(IFS=,; echo "${profiles[*]}")"

################################ build & up ###################################
say "🚢 Building images…"
docker compose -f "$compose_file" $profile_arg pull --quiet || true
docker compose -f "$compose_file" $profile_arg build --pull

say "🔄 Starting stack…"
docker compose --project-name alpha_experience -f "$compose_file" $profile_arg up -d

################################ health gate ##################################
say "⏳ Waiting for orchestrator health"
health_wait 7860 40

################################ success ######################################
printf '\n\033[1;32m🎉 Dashboard → http://localhost:7860\033[0m\n'
echo "📜 Logs      → docker compose -p alpha_experience logs -f"
echo "🛑 Stop      → docker compose -p alpha_experience down"
echo "🧹 Purge     → docker compose -p alpha_experience down -v --remove-orphans"
