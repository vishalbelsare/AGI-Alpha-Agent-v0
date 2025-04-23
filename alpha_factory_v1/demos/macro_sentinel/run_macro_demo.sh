#!/usr/bin/env bash
###############################################################################
#  Macro-Sentinel Demo – Alpha-Factory v1 👁️✨
#  ---------------------------------------------------------------------------
#  One-liner launcher that                                                          │
#    • verifies host prerequisites (Docker Engine + Compose plugin)                │
#    • downloads (or refreshes) the sample macro CSV feeds                         │
#    • intelligently chooses CPU-only vs GPU build (CUDA present)                  │
#    • hydrates a local `.env` with safe defaults                                  │
#    • spins up the stack with deterministic tags                                  │
#    • tails health-check until Gradio is live                                     │
#    • prints helper commands for logs / teardown                                  │
#                                                                                 │
#  The script follows best-practice hardening from                                 │
#  – OpenAI Agents SDK guide §6 (ops & monitoring)                                 │
#  – Google ADK deployment checklist (2025-04)                                     │
###############################################################################
set -Eeuo pipefail

print() { printf "\033[1;36m▶ %s\033[0m\n" "$*"; }
err()   { printf "\033[1;31m🚨 %s\033[0m\n" "$*" >&2; }

demo_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
root_dir="${demo_dir%/*/*}"
compose_file="$demo_dir/docker-compose.macro.yml"
env_file="$demo_dir/config.env"

cd "$root_dir"

######################################################################## util ##
require() {
  command -v "$1" >/dev/null 2>&1 || { err "$1 is required."; exit 1; }
}

has_gpu() {
  docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'
}

health_wait() {
  local service=$1 port=$2
  for i in {1..30}; do
    if curl -s "http://localhost:${port}/__live" | grep -q OK; then return 0; fi
    sleep 2
  done
  err "$service health check timed-out"; exit 1
}

################################################################# prerequisite #
require docker
docker compose version >/dev/null 2>&1 || { err "Docker Compose plugin missing"; exit 1; }

######################################################################### env ##
if [[ ! -f "$env_file" ]]; then
  print "Creating default config.env (edit to add OPENAI_API_KEY)"
  cp "$demo_dir/config.env.sample" "$env_file"
fi

############################################################## sample data ####
print "Syncing sample macro telemetry (offline mode support)…"
mkdir -p "$demo_dir/offline_samples"
curl -sL https://raw.githubusercontent.com/MontrealAI/demo-assets/main/fed_speeches.csv   -o "$demo_dir/offline_samples/fed_speeches.csv"
curl -sL https://raw.githubusercontent.com/MontrealAI/demo-assets/main/yield_curve.csv    -o "$demo_dir/offline_samples/yield_curve.csv"
curl -sL https://raw.githubusercontent.com/MontrealAI/demo-assets/main/stable_flows.csv   -o "$demo_dir/offline_samples/stable_flows.csv"

################################################################ build/run ####
profile_arg=""
if has_gpu; then
  print "🖥️ NVIDIA runtime detected – enabling CUDA profile"
  profile_arg="--profile gpu"
fi

print "🚢 Building images (deterministic tags)…"
docker compose -f "$compose_file" $profile_arg pull --quiet || true
docker compose -f "$compose_file" $profile_arg build --pull

print "🔄 Starting Macro-Sentinel agents…"
docker compose --project-name alpha_macro -f "$compose_file" $profile_arg up -d

print "⏳ Waiting for health check…"
health_wait "orchestrator" 7864

################################################################ success ######
echo -e "\n\033[1;32m🎉 Macro-Sentinel live → http://localhost:7864\033[0m"
echo    "📜 Tail logs          → docker compose -p alpha_macro logs -f"
echo    "🛑 Stop stack         → docker compose -p alpha_macro down"
echo    "🧹 Remove volumes     → docker compose -p alpha_macro down -v"
