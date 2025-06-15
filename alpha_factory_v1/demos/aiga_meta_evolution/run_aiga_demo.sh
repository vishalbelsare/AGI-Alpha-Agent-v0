#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ────────────────────────────────────────────────────────────────
# AI-GA Meta-Evolution – one-command launcher
# Works on: Linux, macOS, WSL2 (Docker Desktop ≥ 4.28)
# Prereqs : Docker Engine 24+  | docker compose plugin OR legacy binary
# Options : --pull   use the signed image from GHCR (no local build)
#           --gpu    enable NVIDIA runtime (if toolkit installed)
#           --logs   follow service logs after start-up
#           --reset  nuke volumes & images (clean slate)
# Docs    : dashboard → http://localhost:7862   |  API → http://localhost:8000/docs
# Stop    : ./run_aiga_demo.sh --stop
# ────────────────────────────────────────────────────────────────
set -Eeuo pipefail

# ------------- CONSTANTS ----------------------------------------------------
DEMO_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
ROOT_DIR="${DEMO_DIR%/*/*}"              # → alpha_factory_v1
COMPOSE_YAML="$DEMO_DIR/docker-compose.aiga.yml"
PROJECT=alpha_aiga
GHCR_IMAGE=ghcr.io/montrealai/alpha-aiga:latest

# ------------- UX helpers ---------------------------------------------------
cinfo()  { printf '\e[32m• %s\e[0m\n' "$*"; }
cwarn()  { printf '\e[33m• %s\e[0m\n' "$*"; }
cfatal() { printf '\e[31m✘ %s\e[0m\n' "$*"; exit 1; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [--pull] [--gpu] [--logs] [--reset] [--stop]

  --pull    skip local build; pull signed image ($GHCR_IMAGE)
  --gpu     enable NVIDIA runtime (needs nvidia-container-toolkit)
  --logs    tail container logs after start-up
  --reset   remove containers, volumes & images for a clean slate
  --stop    graceful shutdown (alias for docker compose down)

EOF
  exit 0
}

# ------------- FLAG PARSE ---------------------------------------------------
PULL=0 GPU=0 LOGS=0 RESET=0 STOP=0
while [[ ${1:-} ]]; do
  case "$1" in
    --pull)  PULL=1 ;;
    --gpu)   GPU=1  ;;
    --logs)  LOGS=1 ;;
    --reset) RESET=1 ;;
    --stop)  STOP=1 ;;
    -h|--help) usage ;;
    *) cfatal "Unknown flag $1" ;;
  esac; shift
done

# ------------- PREREQUISITES -----------------------------------------------
command -v docker >/dev/null 2>&1 || \
  cfatal "Docker is required → https://docs.docker.com/get-docker/"

if docker compose version &>/dev/null;   then DC="docker compose"
elif command -v docker-compose &>/dev/null; then DC="docker-compose"
else cfatal "docker compose plugin not found"; fi

# ------------- RESET / STOP PATHS ------------------------------------------
if (( RESET )); then
  cwarn "Removing AI-GA containers, volumes & images …"
  $DC -p "$PROJECT" down -v --rmi all || true
  exit 0
fi

if (( STOP )); then
  cinfo "Stopping AI-GA demo …"
  $DC -p "$PROJECT" down
  exit 0
fi

# ------------- ENV FILE -----------------------------------------------------
CONFIG_ENV="$DEMO_DIR/config.env"
if [[ ! -f "$CONFIG_ENV" ]]; then
  cinfo "Creating default config.env (edit to add OPENAI_API_KEY)"
  cp "$DEMO_DIR/config.env.sample" "$CONFIG_ENV"
fi

# ------------- START-UP -----------------------------------------------------
cd "$ROOT_DIR"
if ! AUTO_INSTALL_MISSING=1 python "$ROOT_DIR/../check_env.py"; then
  cfatal "Environment check failed. Please resolve the issues and try again."
fi

(( PULL )) && { cinfo "Pulling image $GHCR_IMAGE"; docker pull "$GHCR_IMAGE"; }

GPU_ARGS=()
(( GPU )) && GPU_ARGS=(--compatibility --profile gpu)

cinfo "Launching AI-GA demo (project: $PROJECT)…"
$DC --project-name "$PROJECT" \
    --env-file "$CONFIG_ENV" \
    -f "$COMPOSE_YAML" "${GPU_ARGS[@]}" \
    up -d ${PULL:+--no-build}

# ------------- POST-START MSG ----------------------------------------------
echo
cinfo "Dashboard → http://localhost:7862"
cinfo "OpenAPI  → http://localhost:8000/docs"
cinfo "Stop     → $0 --stop"
echo

(( LOGS )) && $DC -p "$PROJECT" logs -f
