#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# ────────────────────────────────────────────────────────────────
# Alpha‑AGI Business v1 – quick launcher
# Options : --pull  use the signed image from GHCR (skip local build)
#           --gpu   enable NVIDIA runtime (if toolkit installed)
#           --stop  shut down and remove the container
# Usage   : ./run_business_v1_demo.sh [--pull] [--gpu] [--stop]
# ────────────────────────────────────────────────────────────────
set -Eeuo pipefail

# ------------ CONSTANTS ------------------------------------------------------
DEMO_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &>/dev/null && pwd )"
ROOT_DIR="${DEMO_DIR%/*/*}"
IMAGE="alpha_business_v1:latest"
GHCR_IMAGE="ghcr.io/montrealai/alpha-business-v1:latest"
CONTAINER="alpha_business_v1"

# ------------ UX HELPERS -----------------------------------------------------
cinfo()  { printf '\e[32m• %s\e[0m\n' "$*"; }
cfatal() { printf '\e[31m✘ %s\e[0m\n' "$*"; exit 1; }

usage() {
  cat <<EOF
Usage: $(basename "$0") [--pull] [--gpu] [--stop]

  --pull    skip local build and pull signed image ($GHCR_IMAGE)
  --gpu     enable NVIDIA runtime (needs nvidia-container-toolkit)
  --stop    stop and remove the running container

EOF
  exit 0
}

# ------------ FLAG PARSE -----------------------------------------------------
PULL=0 GPU=0 STOP=0
while [[ ${1:-} ]]; do
  case "$1" in
    --pull) PULL=1 ;;
    --gpu)  GPU=1  ;;
    --stop) STOP=1 ;;
    -h|--help) usage ;;
    *) cfatal "Unknown flag $1" ;;
  esac
  shift
done

# ------------ PREREQUISITES --------------------------------------------------
command -v docker >/dev/null 2>&1 || \
  cfatal "Docker is required → https://docs.docker.com/get-docker/"

cd "$ROOT_DIR"
if ! AUTO_INSTALL_MISSING=1 python "$ROOT_DIR/../check_env.py"; then
  cfatal "Environment check failed. Please resolve the issues and try again."
fi

# ------------ STOP PATH ------------------------------------------------------
if (( STOP )); then
  if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER$"; then
    cinfo "Stopping $CONTAINER …"
    docker rm -f "$CONTAINER"
  else
    cinfo "Container $CONTAINER is not running"
  fi
  exit 0
fi

# ------------ IMAGE PREP -----------------------------------------------------
if (( PULL )); then
  cinfo "Pulling image $GHCR_IMAGE"
  docker pull "$GHCR_IMAGE"
  image="$GHCR_IMAGE"
else
  cinfo "Building image $IMAGE"
  docker build -t "$IMAGE" -f alpha_factory_v1/Dockerfile .
  image="$IMAGE"
fi

GPU_ARGS=()
(( GPU )) && GPU_ARGS=(--gpus all)

# ------------ RUN CONTAINER --------------------------------------------------
cinfo "Launching Alpha‑AGI Business demo …"
docker run -d --name "$CONTAINER" -p 7860:7860 -p 8000:8000 \
  -e OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
  "${GPU_ARGS[@]}" "$image" python -m alpha_factory_v1.demos.alpha_agi_business_v1
