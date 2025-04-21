
#!/usr/bin/env bash
set -euo pipefail
IMAGE_REMOTE="ghcr.io/alphafactory/core:v2"
IMAGE_LOCAL="alphafactory/core:v2local"
CONTAINER_NAME="alpha_factory"
PORT="3000"

echo "=== Alpha Factory v2 – Installer ==="
command -v docker >/dev/null || { echo "Docker required."; exit 1; }

GPU_ARGS=""
if command -v nvidia-smi >/dev/null 2>&1 && docker info --format '{{json .Runtimes.nvidia}}' | grep -q nvidia; then
  echo "GPU detected."
  GPU_ARGS="--gpus all"
fi

# pull or build
if docker pull --quiet "$IMAGE_REMOTE"; then
  IMAGE="$IMAGE_REMOTE"
else
  echo "Building locally..."
  docker build --no-cache -t "$IMAGE_LOCAL" .
  IMAGE="$IMAGE_LOCAL"
fi

# restart container
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run -d $GPU_ARGS --name "$CONTAINER_NAME"   -p "$PORT":"$PORT"   -v alphafactory_data:/var/alphafactory   -e OPENAI_API_KEY="${OPENAI_API_KEY:-}"   -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"   "$IMAGE"

echo "Alpha Factory v2 running at http://localhost:${PORT}"
