#!/usr/bin/env bash
set -euo pipefail

demo_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
root_dir="${demo_dir%/*/*}"                    # → alpha_factory_v1
compose_file="$demo_dir/docker-compose.aiga.yml"

cd "$root_dir"

command -v docker >/dev/null 2>&1 || {
  echo "🚨  Docker is required → https://docs.docker.com/get-docker/"; exit 1; }

[[ -f "$demo_dir/config.env" ]] || {
  echo "➕  Creating default config.env (edit to add OPENAI_API_KEY)"; 
  cp "$demo_dir/config.env.sample" "$demo_dir/config.env"; }

echo "🚢  Building & starting AI‑GA demo …"
docker compose --project-name alpha_aiga \
               -f "$compose_file" up -d --build

echo -e "\n🎉  Open http://localhost:7862 for the live AI‑GA dashboard."
echo "🛑  Stop → docker compose -p alpha_aiga down"
