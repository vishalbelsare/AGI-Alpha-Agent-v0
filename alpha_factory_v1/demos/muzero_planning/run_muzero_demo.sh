#!/usr/bin/env bash
set -euo pipefail

demo_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
root_dir="${demo_dir%/*/*}"                       # → alpha_factory_v1
compose="$demo_dir/docker-compose.muzero.yml"

cd "$root_dir"

command -v docker >/dev/null 2>&1 || {
  echo "🚨  Docker is required → https://docs.docker.com/get-docker/"; exit 1; }

[[ -f "$demo_dir/config.env" ]] || {
  echo "➕  Creating default config.env (edit to add OPENAI_API_KEY)"; 
  cp "$demo_dir/config.env.sample" "$demo_dir/config.env"; }

echo "🚢  Building & starting MuZero Planning demo …"
docker compose --project-name alpha_muzero -f "$compose" up -d --build

echo -e "\n🎉  Open http://localhost:7861 for the live MuZero dashboard."
echo "🛑  Stop → docker compose -p alpha_muzero down\n"
