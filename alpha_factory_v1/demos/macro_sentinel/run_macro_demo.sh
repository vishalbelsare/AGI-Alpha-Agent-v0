#!/usr/bin/env bash
set -euo pipefail
demo_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
root_dir="${demo_dir%/*/*}"
compose="$demo_dir/docker-compose.macro.yml"
cd "$root_dir"

command -v docker >/dev/null 2>&1 || { echo "🚨 Install Docker → https://docs.docker.com/get-docker/"; exit 1; }
[[ -f "$demo_dir/config.env" ]] || { cp "$demo_dir/config.env.sample" "$demo_dir/config.env"; }

echo "🚢 Building & starting Macro-Sentinel …"
docker compose --project-name alpha_macro -f "$compose" up -d --build
echo -e "\n🎉 Dashboard → http://localhost:7864"
echo "🛑 Stop     → docker compose -p alpha_macro down"
