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
HOST_PORT=${HOST_PORT:-7861}

# Warn if the requested port is already in use (best-effort check)
if command -v lsof >/dev/null 2>&1 && lsof -i TCP:"${HOST_PORT}" -s TCP:LISTEN >/dev/null 2>&1; then
  echo "🚨  Port ${HOST_PORT} already in use. Set HOST_PORT to an open port." >&2
  exit 1
fi

docker compose --project-name alpha_muzero -f "$compose" up -d --build

echo -e "\n🎉  Open http://localhost:${HOST_PORT} for the live MuZero dashboard."
echo "🛑  Stop → docker compose -p alpha_muzero down\n"
