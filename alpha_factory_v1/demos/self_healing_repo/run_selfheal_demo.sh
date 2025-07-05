#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Note: This research prototype does not deploy real AGI.
set -euo pipefail

demo_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
root_dir="${demo_dir%/*/*}"                      # → alpha_factory_v1
repo_root="${root_dir%/*}"                       # → repository root
compose="$demo_dir/docker-compose.selfheal.yml"

cd "$root_dir"

command -v docker >/dev/null 2>&1 || {
  echo "🚨  Docker is required → https://docs.docker.com/get-docker/"; exit 1; }


docker build -t selfheal-sandbox:latest -f "$repo_root/sandbox.Dockerfile" "$repo_root"

[[ -f "$demo_dir/config.env" ]] || {
  echo "➕  Creating default config.env (edit to add OPENAI_API_KEY)"; 
  cp "$demo_dir/config.env.sample" "$demo_dir/config.env"; }

echo "🚢  Building & starting Self‑Healing Repo demo …"
docker compose --project-name alpha_selfheal -f "$compose" up -d --build

echo -e "\n🎉  Dashboard → http://localhost:7863"
echo "🛑  Stop        → docker compose -p alpha_selfheal down"
