#!/usr/bin/env bash
set -euo pipefail

demo_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
root_dir="${demo_dir%/*/*}"          # points at alpha_factory_v1
compose_file="$demo_dir/docker-compose.experience.yml"

cd "$root_dir"

# 1. prerequisites ------------------------------------------------------------
command -v docker >/dev/null 2>&1 || {
  echo "🚨  Docker not found. Install Docker Desktop ⬇️  https://docs.docker.com/get-docker/";
  exit 1; }

# 2. environment --------------------------------------------------------------
if [[ ! -f "$demo_dir/config.env" ]]; then
  echo "➕  Copying default config.env.sample → config.env"
  cp "$demo_dir/config.env.sample"   "$demo_dir/config.env"
fi

source "$demo_dir/config.env"

# 3. spin‑up ------------------------------------------------------------------
echo "🚢  Building & starting Alpha‑Factory Era‑of‑Experience demo..."
docker compose --project-name alpha_experience -f "$compose_file" up -d --build

echo -e "\n🎉  Demo is live:"
echo "   • Gradio UI → http://localhost:7860"
echo "   • Docs      → $demo_dir/README.md"
echo "   • Logs      → docker compose -p alpha_experience logs -f\n"
