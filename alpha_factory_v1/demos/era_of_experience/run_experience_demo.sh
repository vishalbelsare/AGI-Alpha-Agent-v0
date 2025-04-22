#!/usr/bin/env bash
# Era‑of‑Experience one‑liner — non‑technical friendly
set -euo pipefail

demo_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
root_dir="${demo_dir%/*/*}"                     # → …/alpha_factory_v1
compose_file="$demo_dir/docker-compose.experience.yml"

cd "$root_dir"                                  # required for build context

# ── Prerequisites ────────────────────────────────────────────────────────────
command -v docker >/dev/null 2>&1 || {
  echo "🚨  Docker is not installed. Get it from https://docs.docker.com/get-docker/";
  exit 1; }

# ── Environment ──────────────────────────────────────────────────────────────
if [[ ! -f "$demo_dir/config.env" ]]; then
  echo "➕  First‑time run — creating config.env (edit to add OPENAI_API_KEY)"
  cp "$demo_dir/config.env.sample" "$demo_dir/config.env"
fi

# ── Launch  ──────────────────────────────────────────────────────────────────
echo "🚢  Building & starting Era‑of‑Experience demo …"
docker compose --project-name alpha_experience \
               -f "$compose_file" up -d --build

echo -e "\n🎉  Ready!  Open http://localhost:7860 in your browser."
echo "🔍  Live logs            → docker compose -p alpha_experience logs -f"
echo "🛑  Stop the demo         → docker compose -p alpha_experience down"
