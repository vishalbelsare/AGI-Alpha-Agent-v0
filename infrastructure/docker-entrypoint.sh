#!/usr/bin/env bash
set -e
MODE=${RUN_MODE:-cli}
if [ "$MODE" = "api" ]; then
  exec python -m alpha_factory_v1.demos.alpha_agi_insight_v1.src.interface.api_server
elif [ "$MODE" = "web" ]; then
  exec python -m http.server 3000 --directory /app/src/interface/web_client/dist
else
  exec python -m alpha_factory_v1.demos.alpha_agi_insight_v1.src.interface.cli "$@"
fi
