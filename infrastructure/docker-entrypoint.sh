#!/usr/bin/env bash
set -e
MODE=${RUN_MODE:-web}
if [ "$MODE" = "api" ]; then
  exec python -m alpha_factory_v1.demos.alpha_agi_insight_v1.src.interface.api_server
elif [ "$MODE" = "web" ]; then
  exec streamlit run /app/src/interface/web_app.py \
    --server.port 8501 --server.headless true
else
  exec python -m alpha_factory_v1.demos.alpha_agi_insight_v1.src.interface.cli "$@"
fi
