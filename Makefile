.PHONY: build_web demo-setup demo-run compose-up
build_web:
    pnpm --dir src/interface/web_client install
    pnpm --dir src/interface/web_client run build

demo-setup:
    bash scripts/demo_setup.sh

demo-run:
    @RUN_MODE=${RUN_MODE:-cli}; \
    if [ "$$RUN_MODE" = "web" ]; then \
        .venv/bin/python -m streamlit run alpha_factory_v1/demos/alpha_agi_insight_v0/insight_dashboard.py; \
    else \
        .venv/bin/python -m alpha_factory_v1.demos.alpha_agi_insight_v0 --episodes 5; \
    fi

compose-up:
    docker compose -f docker-compose.yml up --build &
    sleep 2
    python -m webbrowser http://localhost:8080 >/dev/null 2>&1


