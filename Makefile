.PHONY: build_web
build_web:
pnpm --dir src/interface/web_client install
pnpm --dir src/interface/web_client run build

.PHONY: build_web_v1
build_web_v1:
pnpm --dir alpha_factory_v1/demos/alpha_agi_insight_v1/src/interface/web_client install
pnpm --dir alpha_factory_v1/demos/alpha_agi_insight_v1/src/interface/web_client run build

