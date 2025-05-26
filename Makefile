.PHONY: build_web
build_web:
pnpm --dir src/interface/web_client install
pnpm --dir src/interface/web_client run build

