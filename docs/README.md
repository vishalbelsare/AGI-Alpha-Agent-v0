# Project Documentation

## Building the React Dashboard

The React dashboard sources live under `src/interface/web_client`. Build the static assets before serving the API:

```bash
pnpm --dir src/interface/web_client install
pnpm --dir src/interface/web_client run build
```

The compiled files appear in `src/interface/web_client/dist` and are automatically served when running `uvicorn src.interface.api_server:app` with `RUN_MODE=web`.
