This chart deploys the orchestrator and optional web UI.

```bash
helm upgrade --install alpha-demo ./infrastructure/helm-chart \
  --set env.OPENAI_API_KEY=<key> \
  --set env.RUN_MODE=api \
  --set env.AGI_INSIGHT_BUS_PORT=6006
```

Values such as `service.port` or `image` can be overridden with `--set` or a custom `values.yaml` file.
