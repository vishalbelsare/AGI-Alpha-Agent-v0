This chart deploys the α-AGI Insight demo.

Example usage:
```bash
helm upgrade --install insight ./alpha_factory_v1/demos/alpha_agi_insight_v1/infrastructure/helm-chart \
  --set env.OPENAI_API_KEY=<key> \
  --set env.RUN_MODE=api
```
Values such as `service.port` or `image` can be customised via `--set` or a custom values file.
