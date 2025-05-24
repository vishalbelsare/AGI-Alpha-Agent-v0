This chart deploys the demo orchestrator and UI.
Install with:
```bash
helm upgrade --install alpha-demo ./infrastructure/helm-chart \
  --set env.OPENAI_API_KEY=<key>
```
