[See docs/DISCLAIMER_SNIPPET.md](../docs/DISCLAIMER_SNIPPET.md)
This repository is a conceptual research prototype. References to "AGI" and "superintelligence" describe aspirational goals and do not indicate the presence of a real general intelligence. Use at your own risk. Nothing herein constitutes financial advice. MontrealAI and the maintainers accept no liability for losses incurred from using this software.

This chart deploys the orchestrator and optional web UI.

```bash
helm upgrade --install alpha-demo ./infrastructure/helm-chart \
  --set env.OPENAI_API_KEY=<key> \
  --set env.RUN_MODE=api \
  --set env.AGI_INSIGHT_BUS_PORT=6006
# enable persistent storage
#   --set persistence.enabled=true --set persistence.size=5Gi
```

Values such as `service.port` or `image` can be overridden with `--set` or a custom `values.yaml` file.
