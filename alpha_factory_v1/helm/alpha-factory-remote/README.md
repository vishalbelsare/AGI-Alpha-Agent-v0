# alpha-factory-remote Helm Chart

This chart deploys a remote worker pod for the **Alpha‑Factory** swarm. It packages a single worker container alongside optional Prometheus/Grafana monitoring via the `kube-prometheus-stack` dependency.

## Installation
```bash
helm upgrade --install af-remote ./helm/alpha-factory-remote \
  --set env.OPENAI_API_KEY=<key>
```
The worker exposes gRPC + metrics on port `8000`.

## Values
- `image.repository` – container image (default `ghcr.io/montrealai/alpha-factory`)
- `image.tag` – image tag (default `v2`)
- `env` – key/value environment variables passed to the container
- `replicaCount` – number of worker pods
- `workerService` – type and port of the Service exposing the worker
- `spiffe.enabled` – enable SPIFFE sidecar
- `grafanaService` – NodePort configuration for Grafana when Prometheus stack is enabled

## Metrics & Dashboards
When `prometheus.enabled` is true, a `ServiceMonitor` matching label `app.kubernetes.io/name=alpha-factory` is created by the dependency. Grafana is provisioned with a finance dashboard.
