# Alpha Factory Dashboards

This directory contains Grafana dashboards used for monitoring the Alpha Factory
stack. The JSON files can be imported directly into Grafana or automatically
loaded using the provided Docker Compose configuration.

## Usage

1. Ensure [Grafana](https://grafana.com/) is running (see `docker-compose.yml`).
2. Open Grafana in your browser and log in.
3. Navigate to **Dashboards → Import** and upload `alpha_factory_overview.json`.

The default Prometheus data source is expected to be named **Prometheus**.

