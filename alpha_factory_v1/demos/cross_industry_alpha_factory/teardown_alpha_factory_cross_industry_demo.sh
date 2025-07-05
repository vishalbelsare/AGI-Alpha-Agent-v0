#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Remove the demo containers and persistent volumes
set -euo pipefail
COMPOSE_FILE=${COMPOSE_FILE:-alpha_factory_v1/docker-compose.yml}

docker compose -f "$COMPOSE_FILE" down -v
