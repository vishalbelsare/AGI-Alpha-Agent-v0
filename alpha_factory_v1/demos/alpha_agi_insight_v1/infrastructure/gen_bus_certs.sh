#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# gen_bus_certs.sh -- generate a self-signed certificate for the Insight bus
set -euo pipefail

mkdir -p certs
openssl req -x509 -newkey rsa:4096 -nodes \
  -keyout certs/bus.key \
  -out certs/bus.crt \
  -days 365 \
  -subj "/CN=localhost"

printf 'AGI_INSIGHT_BUS_CERT=%s/certs/bus.crt\n' "$(pwd)"
printf 'AGI_INSIGHT_BUS_KEY=%s/certs/bus.key\n' "$(pwd)"
printf 'AGI_INSIGHT_BUS_TOKEN=change_this_token\n'
