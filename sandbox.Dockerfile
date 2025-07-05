# SPDX-License-Identifier: Apache-2.0
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends patch \
    && rm -rf /var/lib/apt/lists/*
