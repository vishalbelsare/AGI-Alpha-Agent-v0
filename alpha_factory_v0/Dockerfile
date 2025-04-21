# syntax=docker/dockerfile:1.6   ← enables BuildKit features everywhere
#######################################################################
#  Global build‑arg (must precede *any* FROM that references it)      #
#######################################################################
ARG BASE_IMAGE=python:3.11-slim-bookworm   # default → CPU‑only layer

#######################################################################
# Stage 0 — Build the Trace‑graph UI (Node / Vite)                    #
#######################################################################
FROM node:20-bookworm-slim AS ui-build

WORKDIR /ui
COPY ui/package*.json ./                    # leverages Docker cache
RUN npm ci --omit=dev
COPY ui/ .
RUN npm run build                           # ⇒ dist/ index.html + assets

#######################################################################
# Stage 1 — Python runtime (CPU by default)                           #
#          Pass                                                    ▼  #
#    --build-arg BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04     #
#          to obtain the CUDA‑enabled image.                           #
#######################################################################
FROM ${BASE_IMAGE} AS runtime

# --------------------------------------------------------------------
# Environment & core build tools
# --------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc git && \
    rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------
# Extra runtime libs for vector stores (faiss / SBERT)
# --------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libopenblas0 \
        libomp5 \
        libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------
# Python deps (reproducible lock file preferred)
# --------------------------------------------------------------------
WORKDIR /app
COPY backend/requirements.txt /tmp/
COPY requirements-lock.txt    /tmp/requirements-lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements-lock.txt

# --------------------------------------------------------------------
# Project source
# --------------------------------------------------------------------
COPY backend/     /app/backend/
COPY entrypoint.sh /app/entrypoint.sh

# --------------------------------------------------------------------
# Pre‑built Trace‑graph UI bundle
# --------------------------------------------------------------------
COPY --from=ui-build /ui/dist/ /app/static/trace/

# --------------------------------------------------------------------
# Lightweight services & entrypoint
# --------------------------------------------------------------------
RUN pip install --no-cache-dir gunicorn rocketry prometheus-client && \
    chmod +x /app/entrypoint.sh

EXPOSE 8000 3000
CMD ["/app/entrypoint.sh"]
