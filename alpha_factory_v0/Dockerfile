# syntax=docker/dockerfile:1.6
##############################################################################
#  Alpha‑Factory ▸ Dockerfile (Option A – Python + Node build)               #
#                                                                            #
#  • Stage 0 builds the Trace‑graph UI with Node/Vite.                        #
#  • Stage 1 is a slim Python runtime (CPU by default).                       #
#    Pass --build‑arg BASE_IMAGE=nvidia/cuda:12.4.0-runtime‑ubuntu22.04       #
#    to obtain a CUDA‑enabled variant.                                        #
##############################################################################

# ────────────────────────────────────────────────────────────────────────────
#  ➊  Global build‑arg MUST appear *before* any FROM that references it
# ────────────────────────────────────────────────────────────────────────────
ARG BASE_IMAGE=python:3.11-slim-bookworm   # default → CPU‑only layer

##############################################################################
#  Stage 0 – Build the Trace‑graph UI (Node / Vite)                           #
##############################################################################
FROM node:20-bookworm-slim AS ui-build

WORKDIR /ui
COPY ui/package*.json ./                        # leverage Docker cache
RUN npm ci --omit=dev
COPY ui/ .
RUN npm run build                               # → dist/ index.html + assets

##############################################################################
#  Stage 1 – Python runtime                                                   #
##############################################################################
FROM ${BASE_IMAGE} AS runtime

# ── Environment ─────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app

# ── System libs & tool‑chain (gcc for wheels, BLAS/OpenMP for faiss) ───────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc git \
        libopenblas0 \
        libomp5 \
        libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies (reproducible lock) ────────────────────────────────
WORKDIR /app
COPY backend/requirements.txt /tmp/
COPY requirements-lock.txt    /tmp/requirements-lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements-lock.txt

# ── Project source code ────────────────────────────────────────────────────
COPY backend/        /app/backend/
COPY entrypoint.sh   /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ── Pre‑built Trace‑graph UI bundle ────────────────────────────────────────
COPY --from=ui-build /ui/dist/ /app/static/trace/

# ── Lightweight runtime helpers ────────────────────────────────────────────
RUN pip install --no-cache-dir \
        gunicorn \
        rocketry \
        prometheus-client

EXPOSE 8000 3000
CMD ["/app/entrypoint.sh"]
