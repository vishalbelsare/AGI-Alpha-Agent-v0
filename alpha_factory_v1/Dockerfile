# syntax=docker/dockerfile:1.6
##############################################################################
#  Alpha‑Factory v1 👁️✨ — Multi‑Agent AGENTIC α‑AGI                         #
#  Production‑grade Dockerfile (CPU/GPU toggle + Trace‑graph UI)             #
#                                                                            #
#  Key upgrades vs original:                                                 #
#  • Installs OpenAI Agents SDK (+ optional ADK & Anthropic)                 #
#  • Keeps UI build stage (Node/Vite) with aggressive caching                #
#  • Single BASE_IMAGE arg lets you switch to CUDA runtime in one flag       #
#      docker build --build-arg BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04 . #
#  • Adds system libs for faiss, open‑blas, etc.                             #
#  • Uses requirements‑lock.txt if present for reproducibility               #
##############################################################################

# ────────────────────────────────────────────────────────────────────────────
#  ➊  Global build‑arg MUST appear *before* any FROM that references it
# ────────────────────────────────────────────────────────────────────────────
ARG BASE_IMAGE=python:3.11-slim-bookworm   # default → CPU‑only layer

##############################################################################
#  Stage 0 – Build the Trace‑graph UI (Node / Vite)                           #
##############################################################################
FROM node:20-bookworm-slim AS ui-build

WORKDIR /ui
COPY ui/package*.json ./                        # leverage Docker cache for npm ci
RUN npm ci --omit=dev
COPY ui/ .
RUN npm run build                               # → dist/ index.html + assets

##############################################################################
#  Stage 1 – Python runtime                                                   #
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
        build-essential \
        git \
        rustc \
        libopenblas0 \
        libomp5 \
        libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

# ── Python dependencies (reproducible lock if available) ───────────────────
WORKDIR /app
# Copy requirements early to maximise Docker cache hits
COPY backend/requirements.txt /tmp/requirements.txt
COPY requirements-lock.txt    /tmp/requirements-lock.txt

RUN set -e; \
    if [ -f /tmp/requirements-lock.txt ]; then \
        pip install --no-cache-dir -r /tmp/requirements-lock.txt; \
    else \
        pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi && \
    # Core agent ecosystem (install here to ensure presence, even if not in req)
    pip install --no-cache-dir --upgrade \
        'openai-agents>=0.1.0' \
        'google-adk>=0.2.0' \
        'anthropic>=0.27.0' \
        ccxt \
        ortools \
        faiss-cpu \
        feedparser \
        newsapi-python \
        gitpython \
        neo4j \
        psycopg2-binary

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

# ── Ports & default CMD ────────────────────────────────────────────────────
EXPOSE 8000 3000
CMD ["/app/entrypoint.sh"]
