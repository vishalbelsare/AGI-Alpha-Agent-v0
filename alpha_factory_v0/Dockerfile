#######################################################################
# Stage 0 — Build the Trace‑graph UI (Node / Vite)                    #
#######################################################################
FROM node:20-bookworm-slim AS ui-build
WORKDIR /ui
COPY ui/package*.json ./                       # ⇢ faster `npm ci`
RUN npm ci --omit=dev
COPY ui/ .
RUN npm run build                              # ⇢ dist/ index.html + assets

#######################################################################
# Stage 1 — Python runtime (CPU by default)                           #
#        Use `--build-arg BASE_IMAGE=nvcr.io/nvidia/cuda:12.4.0‑...`   #
#        if you want the CUDA image instead.                          #
#######################################################################
ARG BASE_IMAGE=python:3.11-slim-bookworm
FROM ${BASE_IMAGE} AS runtime

# --------------------------------------------------------------------
# Environment & system packages
# --------------------------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc git && \
    rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------
# Python deps (use *reproducible* lock file if present)
# --------------------------------------------------------------------
WORKDIR /app
COPY backend/requirements.txt /tmp/
COPY requirements-lock.txt /tmp/requirements-lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements-lock.txt

# --------------------------------------------------------------------
# Project source
# --------------------------------------------------------------------
COPY backend/ /app/backend/
COPY entrypoint.sh /app/entrypoint.sh

# --- ship pre‑built UI ------------------------------------------------
COPY --from=ui-build /ui/dist/ /app/static/trace/

# tini for signal handling (optional but nice)
RUN pip install --no-cache-dir gunicorn rocketry prometheus-client \
 && chmod +x /app/entrypoint.sh

EXPOSE 8000 3000
CMD ["/app/entrypoint.sh"]
