
# -------- Alpha‑Factory multi‑stage build (CPU & optional CUDA) ----------
ARG BASE=python:3.11-slim
FROM ${BASE} as builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential curl

# Copy requirements first to leverage docker cache
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r /tmp/requirements.txt && \
    pip install --prefix=/install uvicorn gunicorn orjson

# ---------- Front‑end (trace‑graph UI) -----------------------------------
FROM node:20-slim as ui-build
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./ 
RUN npm ci --omit=dev
COPY ui/ ./
RUN npm run build   # produces ./dist

# ---------- Runtime image -----------------------------------------------
FROM ${BASE}

LABEL org.opencontainers.image.source="https://github.com/MontrealAI/AGI-Alpha-Agent-v0"
ARG GPU=false

# Optional CUDA support by switching base
# docker build --build-arg BASE=nvidia/cuda:12.4.0-runtime-ubuntu22.04 --build-arg GPU=true -t alphafactory:cuda .
RUN apt-get update && apt-get install -y --no-install-recommends libstdc++6 && rm -rf /var/lib/apt/lists/*

ENV PATH=/install/bin:$PATH \
    PYTHONPATH=/app

COPY --from=builder /install /usr/local
WORKDIR /app

# Source code
COPY backend/ /app/backend/
COPY --from=ui-build /ui/dist /app/static/trace-ui/

# Lightweight entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 3000 8000
CMD ["/app/entrypoint.sh"]
