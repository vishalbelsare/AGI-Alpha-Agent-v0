# SPDX-License-Identifier: Apache-2.0
# ──────────────────────────────────────────────────────────────────────────────
#  OMNI‑Factory • Smart‑City Resilience Demo
#  Production‑grade Dockerfile (file #5)
#  Builds a single container that runs both:
#     • alpha_factory_v1.demos.omni_factory_demo   (core agents + env)
#     • omni_dashboard.py                          (Dash UI on port 8050)
#  Usage:
#     docker build -t omni-factory .
#     docker run -p 8050:8050 omni-factory
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ---------- Metadata ---------------------------------------------------------
LABEL maintainer="Alpha‑Factory Engineering <ops@alpha-factory.ai>"
LABEL org.opencontainers.image.title="OMNI‑Factory Demo"
LABEL org.opencontainers.image.description="Self‑contained open‑ended smart‑city simulation with live dashboard."
LABEL org.opencontainers.image.licenses="Apache-2.0"

# ---------- OS dependencies --------------------------------------------------
RUN apt-get update -qq &&     apt-get install -y --no-install-recommends build-essential libgl1 &&     apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------- Python dependencies ---------------------------------------------
RUN pip install --no-cache-dir --upgrade pip     && pip install --no-cache-dir          dash==2.17.*          pandas==2.2.*          numpy==1.26.*          gymnasium==0.29.*          openai>=1.13,<2.0          plotly==5.*          waitress==3.*

# ---------- Application code -------------------------------------------------
WORKDIR /app
COPY . /app

# ---------- Runtime configuration -------------------------------------------
ENV PYTHONUNBUFFERED=1     DASH_SERVER_PORT=8050     OMNI_LEDGER=/app/omni_ledger.sqlite

EXPOSE 8050

# ---------- Entrypoint -------------------------------------------------------
# Start core loop in background, then launch Dash UI
CMD python -m alpha_factory_v1.demos.omni_factory_demo &     exec python omni_dashboard.py
