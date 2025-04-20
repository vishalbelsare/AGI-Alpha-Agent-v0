###############################################################################
# Alpha‑Factory ‑ Pro                                                         #
#                                                                             #
# Build‑time flags (set by install_alpha_factory_pro.sh):                     #
#   ENABLE_UI=1        → copy /app/ui static front‑end (port 3000)            #
#   ENABLE_TRACE=1     → add FastAPI & Uvicorn for live /ws/trace socket      #
#   INCLUDE_TESTS=1    → keep test‑suite in the image (nice for CI)           #
###############################################################################

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# ─── build‑args with sane defaults ──────────────────────────────────────────
ARG ENABLE_UI=1
ARG ENABLE_TRACE=0
ARG INCLUDE_TESTS=0

# ─── runtime env ────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app \
    ENABLE_UI=${ENABLE_UI} \
    ENABLE_TRACE=${ENABLE_TRACE}

# ─── system packages (always python, node only if UI build needs it) ───────
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ─── python dependencies ────────────────────────────────────────────────────
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    if [ "$ENABLE_TRACE" = "1" ]; then \
        # only when we actually expose the trace‑graph WebSocket
        pip3 install --no-cache-dir "fastapi[all]" "uvicorn[standard]" ; \
    fi

# ─── backend code (always) ──────────────────────────────────────────────────
COPY backend/ /app/backend/

# ─── optional tests ────────────────────────────────────────────────────────
COPY tests/ /tmp/tests
RUN if [ "$INCLUDE_TESTS" = "1" ] && [ -d /tmp/tests ]; then \
        mv /tmp/tests /app/tests ; \
    else \
        rm -rf /tmp/tests ; \
    fi

# ─── optional UI assets ────────────────────────────────────────────────────
COPY ui/ /tmp/ui
RUN if [ "$ENABLE_UI" = "1" ] && [ -d /tmp/ui ]; then \
        mv /tmp/ui /app/ui ; \
    else \
        rm -rf /tmp/ui ; \
    fi

# ─── entrypoint ────────────────────────────────────────────────────────────
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# UI (3000) + API (8000) exposed either way – harmless if closed
EXPOSE 3000 8000

CMD ["/app/entrypoint.sh"]

