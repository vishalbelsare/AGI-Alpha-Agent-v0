#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

# Default to a wheelhouse next to the repository root when available
if [[ -z "${WHEELHOUSE:-}" ]]; then
  default_wheelhouse="$(dirname "$0")/../wheels"
  if [[ -d "$default_wheelhouse" ]]; then
    export WHEELHOUSE="$default_wheelhouse"
  fi
fi

# Support offline installation via WHEELHOUSE
wheel_opts=()
if [[ -n "${WHEELHOUSE:-}" ]]; then
  wheel_opts+=(--no-index --find-links "$WHEELHOUSE")
fi

# Upgrade pip and core build tools
$PYTHON -m pip install --quiet "${wheel_opts[@]}" --upgrade pip setuptools wheel

# Install pip-compile (pip-tools) early so hooks can verify lock files
$PYTHON -m pip install --quiet "${wheel_opts[@]}" pip-tools

# Install package in editable mode
$PYTHON -m pip install --quiet "${wheel_opts[@]}" -e .

# Verify network access when curl is missing. Abort only when the
# Python check fails and WHEELHOUSE is unset.
if ! command -v curl >/dev/null 2>&1; then
  if ! $PYTHON - <<'EOF'
import socket, sys
try:
    socket.gethostbyname("pypi.org")
except Exception:
    sys.exit(1)
EOF
  then
    if [[ -z "${WHEELHOUSE:-}" ]]; then
      echo "Unable to reach pypi.org and WHEELHOUSE is unset" >&2
      exit 1
    fi
  fi
fi

# When FULL_INSTALL=1, install the fully pinned dependencies from the
# deterministic lock file. This path also supports offline installs via a
# wheelhouse. Otherwise install a minimal set of runtime packages for fast
# setup in networked environments.
if [[ "${FULL_INSTALL:-0}" == "1" ]]; then
  $PYTHON -m pip install --quiet "${wheel_opts[@]}" -r requirements.lock
elif [[ "${MINIMAL_INSTALL:-0}" == "1" ]]; then
  packages=(
    pytest
    pytest-benchmark
    prometheus_client
    mypy
    openai
    openai-agents
    google-adk
    anthropic
    fastapi
    opentelemetry-api
    grpcio
    grpcio-tools
    httpx
    uvicorn
    cryptography
    hypothesis
    pytest-httpx
    numpy
    pandas
    playwright
    plotly
    websockets
    click
    requests
  )
  # Deduplicate packages to avoid extra install noise
  mapfile -t packages < <(printf '%s\n' "${packages[@]}" | sort -u)
  $PYTHON -m pip install --quiet "${wheel_opts[@]}" "${packages[@]}"
else
  packages=(
    accelerate
    aiohttp
    anthropic
    backoff
    better-profanity
    ccxt
    chromadb
    click
    cryptography
    ctransformers
    deap
    faiss-cpu
    fastapi
    feedparser
    flask
    gitpython
    google-adk
    grpcio
    grpcio-tools
    gunicorn
    httpx
    litellm
    llama-cpp-python
    neo4j
    networkx
    newsapi-python
    noaa-sdk
    numpy
    openai
    openai-agents
    opentelemetry-api
    opentelemetry-sdk
    orjson
    ortools
    pandas
    playwright
    plotly
    prometheus-client
    psycopg2-binary
    pydantic
    pydantic-settings
    pytest
    python-dotenv
    requests
    rich
    rocketry
    scipy
    sentence-transformers
    sentencepiece
    streamlit
    tiktoken
    transformers
    'uvicorn[standard]'
    websockets
    yfinance
  )
  packages+=(
    pytest-benchmark
    pytest-httpx
    hypothesis
    grpcio-tools
    grpcio
    requests
    pydantic-settings
  )
  # Deduplicate packages to avoid extra install noise
  mapfile -t packages < <(printf '%s\n' "${packages[@]}" | sort -u)
  $PYTHON -m pip install --quiet "${wheel_opts[@]}" "${packages[@]}"
fi

# Validate environment and install any remaining deps
check_env_opts=()
if [[ -n "${WHEELHOUSE:-}" ]]; then
  check_env_opts+=(--wheelhouse "$WHEELHOUSE")
fi
$PYTHON check_env.py --auto-install "${check_env_opts[@]}"

# Verify all dependencies are satisfied and abort on issues
$PYTHON -m pip check

