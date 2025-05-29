#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}

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

# When FULL_INSTALL=1, install the fully pinned dependencies from the
# deterministic lock file. This path also supports offline installs via a
# wheelhouse. Otherwise install a minimal set of runtime packages for fast
# setup in networked environments.
if [[ "${FULL_INSTALL:-0}" == "1" ]]; then
  $PYTHON -m pip install --quiet "${wheel_opts[@]}" -r requirements.lock
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
  if [[ "${DEV_INSTALL:-0}" == "1" ]]; then
    packages+=(
      pytest-benchmark
      pytest-httpx
      hypothesis
      grpcio-tools
      grpcio
      requests
      pydantic-settings
    )
  fi
  $PYTHON -m pip install --quiet "${wheel_opts[@]}" "${packages[@]}"
fi

# Validate environment and install any remaining deps
check_env_opts=()
if [[ -n "${WHEELHOUSE:-}" ]]; then
  check_env_opts+=(--wheelhouse "$WHEELHOUSE")
fi
$PYTHON check_env.py --auto-install "${check_env_opts[@]}"

