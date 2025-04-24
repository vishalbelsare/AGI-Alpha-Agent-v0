#!/usr/bin/env bash
# deploy_alpha_factory_cross_industry_demo.sh
# Alpha-Factory v1 👁️✨ – Cross-Industry AGENTIC α-AGI
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail
IFS=$'\n\t'
shopt -s lastpipe

### ─────────────── 0. CONSTANTS & GLOBALS ───────────────
PROJECT_DIR=${PROJECT_DIR:-"$HOME/alpha_factory_demo"}
REPO_URL="https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git"
BRANCH="main"
COMPOSE_FILE="alpha_factory_v1/docker-compose.yaml"
AGENTS=(finance_agent biotech_agent climate_agent manufacturing_agent policy_agent)
DASH_PORT=9000
RAY_PORT=8265

### ─────────────── 1. PRE-FLIGHT ───────────────
command -v docker >/dev/null || { echo "❌ Docker not installed"; exit 1; }
docker info >/dev/null 2>&1 || { echo "❌ Docker daemon not running"; exit 1; }
command -v git >/dev/null || { echo "❌ git not installed"; exit 1; }

# yq (v4) – install if missing
if ! command -v yq >/dev/null; then
  echo "ℹ️  yq not found – installing locally (~/bin)"
  YQ_BIN="$HOME/bin/yq"
  mkdir -p "$(dirname "$YQ_BIN")"
  curl -sSL https://github.com/mikefarah/yq/releases/latest/download/yq_$(uname -s | tr '[:upper:]' '[:lower:]')_amd64 \
    -o "$YQ_BIN" && chmod +x "$YQ_BIN"
  export PATH="$HOME/bin:$PATH"
fi

### ─────────────── 2. FETCH CODEBASE ───────────────
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
if [ ! -d "AGI-Alpha-Agent-v0/.git" ]; then
  echo "📥  Cloning Alpha-Factory repository …"
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL"
fi
cd AGI-Alpha-Agent-v0
COMMIT_SHA=$(git rev-parse --short HEAD)

### ─────────────── 3. ENV & CONFIG PATCH ───────────────
cat > alpha_factory_v1/.env <<EOF
ALPHA_FACTORY_MODE=cross_industry
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
AGENTS_ENABLED="${AGENTS[*]}"
PROM_PORT=$DASH_PORT
RAY_PORT=$RAY_PORT
PROJECT_SHA=$COMMIT_SHA
EOF
echo "✅  Runtime .env created – agents: ${AGENTS[*]}"

### ─────────────── 4. COMPOSE MUTATIONS ───────────────
# helper: patch via yq safely
patch_yaml() { yq -i "$1" "$COMPOSE_FILE"; }

# offline LLM side-car
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ℹ️  Using offline Mixtral-8x7B (Ollama) backend"
  patch_yaml '.services += {"local-llm": {image:"ollama/ollama:latest", ports:["11434:11434"], volumes:["ollama:/root/.ollama"], environment:{"OLLAMA_MODELS":"mixtral:8x7b-instruct"}}}'
  export OPENAI_API_BASE="http://local-llm:11434/v1"
fi

# MCP policy-engine
mkdir -p alpha_factory_v1/policies
cat > alpha_factory_v1/policies/redteam.json <<'JSON'
{"id":"af_v1_guard","patterns_deny":["(?i)breakout","(?i)privileged","(?i)leak"],"max_tokens":2048}
JSON
patch_yaml '.services += {"policy-engine": {image:"openai/mcp-engine:latest", volumes:["./alpha_factory_v1/policies:/policies:ro"], command:["--policy=/policies/redteam.json"]}}'

# Prometheus + Grafana
patch_yaml '.services += {"prometheus":{image:"prom/prometheus:latest",ports:["9090:9090"]}}'
patch_yaml ".services += {\"grafana\":{image:\"grafana/grafana:latest\",ports:[\"${DASH_PORT}:3000\"],env_file:[\"./alpha_factory_v1/.env\"]}}"

# Ray cluster for learning loop
patch_yaml ".services += {\"ray-head\":{image:\"rayproject/ray:2.10.0\",command:\"ray start --head --dashboard-host 0.0.0.0 --dashboard-port ${RAY_PORT}\",ports:[\"${RAY_PORT}:${RAY_PORT}\"],volumes:[\"ray:/data\"]}}"

# Continuous evaluator / PPO fine-tune
patch_yaml '.services += {"alpha-trainer": {build: {context: "./alpha_factory_v1/continual"}, depends_on:["ray-head","orchestrator"], environment:["RAY_ADDRESS=ray-head:6379"]}}'

# SBOM + cosign hook (runs once after build)
patch_yaml '.services += {"sbom": {image:"anchore/syft:latest",command:["sh","-c","syft packages dir:/app -o spdx-json=/sbom/sbom.json && cosign attest --key=cosign.pub --predicate /sbom/sbom.json $(cat /tmp/IMAGE_ID)"],volumes:["./sbom:/sbom"]}}'

### ─────────────── 5. BUILD & DEPLOY ───────────────
export DOCKER_BUILDKIT=1
echo "🐳  Building images…"
docker compose -f "$COMPOSE_FILE" --env-file alpha_factory_v1/.env pull || true
docker compose -f "$COMPOSE_FILE" --env-file alpha_factory_v1/.env up -d --build

echo "⏳  Waiting for orchestrator readiness…"
for _ in {1..40}; do
  if docker compose exec orchestrator curl -fs http://localhost:8000/healthz >/dev/null 2>&1; then break; fi
  sleep 3
done || { echo "❌ Orchestrator failed health check"; exit 1; }

### ─────────────── 6. CONTINUOUS LEARNING PIPELINE ───────────────
mkdir -p alpha_factory_v1/continual
cat > alpha_factory_v1/continual/ppo_trainer.py <<'PY'
import os, ray, time, json, requests
from ray import tune
from ray.rllib.algorithms import ppo
ray.init(address="auto")
API=os.getenv("API","http://orchestrator:8000/agent")
AGENTS=os.getenv("AGENTS_ENABLED","").split()
def env_creator(config):
  import gymnasium as gym
  import numpy as np, random, requests, json
  class AlphaEnv(gym.Env):
    def __init__(self,agent):
      self.agent=agent; self.observation_space=gym.spaces.Box(0,1,(1,),float); self.action_space=gym.spaces.Discrete(1)
    def reset(self,seed=None,options=None): return np.zeros(1),{}
    def step(self,action):
      r=requests.post(f"{API}/{self.agent}/skill_test",json={"ping":random.random()})
      return np.zeros(1), float(r.status_code==200), True, False, {}
  return AlphaEnv(config["agent"])
for ag in AGENTS:
  tune.Tuner(ppo.PPO, param_space={"env_config":{"agent":ag},"num_workers":0,"framework":"tf2"}).fit()
PY

### ─────────────── 7. CI WORKFLOW BOILERPLATE ───────────────
mkdir -p .github/workflows
cat > .github/workflows/ci.yml <<'YML'
name: alpha-factory-ci
on: [push, pull_request]
jobs:
  smoke:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - name: Launch demo (headless)
        run: |
          ./alpha_factory_v1/demos/cross_industry_alpha_factory/deploy_alpha_factory_cross_industry_demo.sh --ci
      - name: Health probe
        run: curl -fs http://localhost:8000/healthz
YML

### ─────────────── 8. SUCCESS BANNER ───────────────
cat <<EOF
✅  Alpha-Factory Cross-Industry demo is LIVE!
🌐  Grafana  : http://localhost:$DASH_PORT  (admin/admin)
📊  Prometheus: http://localhost:9090
🧩  Ray Dash : http://localhost:$RAY_PORT
👁️   Agents   : ${AGENTS[*]}
EOF
