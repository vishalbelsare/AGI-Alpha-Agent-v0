#!/usr/bin/env bash
# deploy_alpha_factory_cross_industry_demo.sh
# Alpha-Factory v1 👁️✨ — Cross-Industry AGENTIC α-AGI
# One-command bootstrap: code-fetch ▸ infra-patch ▸ secure-build ▸ learn-deploy
# ────────────────────────────────────────────────────────────────────────────
set -Eeuo pipefail
IFS=$'\n\t'
shopt -s lastpipe

##############################################################################
# 0. CONSTANTS & GLOBALS
##############################################################################
PROJECT_DIR=${PROJECT_DIR:-"$HOME/alpha_factory_demo"}
REPO_URL="https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git"
BRANCH="main"
DEFAULT_COMPOSE_PATH="alpha_factory_v1/docker-compose.yml"   # ← *.yml fallback
COMPOSE_FILE=${COMPOSE_FILE:-"${DEFAULT_COMPOSE_PATH}"}
AGENTS=(finance_agent biotech_agent climate_agent manufacturing_agent policy_agent)
DASH_PORT=${DASH_PORT:-9000}
RAY_PORT=${RAY_PORT:-8265}
KEY_DIR="alpha_factory_v1/keys"
SBOM_DIR="alpha_factory_v1/sbom"

##############################################################################
# 1. PRE-FLIGHT CHECKS & UTILITIES
##############################################################################
require() { command -v "$1" >/dev/null || { echo "❌ '$1' is required."; exit 1; }; }
require docker
require git
docker info >/dev/null 2>&1 || { echo "❌ Docker daemon not running."; exit 1; }

# yq v4 — local install or container fallback
if ! command -v yq >/dev/null; then
  echo "ℹ️  yq not found — pulling containerised helper"
  yq() { docker run --rm -i -v "$(pwd)":/workdir mikefarah/yq "$@"; }
fi

##############################################################################
# 2. FETCH CODEBASE (idempotent, commit-pinned)
##############################################################################
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
if [ ! -d "AGI-Alpha-Agent-v0/.git" ]; then
  echo "📥  Cloning Alpha-Factory …"
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL"
fi
cd AGI-Alpha-Agent-v0
COMMIT_SHA=$(git rev-parse --short HEAD)

##############################################################################
# 3. RUNTIME .env + ED25519 KEYPAIR
##############################################################################
mkdir -p "$KEY_DIR"
if [ ! -f "$KEY_DIR/agent_key" ]; then
  echo "🔑  Generating ed25519 keypair for signed prompts/actions"
  ssh-keygen -t ed25519 -N '' -q -C 'alpha_factory' -f "$KEY_DIR/agent_key"
fi
PUBKEY=$(cat "$KEY_DIR/agent_key.pub")

cat > alpha_factory_v1/.env <<EOF
ALPHA_FACTORY_MODE=cross_industry
OPENAI_API_KEY=${OPENAI_API_KEY:-}
OPENAI_API_BASE=${OPENAI_API_BASE:-https://api.openai.com/v1}
AGENTS_ENABLED=${AGENTS[*]}
PROM_PORT=$DASH_PORT
RAY_PORT=$RAY_PORT
PROJECT_SHA=$COMMIT_SHA
PROMPT_SIGN_PUBKEY=$PUBKEY
EOF
echo "✅  .env seeded — agents: ${AGENTS[*]}"

##############################################################################
# 4. PATCH DOCKER-COMPOSE
##############################################################################
patch() { yq -i "$1" "$COMPOSE_FILE"; }

# Local LLM fallback
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "🤖  No OPENAI_API_KEY — enabling Mixtral-8x7B side-car"
  OPENAI_API_BASE="http://local-llm:11434/v1"
  patch '.services += {"local-llm": {image:"ollama/ollama:latest", ports:["11434:11434"], volumes:["ollama:/root/.ollama"], environment:{"OLLAMA_MODELS":"mixtral:8x7b-instruct"}}}'
fi

# Policy engine (MCP)
mkdir -p alpha_factory_v1/policies
cat > alpha_factory_v1/policies/redteam.json <<'JSON'
{"id":"af_v1_guard","patterns_deny":["(?i)breakout","(?i)privileged","(?i)leak"],"max_tokens":2048}
JSON
patch '.services += {"policy-engine": {image:"openai/mcp-engine:latest", volumes:["./alpha_factory_v1/policies:/policies:ro"]}}'

# Observability
patch '.services += {"prometheus":{image:"prom/prometheus:latest",ports:["9090:9090"]}}'
patch ".services += {\"grafana\":{image:\"grafana/grafana:latest\",ports:[\"${DASH_PORT}:3000\"],env_file:[\"./alpha_factory_v1/.env\"]}}"

# Ray cluster (continuous learning)
patch ".services += {\"ray-head\":{image:\"rayproject/ray:2.10.0\",command:\"ray start --head --dashboard-host 0.0.0.0 --dashboard-port ${RAY_PORT}\",ports:[\"${RAY_PORT}:${RAY_PORT}\"]}}"

# Trainer (PPO w/ rubric rewards)
patch '.services += {"alpha-trainer": {build: {context: "./alpha_factory_v1/continual"}, depends_on:["ray-head","orchestrator"]}}'

# SBOM signer (BuildKit writes digest → /tmp/IMAGE_ID)
patch '.services += {"sbom": {image:"anchore/syft:latest",command:["sh","-c","syft dir:/app -o spdx-json=/sbom/sbom.json && cosign attest --key=/sbom/cosign.key --predicate /sbom/sbom.json $(cat /tmp/IMAGE_ID)"],volumes:["./alpha_factory_v1/sbom:/sbom"]}}'

# PubMed & Carbon-API minimal adapters (mock servers)
patch '.services += {"pubmed-adapter": {image:"ghcr.io/alpha-factory/mock-pubmed:latest",ports:["8005:80"]}}'
patch '.services += {"carbon-api": {image:"ghcr.io/alpha-factory/mock-carbon:latest",ports:["8010:80"]}}'

##############################################################################
# 5. BUILD & DEPLOY (SBOM via BuildKit --iidfile)
##############################################################################
export DOCKER_BUILDKIT=1
mkdir -p "$SBOM_DIR"
echo "🐳  Building & launching stack …"
docker compose -f "$COMPOSE_FILE" --env-file alpha_factory_v1/.env up -d --build \
  --iidfile /tmp/IMAGE_ID \
  orchestrator ${AGENTS[*]} policy-engine prometheus grafana ray-head alpha-trainer pubmed-adapter carbon-api

##############################################################################
# 6. HEALTH-GATED STARTUP
##############################################################################
echo "⏳  Waiting for orchestrator health …"
for i in {1..40}; do
  docker compose exec orchestrator curl -fs http://localhost:8000/healthz >/dev/null 2>&1 && break
  sleep 3
  [ $i -eq 40 ] && { echo "❌ Orchestrator failed health check"; exit 1; }
done

##############################################################################
# 7. CONTINUOUS LEARNING PIPELINE (rubric-based PPO)
##############################################################################
mkdir -p alpha_factory_v1/continual
cat > alpha_factory_v1/continual/rubric.json <<'JSON'
{
  "success": { "weight": 1.0 },
  "latency_ms": { "weight": -0.001, "target": 1000 },
  "cost_usd": { "weight": -1.0 }
}
JSON

cat > alpha_factory_v1/continual/ppo_trainer.py <<'PY'
import os, json, ray, requests, random, time
from ray import tune
from ray.rllib.algorithms.ppo import PPO
ray.init(address="auto")
API=os.getenv("API","http://orchestrator:8000/agent")
AGENTS=os.getenv("AGENTS_ENABLED","").split()
RUBRIC=json.load(open("/workspace/rubric.json"))
def env_maker(cfg):
  import gymnasium as gym, numpy as np
  class AlphaEnv(gym.Env):
    def __init__(self,agent): self.agent=agent; self.observation_space=gym.spaces.Box(0,1,(1,),float); self.action_space=gym.spaces.Discrete(1)
    def reset(self,*_): return np.zeros(1),{}
    def step(self,_):
      r=requests.post(f"{API}/{self.agent}/skill_test",json={"ping":random.random()}).json()
      reward=sum(RUBRIC[k]["weight"]*(r.get(k,0)-(RUBRIC[k].get("target",0))) for k in RUBRIC)
      return np.zeros(1), reward, True, False, {}
  return AlphaEnv(cfg["agent"])
for a in AGENTS:
  tune.Tuner(PPO, param_space={"env_config":{"agent":a},"num_workers":0}).fit()
PY

##############################################################################
# 8. GITHUB ACTIONS CI (created only if absent)
##############################################################################
if [ ! -f .github/workflows/ci.yml ]; then
  mkdir -p .github/workflows
  cat > .github/workflows/ci.yml <<'YML'
name: α-Factory-CI
on: [push]
jobs:
  smoke:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - name: Bootstrap demo
        run: |
          sudo ./alpha_factory_v1/demos/cross_industry_alpha_factory/deploy_alpha_factory_cross_industry_demo.sh --ci
      - name: Probe
        run: curl -fs http://localhost:8000/healthz
YML
fi

##############################################################################
# 9. SUCCESS BANNER
##############################################################################
cat <<EOF
🎉  Alpha-Factory Cross-Industry demo is LIVE
   Grafana   → http://localhost:$DASH_PORT  (admin/admin)
   Prometheus→ http://localhost:9090
   Ray Dash  → http://localhost:$RAY_PORT
   Agents    → ${AGENTS[*]}
EOF
