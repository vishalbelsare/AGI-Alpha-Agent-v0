#!/usr/bin/env bash
# Alpha-Factory v1 👁️✨ — Cross-Industry AGENTIC α-AGI
# One-command bootstrap: fetch ▸ patch ▸ secure-build ▸ learn ▸ attest
# ──────────────────────────────────────────────────────────────────────────────
set -Eeuo pipefail
IFS=$'\n\t'
shopt -s lastpipe

##############################################################################
# 0. CONSTANTS
##############################################################################
PROJECT_DIR=${PROJECT_DIR:-"$HOME/alpha_factory_demo"}
REPO_URL="https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git"
BRANCH="main"
DEFAULT_COMPOSE="alpha_factory_v1/docker-compose.yml"
COMPOSE_FILE=${COMPOSE_FILE:-"$DEFAULT_COMPOSE"}   # ← *.yml fallback
AGENTS=(finance_agent biotech_agent climate_agent manufacturing_agent policy_agent)
DASH_PORT=${DASH_PORT:-9000}   # Grafana
RAY_PORT=${RAY_PORT:-8265}     # Ray dashboard
KEY_DIR="alpha_factory_v1/keys"
POLICY_DIR="alpha_factory_v1/policies"
SBOM_DIR="alpha_factory_v1/sbom"
ASSETS_DIR="alpha_factory_v1/demos/cross_industry_alpha_factory/assets"
CONTINUAL_DIR="alpha_factory_v1/continual"
CI_PATH=".github/workflows/ci.yml"

##############################################################################
# 1. PRE-FLIGHT CHECKS
##############################################################################
need() { command -v "$1" >/dev/null || { echo "❌ $1 required"; exit 1; }; }
need docker; need git
docker info >/dev/null 2>&1 || { echo "❌ Docker daemon not running"; exit 1; }
if ! command -v yq >/dev/null; then
  echo "ℹ️  yq not found — using containerised yq"
  yq() { docker run --rm -i -v "$PWD":/workdir mikefarah/yq "$@"; }
fi
if ! command -v cosign >/dev/null; then
  echo "ℹ️  cosign not found — installing (no sudo needed)"
  curl -sSfL https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64 \
    -o "$HOME/bin/cosign" && chmod +x "$HOME/bin/cosign" && export PATH="$HOME/bin:$PATH"
fi

##############################################################################
# 2. FETCH / UPDATE REPOSITORY
##############################################################################
mkdir -p "$PROJECT_DIR"; cd "$PROJECT_DIR"
if [ ! -d "AGI-Alpha-Agent-v0/.git" ]; then
  echo "📥  Cloning Alpha-Factory…"; git clone --depth 1 -b "$BRANCH" "$REPO_URL"
fi
cd AGI-Alpha-Agent-v0; git pull --ff-only
COMMIT_SHA=$(git rev-parse --short HEAD)

##############################################################################
# 3. RUNTIME FILES (.env, keys, policies, assets)
##############################################################################
mkdir -p "$KEY_DIR" "$POLICY_DIR" "$SBOM_DIR" "$ASSETS_DIR" "$CONTINUAL_DIR"
# 3a. Prompt-signature keypair
if [ ! -f "$KEY_DIR/agent_key" ]; then
  echo "🔑  Generating ed25519 keypair"; ssh-keygen -t ed25519 -N '' -q -f "$KEY_DIR/agent_key"
fi
PUBKEY=$(cat "$KEY_DIR/agent_key.pub")

# 3b. cosign keypair for SBOM attestation
if [ ! -f "$SBOM_DIR/cosign.key" ]; then
  echo "🔐  Generating cosign keypair"; cosign generate-key-pair --key "file://$SBOM_DIR/cosign"
fi
COSIGN_PUB=$(cat "$SBOM_DIR/cosign.pub")

# 3c. .env
cat > alpha_factory_v1/.env <<EOF
ALPHA_FACTORY_MODE=cross_industry
OPENAI_API_KEY=${OPENAI_API_KEY:-}
OPENAI_API_BASE=${OPENAI_API_BASE:-https://api.openai.com/v1}
AGENTS_ENABLED=${AGENTS[*]}
PROM_PORT=$DASH_PORT
RAY_PORT=$RAY_PORT
PROJECT_SHA=$COMMIT_SHA
PROMPT_SIGN_PUBKEY=$PUBKEY
COSIGN_PUBKEY=$COSIGN_PUB
EOF

# 3d. Guard-rails policy
cat > "$POLICY_DIR/redteam.json" <<'JSON'
{"id":"af_v1_guard","patterns_deny":["(?i)breakout","(?i)privileged","(?i)leak"],"max_tokens":2048}
JSON

# 3e. Assets placeholder (diagram, dashboards, etc.)
touch "$ASSETS_DIR/.keep"

##############################################################################
# 4. PATCH DOCKER-COMPOSE
##############################################################################
patch() { yq -i "$1" "$COMPOSE_FILE"; }

# Local LLM side-car when no API key
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "🤖  Enabling Mixtral-8x7B side-car"; OPENAI_API_BASE="http://local-llm:11434/v1"
  patch '.services += {"local-llm":{image:"ollama/ollama:latest",ports:["11434:11434"],volumes:["ollama:/root/.ollama"],environment:{"OLLAMA_MODELS":"mixtral:8x7b-instruct"}}}'
fi

patch '.services += {"policy-engine":{image:"openai/mcp-engine:latest",volumes:["./alpha_factory_v1/policies:/policies:ro"]}}'
patch '.services += {"prometheus":{image:"prom/prometheus:latest",ports:["9090:9090"]}}'
patch ".services += {\"grafana\":{image:\"grafana/grafana:latest\",ports:[\"${DASH_PORT}:3000\"]}}"
patch ".services += {\"ray-head\":{image:\"rayproject/ray:2.10.0\",command:\"ray start --head --dashboard-port ${RAY_PORT} --dashboard-host 0.0.0.0\",ports:[\"${RAY_PORT}:${RAY_PORT}\"]}}"
patch '.services += {"pubmed-adapter":{image:"ghcr.io/alpha-factory/mock-pubmed:latest",ports:["8005:80"]},"carbon-api":{image:"ghcr.io/alpha-factory/mock-carbon:latest",ports:["8010:80"]}}'
patch '.services += {"sbom":{image:"anchore/syft:latest",command:["sh","-c","syft dir:/app -o spdx-json=/sbom/sbom.json && cosign attest --key=/sbom/cosign.key --predicate /sbom/sbom.json $(cat /tmp/IMAGE_ID)"],volumes:["./alpha_factory_v1/sbom:/sbom"]}}'
patch '.services += {"alpha-trainer":{build:{context:"./alpha_factory_v1/continual"},depends_on:["ray-head","orchestrator"]}}'

##############################################################################
# 5. TRAINER (PPO + weight reload)
##############################################################################
cat > "$CONTINUAL_DIR/rubric.json" <<'JSON'
{"success":{"weight":1.0},"latency_ms":{"weight":-0.001,"target":1000},"cost_usd":{"weight":-1.0}}
JSON

cat > "$CONTINUAL_DIR/ppo_trainer.py" <<'PY'
import os, json, time, ray, requests, tempfile, shutil, pathlib
from ray import tune
from ray.rllib.algorithms.ppo import PPO

ray.init(address="auto", ignore_reinit_error=True)
API=os.getenv("API","http://orchestrator:8000/agent")
AGENTS=os.getenv("AGENTS_ENABLED","").split()
RUBRIC=json.load(open("/workspace/rubric.json"))
TMP="/tmp/ckpt"

def env_maker(cfg):
  import gymnasium as gym, numpy as np, random, requests, json
  class AlphaEnv(gym.Env):
    def __init__(self,agent): self.agent=agent; self.observation_space=gym.spaces.Box(0,1,(1,),float); self.action_space=gym.spaces.Discrete(1)
    def reset(self,*_): return np.zeros(1),{}
    def step(self,_):
      r=requests.post(f"{API}/{self.agent}/skill_test",json={"ping":random.random()}).json()
      rew=sum(RUBRIC[k]["weight"]*(r.get(k,0)-(RUBRIC[k].get("target",0))) for k in RUBRIC)
      return np.zeros(1), rew, True, False, {}
  return AlphaEnv(cfg["agent"])

for ag in AGENTS:
  result = tune.Tuner(PPO,param_space={"env_config":{"agent":ag},"num_workers":0}).fit()
  # grab best checkpoint
  best = result.get_best_result().checkpoint
  if best:
    ckpt_dir = pathlib.Path(best.path)
    with tempfile.TemporaryDirectory() as tmp:
      shutil.make_archive(f"{tmp}/model", 'zip', ckpt_dir)
      data=open(f"{tmp}/model.zip","rb").read()
    try:
      print("Uploading new weights to orchestrator for",ag)
      requests.post(f"{API}/{ag}/update_model",files={"file":("ckpt.zip",data)})
    except Exception as e:
      print("WARN upload",e)
PY

##############################################################################
# 6. CONTINUOUS-INTEGRATION WORKFLOW (pre-committed)
##############################################################################
mkdir -p .github/workflows
cat > "$CI_PATH" <<'YML'
name: alpha-factory-ci
on: [push, pull_request]
jobs:
  smoke:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Bootstrap demo
        run: |
          sudo bash ./alpha_factory_v1/demos/cross_industry_alpha_factory/deploy_alpha_factory_cross_industry_demo.sh --ci
      - name: Probe orchestrator
        run: curl -fs http://localhost:8000/healthz
YML

##############################################################################
# 7. BUILD & DEPLOY WITH ATTESTATION
##############################################################################
export DOCKER_BUILDKIT=1
echo "🐳  Building & launching (this may take a few minutes)…"
docker compose -f "$COMPOSE_FILE" --env-file alpha_factory_v1/.env up -d --build \
  --iidfile /tmp/IMAGE_ID \
  orchestrator ${AGENTS[*]} policy-engine prometheus grafana ray-head alpha-trainer pubmed-adapter carbon-api

echo "⏳  Health-check orchestrator…"
for i in {1..40}; do
  docker compose exec orchestrator curl -fs http://localhost:8000/healthz >/dev/null 2>&1 && break
  sleep 3; [ $i -eq 40 ] && { echo "❌ Orchestrator failed"; exit 1; }
done

##############################################################################
# 8. AUTO-COMMIT CI & CONTINUAL DIR (no-op if repo clean)
##############################################################################
if [ -z "${CI:-}" ]; then
  git add "$CI_PATH" "$CONTINUAL_DIR" "$ASSETS_DIR" || true
  git commit -m "chore: bootstrap cross-industry demo (auto-generated installer v${COMMIT_SHA})" 2>/dev/null || true
fi

##############################################################################
# 9. DONE
##############################################################################
cat <<EOF
🎉  Alpha-Factory Cross-Industry demo is LIVE
   Grafana    → http://localhost:$DASH_PORT  (admin/admin)
   Prometheus → http://localhost:9090
   Ray Dash   → http://localhost:$RAY_PORT
   Agents     → ${AGENTS[*]}
EOF
