<!--
Era‑of‑Experience Demo
Alpha‑Factory v1 👁️✨ — Multi‑Agent AGENTIC α‑AGI
Out‑learn · Out‑think · Out‑strategise · Out‑execute
© 2025 MONTREAL.AI   Apache‑2.0 License
-->

<h1 align="center">🌌 Era of Experience — Your lifelong‑RL playground</h1>
<p align="center">
 <em>Spin up a self‑improving multi‑agent spine in <strong>one command</strong>.<br>
 Watch it plan, act &amp; learn in real‑time — on your laptop or in the cloud.</em>
</p>

> “AI will eclipse the limits of human‑authored data only when agents <strong>act, observe, and adapt</strong> in the world.” — David Silver &amp; Richard S. Sutton 

This demo distils that manifesto into <strong>Alpha‑Factory v1</strong>. 
Within 60 seconds you will witness an agent <em>rewrite its own playbook</em> every few turns, powered by grounded rewards, long‑range memory and model‑agnostic planning — no dedicated GPU required.

---

## 🛠 Requirements

- **Docker 24+** with the Compose plugin
- At least **4 CPU cores** (or a modest GPU) for smooth local runs
- *(Optional)* `OPENAI_API_KEY` for cloud LLMs — leave blank to use the built‑in Mixtral via Ollama
- If running without `run_experience_demo.sh`, install the
  [`openai-agents`](https://openai.github.io/openai-agents-python/) SDK and
  `gradio` via `pip install openai-agents gradio`.
  Then, you can run the script directly with a command like:
  ```bash
  SAMPLE_DATA_DIR=/path/to/csvs python agent_experience_entrypoint.py

---

## 🚀 Quick‑start (macOS / Windows / Linux)

```bash
git clone https://github.com/MontrealAI/AGI-Alpha-Agent-v0.git
cd AGI-Alpha-Agent-v0/alpha_factory_v1/demos/era_of_experience
python ../../../check_env.py --auto-install  # optional env check
chmod +x run_experience_demo.sh
./run_experience_demo.sh      # ← THAT’S IT
```

Add `--live` to pull in real sensor feeds (wearables, RSS, etc.):

```bash
./run_experience_demo.sh --live
```

1. **Docker Desktop** builds a 300 MB image in ≈ 1 min. 
2. Your browser opens **http://localhost:7860** featuring 
  * live trace‑graph 🪄
  * reward dashboards 📈
  * interactive chat / tool console 💬
  * built‑in alpha detectors (yield curve & supply‑chain) 🔍

> **Offline/Private mode** — leave `OPENAI_API_KEY=` blank in <code>config.env</code>; the stack falls back to <strong>Ollama ✕ Mixtral‑8x7B</strong> and stays air‑gapped.

## Offline Setup

When running without internet access:

1. Pre-download `wearable_daily.csv` and `edu_progress.csv` from the
   <a href="https://github.com/MontrealAI/demo-assets">demo-assets</a> repository.
2. Place both files in `offline_samples/` before executing
   <code>./run_experience_demo.sh</code> so the orchestrator can read them.
3. If the environment check cannot reach PyPI, set `SKIP_ENV_CHECK=1` to skip
   that step:
   ```bash
   SKIP_ENV_CHECK=1 ./run_experience_demo.sh
   ```


### 🔧 Configure &amp; advanced usage

1. Copy the sample environment file and tweak as desired:

   ```bash
   cp config.env.sample config.env
   $EDITOR config.env      # set OPENAI_API_KEY, MODEL_NAME, PG_PASSWORD, LOG_LEVEL, LIVE_FEED, etc.
   ```
   You may override the path for built-in offline samples with
   `SAMPLE_DATA_DIR=/path/to/csvs`.
   Sample CSVs (`wearable_daily.csv`, `edu_progress.csv`) are shipped in
   `offline_samples/` so the demo also works without internet access.

2. Enable real-time collectors and metrics with the `--live` flag:

   ```bash
   ./run_experience_demo.sh --live
   ```

   (equivalent to setting `LIVE_FEED=1` in `config.env`)

   The orchestrator automatically switches to offline mode whenever
   `OPENAI_API_KEY` is left empty.

---

## 🎓 Run on Colab (zero install)

| Notebook | Runtime | Launch |
|----------|---------|--------|
| `colab_era_of_experience.ipynb` | CPU / GPU | <a href="https://colab.research.google.com/github/MontrealAI/AGI-Alpha-Agent-v0/blob/main/alpha_factory_v1/demos/era_of_experience/colab_era_of_experience.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a> |

The notebook installs a lean Python stack (&lt; 120 s), exposes Gradio via ngrok and lets you call tools directly from cells. It automatically verifies the runtime with `check_env.py` and runs the unit tests so you can confirm everything works. Example cells illustrate detecting "alpha" opportunities using the offline yield curve **and** a toy supply‑chain flow snapshot.

---

## ✨ What’s new & why it matters

| Silver &amp; Sutton’s pillar | How we realise it |
|---------------------------|--------------------|
| **Streams of experience** | Infinite generator feeding month‑long synthetic logs |
| **Sensor‑motor actions** | Tools (`web_search`, `plan_meal`, user chat) mutate state |
| **Grounded rewards**   | Plug‑ins: <code>fitness_reward</code>, <code>education_reward</code>, <code>curiosity_reward</code>, … (hot‑reloaded) |
| **Non‑human reasoning**  | Monte‑Carlo Tree Search planner + vector memory — no CoT imitation |

Result: an agent that <strong>evolves faster than you can refresh the page</strong>.

---

## 🛠 Architecture in 60 seconds

```text
┌────────────┐ experience  ┌────────────────┐
│ Generator │ ────────────▶ │ Orchestrator ⚙ │──┐
└────────────┘        └────────────────┘ │ tool‑calls
    ▲               ▲    ▼
 reward│           ┌──────────┐ ┌───────────┐
    │           │ Planner ♟ │ │ Tools  │
    └──────────────────────┴──────────┴─────────────┘
```

* **OpenAI Agents SDK** — composable tool‑calling, function schemas, memory  
* **A2A protocol** — future‑proof multi‑agent hand‑offs  
* **Model Context Protocol** — streaming context for huge traces  
* **Best‑practice guardrails** from OpenAI *Practical Guide to Building Agents*  

---

## 🗂 Repo map

| Path / file | What it does |
|-------------|--------------|
| `agent_experience_entrypoint.py` | boots orchestrator + Gradio |
| `run_experience_demo.sh` | 1‑liner prod launcher (health‑gated) |
| `docker-compose.experience.yml` | orchestrator + Ollama services |
| `reward_backends/` | 🍬 Drop‑in reward plug‑ins (auto‑discovery) |
| `simulation/` | Tiny Gym‑like env stubs (ready to extend) |
| `stub_agents.py` | Minimal agent classes for OpenAI SDK & ADK workflows |
| `colab_era_of_experience.ipynb` | Cloud twin notebook |
| `alpha_report.py` | CLI helper printing current offline alpha signals |

Run it with local CSVs:

```bash
python alpha_report.py --data-dir path/to/offline_samples
```

---

## 🔌 Extending

* **Add a reward**

```bash
cp reward_backends/template.py reward_backends/my_reward.py
$EDITOR reward_backends/my_reward.py   # implement reward()
```

* **Register a tool**

```python
from openai_agents import Tool

@Tool(name="place_trade", description="Execute an equity order on Alpaca")
async def place_trade(ticker:str, qty:int, side:str="BUY"): ...
```

This demo ships with a minimal example:

```python
@Tool("detect_yield_curve_alpha", "Assess yield curve inversion using offline data.")
async def detect_yield_curve_alpha_tool():
    return {"alpha": detect_yield_curve_alpha()}

@Tool("detect_supply_chain_alpha", "Check for potential supply-chain disruptions using offline data.")
async def detect_supply_chain_alpha_tool(threshold: float = 50.0):
    return {"alpha": detect_supply_chain_alpha(threshold)}
```

* **Run in simulation**

The `simulation` package ships with `SimpleExperienceEnv`, a tiny
Gym-like environment for experimenting with offline loops:

```python
from alpha_factory_v1.demos.era_of_experience.simulation import SimpleExperienceEnv

env = SimpleExperienceEnv()
state = env.reset()
for _ in range(3):
    state, reward, done, info = env.step("act")
    print(state, reward)
```

* **Prototype a custom agent**

  `stub_agents.py` contains minimal classes
  (`ExperienceAgent`, `FederatedExperienceAgent`) illustrating how to build
  on the OpenAI SDK and Google ADK respectively.


* **Cluster‑scale**

```bash
docker compose --profile gpu --scale orchestrator=4 up --build
```

Shared Redis memory + A2A = emergent cooperation.

---

## 🛡 Security & Compliance

* Non‑root container; no Docker‑in‑Docker. 
* Secrets isolated in `config.env`, never baked into images. 
* Opt‑in telemetry only; default is **OFF**. 
* `/__live` returns **200 OK** for K8s, Traefik, Nginx health probes. 
* <code>safety_compliance_reward.py</code> penalises violations and rewards self‑correction.

---

## 📈 Benchmarks (o3‑mini, 8×CPU vCPU)

| Metric | 1‑agent | 4‑agent swarm |
|--------|---------|---------------|
| Decisions / min | 38 | 124 |
| Avg reward | 0.43 | 0.57 |
| Latency P50 | 520 ms | 730 ms |

*(Synthetic workload; see `benchmarks/` for scripts)*

---

## ✅ Tests

Verify the demo locally with Python's builtin test runner:

```bash
python -m unittest tests.test_era_experience
```

Run `python ../../../check_env.py --auto-install` first to ensure optional
packages like `pytest` and `openai-agents` are available.

---

## 🗺 Road‑map

- [ ] Plug‑and‑play Gym‑Retrowrapper for atari‑style sims 
- [ ] Vector‑DB eviction policy learning 
- [ ] Live reward tuning UI 
- [ ] WASM build for edge devices 

---

## 📜 License

Apache 2.0. By using this repo you agree to cite **Montreal.AI Alpha‑Factory** if you build on top.

> **Alpha‑Factory** — forging intelligence that *out‑learns, out‑thinks, out‑executes*.
