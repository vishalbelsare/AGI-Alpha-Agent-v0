# alpha_asi_world_model_demo.py – Alpha-Factory v1 👁️✨
# =============================================================================
# Fully-agentic α-AGI demo generating an open-ended curriculum of synthetic
# worlds and training general agents toward α-ASI.  Works standalone or inside
# Alpha-Factory.  Python 3.11+, PyTorch ≥2.2, no external services required.
# -----------------------------------------------------------------------------
# Quick start………………  python -m alpha_asi_world_model_demo --demo
# Docker…………………  python -m alpha_asi_world_model_demo --emit-docker
# K8s/Helm………………  python -m alpha_asi_world_model_demo --emit-helm
# UI……………………………  open http://localhost:7860  (after launch)
# =============================================================================
from __future__ import annotations
import argparse, asyncio, importlib, inspect, json, os, random, sys, threading, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch import optim

# -----------------------------------------------------------------------------
# ==========  A2A BUS + AGENT REGISTRY  =======================================
# -----------------------------------------------------------------------------
class A2ABus:
    _subs: Dict[str, List[Callable[[dict], None]]] = {}
    _lock = threading.Lock()

    @classmethod
    def publish(cls, topic: str, msg: dict):
        with cls._lock:
            for cb in list(cls._subs.get(topic, [])):
                try:
                    cb(msg)
                except Exception as e:
                    print(f"[A2A] handler error on {topic}: {e}")

    @classmethod
    def subscribe(cls, topic: str, cb: Callable[[dict], None]):
        with cls._lock:
            cls._subs.setdefault(topic, []).append(cb)

# -----------------------------------------------------------------------------
class BaseAgent:
    """Minimal interface every Alpha-Factory agent follows."""
    name: str
    def __init__(self, name: str):
        self.name = name
        A2ABus.subscribe(self.name, self._on_msg)
    # --------------------------------------------------------------
    def _on_msg(self, msg: dict):
        try: self.handle(msg)
        except Exception as e: print(f"[{self.name}] error: {e}")
    # --------------------------------------------------------------
    def handle(self, msg: dict): raise NotImplementedError
    def emit(self, topic: str, msg: dict): A2ABus.publish(topic, msg)

# -----------------------------------------------------------------------------
# Dynamically load ≥5 real agents if present, else fallback to stubs.
REAL_AGENT_PATHS: Sequence[str] = [
    "alpha_factory_v1.backend.agents.planning_agent.PlanningAgent",
    "alpha_factory_v1.backend.agents.research_agent.ResearchAgent",
    "alpha_factory_v1.backend.agents.strategy_agent.StrategyAgent",
    "alpha_factory_v1.backend.agents.market_agent.MarketAnalysisAgent",
    "alpha_factory_v1.backend.agents.codegen_agent.CodeGenAgent",
    "alpha_factory_v1.backend.agents.safety_agent.SafetyAgent",
    "alpha_factory_v1.backend.agents.memory_agent.MemoryAgent",
]
AGENTS: Dict[str, BaseAgent] = {}

def _register_agent(path: str):
    mod, cls_name = path.rsplit(".", 1)
    try:
        cls = getattr(importlib.import_module(mod), cls_name)
        agent: BaseAgent = cls()
        print(f"[BOOT] loaded real agent {agent.name}")
    except Exception as e:
        # --- stub --------------------------------------------------
        class Stub(BaseAgent):
            def handle(self, msg):  # pragma: no cover
                print(f"[Stub:{cls_name}] {msg}")
        agent = Stub(cls_name)
        print(f"[BOOT] stubbed {cls_name}  ({e})")
    AGENTS[agent.name] = agent

for p in REAL_AGENT_PATHS[:5]:  # guarantee ≥5
    _register_agent(p)

# Make stubs for any missing required names (ensures ≥ 5 agents active)
while len(AGENTS) < 5:
    i = len(AGENTS) + 1
    class Fallback(BaseAgent):
        def handle(self, msg): print(f"[Fallback{i}] {msg}")
    AGENTS[f"Fallback{i}"] = Fallback(f"Fallback{i}")

# -----------------------------------------------------------------------------
# ==========  MUZERO NETWORK  ==================================================
# -----------------------------------------------------------------------------
class Repr(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__(); self.l = nn.Linear(obs_dim, hidden)
    def forward(self, x): return torch.tanh(self.l(x))

class Dyn(nn.Module):
    def __init__(self, hidden: int = 128, act_dim: int = 8):
        super().__init__(); self.r = nn.Linear(hidden+act_dim, 1); self.h = nn.Linear(hidden+act_dim, hidden)
    def forward(self, h, a):
        x = torch.cat([h, a], -1)
        return self.r(x), torch.tanh(self.h(x))

class Pred(nn.Module):
    def __init__(self, hidden: int = 128, act_dim: int = 8):
        super().__init__(); self.v = nn.Linear(hidden,1); self.p = nn.Linear(hidden, act_dim)
    def forward(self, h): return self.v(h), F.log_softmax(self.p(h), -1)

class MuZero(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__(); self.repr = Repr(obs_dim); self.dyn = Dyn(act_dim=act_dim); self.pred = Pred(act_dim=act_dim)
    def initial(self, obs):
        h = self.repr(obs); v,p = self.pred(h); return h,v,p
    def recurrent(self, h, a_onehot):
        r,h2 = self.dyn(h, a_onehot); v,p = self.pred(h2); return h2,r,v,p

# -----------------------------------------------------------------------------
# ==========  MINI-WORLD + POET GENERATOR  =====================================
# -----------------------------------------------------------------------------
@dataclass
class MiniWorld:
    size: int = 6
    obstacles: List[Tuple[int,int]] = field(default_factory=list)
    goal: Tuple[int,int] = (5,5)
    agent: Tuple[int,int] = (0,0)
    def reset(self): self.agent = (0,0); return self._obs()
    def _clip(self, v): return max(0, min(self.size-1, v))
    def step(self, act: int):
        dx,dy = [(0,1),(1,0),(0,-1),(-1,0)][act%4]
        nx,ny = self._clip(self.agent[0]+dx), self._clip(self.agent[1]+dy)
        if (nx,ny) in self.obstacles: nx,ny = self.agent
        self.agent = (nx,ny); done = self.agent==self.goal
        return self._obs(), (1.0 if done else -0.01), done, {}
    def _obs(self):
        o = np.zeros(self.size*self.size, dtype=np.float32)
        o[self.agent[0]*self.size+self.agent[1]] = 1.0
        return o

class POETGenerator:
    def __init__(self): self.pool: List[MiniWorld] = []
    def propose(self) -> MiniWorld:
        size = random.randint(5,8)
        obstacles = {(random.randint(1,size-2), random.randint(1,size-2)) for _ in range(random.randint(0, size))}
        env = MiniWorld(size, list(obstacles), (size-1, size-1))
        self.pool.append(env); return env

# -----------------------------------------------------------------------------
# ==========  LEARNER  =========================================================
# -----------------------------------------------------------------------------
class Learner:
    def __init__(self, env: MiniWorld):
        self.env = env
        self.net = MuZero(obs_dim=env.size**2, act_dim=4).to("cuda" if torch.cuda.is_available() else "cpu")
        self.opt = optim.Adam(self.net.parameters(), 1e-3)
        self.buffer: List[Tuple[np.ndarray, float]] = []

    def act(self, obs, eps=0.25):
        with torch.no_grad():
            h,v,p = self.net.initial(torch.tensor(obs).float())
        return random.randint(0,3) if random.random()<eps else int(torch.argmax(p).item())

    def remember(self, obs, reward):
        self.buffer.append((obs, reward))
        if len(self.buffer) > 50_000: self.buffer.pop(0)

    def train(self, batch=128):
        if len(self.buffer) < batch: return 0.0
        sample = random.sample(self.buffer, batch)
        obs,r = zip(*sample)
        h,v,_ = self.net.initial(torch.tensor(obs).float())
        loss = F.mse_loss(v.squeeze(), torch.tensor(r).float())
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return float(loss.item())

# -----------------------------------------------------------------------------
# ==========  ORCHESTRATOR  ====================================================
# -----------------------------------------------------------------------------
class Orchestrator:
    def __init__(self):
        self.gen = POETGenerator()
        self.env = self.gen.propose()
        self.learner = Learner(self.env)
        self.stop = False
        A2ABus.subscribe("orch", self._on_cmd)
        print("[BOOT] orchestrator online")

    def _on_cmd(self, msg: dict):
        cmd = msg.get("cmd")
        if cmd == "new_env":
            self.env = self.gen.propose()
            print("[SYS] new environment spawned")
        elif cmd == "stop": self.stop = True

    # main loop ---------------------------------------------------------------
    def run(self, steps=100_000):
        obs = self.env.reset()
        for t in range(steps):
            if self.stop: break
            a = self.learner.act(obs)
            obs_,rew,done,_ = self.env.step(a)
            self.learner.remember(obs, rew)
            loss = self.learner.train()
            obs = self.env.reset() if done else obs_
            if t%100 == 0:
                A2ABus.publish("ui", {"t":t,"r":rew,"loss":loss})
        print("[SYS] orchestrator loop exit")

# -----------------------------------------------------------------------------
# ==========  FASTAPI + WS UI  =================================================
# -----------------------------------------------------------------------------
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Alpha-ASI World-Model")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
orch: Optional[Orchestrator] = None

class Cmd(BaseAgent): pass  # pydantic stub unused

@app.on_event("startup")
async def _start():
    global orch
    orch = Orchestrator()
    threading.Thread(target=orch.run, daemon=True).start()

@app.post("/command")
async def command(cmd: Dict[str,str]):
    A2ABus.publish("orch", cmd)
    return {"ok": True}

@app.get("/agents")
async def agents(): return list(AGENTS.keys())

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept(); q: List[dict] = []
    A2ABus.subscribe("ui", lambda m: q.append(m))
    try:
        while True:
            if q: await ws.send_text(json.dumps(q.pop(0)))
            await asyncio.sleep(0.1)
    except Exception: pass

# -----------------------------------------------------------------------------
# ==========  DEVOPS HELPERS  ==================================================
# -----------------------------------------------------------------------------
DOCKER_TXT = """
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir fastapi uvicorn gunicorn torch numpy pydantic
EXPOSE 7860
CMD ["python","-m","alpha_asi_world_model_demo","--demo","--host","0.0.0.0","--port","7860"]
"""

HELM_VALUES = """
replicaCount: 1
image:
  repository: alpha_asi_world_model
  tag: latest
service:
  type: ClusterIP
  port: 80
"""

def emit_docker(fp: Path = Path("Dockerfile")):
    fp.write_text(DOCKER_TXT); print("Dockerfile emitted →", fp)

def emit_helm(dir_: Path = Path("helm_chart")):
    dir_.mkdir(exist_ok=True)
    (dir_/ "values.yaml").write_text(HELM_VALUES)
    (dir_/ "Chart.yaml").write_text("apiVersion: v2\nname: alpha-asi-demo\nversion: 0.1.0")
    print("Helm chart emitted →", dir_)

# -----------------------------------------------------------------------------
# ==========  CLI  =============================================================
# -----------------------------------------------------------------------------
def _cli():
    p = argparse.ArgumentParser("alpha_asi_world_model_demo")
    p.add_argument("--demo", action="store_true", help="run FastAPI UI server")
    p.add_argument("--emit-docker", action="store_true", help="write Dockerfile")
    p.add_argument("--emit-helm", action="store_true", help="write helm chart")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    if args.emit_docker: emit_docker(); return
    if args.emit_helm: emit_helm(); return
    if args.demo:
        uvicorn.run("alpha_asi_world_model_demo:app", host=args.host, port=args.port, log_level="info")
        return
    p.print_help()

if __name__ == "__main__": _cli()
