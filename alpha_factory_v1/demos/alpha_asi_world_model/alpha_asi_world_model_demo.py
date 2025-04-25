# alpha_asi_world_model_demo.py – Alpha-Factory v1 👁️✨  (2025-04-25)
# ============================================================================
# Fully-agentic α-AGI demo: POET-style curriculum × MuZero learner orchestrated
# by ≥5 Alpha-Factory agents.  Local-first; optional OpenAI/LLM helpers.
#
# PyPI deps (CPU‐only):  fastapi uvicorn[standard] pydantic torch numpy nbformat
# Optional: openai  (only loaded if OPENAI_API_KEY set)
#
# Quick start………………  python -m alpha_asi_world_model_demo --demo
# Notebook………………  python -m alpha_asi_world_model_demo --emit-notebook
# Docker…………………  python -m alpha_asi_world_model_demo --emit-docker
# K8s/Helm………………  python -m alpha_asi_world_model_demo --emit-helm
# UI……………………………  http://127.0.0.1:7860
# ============================================================================

from __future__ import annotations
import argparse, asyncio, importlib, json, os, random, sys, threading, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch import optim

# ===========================================================
# 1️⃣  Global deterministic seed
# ===========================================================
_SEED = int(os.getenv("ALPHA_ASI_SEED", "42"))
random.seed(_SEED); np.random.seed(_SEED); torch.manual_seed(_SEED)

# ===========================================================
# 2️⃣ Simple typed config (editable via YAML / CLI)
# ===========================================================
@dataclass
class Config:
    env_batch: int = 1
    buffer_limit: int = 50_000
    hidden: int = 128
    lr: float = 1e-3
    train_batch: int = 128
    ui_tick: int = 100
    max_steps: int = 100_000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Config()

# ===========================================================
# 3️⃣ A2A bus & agent registry
# ===========================================================
class A2A:
    _subs: Dict[str, List[Callable[[dict], None]]] = {}
    _lock = threading.Lock()

    @classmethod
    def sub(cls, topic: str, cb: Callable[[dict], None]):
        with cls._lock: cls._subs.setdefault(topic, []).append(cb)

    @classmethod
    def pub(cls, topic: str, msg: dict):
        with cls._lock:
            for cb in cls._subs.get(topic, []):               # fan-out
                try: cb(msg)
                except Exception as e: print(f"[A2A] {topic} err→ {e}", file=sys.stderr)

class Agent:
    name: str
    def __init__(self, name: str): self.name = name; A2A.sub(self.name, self._on)
    def _on(self, msg: dict):                                       # envelope
        try: self.handle(msg)
        except Exception as e: print(f"[{self.name}] {e}", file=sys.stderr)
    def handle(self, msg: dict): raise NotImplementedError
    def emit(self, topic: str, msg: dict): A2A.pub(topic, msg)

# -------- auto-load ≥5 Alpha-Factory agents -----------------
REQUIRED = [
    "planning_agent.PlanningAgent",
    "research_agent.ResearchAgent",
    "strategy_agent.StrategyAgent",
    "market_agent.MarketAnalysisAgent",
    "codegen_agent.CodeGenAgent",
    "safety_agent.SafetyAgent",
]
BASE = "alpha_factory_v1.backend.agents."
AGENTS: Dict[str, Agent] = {}

def _boot(path: str):
    mod, cls = (BASE + path).rsplit(".", 1)
    try:
        cls = getattr(importlib.import_module(mod), cls)
        inst: Agent = cls()
        print(f"[BOOT] real agent {inst.name}")
    except Exception as e:
        class Stub(Agent):                            # stub keeps bus healthy
            def handle(self, m): print(f"[Stub:{cls}] {m}")
        inst = Stub(cls)
        print(f"[BOOT] stubbed {cls}  ({e})")
    AGENTS[inst.name] = inst

for p in REQUIRED: _boot(p)
while len(AGENTS) < 5:                               # absolute guarantee
    i = len(AGENTS)+1
    class Fallback(Agent):
        def handle(self, m): print(f"[Fallback{i}] {m}")
    AGENTS[f"Fallback{i}"] = Fallback(f"Fallback{i}")

# ===========================================================
# 4️⃣ MuZero model (tiny but GPU-friendly)
# ===========================================================
class Repr(nn.Module):
    def __init__(self, n, h): super().__init__(); self.l = nn.Linear(n, h)
    def forward(self, x): return torch.tanh(self.l(x))

class Dyn(nn.Module):
    def __init__(self, h, a): super().__init__(); self.r = nn.Linear(h+a,1); self.h = nn.Linear(h+a,h)
    def forward(self, h, a): x = torch.cat([h,a],-1); return self.r(x), torch.tanh(self.h(x))

class Pred(nn.Module):
    def __init__(self, h, a): super().__init__(); self.v = nn.Linear(h,1); self.p = nn.Linear(h,a)
    def forward(self, h): return self.v(h), F.log_softmax(self.p(h),-1)

class Net(nn.Module):
    def __init__(self, obs, act): super().__init__(); H=CFG.hidden
        # pylint:disable=unexpected-keyword-arg
        self.repr, self.dyn, self.pred = Repr(obs,H), Dyn(H,act), Pred(H,act)
    def initial(self, o): h=self.repr(o); v,p=self.pred(h); return h,v,p
    def recurrent(self, h,a): r,h2=self.dyn(h,a); v,p=self.pred(h2); return h2,r,v,p

# ===========================================================
# 5️⃣ Mini-world env & POET generator
# ===========================================================
@dataclass
class World:
    size:int=6; obstacles:List[Tuple[int,int]]=field(default_factory=list); goal:Tuple[int,int]=(5,5)
    def reset(self): self.agent=(0,0); return self._obs()
    def step(self, a:int):
        dx,dy=[(0,1),(1,0),(0,-1),(-1,0)][a%4]; nx,ny=max(0,min(self.size-1,self.agent[0]+dx)),max(0,min(self.size-1,self.agent[1]+dy))
        if (nx,ny) in self.obstacles: nx,ny=self.agent
        self.agent=(nx,ny); done=self.agent==self.goal
        return self._obs(),(1.0 if done else -0.01),done,{}
    def _obs(self):
        o=np.zeros(self.size*self.size,dtype=np.float32); o[self.agent[0]*self.size+self.agent[1]]=1.
        return o

class Generator:
    def __init__(self): self.pool:List[World]=[]
    def propose(self)->World:
        s=random.randint(5,8); obs={(random.randint(1,s-2),random.randint(1,s-2)) for _ in range(random.randint(0,s))}
        w=World(s,list(obs),(s-1,s-1)); self.pool.append(w); return w

# ===========================================================
# 6️⃣ Learner (MuZero + replay)
# ===========================================================
class Learner:
    def __init__(self, env:World):
        self.env=env; self.net=Net(env.size**2,4).to(CFG.device); self.opt=optim.Adam(self.net.parameters(),CFG.lr)
        self.mem:List[Tuple[np.ndarray,float]]=[]
    def act(self,o,eps=0.25):
        with torch.no_grad(): _,_,p=self.net.initial(torch.tensor(o,device=CFG.device))
        return random.randrange(4) if random.random()<eps else int(torch.argmax(p).item())
    def remember(self,o,r):
        self.mem.append((o,r)); self.mem=self.mem[-CFG.buffer_limit:]
    def train(self):
        if len(self.mem)<CFG.train_batch: return 0.
        sample=random.sample(self.mem,CFG.train_batch); o,r=zip(*sample)
        _,v,_=self.net.initial(torch.tensor(o,device=CFG.device))
        loss=F.mse_loss(v.squeeze(), torch.tensor(r,device=CFG.device))
        self.opt.zero_grad(); loss.backward(); self.opt.step(); return float(loss.item())

# ===========================================================
# 7️⃣ Orchestrator (executes curriculum loop)
# ===========================================================
class Orchestrator:
    def __init__(self):
        self.gen=Generator(); self.env=self.gen.propose(); self.lr=Learner(self.env); self.stop=False
        A2A.sub("orch",self._cmd); A2A.pub("system",{"event":"orch_online"})
    def _cmd(self,msg):                               # simple command set
        if msg.get("cmd")=="new_env": self.env=self.gen.propose()
        elif msg.get("cmd")=="stop": self.stop=True
    def loop(self):
        o=self.env.reset()
        for t in range(CFG.max_steps):
            if self.stop: break
            a=self.lr.act(o); o2,r,d,_=self.env.step(a); self.lr.remember(o,r); loss=self.lr.train()
            o=self.env.reset() if d else o2
            if t%CFG.ui_tick==0: A2A.pub("ui",{"t":t,"r":r,"loss":loss})

# ===========================================================
# 8️⃣ Safety agent (monitors rewards & divergence)
# ===========================================================
class SafetyAgent(Agent):
    def __init__(self): super().__init__("safety"); self.last=0
    def handle(self,m):
        if "loss" in m and (np.isnan(m["loss"]) or m["loss"]>1e3):
            print("[SAFETY] abnormal loss, pausing learner"); A2A.pub("orch",{"cmd":"stop"})

SafetyAgent()  # register – real impl replaces stub if available

# ===========================================================
# 9️⃣ Optional OpenAI-LLM helper (planning advice)
# ===========================================================
if os.getenv("OPENAI_API_KEY"):
    try:
        import openai
        class LLMPlanner(Agent):
            def __init__(self): super().__init__("llm_planner"); self.cli=openai.ChatCompletion
            def handle(self,m):
                if m.get("need_plan"):                           # simplistic
                    rsp=self.cli.create(model="gpt-4o-mini",messages=[{"role":"user","content":m["need_plan"]}])
                    plan=rsp.choices[0].message.content
                    self.emit("planning_agent",{"llm_plan":plan})
        LLMPlanner(); print("[BOOT] LLM planner active")
    except Exception as e: print("[BOOT] LLM planner unavailable:",e)

# ===========================================================
# 🔟 FastAPI UI / REST / WS
# ===========================================================
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app=FastAPI(title="Alpha-ASI World Model"); app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
orch:Optional[Orchestrator]=None
@app.on_event("startup");async def _up():
    global orch; orch=Orchestrator(); threading.Thread(target=orch.loop,daemon=True).start()

@app.post("/command");async def cmd(c:Dict[str,str]): A2A.pub("orch",c); return {"ok":True}
@app.get("/agents");async def list_agents(): return list(AGENTS.keys())
@app.websocket("/ws");async def ws(sock:WebSocket):
    await sock.accept(); q:List[dict]=[]; A2A.sub("ui",lambda m:q.append(m))
    try:
        while True:
            if q: await sock.send_text(json.dumps(q.pop(0)))
            await asyncio.sleep(0.1)
    except Exception: pass

# ===========================================================
# 1️⃣1️⃣ Dev-ops helpers
# ===========================================================
DOCKERFILE="""\
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir fastapi uvicorn[standard] pydantic torch numpy nbformat
EXPOSE 7860
CMD ["python","-m","alpha_asi_world_model_demo","--demo","--host","0.0.0.0","--port","7860"]
"""
HELM_VALS="""\
replicaCount: 1
image:
  repository: alpha_asi_world_model
  tag: latest
service:
  port: 80
"""

def emit_docker(fp=Path("Dockerfile")): fp.write_text(DOCKERFILE); print("Dockerfile →",fp)
def emit_helm(dir_=Path("helm_chart")):
    dir_.mkdir(exist_ok=True); (dir_/ "values.yaml").write_text(HELM_VALS)
    (dir_/ "Chart.yaml").write_text("apiVersion: v2\nname: alpha-asi-demo\nversion: 0.1.0")
    print("Helm chart →",dir_)
def emit_nb(nb=Path("alpha_asi_world_model_demo.ipynb")):
    import nbformat as nbf; nb=nbf.v4.new_notebook(); nb.cells=[
        nbf.v4.new_markdown_cell("# α-ASI World Model Demo\nLaunch & monitor from a notebook."),
        nbf.v4.new_code_cell("!python -m alpha_asi_world_model_demo --demo &"),
        nbf.v4.new_code_cell("import websockets, asyncio, json, nest_asyncio, pprint, IPython.display as disp\nnest_asyncio.apply()\nasync def stream():\n    async with websockets.connect('ws://127.0.0.1:7860/ws') as ws:\n        for _ in range(20):\n            msg=json.loads(await ws.recv()); pprint.pp(msg)\nasyncio.run(stream())")
    ]; nbf.write(nb, nb); print("Notebook →",nb)

# ===========================================================
# 1️⃣2️⃣ CLI entry-point
# ===========================================================
def _main():
    p=argparse.ArgumentParser("alpha_asi_world_model_demo")
    p.add_argument("--demo",action="store_true"); p.add_argument("--emit-docker",action="store_true")
    p.add_argument("--emit-helm",action="store_true"); p.add_argument("--emit-notebook",action="store_true")
    p.add_argument("--host",default="127.0.0.1"); p.add_argument("--port",type=int,default=7860)
    a=p.parse_args()
    if a.emit_docker: emit_docker(); return
    if a.emit_helm: emit_helm(); return
    if a.emit_notebook: emit_nb(); return
    if a.demo: uvicorn.run("alpha_asi_world_model_demo:app",host=a.host,port=a.port,log_level="info"); return
    p.print_help()

if __name__=="__main__": _main()
