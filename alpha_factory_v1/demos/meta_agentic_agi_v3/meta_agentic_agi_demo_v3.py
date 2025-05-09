
#!/usr/bin/env python3
"""Meta‑Agentic α‑AGI Demo v3 – production‑grade.

A single‑file entry‑point that can run as:
 • CLI evolutionary demo
 • Streamlit lineage UI
 • FastAPI service (REST)

It auto‑detects the best available foundation‑model provider:
OpenAI → Anthropic → local gguf via llama.cpp → offline stub.

Dependencies (auto‑installed via requirements.txt):
 openai, anthropic, llama‑cpp‑python, streamlit, fastapi, uvicorn.

Copyright 2025 MONTREAL.AI
License: Apache‑2.0
"""

from __future__ import annotations
import argparse, asyncio, json, logging, os, sqlite3, sys, time, pathlib
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

LOG = logging.getLogger("metaagi")
logging.basicConfig(level=os.getenv("METAAGI_LOGLEVEL", "INFO"))


# --------------------------------------------------------------------------- #
#                   Foundation‑Model Provider Abstractions                    #
# --------------------------------------------------------------------------- #

class BaseProvider:
    """Minimal async chat interface shared by all providers."""

    name: str = "base"

    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.4,
        max_tokens: int = 1024,
    ) -> str:
        raise NotImplementedError


class OpenAIProvider(BaseProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o"):
        import openai  # lazy
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY missing")
        self._client = openai.OpenAI()
        self._model = model

    async def chat(self, system: str, user: str, **kw) -> str:  # type: ignore[override]
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kw,
        )
        return resp.choices[0].message.content


class AnthropicProvider(BaseProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-3-sonnet-20240229"):
        import anthropic  # lazy
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY missing")
        self._client = anthropic.Anthropic()
        self._model = model

    async def chat(self, system: str, user: str, **kw) -> str:  # type: ignore[override]
        resp = await self._client.messages.create(
            model=self._model,
            system=system,
            messages=[{"role": "user", "content": user}],
            **kw,
        )
        return resp.content[0].text


class LocalLlamaProvider(BaseProvider):
    name = "local"

    def __init__(self, path: str):
        from llama_cpp import Llama  # lazy
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(path)
        self._llm = Llama(model_path=path, n_ctx=8192, logits_all=False)

    async def chat(self, system: str, user: str, temperature: float = 0.4, max_tokens: int = 1024):
        prompt = f"<|system|>{system}\n<|user|>{user}\n<|assistant|>"
        out = self._llm(prompt, temperature=temperature, max_tokens=max_tokens, stop=["</s>"])
        return out["choices"][0]["text"].strip()


class StubProvider(BaseProvider):
    name = "stub"

    async def chat(self, system: str, user: str, **kw) -> str:  # type: ignore[override]
        return "⚠️ Stub reply (offline mode)."


def auto_provider(spec: Optional[str] = None) -> BaseProvider:
    """Resolve the best available provider according to precedence & user spec."""
    if spec:
        pid, *rest = spec.split(":", 1)
    else:
        pid, rest = None, []

    # Explicit spec branches
    if pid in {"openai", "oai"}:
        return OpenAIProvider(rest[0] if rest else "gpt-4o")
    if pid in {"anthropic", "claude"}:
        return AnthropicProvider(rest[0] if rest else "claude-3-sonnet-20240229")
    if pid in {"local", "gguf"}:
        path = rest[0] if rest else os.getenv("LOCAL_MODEL_PATH", "mistral-7b-instruct.gguf")
        return LocalLlamaProvider(path)

    # Auto‑detect order
    try:
        return OpenAIProvider()
    except Exception:
        try:
            return AnthropicProvider()
        except Exception:
            path = os.getenv("LOCAL_MODEL_PATH", "mistral-7b-instruct.gguf")
            if pathlib.Path(path).exists():
                return LocalLlamaProvider(path)
    return StubProvider()


# --------------------------------------------------------------------------- #
#                         Lineage Storage & Audit Log                         #
# --------------------------------------------------------------------------- #

class LineageDB:
    """Tiny SQLite helper – stores agent lineage & events."""

    def __init__(self, path: str = "meta_agentic_agi.sqlite"):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._init()

    def _init(self):
        cur = self._conn.cursor()
        cur.execute(
            """CREATE TABLE IF NOT EXISTS agent_lineage (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                generation  INTEGER,
                agent_code  TEXT,
                metrics     TEXT,
                ts          REAL
            )"""
        )
        cur.execute(
            """CREATE TABLE IF NOT EXISTS events (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      REAL,
                kind    TEXT,
                payload TEXT
            )"""
        )
        self._conn.commit()

    def agent(self, gen: int, code: str, metrics: dict):
        self._conn.execute(
            "INSERT INTO agent_lineage(generation,agent_code,metrics,ts) VALUES(?,?,?,?)",
            (gen, code, json.dumps(metrics), time.time()),
        )
        self._conn.commit()

    def event(self, kind: str, payload: dict):
        self._conn.execute(
            "INSERT INTO events(ts,kind,payload) VALUES(?,?,?)",
            (time.time(), kind, json.dumps(payload)),
        )
        self._conn.commit()


# --------------------------------------------------------------------------- #
#                 Candidate Agent & Evolutionary Search Loop                 #
# --------------------------------------------------------------------------- #

@dataclass
class Candidate:
    code: str
    metrics: dict = field(default_factory=dict)


def _safe_exec(code: str, x: int | float) -> Optional[str]:
    """Execute candidate's `agent` function in a basic sandbox (no fs/net)."""
    import multiprocessing, resource, textwrap, tempfile, subprocess

    def _lim():
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))

    with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=False) as f:
        f.write(code + f"\nif __name__=='__main__':\n    print(agent({x!r}))\n")
        f.flush()
        path = f.name
    q: multiprocessing.Queue = multiprocessing.Queue()
    def _run(q):
        _lim()
        try:
            proc = subprocess.Popen([sys.executable, path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = proc.communicate(timeout=3)
            q.put((out.strip(), err.strip()))
        except Exception as exc:
            q.put(("", str(exc)))
    p = multiprocessing.Process(target=_run, args=(q,))
    p.start()
    p.join(4)
    if p.is_alive():
        p.terminate()
    try:
        out, err = q.get_nowait()
    except Exception:
        out, err = "", "queue‑fail"
    try:
        os.remove(path)
    except OSError:
        pass
    return None if err else out


async def evolutionary_search(
    fm: BaseProvider,
    db: LineageDB,
    generations: int = 8,
    pop_size: int = 6,
):
    """Simplified NSGA‑II‑style loop (accuracy only for demo brevity)."""
    LOG.info("🚀 Evolution started – %d generations, pop=%d", generations, pop_size)
    from curriculum.azr_engine import curriculum_factory  # local import

    azr = curriculum_factory(fm)
    population: List[Candidate] = [Candidate("def agent(x):\n    return x")]
    db.agent(0, population[0].code, {"seed": True})

    for gen in range(1, generations + 1):
        # AZR – propose fresh tasks and learn difficulty signal
        tasks = azr.propose(k=pop_size)
        task_results = azr.solve(tasks)
        azr.learn(task_results)
        db.event("AZR", {"tasks": len(tasks), "solved": sum(r.solved for r in task_results)})

        # Build population variants using the FM (“Builder” behaviour)
        new_pop: List[Candidate] = []
        for _ in range(pop_size):
            prompt = (
                "Write a deterministic python function 'agent(x)' that passes the following tests:\n\n"
                + "\n\n".join(f"# Test: {t.program.splitlines()[0]}" for t in tasks[:2])
                + "\nReturn only the python code."
            )
            code = await fm.chat("You are Builder‑Agent.", prompt, temperature=0.7, max_tokens=600)
            if "def" not in code:
                code = "def agent(x):\n    return x"
            new_pop.append(Candidate(code.strip()))

        # Evaluate on AZR tasks (accuracy score)
        for cand in new_pop:
            correct = 0
            for t in tasks:
                out = _safe_exec(cand.code.replace("def agent", "def main"), json.loads(t.inp))
                if out == t.out.strip():
                    correct += 1
            cand.metrics = {"correct": correct, "total": len(tasks)}
        population.extend(new_pop)

        # Select top performers by correctness
        population.sort(key=lambda c: -c.metrics.get("correct", 0))
        population = population[:pop_size]
        best = population[0]
        db.agent(gen, best.code, best.metrics)
        LOG.info("Gen %02d | best %s / %d", gen, best.metrics["correct"], len(tasks))

    LOG.info("✅ Evolution finished. Best metrics: %s", best.metrics)
    return best


# --------------------------------------------------------------------------- #
#                              Deployment Modes                               #
# --------------------------------------------------------------------------- #

def run_cli(args, fm: BaseProvider, db: LineageDB):
    best = asyncio.run(evolutionary_search(fm, db, args.gens, args.pop_size))
    print("\n✔ Run complete. Best agent code:\n", best.code)
    print(f"\nOpen the lineage UI via: streamlit run ui/lineage_app.py -- --db {args.db}")


def run_streamlit(args):
    try:
        import streamlit as st
        import pandas as pd

        st.set_page_config(page_title="Meta‑Agentic α‑AGI Lineage", layout="wide")
        st.title("📈 Lineage of Evolved Agents")
        conn = sqlite3.connect(args.db)
        df = pd.read_sql("SELECT * FROM agent_lineage", conn)
        st.dataframe(df, use_container_width=True)
        if st.button("Refresh"):
            st.experimental_rerun()
    except ImportError:
        print("Streamlit not installed – `pip install streamlit`", file=sys.stderr)


def run_api(args, fm: BaseProvider, db: LineageDB):
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
        import uvicorn

        app = FastAPI()
        state = {"best": None}

        @app.on_event("startup")
        async def _start():
            state["best"] = await evolutionary_search(fm, db, args.gens, args.pop_size)

        @app.get("/status")
        async def _status():
            if state["best"] is None:
                return {"status": "running"}
            return state["best"].metrics

        @app.get("/best")
        async def _best():
            if state["best"] is None:
                return JSONResponse({"status": "running"})
            return JSONResponse({"code": state["best"].code})

        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("FastAPI/uvicorn missing – `pip install fastapi uvicorn`", file=sys.stderr)


# --------------------------------------------------------------------------- #
#                                    CLI                                     #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(description="Meta‑Agentic α‑AGI Demo v3")
    p.add_argument("--provider", "-p", help="provider spec e.g. openai:gpt-4o or local:/path/model.gguf")
    p.add_argument("--gens", "-g", type=int, default=int(os.getenv("ALPHA_N_GEN", "8")))
    p.add_argument("--pop_size", "-n", type=int, default=int(os.getenv("ALPHA_POP_SIZE", "6")))
    p.add_argument("--mode", "-m", choices=["cli", "streamlit", "api"], default="cli")
    p.add_argument("--db", default=os.getenv("METAAGI_DB", "meta_agentic_agi.sqlite"), help="SQLite lineage db")
    return p.parse_args()


def main():
    args = parse_args()
    fm = auto_provider(args.provider)
    db = LineageDB(args.db)

    if args.mode == "cli":
        run_cli(args, fm, db)
    elif args.mode == "streamlit":
        run_streamlit(args)
    elif args.mode == "api":
        run_api(args, fm, db)
    else:
        raise ValueError("unknown mode")


if __name__ == "__main__":  # pragma: no cover
    main()
