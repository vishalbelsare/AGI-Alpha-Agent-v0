import os, asyncio, json, pathlib, random, sqlite3, typing as t
from datetime import datetime
from alpha_factory_v1.third_party.ADAS.core.search import (
    search, evaluate_forward_fn, Info
)  # unchanged upstream
from .provider import ChatLLM, UnsupportedProvider
from .objectives import multi_objective_pareto
DB = pathlib.Path(__file__).with_suffix(".sqlite")

# ---------- Key-less fall-back ----------
PROVIDER = os.getenv("LLM_PROVIDER")  # e.g. 'openai:gpt-4o' | 'anthropic:claude-3' | 'ollama:yi'
try:
    llm = ChatLLM(PROVIDER) if PROVIDER else UnsupportedProvider()
except UnsupportedProvider as e:
    print(f"⚠️  {e}; ADAS disabled, but rest of Alpha-Factory is intact.")
    exit(0)

# ---------- Meta-Agentic α-AGI demo ----------
async def main(max_gens: int = 5):
    archive: list[dict] = []
    db = sqlite3.connect(DB); _init_db(db)
    for gen in range(max_gens):
        # 1.  Ask meta-agent (LLM) to write a new agent
        meta_prompt = _build_meta_prompt(archive)
        candidate = llm.chat(meta_prompt)
        agent_code = candidate["code"]

        # 2. Evaluate on multiple objectives
        scores = evaluate_forward_fn(agent_code)  # accuracy list
        fitness = multi_objective_pareto(
            accuracy=sum(scores)/len(scores),
            cost=llm.last_cost,
            latency=llm.last_latency
        )

        # 3. Persist & stream to UI
        entry = {
            "id": random.randint(1, 1e9),
            "gen": gen,
            "code": agent_code,
            "fitness": fitness,
            "ts": datetime.utcnow().isoformat()
        }
        archive.append(entry)
        _write_db(db, entry)

        # 4. Select archive (Pareto front)
        archive = sorted({e["id"]: e for e in archive}.values(),
                         key=lambda x: x["fitness"]["rank"])[:128]

    print("🎉 Meta-Agentic search complete; browse http://localhost:8000")
    os.system(f"python -m uvicorn alpha_factory_v1.demos.meta_agentic_agi.ui.app:app")

# ---------- helpers ----------
def _build_meta_prompt(arc):  # minimal; bespoke prompt lives in prompt/ dir
    context = json.dumps([{k: v for k, v in a.items() if k != "code"} for a in arc][-5:])
    return f"""You are a meta-agent. Invent a new Python function `forward` that improves
all objectives. Past elite agents: {context}"""

def _init_db(db):
    db.executescript("""CREATE TABLE IF NOT EXISTS lineage(
        id INT PRIMARY KEY, gen INT, fitness TEXT, ts TEXT, code TEXT);""")

def _write_db(db, e):
    db.execute("INSERT INTO lineage VALUES (?,?,?,?,?)",
               (e["id"], e["gen"], json.dumps(e["fitness"]), e["ts"], e["code"]))
    db.commit()

if __name__ == "__main__": asyncio.run(main())
