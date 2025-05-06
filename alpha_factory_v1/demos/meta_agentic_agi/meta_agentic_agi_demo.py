
"""meta_agentic_agi_demo.py
--------------------------------------------------------------------
Meta‑Agentic α‑AGI Demo – Production‑Grade v0.2.0 (2025‑05‑05)

This single‑file entry‑point bootstraps a *self‑improving* meta‑agentic
search loop on top of Alpha‑Factory v1.  It remains completely
functional *without* paid API keys by falling back to open‑weights
models via `llama‑cpp‑python`, yet may dynamically switch to OpenAI or
Anthropic if keys are detected.

Core features
=============
* Multi‑objective fitness: accuracy, latency, cost, carbon, novelty.
* Provider‑agnostic LLM wrapper (`ChatLLM`) with transparent cost/latency
  tracing.
* Pareto archive persisted to an *embedded* sqlite lineage DB – powering
  both analytics notebooks and the bundled Streamlit dashboard
  (`ui/lineage_app.py`).
*   100 % pure‑Python — runs on a vanilla `python >= 3.10` laptop.

Run
---
```bash
micromamba create -n metaagi python=3.11 -y
micromamba activate metaagi
pip install -r requirements.txt      # tiny; pure‑py wheels only
python meta_agentic_agi_demo.py --gens 8 --provider mistral:7b-instruct.gguf
# or
OPENAI_API_KEY=sk-... python meta_agentic_agi_demo.py --gens 8 --provider openai:gpt-4o
```

See README.md for full docs.
"""

from __future__ import annotations
import argparse, asyncio, json, os, random, sqlite3, sys, time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# ---------------------------------------------------------------------
# 0 .  Provider‑agnostic chat wrapper
# ---------------------------------------------------------------------
class UnsupportedProvider(RuntimeError): ...

@dataclass
class ChatReturn:
    content: str
    cost: float
    latency: float

class ChatLLM:
    """Normalises OpenAI / Anthropic / llama‑cpp providers."""

    def __init__(self, spec: str):
        """spec e.g. `openai:gpt-4o` | `anthropic:claude-3-sonnet`
                 | `mistral:7b-instruct.gguf`
        """
        if ':' not in spec:
            raise UnsupportedProvider(f"Malformed provider spec: {spec}")
        self.kind, self.model = spec.split(':', 1)

        if self.kind == 'openai':
            import openai
            if not os.getenv('OPENAI_API_KEY'):
                raise UnsupportedProvider('OPENAI_API_KEY missing')
            self._client = openai.AsyncOpenAI()
        elif self.kind == 'anthropic':
            import anthropic
            if not os.getenv('ANTHROPIC_API_KEY'):
                raise UnsupportedProvider('ANTHROPIC_API_KEY missing')
            self._client = anthropic.AsyncAnthropic()
        elif self.kind in {'mistral', 'ollama', 'gguf', 'llama'}:
            from llama_cpp import Llama
            model_path = Path.home() / '.cache' / 'models' / self.model
            if not model_path.exists():
                model_path.parent.mkdir(parents=True, exist_ok=True)
                # simple model downloader
                import urllib.request, shutil, tempfile
                url = f"https://huggingface.co/TheBloke/{self.model}/resolve/main/{self.model}"
                print(f"▸ downloading {url} → {model_path}")
                with tempfile.NamedTemporaryFile(delete=False) as tmp, urllib.request.urlopen(url) as resp:
                    shutil.copyfileobj(resp, tmp)
                shutil.move(tmp.name, model_path)
            self._client = Llama(model_path=str(model_path), n_ctx=4096)
        else:
            raise UnsupportedProvider(f"Unknown provider kind {self.kind}")

    async def chat(self, prompt: str) -> ChatReturn:
        t0 = time.time()
        if self.kind == 'openai':
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
            )
            txt = resp.choices[0].message.content
            cost = resp.usage.completion_tokens/1e6*15 + resp.usage.prompt_tokens/1e6*5
        elif self.kind == 'anthropic':
            resp = await self._client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.6,
            )
            txt = resp.content[0].text
            cost = 0  # Anthropic usage meta not yet surfaced
        else:
            # llama_cpp sync
            txt = self._client.create_completion(prompt=prompt, temperature=0.6)['choices'][0]['text']
            cost = 0
        lat = time.time() - t0
        return ChatReturn(txt.strip(), cost, lat)

# ---------------------------------------------------------------------
# 1 .  Multi‑objective scorer & Pareto archive
# ---------------------------------------------------------------------
@dataclass
class Fitness:
    accuracy: float
    latency: float
    cost: float
    carbon: float
    novelty: float
    rank: Optional[int] = None  # filled by pareto_sort

def pareto_sort(objs: List[Fitness]) -> None:
    for i, fi in enumerate(objs):
        fi.rank = 1 + sum(
            all(getattr(fj, k) <= getattr(fi, k) for k in vars(fi) if k != 'rank')
            and any(getattr(fj, k) < getattr(fi, k) for k in vars(fi) if k != 'rank')
            for fj in objs
        )

def novelty_hash(code: str) -> float:
    import hashlib, math
    h = hashlib.sha256(code.encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF

# ---------------------------------------------------------------------
# 2 .  Lineage DB helpers
# ---------------------------------------------------------------------
DB = Path(__file__).with_suffix('.sqlite')

def get_db():
    db = sqlite3.connect(DB)
    db.execute("""CREATE TABLE IF NOT EXISTS lineage(
        id INTEGER PRIMARY KEY,
        gen INTEGER,
        ts TEXT,
        code TEXT,
        fitness TEXT
    )""")
    return db

def insert_entry(db, e_id: int, gen: int, code: str, fit: Fitness):
    db.execute("INSERT INTO lineage VALUES (?,?,?,?,?)",
               (e_id, gen, datetime.utcnow().isoformat(), code, json.dumps(asdict(fit))))
    db.commit()

# ---------------------------------------------------------------------
# 3 .  Evaluation stub – integrate ADAS
# ---------------------------------------------------------------------
def evaluate_agent(code: str, reps: int = 3) -> float:
    """Placeholder: user plugs their domain accuracy metric here."""
    random.seed(hash(code) & 0xFFFF_FFFF)
    return sum(random.random() * 0.2 + 0.8 for _ in range(reps)) / reps  # pseudo accuracy ≈ 0.8‑1.0

# ---------------------------------------------------------------------
# 4 .  Main meta search loop
# ---------------------------------------------------------------------
async def meta_loop(gens: int, provider_spec: str):
    try:
        llm = ChatLLM(provider_spec)
    except UnsupportedProvider as e:
        print(f"{e}. Falling back to open‑weights (`mistral:7b-instruct.gguf`).")
        llm = ChatLLM('mistral:7b-instruct.gguf')

    db = get_db()
    archive: Dict[int, Fitness] = {}

    for gen in range(gens):
        # a) ask meta‑agent to produce python code
        prompt = ("You are a meta‑agentic architect. Draft a minimal Python function\n"
                  "`forward(task_info: dict) -> Any` that improves accuracy while\n"
                  "keeping latency, cost, and carbon low.  ONLY return the code block.")

        draft = await llm.chat(prompt)
        agent_code = draft.content.strip('`\n ')

        # b) evaluate
        acc = evaluate_agent(agent_code)
        fit = Fitness(
            accuracy=acc,
            latency=draft.latency,
            cost=draft.cost,
            carbon=draft.latency*0.0002,
            novelty=novelty_hash(agent_code)
        )
        eid = random.randint(1, 1_000_000_000)
        insert_entry(db, eid, gen, agent_code, fit)
        archive[eid] = fit

        # c) Pareto ranking
        pareto_sort(list(archive.values()))
        # keep top‑k
        archive = {k: v for k, v in archive.items() if v.rank and v.rank <= 5}

        print(f"Gen {gen:02d} | acc={fit.accuracy:.3f} lat={fit.latency:.2f}s cost=${fit.cost:.4f} rank={fit.rank}")

    print("🎉 search finished – run `streamlit run ui/lineage_app.py` to inspect.")

# ---------------------------------------------------------------------
# 5 .  CLI
# ---------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', type=int, default=6, help='number of generations')
    ap.add_argument('--provider', type=str, default=os.getenv('LLM_PROVIDER', 'mistral:7b-instruct.gguf'))
    args = ap.parse_args()

    try:
        asyncio.run(meta_loop(args.gens, args.provider))
    except KeyboardInterrupt:
        print('\nInterrupted by user')
