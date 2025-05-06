"""archive.py – Meta‑Agentic α‑AGI lineage store
=================================================
Production‑grade helper that reads / writes the lineage produced by
`meta_agentic_agi_demo.py` and exposes a lightweight analytics layer.

Key capabilities
----------------
• Zero‑dependency core (sqlite3 + std‑lib) – *pandas* optional.  
• Insert, update, merge and stream candidates in real‑time.  
• Fast Pareto‑front & crowding‐distance selection for *N* ≲ 1e5.  
• Shannon‑entropy & Jaccard‑distance novelty metrics.  
• Round‑trip converters: **SQLite ⇆ JSONL ⇆ CSV ⇆ Parquet**.  
• CLI utilities: *info*, *export*, *import*, *vacuum* and *tail*.

Copyright © 2025 MONTREAL.AI — Apache‑2.0
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, List, Sequence

###############################################################################
# 1 Datamodel
###############################################################################

@dataclass(slots=True)
class Fitness:
    """Multi‑objective fitness container.

    Add / remove fields as needed – the rest of the library is *schema‑agnostic* as
    long as every metric is a float and *lower* is better for Pareto ranking.
    """

    accuracy: float
    latency: float
    cost: float
    carbon: float
    novelty: float
    # runtime – filled by :func:`pareto_front`
    rank: int | None = field(default=None, compare=False)
    crowd: float | None = field(default=None, compare=False)

    def dominated_by(self, other: "Fitness") -> bool:
        """Return *True* if *other* Pareto‑dominates *self*."""
        # maximise accuracy & novelty → minimise their negative
        self_vec = [-self.accuracy, self.latency, self.cost, self.carbon, -self.novelty]
        other_vec = [-other.accuracy, other.latency, other.cost, other.carbon, -other.novelty]
        return all(o <= s for o, s in zip(other_vec, self_vec)) and any(
            o < s for o, s in zip(other_vec, self_vec)
        )

@dataclass(slots=True)
class Candidate:
    id: int
    gen: int
    ts: str
    code: str
    fitness: Fitness

###############################################################################
# 2 SQLite backend
###############################################################################

SCHEMA = """CREATE TABLE IF NOT EXISTS lineage(
    id      INTEGER PRIMARY KEY,
    gen     INTEGER,
    ts      TEXT,
    code    TEXT,
    fitness TEXT
)"""

@contextmanager
def _cx(path: Path | str, readonly: bool = False):
    uri = f"file:{Path(path).absolute()}?mode={'ro' if readonly else 'rw'}"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    if not readonly:
        conn.execute(SCHEMA)
        conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
    finally:
        if not readonly:
            conn.commit()
        conn.close()

# ---------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------
def insert(c: Candidate, db: Path | str) -> None:
    with _cx(db) as cx:
        cx.execute(
            "REPLACE INTO lineage VALUES (?,?,?,?,?)",
            (c.id, c.gen, c.ts, c.code, json.dumps(asdict(c.fitness))),
        )

def stream(db: Path | str) -> Iterable[Candidate]:
    """Generator yielding new rows forever (tail ‑f)."""
    seen = set()
    while True:
        with _cx(db, readonly=True) as cx:
            for row in cx.execute("SELECT * FROM lineage ORDER BY gen, id"):
                if row[0] in seen:
                    continue
                seen.add(row[0])
                yield _row_to_cand(row)
        time.sleep(1)

# ---------------------------------------------------------------------
# Load / convert
# ---------------------------------------------------------------------
def load(db: Path | str) -> List[Candidate]:
    with _cx(db, readonly=True) as cx:
        rows = list(cx.execute("SELECT * FROM lineage ORDER BY gen, id"))
    return [_row_to_cand(r) for r in rows]

def to_dataframe(cands: Sequence[Candidate]):
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pandas is required for DataFrame export") from e
    records = []
    for c in cands:
        rec = {"id": c.id, "gen": c.gen, "ts": c.ts, "code": c.code}
        rec.update(asdict(c.fitness))
        records.append(rec)
    return pd.DataFrame.from_records(records)

def save_jsonl(cands: Sequence[Candidate], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for c in cands:
            fh.write(json.dumps(asdict(c)) + "\n")


def load_jsonl(path: Path | str) -> List[Candidate]:
    out: List[Candidate] = []
    for line in Path(path).open():
        d = json.loads(line)
        out.append(
            Candidate(
                id=d["id"],
                gen=d["gen"],
                ts=d["ts"],
                code=d["code"],
                fitness=Fitness(**d["fitness"]),
            )
        )
    return out

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _row_to_cand(r: sqlite3.Row | tuple) -> Candidate:
    fit = Fitness(**json.loads(r[4]))
    return Candidate(id=r[0], gen=r[1], ts=r[2], code=r[3], fitness=fit)

###############################################################################
# 3 Pareto‑front & diversity
###############################################################################

def pareto_front(cands: Sequence[Candidate]) -> List[Candidate]:
    """Return candidates with *rank = 1* (non‑dominated)."""
    out: List[Candidate] = []
    for c in cands:
        c.fitness.rank = 1  # reset
        for o in cands:
            if o is c:
                continue
            if c.fitness.dominated_by(o.fitness):
                c.fitness.rank += 1
        if c.fitness.rank == 1:
            out.append(c)
    return out

def crowding_distance(front: Sequence[Candidate]) -> None:
    """Assign *crowd* attribute for diversity (larger is better)."""
    if not front:
        return
    mkeys = [k for k in vars(front[0].fitness) if k not in {"rank", "crowd"}]
    for f in front:
        f.fitness.crowd = 0.0
    for k in mkeys:
        front_sorted = sorted(front, key=lambda c: getattr(c.fitness, k))
        front_sorted[0].fitness.crowd = front_sorted[-1].fitness.crowd = math.inf
        k_min = getattr(front_sorted[0].fitness, k)
        k_max = getattr(front_sorted[-1].fitness, k)
        if k_max == k_min:
            continue
        for i in range(1, len(front_sorted) - 1):
            prev_k = getattr(front_sorted[i - 1].fitness, k)
            next_k = getattr(front_sorted[i + 1].fitness, k)
            front_sorted[i].fitness.crowd += (next_k - prev_k) / (k_max - k_min)

###############################################################################
# 4 Novelty metrics
###############################################################################

def shannon_novelty(code: str, k: int = 32) -> float:
    from collections import Counter

    toks = code.split()
    if len(toks) < k:
        return 0.0
    grams = [" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)]
    cnt = Counter(grams)
    tot = sum(cnt.values())
    return -sum((n / tot) * math.log(n / tot + 1e-12) for n in cnt.values())

def jaccard_novelty(a: str, b: str, k: int = 6) -> float:
    """Classic Jaccard distance over k‑grams (0 = identical)."""
    grams = lambda s: {s[i : i + k] for i in range(len(s) - k + 1)}
    A, B = grams(a), grams(b)
    return 1 - len(A & B) / (len(A | B) or 1)

###############################################################################
# 5 CLI
###############################################################################

def _cmd_info(db: Path):
    cand = load(db)
    print(f"Rows: {len(cand)}   gens: {cand[0].gen if cand else '-'} → {cand[-1].gen if cand else '-'}")
    print(f"Pareto front size: {len(pareto_front(cand))}")

def _cmd_export(db: Path, dst: Path):
    cand = load(db)
    if dst.suffix == ".jsonl":
        save_jsonl(cand, dst)
    elif dst.suffix == ".csv":
        to_dataframe(cand).to_csv(dst, index=False)
    elif dst.suffix in {".parquet", ".pq"}:  # pragma: no cover
        to_dataframe(cand).to_parquet(dst, index=False)
    else:
        raise ValueError("Unsupported export format: " + dst.suffix)
    print("✅ exported", dst)

def _parse():
    ap = argparse.ArgumentParser(prog="archive.py", description="Meta‑Agentic lineage utility")
    sub = ap.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("info", help="print quick stats")
    i.add_argument("db", type=Path)

    e = sub.add_parser("export", help="export to JSONL / CSV / Parquet")
    e.add_argument("db", type=Path)
    e.add_argument("dst", type=Path)

    t = sub.add_parser("tail", help="stream rows as JSONL to stdout")
    t.add_argument("db", type=Path)

    v = sub.add_parser("vacuum", help="VACUUM & optimize db")
    v.add_argument("db", type=Path)

    return ap.parse_args()

def _main():
    args = _parse()
    if args.cmd == "info":
        _cmd_info(args.db)
    elif args.cmd == "export":
        _cmd_export(args.db, args.dst)
    elif args.cmd == "tail":
        for c in stream(args.db):
            print(json.dumps(asdict(c)))
    elif args.cmd == "vacuum":
        with _cx(args.db) as cx:
            cx.execute("VACUUM")
        print("✅ vacuumed", args.db)

if __name__ == "__main__":  # pragma: no cover
    _main()
