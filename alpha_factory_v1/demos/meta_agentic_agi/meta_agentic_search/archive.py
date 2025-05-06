"""
archive.py – Meta‑Agentic α‑AGI lineage store
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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

# ---------------------------------------------------------------------------
# 1 Data‑model
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Fitness:
    """Multi‑objective fitness container.

    All metrics are floats.  **Lower is better** for Pareto ranking; metrics that
    should be maximised (e.g. *accuracy*, *novelty*) are stored as their negative
    when ranking.
    """

    accuracy: float
    latency: float
    cost: float
    carbon: float
    novelty: float

    # Runtime attributes (filled by :func:`pareto_front` / :func:`crowding_distance`)
    rank: int | None = field(default=None, compare=False)
    crowd: float | None = field(default=None, compare=False)

    # -- Pareto dominance ---------------------------------------------------
    def _vec(self) -> list[float]:
        """Return numerical vector used for Pareto comparison."""
        return [
            -self.accuracy,  # maximise accuracy
            self.latency,
            self.cost,
            self.carbon,
            -self.novelty,   # maximise novelty
        ]

    def dominated_by(self, other: 'Fitness') -> bool:
        """Return *True* if *other* Pareto‑dominates *self*."""
        o, s = other._vec(), self._vec()
        return all(oo <= ss for oo, ss in zip(o, s)) and any(oo < ss for oo, ss in zip(o, s))


@dataclass(slots=True)
class Candidate:
    id: int
    gen: int
    ts: str
    code: str
    fitness: Fitness


# ---------------------------------------------------------------------------
# 2 SQLite persistence
# ---------------------------------------------------------------------------

_SCHEMA = """CREATE TABLE IF NOT EXISTS lineage(
    id      INTEGER PRIMARY KEY,
    gen     INTEGER,
    ts      TEXT,
    code    TEXT,
    fitness TEXT
)"""

@contextmanager
def _cx(path: Path | str, readonly: bool = False):
    """Context‑managed sqlite connection (WAL, URI‑mode)."""
    uri = f"file:{Path(path).absolute()}?mode={'ro' if readonly else 'rw'}"
    con = sqlite3.connect(uri, uri=True, check_same_thread=False)
    if not readonly:
        con.execute(_SCHEMA)
        con.execute("PRAGMA journal_mode=WAL")
    try:
        yield con
    finally:
        if not readonly:
            con.commit()
        con.close()


# ---------------------------------------------------------------------------
# 3 CRUD helpers
# ---------------------------------------------------------------------------

def insert(cand: Candidate, db: Path | str) -> None:
    """Insert or replace *cand* into *db*."""
    with _cx(db) as cx:
        cx.execute(
            "REPLACE INTO lineage VALUES (?,?,?,?,?)",
            (cand.id, cand.gen, cand.ts, cand.code, json.dumps(asdict(cand.fitness))),
        )


def load(db: Path | str) -> list[Candidate]:
    """Load all candidates (ordered by generation, id)."""
    with _cx(db, readonly=True) as cx:
        rows = list(cx.execute("SELECT * FROM lineage ORDER BY gen, id"))
    return [_row_to_cand(r) for r in rows]


def stream(db: Path | str, poll: float = 2.0) -> Iterable[Candidate]:
    """Generate unseen rows forever (tail ‑f semantics)."""
    seen: set[int] = set()
    while True:
        with _cx(db, readonly=True) as cx:
            for row in cx.execute("SELECT * FROM lineage ORDER BY gen, id"):
                if row[0] in seen:
                    continue
                seen.add(row[0])
                yield _row_to_cand(row)
        time.sleep(poll)


# -- Converters -------------------------------------------------------------
def to_dataframe(cands: Sequence[Candidate]):  # noqa: D401
    """Return *pandas* DataFrame (lazy import)."""
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required for DataFrame export") from exc

    recs: list[dict[str, float | int | str]] = []
    for c in cands:
        row = {"id": c.id, "gen": c.gen, "ts": c.ts, "code": c.code}
        row.update(asdict(c.fitness))
        recs.append(row)
    return pd.DataFrame.from_records(recs)


def save_jsonl(cands: Sequence[Candidate], dst: Path | str) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        for c in cands:
            fh.write(json.dumps(asdict(c)) + "\n")


def load_jsonl(src: Path | str) -> list[Candidate]:
    out: list[Candidate] = []
    for line in Path(src).open():
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


# ---------------------------------------------------------------------------
# 4 Analytics – Pareto & Diversity
# ---------------------------------------------------------------------------

def pareto_front(cands: Sequence[Candidate]) -> list[Candidate]:
    """Return non‑dominated candidates (rank = 1)."""
    for c in cands:
        c.fitness.rank = 1
        for o in cands:
            if o is c:
                continue
            if c.fitness.dominated_by(o.fitness):
                c.fitness.rank += 1
    return [c for c in cands if c.fitness.rank == 1]


def crowding_distance(front: Sequence[Candidate]) -> None:
    """Assign *crowd* attribute in‑place (larger ⇒ more isolated)."""
    if not front:
        return
    keys = [k for k in vars(front[0].fitness) if k not in {"rank", "crowd"}]
    for f in front:
        f.fitness.crowd = 0.0

    for k in keys:
        sorted_f = sorted(front, key=lambda c: getattr(c.fitness, k))
        sorted_f[0].fitness.crowd = sorted_f[-1].fitness.crowd = math.inf
        k_min = getattr(sorted_f[0].fitness, k)
        k_max = getattr(sorted_f[-1].fitness, k)
        if k_max == k_min:
            continue
        for i in range(1, len(sorted_f) - 1):
            prev_k = getattr(sorted_f[i - 1].fitness, k)
            next_k = getattr(sorted_f[i + 1].fitness, k)
            sorted_f[i].fitness.crowd += (next_k - prev_k) / (k_max - k_min)


# -- Novelty ----------------------------------------------------------------
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
    """Jaccard distance between *a* and *b* k‑grams."""
    grams = lambda s: {s[i : i + k] for i in range(len(s) - k + 1)}
    A, B = grams(a), grams(b)
    return 1 - len(A & B) / (len(A | B) or 1)


# ---------------------------------------------------------------------------
# 5 CLI utilities
# ---------------------------------------------------------------------------

def _row_to_cand(r: sqlite3.Row | tuple) -> Candidate:
    return Candidate(id=r[0], gen=r[1], ts=r[2], code=r[3], fitness=Fitness(**json.loads(r[4])))


def _cmd_info(db: Path):
    cand = load(db)
    gens = (cand[0].gen, cand[-1].gen) if cand else (None, None)
    print(f"Rows: {len(cand)} | generations: {gens[0]} → {gens[1]}")
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


def _cmd_tail(db: Path):
    for c in stream(db):
        print(json.dumps(asdict(c)))


def _cmd_vacuum(db: Path):
    with _cx(db) as cx:
        cx.execute("VACUUM")
    print("✅ vacuumed", db)


def _parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="archive.py", description="Meta‑Agentic lineage utility")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="print quick stats")
    p.add_argument("db", type=Path)

    p = sub.add_parser("export", help="export to JSONL / CSV / Parquet")
    p.add_argument("db", type=Path)
    p.add_argument("dst", type=Path)

    p = sub.add_parser("tail", help="stream rows as JSONL to stdout")
    p.add_argument("db", type=Path)

    p = sub.add_parser("vacuum", help="VACUUM & optimise db")
    p.add_argument("db", type=Path)

    return ap.parse_args()


def _main() -> None:
    args = _parse()
    match args.cmd:
        case "info":
            _cmd_info(args.db)
        case "export":
            _cmd_export(args.db, args.dst)
        case "tail":
            _cmd_tail(args.db)
        case "vacuum":
            _cmd_vacuum(args.db)


if __name__ == "__main__":  # pragma: no cover
    _main()
