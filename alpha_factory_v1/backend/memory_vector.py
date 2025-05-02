
"""
alpha_factory_v1.backend.memory_vector
======================================

**Vector Memory Fabric – Production‑grade, fault‑tolerant, self‑contained.**

• Primary storage   : PostgreSQL + pgvector (persistent, scalable)  
• Seamless fallback : FAISS‑in‑RAM (fast, ephemeral)  
• Last‑resort       : Pure‑numpy brute‑force (always available)  

Embedding backend:
    1. OpenAI ``text-embedding-3-small`` (if ``OPENAI_API_KEY`` is set)  
    2. Sentence‑Transformers ``all‑MiniLM‑L6‑v2`` (small CPU model)  

This file is **stand‑alone** – it has *no* intra‑project imports, so it can
be unit‑tested or reused in isolation.  All optional dependencies are
gracefully degraded; *nothing* here will ever raise ImportError at import
time.

---------------------------------------------------------------------------
Quick‑start
---------------------------------------------------------------------------
>>> from alpha_factory_v1.backend.memory_vector import VectorMemory
>>> mem = VectorMemory()                         # auto‑detects backend
>>> mem.add("finance", "Bought BTC at 62 000 USD")  # store
>>> mem.search("btc purchase", k=3)            # similarity lookup
[('finance', 'Bought BTC at 62000 USD', 0.9123)]

---------------------------------------------------------------------------
CLI self‑test (runs even without Postgres/FAISS/OpenAI)
---------------------------------------------------------------------------
$ python -m alpha_factory_v1.backend.memory_vector --demo

"""
from __future__ import annotations

# --------------------------------------------------------------------- #
# Standard library                                                      #
# --------------------------------------------------------------------- #
import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as _np

# --------------------------------------------------------------------- #
# Optional third‑party deps (each individually gated)                   #
# --------------------------------------------------------------------- #
try:
    import openai  # type: ignore
    _HAS_OPENAI = True and os.getenv("OPENAI_API_KEY")
except Exception:  # pragma: no cover
    _HAS_OPENAI = False

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore
    _HAS_PG = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_PG = False

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_FAISS = False

try:
    from prometheus_client import Counter, Gauge  # type: ignore
    _METRIC_ADD = Counter("af_mem_add_total", "Memories added", ["backend"])
    _METRIC_QUERY = Counter("af_mem_query_total", "Memory queries", ["backend"])
    _METRIC_SIZE = Gauge("af_mem_size", "Total memories stored", ["backend"])
except ModuleNotFoundError:  # pragma: no cover
    def _noop(*_a, **_kw):  # type: ignore
        class _D:  # pylint: disable=too-few-public-methods
            def labels(self, *_l): return self
            def inc(self, *_i): pass
            def set(self, *_v): pass
        return _D()
    _METRIC_ADD = _METRIC_QUERY = _METRIC_SIZE = _noop()

# --------------------------------------------------------------------- #
logger = logging.getLogger("alpha_factory.memory_vector")
if not logger.handlers:  # avoid duplicate in case of re‑import
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

_DIM = 384  # embedding dimension (MiniLM)

# =========================== Embedding layer ========================= #
_SBERT_MODEL: Optional[SentenceTransformer] = None

def _get_sbert() -> "SentenceTransformer":
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        if SentenceTransformer is None:
            raise RuntimeError("Sentence‑Transformers not installed – run `pip install sentence-transformers` or set OPENAI_API_KEY.")
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL

def _l2(a: _np.ndarray) -> _np.ndarray:
    return a / _np.linalg.norm(a, ord=2, axis=1, keepdims=True)

def _embed(texts: Sequence[str]) -> _np.ndarray:
    """Return L2‑normalised embeddings for *texts*."""
    # Prefer OpenAI API – production quality & multilingual
    if _HAS_OPENAI:
        try:
            rsp = openai.embeddings.create(  # type: ignore
                model="text-embedding-3-small",
                input=list(texts),
            )
            mat = _np.asarray([d.embedding for d in rsp.data], dtype="float32")
            return _l2(mat)
        except Exception:  # pragma: no cover
            logger.exception("OpenAI embedding failed – falling back to SBERT")
    # Local SBERT
    model = _get_sbert()
    mat = model.encode(list(texts), normalize_embeddings=True)
    return _np.asarray(mat, dtype="float32")

# =========================== Storage backends ======================== #
class _PostgresStore:
    """Persistent pgvector‑backed storage."""
    def __init__(self, dsn: str):
        self._conn = psycopg2.connect(dsn)
        self._ensure_schema()
        _METRIC_SIZE.labels("postgres").set(self.__len__())

    def _ensure_schema(self) -> None:
        with self._conn, self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS memories(
                        id        BIGSERIAL PRIMARY KEY,
                        agent     TEXT,
                        embedding VECTOR({_DIM}),
                        content   TEXT,
                        ts        TIMESTAMPTZ DEFAULT now()
                    )""")
            cur.execute("CREATE INDEX IF NOT EXISTS mem_vec_idx ON memories USING ivfflat(embedding vector_cosine_ops)")
            cur.execute("CREATE INDEX IF NOT EXISTS mem_agent_ts_idx ON memories(agent, ts DESC)")

    # CRUD ------------------------------------------------------------ #
    def add(self, agent: str, vecs: _np.ndarray, texts: List[str]) -> None:
        rows = [(agent, v.tolist(), t) for v, t in zip(vecs, texts)]
        with self._conn, self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "INSERT INTO memories(agent, embedding, content) VALUES (%s,%s,%s)",
                rows,
                page_size=100,
            )
        _METRIC_ADD.labels("postgres").inc(len(rows))
        _METRIC_SIZE.labels("postgres").set(self.__len__())

    def query(self, vec: _np.ndarray, k: int) -> List[Tuple[str, str, float]]:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT agent, content,
                           1 - (embedding <=> %s::vector) AS score
                       FROM memories
                       ORDER BY embedding <=> %s::vector
                       LIMIT %s""", (vec.tolist(), vec.tolist(), k)
            )
            return cur.fetchall()

    def __len__(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM memories")
            return int(cur.fetchone()[0])

class _FaissStore:
    """Fast in‑RAM FAISS (or numpy) store."""
    def __init__(self):
        self._index = faiss.IndexFlatIP(_DIM) if _HAS_FAISS else None
        self._texts: List[Tuple[str, str]] = []
        self._vecs: List[_np.ndarray] = []

    def add(self, agent: str, vecs: _np.ndarray, texts: List[str]) -> None:
        self._texts.extend((agent, t) for t in texts)
        if self._index is not None:
            self._index.add(vecs)
        self._vecs.append(vecs)
        _METRIC_ADD.labels("faiss" if self._index is not None else "numpy").inc(len(texts))
        _METRIC_SIZE.labels("faiss" if self._index is not None else "numpy").set(self.__len__())

    def query(self, vec: _np.ndarray, k: int) -> List[Tuple[str, str, float]]:
        if self._index is not None and self._index.ntotal:
            D, I = self._index.search(vec, min(k, self._index.ntotal))
            return [
                (*self._texts[idx], float(score))
                for idx, score in zip(I[0], D[0])
            ]
        # brute‑force dot‑product
        if not self._vecs:
            return []
        mat = _np.vstack(self._vecs)
        sims = mat @ vec.T
        order = sims[:, 0].argsort()[::-1][:k]
        return [
            (*self._texts[i], float(sims[i, 0]))
            for i in order
        ]

    def __len__(self) -> int:
        if self._index is not None:
            return self._index.ntotal
        return sum(v.shape[0] for v in self._vecs)

# ============================= Public API ============================ #
class VectorMemory:
    """Unified interface wrapping whichever backend is available."""
    def __init__(self, dsn: Optional[str] = None):
        dsn = dsn or os.getenv("PG_DSN") or os.getenv("DATABASE_URL")
        if dsn and _HAS_PG:
            self._store: _PostgresStore | _FaissStore = _PostgresStore(dsn)
            self.backend = "postgres"
        else:
            self._store = _FaissStore()
            self.backend = "faiss" if _HAS_FAISS else "numpy"
            if self.backend != "postgres":
                logger.warning("VectorMemory running in non‑persistent %s mode", self.backend)
                if self.backend == "numpy":
                    logger.warning("Performance will be sub‑optimal – install faiss‑cpu or connect Postgres+pgvector")
        _METRIC_SIZE.labels(self.backend).set(len(self))

    # --------------------------- operations ------------------------- #
    def add(self, agent: str, content: str | Iterable[str]) -> None:
        texts = [content] if isinstance(content, str) else list(content)
        vecs = _embed(texts)
        self._store.add(agent, vecs, texts)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        _METRIC_QUERY.labels(self.backend).inc()
        vec = _embed([query])
        return self._store.query(vec, k)

    # helper utilities ---------------------------------------------- #
    def __len__(self) -> int:
        return self._store.__len__()

    def flush(self) -> None:
        """**DANGEROUS**: wipe all memories (only supported in Postgres)."""
        if isinstance(self._store, _PostgresStore):
            with self._store._conn, self._store._conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE memories")
            _METRIC_SIZE.labels("postgres").set(0)
        else:
            raise RuntimeError("Flush only supported for persistent Postgres backend")

# ================================ CLI =============================== #
if __name__ == "__main__":
    import argparse, pprint, random, time  # noqa:  E402

    ap = argparse.ArgumentParser("vector‑memory CLI")
    ap.add_argument("--dsn", help="Postgres DSN (optional)")
    ap.add_argument("--demo", action="store_true", help="run quick demo")
    ns = ap.parse_args()

    vm = VectorMemory(ns.dsn)
    if ns.demo:
        print("Backend:", vm.backend)
        samples = ["the quick brown fox", "jumps over", "the lazy dog"]
        random.shuffle(samples)
        vm.add("demo", samples)
        time.sleep(0.2)  # simulate latency
        hits = vm.search("quick fox", k=2)
        pprint.pp(hits)
    else:
        ap.print_help()
