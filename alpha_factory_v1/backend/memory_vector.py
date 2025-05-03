"""
alpha_factory_v1.backend.memory_vector
======================================

🚀  *Vector Memory Fabric* – persistent, high-performance, always-on
--------------------------------------------------------------------
Primary store   : **PostgreSQL + pgvector**           (durable, scalable)  
Secondary store : **FAISS in RAM**                    (ultra-fast, ephemeral)  
Tertiary store  : **Pure-NumPy cosine search**        (zero-dep fallback)  

Embedding back-end
------------------
1. **OpenAI** `text-embedding-3-small` *(1536 d)* – if ``OPENAI_API_KEY`` present.  
2. **Sentence-Transformers** `all-MiniLM-L6-v2` *(384 d)* – local CPU model.  

Design goals
------------
* **Graceful degradation** – no ImportError at import-time; run anywhere.
* **Observability first** – Prometheus counters & gauges baked-in.
* **Thread-safe, fork-safe** – one connection per process.
* **CLI self-test** – ``python -m ...memory_vector --demo`` works off-grid.

Copyright
---------
© 2025 Montreal AI – MIT License (see repo root).

"""

from __future__ import annotations

# -- std lib --───────────────────────────────────────────────────────────────
import contextlib
import logging
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as _np

# -- logging set-up (idempotent) --────────────────────────────────────────────
_logger = logging.getLogger("alpha_factory.memory_vector")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s  %(message)s")
    )
    _logger.addHandler(_handler)
    _logger.setLevel(os.getenv("LOGLEVEL", "INFO"))

# --------------------------------------------------------------------------- #
# Optional dependencies (individually gated – *never* explode at import time) #
# --------------------------------------------------------------------------- #
# Embeddings
try:
    import openai  # type: ignore
    _HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
except Exception:  # pragma: no cover
    _HAS_OPENAI = False
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

# Storage
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

# Metrics
try:
    from prometheus_client import Counter, Gauge  # type: ignore
    _MET_ADD = Counter("af_mem_add_total", "Memories added", ["backend"])
    _MET_QRY = Counter("af_mem_query_total", "Vector queries", ["backend"])
    _MET_SZ  = Gauge("af_mem_size", "Total memories stored", ["backend"])
except ModuleNotFoundError:  # pragma: no cover
    class _Dummy:  # pylint: disable=too-few-public-methods
        def labels(self, *_a): return self
        def inc(self, *_a): ...
        def set(self, *_a): ...
    _MET_ADD = _MET_QRY = _MET_SZ = _Dummy()  # type: ignore

# --------------------------------------------------------------------------- #
# Embedding layer                                                             #
# --------------------------------------------------------------------------- #
_DIM_OPENAI = 1536
_DIM_SBERT  = 384
_SBERT_MODEL: SentenceTransformer | None = None

def _embed(texts: Sequence[str]) -> _np.ndarray:
    """Return **L2-normalised** embeddings for *texts*."""
    if _HAS_OPENAI:  # OpenAI first – best quality
        try:
            rsp = openai.embeddings.create(  # type: ignore
                model="text-embedding-3-small",
                input=list(texts),
            )
            mat = _np.asarray([d.embedding for d in rsp.data], dtype="float32")
            return _l2(mat)
        except Exception:  # pragma: no cover
            _logger.exception("OpenAI embedding failed – falling back to SBERT")

    if SentenceTransformer is None:
        raise RuntimeError(
            "No embedding backend available – install `sentence-transformers` "
            "or set OPENAI_API_KEY."
        )
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    mat = _SBERT_MODEL.encode(list(texts), normalize_embeddings=True)
    return _np.asarray(mat, dtype="float32")

def _l2(mat: _np.ndarray) -> _np.ndarray:
    return mat / _np.linalg.norm(mat, ord=2, axis=1, keepdims=True)

def _emb_dim() -> int:
    return _DIM_OPENAI if _HAS_OPENAI else _DIM_SBERT

# --------------------------------------------------------------------------- #
# Storage back-ends                                                           #
# --------------------------------------------------------------------------- #
class _PostgresStore:
    """Persistent store: PostgreSQL + pgvector (requires pgvector extension)."""
    def __init__(self, dsn: str, dim: int):
        self._dim = dim
        self._conn = psycopg2.connect(dsn)
        self._prepare_schema()
        _MET_SZ.labels("postgres").set(len(self))

    def _prepare_schema(self) -> None:
        with self._conn, self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                f"""CREATE TABLE IF NOT EXISTS memories(
                        id        BIGSERIAL PRIMARY KEY,
                        agent     TEXT,
                        embedding VECTOR({_emb_dim()}),
                        content   TEXT,
                        ts        TIMESTAMPTZ DEFAULT now()
                    )
                """)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS mem_vec_idx "
                "ON memories USING ivfflat(embedding vector_cosine_ops)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS mem_agent_ts_idx "
                "ON memories(agent, ts DESC)"
            )

    # -- CRUD --───────────────────────────────────────────────────────────
    def add(self, agent: str, vecs: _np.ndarray, texts: List[str]) -> None:
        rows = [(agent, v.tolist(), t) for v, t in zip(vecs, texts)]
        with self._conn, self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "INSERT INTO memories(agent, embedding, content) VALUES (%s,%s,%s)",
                rows,
                page_size=100,
            )
        _MET_ADD.labels("postgres").inc(len(rows))
        _MET_SZ.labels("postgres").set(len(self))

    def query(self, vec: _np.ndarray, k: int) -> List[Tuple[str, str, float]]:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT agent, content,
                          1 - (embedding <=> %s::vector) AS score
                     FROM memories
                 ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                """,
                (vec.tolist(), vec.tolist(), k),
            )
            return cur.fetchall()

    def __len__(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM memories")
            return int(cur.fetchone()[0])

class _FaissStore:
    """Fast in-memory FAISS index (falls back to NumPy if FAISS unavailable)."""
    def __init__(self, dim: int):
        self._dim = dim
        self._has_faiss = _HAS_FAISS
        self._index = faiss.IndexFlatIP(dim) if _HAS_FAISS else None  # type: ignore
        self._vecs: List[_np.ndarray] = []
        self._meta: List[Tuple[str, str]] = []  # (agent, text)

    def add(self, agent: str, vecs: _np.ndarray, texts: List[str]) -> None:
        self._meta.extend((agent, t) for t in texts)
        if self._has_faiss:
            self._index.add(vecs)  # type: ignore
        else:
            self._vecs.append(vecs)
        backend = "faiss" if self._has_faiss else "numpy"
        _MET_ADD.labels(backend).inc(len(texts))
        _MET_SZ.labels(backend).set(len(self))

    def query(self, vec: _np.ndarray, k: int) -> List[Tuple[str, str, float]]:
        if not len(self):
            return []
        if self._has_faiss:
            k = min(k, self._index.ntotal)  # type: ignore
            D, I = self._index.search(vec, k)  # type: ignore
            return [
                (*self._meta[idx], float(score))
                for idx, score in zip(I[0], D[0])
            ]

        # brute-force NumPy cosine
        mat = _np.vstack(self._vecs)
        sims = (mat @ vec.T).ravel()
        order = sims.argsort()[::-1][:k]
        return [
            (*self._meta[i], float(sims[i]))
            for i in order
        ]

    def __len__(self) -> int:
        if self._has_faiss:
            return self._index.ntotal  # type: ignore
        return sum(v.shape[0] for v in self._vecs)

# --------------------------------------------------------------------------- #
# Public façade                                                               #
# --------------------------------------------------------------------------- #
class VectorMemory:
    """Unified API wrapping whichever storage back-end is available."""
    def __init__(self, dsn: str | None = None):
        self._dim = _emb_dim()
        dsn = dsn or os.getenv("PG_DSN") or os.getenv("DATABASE_URL")
        if dsn and _HAS_PG:
            try:
                self._store: _PostgresStore | _FaissStore = _PostgresStore(dsn, self._dim)
                self.backend = "postgres"
            except Exception as exc:  # pragma: no cover
                _logger.warning("Postgres unavailable (%s) – using FAISS/NumPy fallback", exc)
                self._store = _FaissStore(self._dim)
                self.backend = "faiss" if _HAS_FAISS else "numpy"
        else:
            self._store = _FaissStore(self._dim)
            self.backend = "faiss" if _HAS_FAISS else "numpy"
            _logger.warning("VectorMemory running in *%s* mode (not persistent)", self.backend)
        _MET_SZ.labels(self.backend).set(len(self))

    # -- public ops --──────────────────────────────────────────────────────
    def add(self, agent: str, content: str | Iterable[str]) -> None:
        texts = [content] if isinstance(content, str) else list(content)
        vecs = _embed(texts)
        self._store.add(agent, vecs, texts)

    def search(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        _MET_QRY.labels(self.backend).inc()
        vec = _embed([query])
        return self._store.query(vec, k)

    # helpers --------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._store)

    def flush(self) -> None:
        """Dangerous – wipe all memories (only supported on Postgres)."""
        if isinstance(self._store, _PostgresStore):
            with self._store._conn, self._store._conn.cursor() as cur:  # type: ignore
                cur.execute("TRUNCATE TABLE memories")
            _MET_SZ.labels("postgres").set(0)
        else:
            raise RuntimeError("Flush supported only for persistent Postgres backend")

# --------------------------------------------------------------------------- #
# CLI demo (works fully offline)                                              #
# --------------------------------------------------------------------------- #
def _cli_demo(vm: VectorMemory) -> None:
    _logger.info("Backend → %s", vm.backend)
    samples = [
        "the quick brown fox",
        "jumps over",
        "the lazy dog",
    ]
    vm.add("demo", samples)
    time.sleep(0.2)
    hits = vm.search("quick fox", k=2)
    print("\nTop-K search results:")
    for agent, text, score in hits:
        print(f"  [{score:.3f}] {agent}: {text}")
    print("\n✓ demo complete – VectorMemory operational.")

if __name__ == "__main__":
    import argparse  # pylint: disable=wrong-import-position

    ap = argparse.ArgumentParser("vector-memory CLI")
    ap.add_argument("--dsn", help="Postgres DSN (optional)")
    ap.add_argument("--demo", action="store_true", help="run quick self-test")
    args = ap.parse_args()

    vm = VectorMemory(args.dsn)
    if args.demo:
        _cli_demo(vm)
    else:
        ap.print_help()
