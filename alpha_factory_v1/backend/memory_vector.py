
"""
alpha_factory_v1.backend.memory_vector
======================================

Vector‑memory layer for **Alpha‑Factory v1 👁️✨**.

*   Primary backend — **PostgreSQL + pgvector** (best‑in‑class production setup)
*   Seamless fallback — **FAISS‑in‑RAM** or **brute‑force** (no external deps)
*   Embedding engine — auto‑selects **OpenAI `text‑embedding‑3‑small`**
    when `OPENAI_API_KEY` is set, otherwise falls back to
    *Sentence‑Transformers* (`all‑MiniLM‑L6‑v2`) – tiny & CPU‑friendly.

This module is intentionally **self‑contained** (no imports from the rest of
Alpha‑Factory) so that it can be unit‑tested and reused in isolation.

Example
-------
```python
from alpha_factory_v1.backend.memory_vector import VectorMemory

mem = VectorMemory()                       # auto‑connect
mem.add(agent="finance", content="Bought BTC at 62k")
hits = mem.search("BTC purchase", k=3)   # → [(agent, content, score), …]
```
"""

from __future__ import annotations

###############################################################################
# Standard‑library                                                            #
###############################################################################
import contextlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

###############################################################################
# Third‑party (import guarded – module still works if optional deps missing)  #
###############################################################################
try:                      # 🔌 OpenAI embeddings (preferred when available)
    import openai
    _OPENAI = os.getenv("OPENAI_API_KEY") is not None
except Exception:         # pragma: no cover
    _OPENAI = False       # type: ignore[assignment]

try:                      # 📚 Sentence‑Transformers fallback
    from sentence_transformers import SentenceTransformer           # type: ignore
except ModuleNotFoundError:                                          # pragma: no cover
    SentenceTransformer = None                                       # type: ignore

try:                      # 🐘 PostgreSQL & pgvector
    import psycopg2                                                 # type: ignore
    import psycopg2.extras                                          # type: ignore
    _HAS_PG = True
except ModuleNotFoundError:                                         # pragma: no cover
    _HAS_PG = False

try:                      # ⚡ FAISS CPU index
    import faiss                                                 # type: ignore
    _HAS_FAISS = True
except ModuleNotFoundError:                                         # pragma: no cover
    _HAS_FAISS = False

import numpy as _np  # numpy is a hard dep – required for array ops


###############################################################################
# Logging                                                                     #
###############################################################################
logger = logging.getLogger("alpha_factory.memory_vector")
logger.setLevel(logging.INFO)


###############################################################################
# Embedding helper                                                            #
###############################################################################
_DIM = 384  # default dimensionality when using MiniLM

def _embed(texts: Sequence[str]) -> "_np.ndarray":  # → shape (n, dim)
    """Return **L2‑normalised** embeddings for *1 or more* texts."""
    if _OPENAI:
        # batch call (OpenAI supports up to 2048 tokens / 2048 texts per req)
        try:
            rsp = openai.embeddings.create(model="text-embedding-3-small", input=list(texts))  # type: ignore
            mats = _np.asarray([d.embedding for d in rsp.data], dtype="float32")
        except Exception:                                   # pragma: no cover
            logger.exception("OpenAI embed failed – falling back to SBERT")
            return _embed_sbert(texts)
        return _l2_normalise(mats)

    return _embed_sbert(texts)


def _embed_sbert(texts: Sequence[str]) -> "_np.ndarray":
    if SentenceTransformer is None:
        raise RuntimeError("Sentence‑Transformers missing – run `pip install sentence-transformers` or set OPENAI_API_KEY.")
    model = _get_sbert_model()
    mats = _np.asarray(model.encode(list(texts), normalize_embeddings=True), dtype="float32")
    return mats


_SBERT_MODEL: Optional[SentenceTransformer] = None
def _get_sbert_model() -> "SentenceTransformer":   # lazy‑load once
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # noqa: S113
    return _SBERT_MODEL


def _l2_normalise(a: "_np.ndarray") -> "_np.ndarray":
    return a / _np.linalg.norm(a, ord=2, axis=1, keepdims=True)


###############################################################################
# Storage backends                                                            #
###############################################################################
class _PostgresStore:
    """pgvector‑powered persistent store."""

    def __init__(self, dsn: str):
        self._conn = psycopg2.connect(dsn)
        self._ensure_schema()

    # --------------------------------------------------------------------- #
    def _ensure_schema(self) -> None:
        with self._conn, self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """CREATE TABLE IF NOT EXISTS memories(
                        id       BIGSERIAL PRIMARY KEY,
                        agent    TEXT,
                        embedding  VECTOR(%s),
                        content  TEXT,
                        ts       TIMESTAMPTZ DEFAULT now()
                    )""", (_DIM,))
            cur.execute("CREATE INDEX IF NOT EXISTS mem_embedding_idx ON memories USING ivfflat(embedding vector_cosine_ops)")
            cur.execute("CREATE INDEX IF NOT EXISTS mem_agent_ts_idx   ON memories(agent, ts DESC)")

    # --------------------------------------------------------------------- #
    def add(self, agent: str, embeds: "_np.ndarray", contents: List[str]) -> None:
        assert embeds.shape[0] == len(contents)
        vec_list = embeds.tolist()
        rows = [(agent, v, c) for v, c in zip(vec_list, contents)]
        with self._conn, self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                "INSERT INTO memories(agent, embedding, content) VALUES (%s,%s,%s)",
                rows,
                page_size=100,
            )

    # --------------------------------------------------------------------- #
    def query(self, embed: "_np.ndarray", k: int) -> List[Tuple[str, str, float]]:
        with self._conn.cursor() as cur:
            cur.execute(
                """SELECT agent, content,
                           1 - (embedding <=> %s::vector) AS score
                       FROM memories
                       ORDER BY embedding <=> %s::vector
                       LIMIT %s""", (embed.tolist(), embed.tolist(), k)
            )
            return cur.fetchall()


class _FaissStore:
    """In‑memory FAISS (or brute‑force) store – non‑persistent."""
    def __init__(self):
        if _HAS_FAISS:
            self._index = faiss.IndexFlatIP(_DIM)   # cosine similarity via L2‑normalised dot
        else:
            self._index = None
        self._buf  : List[Tuple[str,str]] = []
        self._vecs : List["_np.ndarray"] = []

    # ------------------------------------------------------------------ #
    def add(self, agent: str, embeds: "_np.ndarray", contents: List[str]) -> None:
        self._buf.extend((agent, c) for c in contents)
        if self._index is not None:
            self._index.add(embeds)
        self._vecs.append(embeds)

    # ------------------------------------------------------------------ #
    def query(self, embed: "_np.ndarray", k: int) -> List[Tuple[str, str, float]]:
        if self._index is not None and self._index.ntotal > 0:
            D, I = self._index.search(embed, min(k, self._index.ntotal))
            return [
                (*self._buf[idx], float(score))
                for idx, score in zip(I[0], D[0])
            ]
        # brute‑force fallback (slow but always works)
        if not self._vecs:
            return []
        mat = _np.vstack(self._vecs)
        sims = mat @ embed.T
        topk_idx = sims[:,0].argsort()[::-1][:k]
        return [
            (*self._buf[i], float(sims[i,0]))
            for i in topk_idx
        ]


###############################################################################
# Public API                                                                  #
###############################################################################
class VectorMemory:
    """Unified vector‑memory wrapper (Postgres → FAISS → brute‑force)."""

    def __init__(self, dsn: Optional[str] = None):
        dsn = dsn or os.getenv("PG_DSN") or os.getenv("DATABASE_URL")
        self._store: "_PostgresStore | _FaissStore"
        if dsn and _HAS_PG:
            self._store = _PostgresStore(dsn)
            logger.info("VectorMemory: using Postgres pgvector backend")
        else:
            self._store = _FaissStore()
            logger.warning("VectorMemory: using in‑memory backend – data non‑persistent")

    # ------------------------------------------------------------------ #
    def add(self, agent: str, content: str | Iterable[str]) -> None:
        """Add *one or many* pieces of text for *agent*.

        Parameters
        ----------
        agent   – name/id of the agent (used for filtering / analytics)
        content – str or iterable[str]
        """
        if isinstance(content, str):
            texts = [content]
        else:
            texts = list(content)
        embeds = _embed(texts)
        self._store.add(agent, embeds, texts)

    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        """Return *(agent, content, similarity)* triples, highest‑score first."""
        q_vec = _embed([query])
        return self._store.query(q_vec, k)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        """Return number of stored memories (approx for FAISS)."""
        if isinstance(self._store, _PostgresStore):
            with self._store._conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM memories")
                return int(cur.fetchone()[0])
        elif isinstance(self._store, _FaissStore) and self._store._vecs:
            return sum(v.shape[0] for v in self._store._vecs)
        return 0  # empty


###############################################################################
# CLI (test)                                                                   #
###############################################################################
if __name__ == "__main__":  # quick‑and‑dirty self‑test
    import argparse, sys, pprint                                 # noqa: E402
    ap = argparse.ArgumentParser("vector‑memory quick‑test")
    ap.add_argument("--dsn", help="Postgres DSN (optional)")
    ns = ap.parse_args()

    mem = VectorMemory(ns.dsn)
    mem.add("demo", ["Sky is blue", "BTC hits 70k", "AI is the future"])
    pprint.pp(mem.search("bitcoin price", k=2))
