"""
Vector memory wrapper – Postgres/pgvector first, FAISS fallback.

Usage:
    mem = VectorMemory()
    mem.add("finance", "Bought BTC at 62000")
    mem.search("BTC purchase", k=3)
"""

from __future__ import annotations
import os
import contextlib
from typing import List, Tuple

import numpy as np

try:
    import psycopg2
    HAS_PG = True
except ImportError:  # no Postgres client → fallback
    HAS_PG = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from sentence_transformers import SentenceTransformer  # small, CPU-friendly

_EMBED = SentenceTransformer("all-MiniLM-L6-v2")  # 384-dim

class VectorMemory:
    def __init__(self) -> None:
        self.pg_dsn = os.getenv("PG_DSN")
        self._conn = None
        self._faiss_index = None
        if HAS_PG and self.pg_dsn:
            self._setup_postgres()
        elif HAS_FAISS:
            self._setup_faiss()
        else:
            raise RuntimeError(
                "VectorMemory requires either Postgres+pgvector or faiss-cpu."
            )

    # ---------- Postgres ----------
    def _setup_postgres(self) -> None:
        self._conn = psycopg2.connect(self.pg_dsn)
        cur = self._conn.cursor()
        cur.execute(
            "CREATE EXTENSION IF NOT EXISTS vector;"
            "CREATE TABLE IF NOT EXISTS memories("
            "id SERIAL PRIMARY KEY, agent TEXT, embedding vector(384), "
            "content TEXT, ts TIMESTAMPTZ DEFAULT now());"
        )
        self._conn.commit()

    # ---------- FAISS ----------
    def _setup_faiss(self) -> None:
        self._faiss_index = faiss.IndexFlatL2(384)  # simple L2
        self._texts: List[Tuple[str, str]] = []      # (agent, content)

    # ---------- Public API ----------
    def add(self, agent: str, content: str) -> None:
        vec = _EMBED.encode(content, normalize_embeddings=True)
        if self._conn:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO memories(agent, embedding, content) VALUES (%s, %s, %s)",
                (agent, list(vec), content),
            )
            self._conn.commit()
        else:  # FAISS
            self._faiss_index.add(np.array([vec]).astype("float32"))
            self._texts.append((agent, content))

    def search(self, query: str, k: int = 5) -> List[Tuple[str, str, float]]:
        vec = _EMBED.encode(query, normalize_embeddings=True)
        if self._conn:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT agent, content, 1 - (embedding <=> %s::vector) AS sim "
                "FROM memories ORDER BY embedding <=> %s::vector LIMIT %s",
                (list(vec), list(vec), k),
            )
            return cur.fetchall()
        else:
            D, I = self._faiss_index.search(
                np.array([vec]).astype("float32"), k
            )
            return [
                (*self._texts[idx], 1 - float(dist))
                for idx, dist in zip(I[0], D[0])
                if idx != -1
            ]
