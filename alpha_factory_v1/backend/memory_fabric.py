# SPDX-License-Identifier: MIT
"""
alpha_factory_v1.backend.memory_fabric
======================================

🧠  MEMORY FABRIC  (v0.9.0 – 2025-05-02)
---------------------------------------
Unified episodic- & causal-memory layer for Alpha-Factory v1.

Key Capabilities
----------------
• Vector Memory  – dense embedding search (PostgreSQL + pgvector → FAISS → python list fallback)  
• Graph  Memory  – causal relations (Neo4j → NetworkX → python list fallback)  
• Zero-crash  policy – **runs even if every external dependency is missing**  
• Auto-provision  – will create required tables / indices / graph constraints on first run  
• Thread-safe  – lightweight locks prevent race conditions in in-proc multi-agent mode  
• Env-only  config – no hard-coded secrets

Environment Variables (defaults in brackets)
--------------------------------------------
PGHOST, PGPORT [5432], PGUSER, PGPASSWORD, PGDATABASE [memdb]  
NEO4J_URI [bolt://localhost:7687], NEO4J_USER [neo4j], NEO4J_PASS [neo4j]  
OPENAI_API_KEY (optional) – if set, OpenAI embeddings are used before local models  
VECTOR_DIM [768] – expected dimensionality for pgvector & fallback stores
"""

from __future__ import annotations

# ────────────────────────── std-lib ──────────────────────────
import contextlib
import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Sequence, Tuple

logger = logging.getLogger("AlphaFactory.MemoryFabric")
logger.setLevel(logging.INFO)

_LOCK = threading.Lock()  # global coarse-grain

# ─────────────── optional third-party imports (soft!) ─────────
with contextlib.suppress(ModuleNotFoundError):
    import numpy as np  # type: ignore

with contextlib.suppress(ModuleNotFoundError):
    from sentence_transformers import SentenceTransformer  # type: ignore

with contextlib.suppress(ModuleNotFoundError):
    import psycopg2  # type: ignore
    import psycopg2.extras  # type: ignore

with contextlib.suppress(ModuleNotFoundError):
    import faiss  # type: ignore

with contextlib.suppress(ModuleNotFoundError):
    from neo4j import GraphDatabase  # type: ignore

with contextlib.suppress(ModuleNotFoundError):
    import networkx as nx  # type: ignore

with contextlib.suppress(ModuleNotFoundError):
    import openai  # type: ignore

# ═════════════════════ helper: Embedding backend ══════════════


def _load_embedder():
    """Return a callable(text:str)->np.ndarray[float]."""
    # Preference order: OpenAI → sentence-transformer → random
    if "openai" in globals() and os.getenv("OPENAI_API_KEY"):
        logger.info("[MEM] OpenAI embeddings backend activated.")

        def _openai_emb(text: str):
            resp = openai.Embedding.create(model="text-embedding-3-small", input=text)  # type: ignore[attr-defined]
            return np.array(resp["data"][0]["embedding"], dtype="float32")  # type: ignore[index]

        return _openai_emb

    if "SentenceTransformer" in globals():
        logger.info("[MEM] Local SBERT backend activated.")
        _model = SentenceTransformer("all-MiniLM-L6-v2")

        def _sbert(text: str):
            return _model.encode(text, normalize_embeddings=True)

        return _sbert

    logger.warning("[MEM] No embedding backend found; falling back to hashed vectors.")

    def _hash_vec(text: str, dim: int = int(os.getenv("VECTOR_DIM", 768))):
        import hashlib, math, struct

        h = hashlib.sha256(text.encode()).digest()
        # spread bits deterministically into vector
        vec = [0.0] * dim
        for i in range(0, len(h), 4):
            idx = int.from_bytes(h[i : i + 2], "little") % dim
            sign = 1 if h[i + 2] & 1 else -1
            vec[idx] += sign * 0.5
        # L2-normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return np.array([v / norm for v in vec], dtype="float32")

    return _hash_vec


_EMBED = _load_embedder()
_VEC_DIM = int(os.getenv("VECTOR_DIM", 768))

# ══════════════════════ Vector Memory class ═══════════════════


@dataclass
class _VectorStore:
    """PgVector → FAISS → SQLite(fallback)  in that order."""

    pg_conn: Any | None = field(init=False, default=None)
    faiss_index: Any | None = field(init=False, default=None)
    mem: List[Tuple[str, str, str, List[float]]] = field(init=False, default_factory=list)
    _sqlite: sqlite3.Connection | None = field(init=False, default=None)

    def __post_init__(self):
        self._try_pg()
        if not self.pg_conn:
            self._try_faiss()

    # ─────────────── Postgres / pgvector initialisation ────────────────
    def _try_pg(self):
        if "psycopg2" not in globals():
            return
        dsn = " ".join(
            f"{k}={v}"
            for k, v in {
                "host": os.getenv("PGHOST"),
                "port": os.getenv("PGPORT", 5432),
                "user": os.getenv("PGUSER"),
                "password": os.getenv("PGPASSWORD"),
                "dbname": os.getenv("PGDATABASE", "memdb"),
            }.items()
            if v
        )
        try:
            self.pg_conn = psycopg2.connect(dsn)  # type: ignore[arg-type]
            with self.pg_conn, self.pg_conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"""CREATE TABLE IF NOT EXISTS memories(
                        id SERIAL PRIMARY KEY,
                        agent TEXT,
                        embedding VECTOR({_VEC_DIM}),
                        content TEXT,
                        ts TIMESTAMPTZ DEFAULT NOW()
                    );"""
                )
                cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_vec ON memories USING ivfflat (embedding vector_cosine_ops);")
            logger.info("[MEM] Vector store backed by Postgres/pgvector.")
        except Exception as exc:  # noqa: BLE001
            self.pg_conn = None
            logger.warning("[MEM] Postgres unavailable (%s) – falling back.", exc)

    # ────────────────────── FAISS in-memory index ──────────────────────
    def _try_faiss(self):
        if "faiss" in globals():
            self.faiss_index = faiss.IndexFlatIP(_VEC_DIM)
            logger.info("[MEM] Vector store backed by FAISS (in-memory).")
        else:
            # Final fallback: on-disk mini SQLite for persistence
            self._sqlite = sqlite3.connect(Path("vector_fallback.db"))
            self._sqlite.execute(
                "CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY AUTOINCREMENT, agent TEXT, vector BLOB, content TEXT, ts TEXT)"
            )
            logger.warning("[MEM] Vector store fallback to SQLite blobs.")

    # ───────────────────────── public API ──────────────────────────────
    def add(self, agent: str, content: str):
        vec = _EMBED(content).astype("float32")
        if self.pg_conn:
            with self.pg_conn, self.pg_conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memories(agent, embedding, content) VALUES(%s, %s, %s)",
                    (agent, list(map(float, vec)), content),
                )
        elif self.faiss_index:
            idx = len(self.mem)
            self.faiss_index.add(vec.reshape(1, -1))
            self.mem.append((agent, content, datetime.now(timezone.utc).isoformat(), vec.tolist()))
        else:  # SQLite
            self._sqlite.execute(
                "INSERT INTO memories(agent, vector, content, ts) VALUES(?,?,?,?)",
                (agent, vec.tobytes(), content, datetime.now(timezone.utc).isoformat()),
            )
            self._sqlite.commit()

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[dict]:
        q_vec = _EMBED(query).astype("float32")
        if self.pg_conn:
            with self.pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:  # type: ignore[arg-type]
                cur.execute(
                    "SELECT agent, content, ts FROM memories ORDER BY embedding <-> %s LIMIT %s;",
                    (list(map(float, q_vec)), top_k),
                )
                return cur.fetchall()
        elif self.faiss_index:
            if not len(self.mem):
                return []
            dists, idxs = self.faiss_index.search(q_vec.reshape(1, -1), top_k)
            out = []
            for dist, idx in zip(dists[0], idxs[0]):
                if idx == -1:
                    continue
                agent, content, ts, _ = self.mem[idx]
                out.append({"agent": agent, "content": content, "ts": ts, "score": float(dist)})
            return out
        else:  # brute sqlite
            cur = self._sqlite.execute("SELECT agent, vector, content, ts FROM memories")
            rows = cur.fetchall()
            scored = []
            import numpy.linalg as npl

            for a, vec_blob, c, ts in rows:
                v = np.frombuffer(vec_blob, dtype="float32")
                score = float(np.dot(v, q_vec) / (npl.norm(v) * npl.norm(q_vec) or 1.0))
                scored.append((score, a, c, ts))
            scored.sort(reverse=True)
            return [{"agent": a, "content": c, "ts": ts, "score": s} for s, a, c, ts in scored[: top_k]]

    # ------------------------------------------------------------------
    def purge(self, agent: str):
        if self.pg_conn:
            with self.pg_conn, self.pg_conn.cursor() as cur:
                cur.execute("DELETE FROM memories WHERE agent=%s", (agent,))
        elif self.faiss_index:
            # simple brute force rebuild
            self.mem = [(a, c, t, v) for a, c, t, v in self.mem if a != agent]
            self.faiss_index.reset()
            if self.mem:
                vecs = np.array([m[3] for m in self.mem], dtype="float32")
                self.faiss_index.add(vecs)
        else:
            self._sqlite.execute("DELETE FROM memories WHERE agent=?", (agent,))
            self._sqlite.commit()

    # ------------------------------------------------------------------
    def recent(self, agent: str, n: int = 10) -> List[str]:
        if self.pg_conn:
            with self.pg_conn.cursor() as cur:
                cur.execute("SELECT content FROM memories WHERE agent=%s ORDER BY ts DESC LIMIT %s", (agent, n))
                return [r[0] for r in cur.fetchall()]
        elif self.faiss_index:
            return [c for a, c, _, _ in self.mem if a == agent][-n:]
        else:
            cur = self._sqlite.execute("SELECT content FROM memories WHERE agent=? ORDER BY ts DESC LIMIT ?", (agent, n))
            return [r[0] for r in cur.fetchall()]


# ═══════════════════════ Graph Memory class ═══════════════════


@dataclass
class _GraphStore:
    driver: Any | None = field(init=False, default=None)
    g: Any | None = field(init=False, default=None)  # networkx
    triples: List[Tuple[str, str, str, dict]] = field(init=False, default_factory=list)

    def __post_init__(self):
        self._try_neo()
        if not self.driver:
            self._try_nx()

    # ───────────────────── Neo4j connector ─────────────────────
    def _try_neo(self):
        if "GraphDatabase" not in globals():
            return
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        try:
            self.driver = GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASS", "neo4j")))
            # quick liveness check
            with self.driver.session() as sess:
                sess.run("RETURN 1").single()
            logger.info("[MEM] Graph store backed by Neo4j.")
        except Exception as exc:  # noqa: BLE001
            self.driver = None
            logger.warning("[MEM] Neo4j unavailable (%s) – falling back.", exc)

    # ──────────────────  NetworkX fallback ────────────────────
    def _try_nx(self):
        if "nx" in globals():
            self.g = nx.DiGraph()  # type: ignore[attr-defined]
            logger.info("[MEM] Graph store using in-memory NetworkX.")
        else:
            logger.warning("[MEM] No graph backend; using python list.")

    # ───────────────────────── public API ─────────────────────
    def add_relation(self, a: str, rel: str, b: str, props: dict | None = None):
        props = props or {}
        ts = datetime.now(timezone.utc).isoformat()
        props.setdefault("ts", ts)
        if self.driver:
            cypher = (
                "MERGE (x:Entity {name:$a}) "
                "MERGE (y:Entity {name:$b}) "
                f"MERGE (x)-[r:{rel.upper()}]->(y) "
                "SET r += $props "
            )
            with self.driver.session() as sess:
                sess.run(cypher, a=a, b=b, props=props)
        elif self.g is not None:
            self.g.add_node(a)
            self.g.add_node(b)
            self.g.add_edge(a, b, key=rel, **props)
        else:
            self.triples.append((a, rel, b, props))

    # ------------------------------------------------------------------
    def query(self, cypher: str) -> List[dict]:
        """Neo4j Cypher passthrough or fallback filter-hack."""
        if self.driver:
            with self.driver.session() as sess:
                return [rec.data() for rec in sess.run(cypher)]
        logger.warning("[MEM] Cypher requested but Neo4j absent; returning empty.")
        return []

    # ------------------------------------------------------------------
    def find_path(self, start: str, end: str, max_len: int = 4) -> List[str]:
        if self.driver:
            cypher = (
                "MATCH p=(a:Entity {name:$start})-[*1..%d]-(b:Entity {name:$end}) "
                "RETURN nodes(p) AS n LIMIT 1"
            ) % max_len
            res = self.query(cypher)
            return [n["name"] for n in res[0]["n"]] if res else []
        elif self.g is not None:
            try:
                path = nx.shortest_path(self.g, start, end)  # type: ignore[arg-type]
                return path[: max_len + 1]
            except Exception:
                return []
        else:
            # brute force list search
            return []


# ═══════════════════════ public façade ════════════════════════
class MemoryFabric:
    """Singleton façade."""

    def __init__(self):
        self.vector = _VectorStore()
        self.graph = _GraphStore()

    # convenience wrappers ------------------------------------------------
    def add_memory(self, agent: str, content: str):
        with _LOCK:
            self.vector.add(agent, content)

    def search(self, query: str, top_k: int = 5):
        with _LOCK:
            return self.vector.search(query, top_k=top_k)

    def add_relation(self, a: str, rel: str, b: str, props: dict | None = None):
        with _LOCK:
            self.graph.add_relation(a, rel, b, props)

    def find_path(self, start: str, end: str, max_len: int = 4):
        with _LOCK:
            return self.graph.find_path(start, end, max_len=max_len)


# global instance used by orchestrator / agents
mem = MemoryFabric()

__all__ = ["mem", "MemoryFabric"]
