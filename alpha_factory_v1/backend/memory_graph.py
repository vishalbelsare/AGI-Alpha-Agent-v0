"""alpha_factory_v1.backend.memory_graph
========================================

Causal-graph memory layer for *Alpha-Factory v1 👁️✨*.

Key design goals
----------------
1. **Never crash the Orchestrator** – if Neo4j is unreachable, we drop-in an
   in-process graph that implements (a sensible subset of) the same API.
2. **Maximum portability** – all heavy dependencies are soft-imports; the code
   continues to run (with graceful capability degradation) on bare-bones
   Python 3.10-slim or even edge ARM devices.
3. **Observability built-in** – Prometheus counters, gauges & basic latency
   histogram let operators watch the causal-graph health in real time.
4. **Thread-safe & async-friendly** – a global RLock serialises state-mutating
   ops while allowing concurrent *read* access from multiple agents.
5. **Developer ergonomics** – a thin, intuitive surface API hides the Neo4j/
   NetworkX differences so agent authors can stay focused on their domain.

Example
-------
>>> from alpha_factory_v1.backend.memory_graph import GraphMemory
>>> g = GraphMemory()                             # auto-connect to Neo4j if env set
>>> g.add("EnergySpike", "CAUSES", "PriceRise", {"delta_alpha": 1e6})
>>> g.find_path("EnergySpike", "PriceRise")
['EnergySpike', 'PriceRise']

CLI smoke test (uses env var creds if present):
$ python -m alpha_factory_v1.backend.memory_graph --verbose
"""
from __future__ import annotations

# ────────────────────────────── stdlib ────────────────────────────
import json
import logging
import os
import pathlib
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ───────────────────────────── soft-imports ───────────────────────
try:
    from neo4j import GraphDatabase, basic_auth  # type: ignore
    _HAS_NEO4J = True
except Exception:  # pragma: no cover
    _HAS_NEO4J = False

try:
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:  # pragma: no cover
    _HAS_NX = False

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
    )  # type: ignore
    _HAS_PROM = True
except Exception:  # pragma: no cover
    _HAS_PROM = False

# ────────────────────────────── logging ───────────────────────────
logger = logging.getLogger("alpha_factory.memory_graph")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ──────────────────────────── Prometheus ──────────────────────────
if _HAS_PROM:
    _REL_ADD_TOTAL = Counter("af_graph_relation_add_total", "Relations inserted")
    _NODE_UPSERT_TOTAL = Counter("af_graph_node_upsert_total", "Nodes upserted")
    _QUERY_COUNT = Counter("af_graph_query_total", "Cypher / fallback queries")
    _QUERY_LAT = Histogram("af_graph_query_latency_seconds", "Query latency")
    _NODE_GAUGE = Gauge("af_graph_nodes", "Current node count")
    _EDGE_GAUGE = Gauge("af_graph_edges", "Current edge count")
else:  # pragma: no cover
    class _NoOp:  # noqa: D401
        def __getattr__(self, *_a, **_k):
            return lambda *a, **k: None

    _REL_ADD_TOTAL = _NODE_UPSERT_TOTAL = _QUERY_COUNT = _QUERY_LAT = (
        _NODE_GAUGE
    ) = _EDGE_GAUGE = _NoOp()  # type: ignore

# ───────────────────────────── globals ────────────────────────────
_LOCK = threading.RLock()
_RANDOM = random.Random(42)  # deterministic jitter

# ────────────────────────── helper utils ──────────────────────────
def _jitter_sleep(attempt: int) -> None:
    time.sleep((2 ** attempt) * 0.05 + _RANDOM.random() * 0.05)


@contextmanager
def _neo_session(driver, database: str):
    """Context manager with 3-retry exponential back-off."""
    for attempt in range(3):
        try:
            with driver.session(database=database) as session:
                yield session
                return
        except Exception as exc:  # pragma: no cover
            logger.warning("Neo4j error (%s) – retry %d/3", exc, attempt + 1)
            _jitter_sleep(attempt)
    raise RuntimeError("Neo4j unreachable after 3 attempts")


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


# ────────────────────────────── class ─────────────────────────────
class GraphMemory:
    """Unified causal-graph memory.

    Parameters
    ----------
    uri, user, password :
        Explicit Neo4j connection details.  If *any* are missing **or** the
        `neo4j` driver isn’t available, the class transparently falls back to
        an in-process (non-persistent) NetworkX graph.
    database :
        Neo4j DB name (default ``neo4j``).
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        *,
        database: str | None = None,
    ) -> None:
        uri = uri or os.getenv("NEO4J_URI")
        user = user or os.getenv("NEO4J_USER")
        password = password or os.getenv("NEO4J_PASS")
        self._database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        self._driver = None
        self._graph: "nx.MultiDiGraph | None"

        if _HAS_NEO4J and uri and user and password:
            try:
                self._driver = GraphDatabase.driver(
                    uri,
                    auth=basic_auth(user, password),
                    max_connection_pool_size=10,
                )
                self._ensure_schema()
                logger.info("GraphMemory connected to Neo4j @ %s", uri)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "Neo4j connect failed (%s) – switching to in-memory mode", exc
                )
                self._driver = None

        if self._driver is None:
            if _HAS_NX:
                self._graph = nx.MultiDiGraph()
            else:  # super-minimal stub
                class _Stub:  # noqa: D401
                    nodes: set[str] = set()
                    edges: list[tuple[str, str, dict[str, Any]]] = []

                    def add_node(self, n):
                        self.nodes.add(n)

                    def add_edge(self, u, v, key=None, **d):
                        self.edges.append((u, v, d))

                    def number_of_nodes(self):
                        return len(self.nodes)

                    def number_of_edges(self):
                        return len(self.edges)

                self._graph = _Stub()  # type: ignore
            logger.warning("GraphMemory running with in-memory graph – data volatile")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def add(  # noqa: D401
        self,
        src: str,
        rel: str,
        dst: str,
        props: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add (or merge) a relation.

        `src` and `dst` are **auto-created** if absent.
        """
        props = props or {}
        with _LOCK:
            self._node_upsert(src)
            self._node_upsert(dst)
            if self._driver:
                cypher = (
                    "MATCH (a:Entity {name:$src}), (b:Entity {name:$dst}) "
                    f"MERGE (a)-[r:{rel}]->(b) "
                    "SET r += $props"
                )
                with _neo_session(self._driver, self._database) as sess:
                    sess.run(cypher, src=src, dst=dst, props=props)
                self._refresh_gauges()
            else:  # NX path
                self._graph.add_edge(src, dst, key=rel, **props)  # type: ignore[arg-type]
                self._refresh_gauges_nx()
        _REL_ADD_TOTAL.inc()

    # ------------------------------------------------------------------
    def batch_add(
        self,
        triples: Iterable[Tuple[str, str, str]],
        default_props: Optional[Dict[str, Any]] = None,
    ) -> None:
        """High-throughput bulk inserter (single round-trip Neo4j)."""
        default_props = default_props or {}
        with _LOCK:
            if self._driver:
                cypher = (
                    "UNWIND $rows AS row "
                    "MERGE (a:Entity {name:row[0]}) "
                    "MERGE (b:Entity {name:row[2]}) "
                    "MERGE (a)-[r:`%s`]->(b) "
                    "SET r += $props" % "%s"
                )
                # we need to split by relation type because Neo4j‘s APOC isn’t
                # guaranteed; simple loop over groups keeps it dependency-free.
                by_rel: Dict[str, List[Tuple[str, str, str]]] = {}
                for s, r, d in triples:
                    by_rel.setdefault(r, []).append((s, r, d))
                with _neo_session(self._driver, self._database) as sess:
                    for rel, rows in by_rel.items():
                        sess.run(
                            cypher % rel,
                            rows=[list(t) for t in rows],
                            props=default_props,
                        )
                self._refresh_gauges()
            else:
                for s, r, d in triples:
                    self._graph.add_edge(s, d, key=r, **default_props)  # type: ignore[arg-type]
                self._refresh_gauges_nx()
        _REL_ADD_TOTAL.inc(len(list(triples)))

    # ------------------------------------------------------------------
    @_QUERY_LAT.time()  # type: ignore[arg-type]
    def query(self, cypher: str, **kwargs: Any) -> List[Tuple[Any, ...]]:
        """Run raw Cypher (Neo4j) – or crude fallback filter."""
        _QUERY_COUNT.inc()
        if self._driver:
            with _neo_session(self._driver, self._database) as sess:
                records = sess.run(cypher, **kwargs)
                return [tuple(r.values()) for r in records]
        # minimal fallback: only a couple patterns understood
        return self._fallback_query(cypher)

    # ------------------------------------------------------------------
    def find_path(self, src: str, dst: str, max_depth: int = 4) -> List[str]:
        """Shortest path (names) between two nodes."""
        if self._driver:
            cypher = (
                "MATCH p=shortestPath((a:Entity {name:$s})-[*..%d]->"
                "(b:Entity {name:$d})) "
                "RETURN [n IN nodes(p)|n.name] AS p LIMIT 1" % max_depth
            )
            res = self.query(cypher, s=src, d=dst)
            return res[0][0] if res else []
        if _HAS_NX and hasattr(self._graph, "shortest_path"):  # type: ignore[attr-defined]
            try:
                return nx.shortest_path(self._graph, src, dst)  # type: ignore[arg-type]
            except Exception:
                return []
        return []

    # ------------------------------------------------------------------
    def neighbours(self, node: str, *, rel: str | None = None) -> List[str]:
        """Return outgoing neighbour names (optionally filter by rel)."""
        if self._driver:
            if rel:
                cypher = (
                    "MATCH (:Entity {name:$n})-[:%s]->(m) RETURN m.name" % rel
                )
                return [r[0] for r in self.query(cypher, n=node)]
            cypher = "MATCH (:Entity {name:$n})-->(m) RETURN DISTINCT m.name"
            return [r[0] for r in self.query(cypher, n=node)]
        # NX path
        if rel:
            return [
                v
                for _, v, k in self._graph.out_edges(node, keys=True)  # type: ignore[attr-defined]
                if k == rel
            ]
        return list(self._graph.successors(node))  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    def export_graphml(self, path: str | pathlib.Path) -> None:
        """Dump graph to GraphML (works in both backends)."""
        p = pathlib.Path(path)
        if self._driver:
            g_nx = nx.MultiDiGraph()  # type: ignore[arg-type]
            # naive copy – good enough for demo-scale graphs
            for n, in self.query("MATCH (n) RETURN n.name"):
                g_nx.add_node(n)
            for s, r, d in self.query("MATCH (a)-[r]->(b) RETURN a.name,r,b.name"):
                g_nx.add_edge(s, d, key=r)
            nx.write_graphml(g_nx, p)
        else:
            nx.write_graphml(self._graph, p)  # type: ignore[arg-type]
        logger.info("Graph exported → %s", p)

    def import_graphml(self, path: str | pathlib.Path) -> None:
        """Load GraphML file into current backend (additive)."""
        p = pathlib.Path(path)
        g_in = nx.read_graphml(p)  # type: ignore[arg-type]
        triples = [(u, d["key"], v) for u, v, d in g_in.edges(data=True)]
        self.batch_add(triples)
        logger.info("Graph imported from %s (%d triples)", p, len(triples))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        if self._driver:
            return _safe_int(
                self.query("MATCH ()-[r]->() RETURN count(r) AS c")[0][0]
            )
        return self._graph.number_of_edges()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _node_upsert(self, name: str) -> None:
        _NODE_UPSERT_TOTAL.inc()
        if self._driver:
            with _neo_session(self._driver, self._database) as sess:
                sess.run("MERGE (:Entity {name:$n})", n=name)
        else:
            self._graph.add_node(name)  # type: ignore[attr-defined]

    def _ensure_schema(self) -> None:
        with _neo_session(self._driver, self._database) as sess:
            sess.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) "
                "REQUIRE e.name IS UNIQUE"
            )

    def _refresh_gauges(self) -> None:
        n = _safe_int(
            self.query("MATCH (n) RETURN count(n) AS c")[0][0]
        )
        e = _safe_int(
            self.query("MATCH ()-[r]->() RETURN count(r) AS c")[0][0]
        )
        _NODE_GAUGE.set(n)
        _EDGE_GAUGE.set(e)

    def _refresh_gauges_nx(self) -> None:
        _NODE_GAUGE.set(self._graph.number_of_nodes())  # type: ignore[attr-defined]
        _EDGE_GAUGE.set(self._graph.number_of_edges())  # type: ignore[attr-defined]

    def _fallback_query(self, cypher: str) -> List[Tuple[Any, ...]]:
        """Extremely naive fallback – recognises two patterns."""
        if "delta_alpha" in cypher and ">" in cypher:
            thresh = float(cypher.split(">")[-1].split()[0])
            return [
                (u, v, d)
                for u, v, d in self._graph.edges(data=True)  # type: ignore[attr-defined]
                if d.get("delta_alpha", 0) > thresh
            ]
        # default – brute force dump
        return list(self._graph.edges(data=True))  # type: ignore[attr-defined]

# ────────────────────────────── CLI test ──────────────────────────
if __name__ == "__main__":  # pragma: no cover
    import argparse
    from pprint import pprint

    ap = argparse.ArgumentParser(
        "graph-memory quick-test (Neo4j if env set, else in-memory)"
    )
    ap.add_argument("--verbose", action="store_true")
    ns = ap.parse_args()
    if ns.verbose:
        logger.setLevel(logging.DEBUG)

    g = GraphMemory()
    g.add("AlphaEvent", "CAUSES", "BetaOutcome", {"delta_alpha": 42})
    pprint(g.query("MATCH (a)-[r]->(b) RETURN a.name,r,b.name LIMIT 5"))
    print("Shortest path:", g.find_path("AlphaEvent", "BetaOutcome"))
    print("Graph size:", len(g))
