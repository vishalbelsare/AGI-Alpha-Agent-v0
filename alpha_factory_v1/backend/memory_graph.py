
"""alpha_factory_v1.backend.memory_graph
=========================================

**Causal‑graph memory layer** for *Alpha‑Factory v1 👁️✨*.

*   Primary backend — **Neo4j** (Bolt 4+) for high‑performance
    labeled‑property graph storage and Cypher querying.
*   Seamless fallback — **in‑memory NetworkX MultiDiGraph** when Neo4j
    is not reachable or the Python driver is unavailable.
*   Observability — Prometheus counters & gauges expose live
    node/edge counts and operation totals.
*   Thread‑safety — internally serialised via a re‑entrant lock so
    multiple agents can write concurrently from a shared process.
*   Zero‑downtime schema — auto‑creates uniqueness constraints on
    the *Entity.name* label while the database is online.

Example
-------
```python
from alpha_factory_v1.backend.memory_graph import GraphMemory

g = GraphMemory()  # auto‑connect
g.add("EnergySpike", "CAUSES", "PriceRise", {"delta_alpha": 1e6})

# simple Cypher
print(g.query("MATCH (a)-[r:CAUSES]->(b) RETURN a.name, b.name, r.delta_alpha"))

# high‑level helper
path = g.find_path("EnergySpike", "PriceRise", max_depth=3)
print("Path:", path)
```

The design purposefully keeps **all external imports optional** so the
broader Alpha‑Factory stack can still start in air‑gapped /
resource‑constrained environments.
"""

from __future__ import annotations

###############################################################################
# Standard‑library                                                            #
###############################################################################
import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

###############################################################################
# Third‑party (soft imports)                                                  #
###############################################################################
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
    # we will fallback to a very small stub if networkx absent
    class _StubGraph:  # noqa: D401
        """Tiny replacement used only when networkx is missing."""
        def __init__(self) -> None:
            self.nodes: set[str] = set()
            self.edges: list[tuple[str, str, dict[str, Any]]] = []

        def add_node(self, n: str) -> None:  # noqa: D401
            self.nodes.add(n)

        def add_edge(self, u: str, v: str, **props: Any) -> None:
            self.edges.append((u, v, props))

        def edges_iter(self):
            return self.edges
    nx = _StubGraph()  # type: ignore

try:
    from prometheus_client import Counter, Gauge  # type: ignore
    _HAS_PROM = True
except Exception:  # pragma: no cover
    _HAS_PROM = False

###############################################################################
# Logging                                                                     #
###############################################################################
logger = logging.getLogger("alpha_factory.memory_graph")
logger.setLevel(logging.INFO)

###############################################################################
# Prometheus metrics                                                          #
###############################################################################
if _HAS_PROM:
    _REL_ADD_TOTAL = Counter("af_graph_rel_add_total", "Relations added to GraphMemory")
    _QUERY_TOTAL   = Counter("af_graph_query_total",   "Cypher / NX queries executed")
    _NODES_GAUGE   = Gauge( "af_graph_nodes",          "Total nodes tracked" )
    _EDGES_GAUGE   = Gauge( "af_graph_edges",          "Total edges tracked" )
else:  # stub no‑op objects
    class _NoOp:  # noqa: D401
        def __getattr__(self, name):  # noqa: D401
            return lambda *a, **k: None
    _REL_ADD_TOTAL = _QUERY_TOTAL = _NODES_GAUGE = _EDGES_GAUGE = _NoOp()  # type: ignore

###############################################################################
# Helper utilities                                                            #
###############################################################################
_LOCK = threading.RLock()

def _update_gauges(node_cnt: int, edge_cnt: int) -> None:
    """Update Prometheus gauges, guarded for missing client."""
    _NODES_GAUGE.set(node_cnt)
    _EDGES_GAUGE.set(edge_cnt)

@contextmanager
def _neo_session(driver):
    """Context manager that retries transient Neo4j errors."""
    for attempt in range(3):
        try:
            with driver.session() as session:
                yield session
                return
        except Exception as exc:  # pragma: no cover
            logger.warning("Neo4j session error (%s) – retry %d/3", exc, attempt + 1)
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("Neo4j is unreachable after 3 attempts")

###############################################################################
# Core class                                                                  #
###############################################################################
class GraphMemory:
    """Causal graph memory with Neo4j primary & NetworkX fallback."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        """Initialise backend; env vars are used when args omitted.

        Parameters
        ----------
        uri, user, password
            Connection details for Neo4j (bolt/neo4j scheme). If *any*
            param is missing or Neo4j driver import fails, falls back to
            in‑memory graph.
        database
            Target Neo4j database (defaults to "neo4j").
        """
        self._driver = None
        self._nx: "nx.MultiDiGraph | _StubGraph"

        uri = uri or os.getenv("NEO4J_URI")
        user = user or os.getenv("NEO4J_USER")
        password = password or os.getenv("NEO4J_PASS")
        database = database or os.getenv("NEO4J_DATABASE", "neo4j")

        if _HAS_NEO4J and uri and user and password:
            try:
                self._driver = GraphDatabase.driver(uri, auth=basic_auth(user, password))
                self._db = database
                # ensure basic schema – uniqueness on Entity.name
                with _neo_session(self._driver) as sess:
                    sess.run(
                        "CREATE CONSTRAINT entity_name IF NOT EXISTS
                         ON (e:Entity) ASSERT e.name IS UNIQUE"
                    )
                logger.info("GraphMemory: connected to Neo4j @ %s", uri)
            except Exception as exc:  # pragma: no cover
                logger.warning("Neo4j connection failed (%s) – falling back to NetworkX", exc)
                self._driver = None

        # fallback path
        if self._driver is None:
            self._nx = nx.MultiDiGraph() if _HAS_NX else nx  # type: ignore
            logger.warning("GraphMemory: using in‑memory graph – data non‑persistent")

    # ------------------------------------------------------------------ #
    def add(
        self,
        src: str,
        rel: str,
        dst: str,
        props: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add (or upsert) a directed relation with optional properties."""
        props = props or {}
        with _LOCK:
            if self._driver:
                cypher = (
                    "MERGE (a:Entity {name:$src}) "
                    "MERGE (b:Entity {name:$dst}) "
                    f"MERGE (a)-[r:{rel}]->(b) "
                    "SET r += $props"
                )
                with _neo_session(self._driver) as sess:
                    sess.run(cypher, src=src, dst=dst, props=props, db=self._db)
                # simple counts via lightweight query to keep gauges correct
                _update_gauges(
                    self._scalar("MATCH (n) RETURN count(n)"),
                    self._scalar("MATCH ()-[r]->() RETURN count(r)"),
                )
            else:
                self._nx.add_edge(src, dst, key=rel, **props)
                _update_gauges(self._nx.number_of_nodes(), self._nx.number_of_edges())
        _REL_ADD_TOTAL.inc()

    # ------------------------------------------------------------------ #
    def query(self, cypher: str) -> List[Tuple[Any, ...]]:
        """Run a Cypher query (Neo4j) or approximate against NetworkX."""
        _QUERY_TOTAL.inc()
        if self._driver:
            with _neo_session(self._driver) as sess:
                res = sess.run(cypher, db=self._db)
                return [tuple(r.values()) for r in res]
        # very naive parser for simple patterns in fallback
        return self._fallback_cypher(cypher)

    # ------------------------------------------------------------------ #
    def find_path(
        self,
        src: str,
        dst: str,
        max_depth: int = 4,
    ) -> List[str]:
        """Return a list of node names representing one shortest path."""
        if self._driver:
            cypher = (
                "MATCH p=shortestPath((a:Entity {name:$src})-[*..%d]->(b:Entity {name:$dst})) "
                "RETURN [n IN nodes(p) | n.name] AS path LIMIT 1" % max_depth
            )
            res = self.query(cypher)
            return res[0][0] if res else []
        if hasattr(self._nx, "shortest_path"):
            try:
                path = nx.shortest_path(self._nx, src, dst)  # type: ignore
                return path
            except Exception:
                return []
        return []

    # ------------------------------------------------------------------ #
    def neighbours(self, node: str, rel: Optional[str] = None) -> List[str]:
        """Return outgoing neighbours of *node* (optionally filter by relation label)."""
        if self._driver:
            if rel:
                cypher = (
                    "MATCH (:Entity {name:$n})-[:%s]->(m) RETURN m.name" % rel
                )
                return [r[0] for r in self.query(cypher, n=node)]  # type: ignore[arg-type]
            cypher = "MATCH (:Entity {name:$n})-->(m) RETURN DISTINCT m.name"
            return [r[0] for r in self.query(cypher, n=node)]  # type: ignore[arg-type]
        # NX fallback
        if rel:
            return [
                v
                for _, v, k, d in self._nx.out_edges(node, keys=True, data=True)  # type: ignore[attr-defined]
                if k == rel
            ]
        return list(self._nx.successors(node))  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        if self._driver:
            return self._scalar("MATCH ()-[r]->() RETURN count(r)")
        return self._nx.number_of_edges()  # type: ignore[attr-defined]

    # ------------------------------------------------------------------ #
    # Internal utils
    # ------------------------------------------------------------------ #
    def _scalar(self, cypher: str, **params: Any) -> int:
        """Run cypher returning single scalar int."""
        with _neo_session(self._driver) as sess:
            rec = sess.run(cypher, **params, db=self._db).single()
            return int(rec.values()[0]) if rec else 0

    def _fallback_cypher(self, q: str) -> List[Tuple[Any, ...]]:
        """Extremely limited parser for simple MATCH ... RETURN queries."""
        if "delta_alpha" in q and ">" in q:
            # pattern MATCH ... WHERE r.delta_alpha > X
            thresh = float(q.split("delta_alpha")[-1].split(">")[-1].split()[0])
            return [
                (u, v, d)
                for u, v, d in self._nx.edges(data=True)  # type: ignore[attr-defined]
                if d.get("delta_alpha", 0) > thresh
            ]
        # default: return all triples
        return list(self._nx.edges(data=True))  # type: ignore[attr-defined]

###############################################################################
# CLI for manual smoke‑test                                                   #
###############################################################################
if __name__ == "__main__":  # pragma: no cover
    import argparse, pprint, sys

    ap = argparse.ArgumentParser("graph‑memory quick‑test")
    ap.add_argument("--uri", help="Neo4j bolt URI e.g. bolt://localhost:7687")
    ap.add_argument("--user", help="Neo4j user")
    ap.add_argument("--pwd", help="Neo4j password")
    ns = ap.parse_args()

    gmem = GraphMemory(ns.uri, ns.user, ns.pwd)
    gmem.add("TestEvent", "CAUSES", "Outcome", {"delta_alpha": 123})
    pprint.pp(gmem.query("MATCH (a)-[r:CAUSES]->(b) RETURN a.name, b.name, r.delta_alpha"))
    print("len(graph) =", len(gmem))
