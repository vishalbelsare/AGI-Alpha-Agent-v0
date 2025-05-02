"""
Graph memory – Neo4j first, NetworkX fallback.

Example:
    g = GraphMemory()
    g.add("EnergySpike", "CAUSES", "PriceRise", {"delta_alpha": 1e6})
    g.query("MATCH (e:Event)-[r:CAUSES]->(p:Price) RETURN e,p,r LIMIT 5")
"""

from __future__ import annotations
import os
from typing import Dict, Any, List, Tuple

try:
    from neo4j import GraphDatabase  # type: ignore
    HAS_NEO = True
except ImportError:
    HAS_NEO = False

import networkx as nx

class GraphMemory:
    def __init__(self) -> None:
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        pwd = os.getenv("NEO4J_PASS")
        if HAS_NEO and uri and user and pwd:
            self._driver = GraphDatabase.driver(uri, auth=(user, pwd))
            self._nx = None
        else:
            self._driver = None
            self._nx = nx.MultiDiGraph()

    def add(
        self,
        src: str,
        rel: str,
        dst: str,
        props: Dict[str, Any] | None = None,
    ) -> None:
        props = props or {}
        if self._driver:
            with self._driver.session() as sess:
                sess.run(
                    "MERGE (a:Entity {name:$src}) "
                    "MERGE (b:Entity {name:$dst}) "
                    "MERGE (a)-[r:%s $props]->(b)" % rel,
                    src=src,
                    dst=dst,
                    props=props,
                )
        else:
            self._nx.add_edge(src, dst, key=rel, **props)

    def query(self, cypher: str) -> List[Tuple[Any, ...]]:
        if self._driver:
            with self._driver.session() as sess:
                return [tuple(r.values()) for r in sess.run(cypher)]
        else:
            # crude pattern: return all edges with property filter
            # e.g., "delta_alpha>1000000"
            if "delta_alpha" in cypher:
                thresh = float(cypher.split(">")[-1])
                return [
                    (u, v, d)
                    for u, v, d in self._nx.edges(data=True)
                    if d.get("delta_alpha", 0) > thresh
                ]
            return list(self._nx.edges(data=True))
