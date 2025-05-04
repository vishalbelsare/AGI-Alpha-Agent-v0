
#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""phylo_tree_smart_city.py
──────────────────────────────────────────────────────────────────────────────
Visualise the *evolution* of smart‑city disruption scenarios archived by
**OMNI‑Factory** as a phylogenetic‑style tree.

• Reads `omni_ledger.sqlite` (same directory) – or falls back to 4 samples.
• TF‑IDF + cosine agglomerative clustering builds semantic branches.
• Node radius ∝ average reward (larger = stronger “Alpha” value).
• Saves high‑resolution PNG `phylo_tree.png` next to the script.

Quick‑start
───────────
    pip install matplotlib networkx scikit-learn
    python phylo_tree_smart_city.py

If GraphViz is installed, the plot uses a tidy hierarchical layout;
otherwise it gracefully falls back to a spring layout.

This file is self‑contained, safe (read‑only DB access) and platform‑agnostic.
"""

from __future__ import annotations
import contextlib, sqlite3, math, pathlib, sys, os

# ─── Third‑party deps ──────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    import networkx as nx
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_distances
except ImportError as exc:      # pragma: no cover
    miss = str(exc).split("'")[-2]
    print(f"[ERROR] Missing dependency: {miss}. "
          "Run  pip install matplotlib networkx scikit-learn")
    sys.exit(1)

# ─── Constants ─────────────────────────────────────────────────────────────
LEDGER_PATH = pathlib.Path(__file__).with_name("omni_ledger.sqlite")
OUT_FILE    = pathlib.Path(__file__).with_name("phylo_tree.png")
MAX_CLUSTERS = 6               # soft cap for readability

# ─── Helpers ───────────────────────────────────────────────────────────────
def _load_ledger() -> list[tuple[str, float]]:
    """Return list of (scenario sentence, avg_reward)."""
    if not LEDGER_PATH.exists():
        # demo fallback
        return [
            ("Flash‑flood closes two bridges at rush hour", 42.0),
            ("Cyber‑attack on traffic‑light network",       55.0),
            ("Record heatwave threatens rolling brown‑outs",65.0),
            ("Protest blocks downtown core",                51.0),
        ]
    rows: list[tuple[str, float]] = []
    with sqlite3.connect(LEDGER_PATH) as conn:
        cur = conn.execute("SELECT scenario, avg_reward FROM ledger")
        rows = cur.fetchall()
    return rows

def _cluster(sentences: list[str]) -> list[int]:
    """Return cluster labels using distance threshold heuristic."""
    vec = TfidfVectorizer().fit_transform(sentences)
    # distance matrix (cosine)
    dist = cosine_distances(vec)
    # decide n_clusters via distance threshold to avoid tiny clusters
    # cap to MAX_CLUSTERS for readability
    clustering = AgglomerativeClustering(
        affinity='precomputed',
        linkage='average',
        distance_threshold=0.75,
        n_clusters=None,
    )
    clustering.fit(dist)
    # If too many clusters collapse smallest
    labels = clustering.labels_
    uniq = sorted(set(labels))
    if len(uniq) > MAX_CLUSTERS:
        # merge extras into nearest existing
        mapping = {old:i for i, old in enumerate(uniq[:MAX_CLUSTERS])}
        next_id = MAX_CLUSTERS - 1
        for old in uniq[MAX_CLUSTERS:]:
            mapping[old] = next_id
        labels = [mapping[l] for l in labels]
    return labels

def _build_graph(rows: list[tuple[str,float]], labels: list[int]) -> "nx.DiGraph":
    G = nx.DiGraph()
    G.add_node("root", size=0)           # phantom root
    clusters: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(idx)

    # Add internal nodes (clusters) and leaves (scenarios)
    for c_id, members in clusters.items():
        cluster_name = f"cluster_{c_id}"
        # mean reward for cluster size representation
        mean_r = sum(rows[i][1] for i in members) / len(members)
        G.add_node(cluster_name, size=mean_r)
        G.add_edge("root", cluster_name)
        for m in members:
            desc, reward = rows[m]
            node_id = f"leaf_{m}"
            G.add_node(node_id, label=desc, size=reward)
            G.add_edge(cluster_name, node_id)
    return G

def _draw(G: "nx.DiGraph") -> None:
    import matplotlib.pyplot as plt

    # prefer graphviz layout if available
    pos = None
    with contextlib.suppress(Exception):
        from networkx.drawing.nx_agraph import graphviz_layout
        pos = graphviz_layout(G, prog="dot")
    if pos is None:
        pos = nx.spring_layout(G, seed=0)

    sizes = [max(100, G.nodes[n]["size"]*5) for n in G.nodes]
    labels = {
        n: (G.nodes[n].get("label") or n.replace("cluster_", "C").replace("leaf_", "")) 
        for n in G.nodes if not n == "root"
    }

    # draw
    plt.figure(figsize=(10, 8), dpi=150)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#5e85f2", alpha=0.9)
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family="sans-serif")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_FILE, dpi=300)
    print(f"[OK] Phylogenetic tree saved → {OUT_FILE.resolve()}")
    plt.close()

# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    rows = _load_ledger()
    if not rows:
        print("[WARN] No scenarios found.")
        return
    sentences = [r[0] for r in rows]
    labels    = _cluster(sentences)
    G         = _build_graph(rows, labels)
    _draw(G)

if __name__ == "__main__":
    main()
