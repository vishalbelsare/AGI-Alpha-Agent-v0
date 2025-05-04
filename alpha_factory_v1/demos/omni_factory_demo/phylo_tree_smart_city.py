#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""OMNI‑Factory · Smart‑City Scenario Phylogenetic Tree
════════════════════════════════════════════════════════
Visualise the *evolutionary landscape* of disruption scenarios tackled by
**Alpha‑Factory v1** as a radial phylogenetic‑style tree.  Each leaf node
represents a scenario sentence stored in ``omni_ledger.sqlite`` (or sample
data if absent).  Branches are created via semantic agglomerative clustering
(TF‑IDF by default, or ``sentence‑transformers`` / OpenAI embeddings if
available) and their radii encode the *economic value* (average reward).

Highlights
──────────
 • Zero external writes – read‑only access to the SQLite ledger.
 • Runs *offline‑first* (no API keys); uses stronger embeddings if present.
 • Graphviz ‘dot’ layout when available; falls back to deterministic spring.
 • PNG & SVG output   → ``phylo_tree.[png|svg]`` next to the script.
 • Optional CLI switches   → ``--db path``  ``--out path``  ``--format svg``.

Usage
─────
    pip install matplotlib networkx scikit-learn
    # (optional) pip install sentence-transformers
    python phylo_tree_smart_city.py               # PNG, default TF‑IDF

    python phylo_tree_smart_city.py --format svg  # high‑res SVG

Environment
───────────
OMNI_EMBEDDING = "tfidf" | "sbert" | "openai"      (default: auto)

The script auto‑selects the best embedding backend in this order:
(1) OpenAI embeddings (if key + internet) → (2) sentence‑transformers
→ (3) TF‑IDF fallback.  You can pin a backend via the env‑var above.

This file is self‑contained, PEP‑517 compliant, and platform‑agnostic.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses as _dc
import datetime as _dt
import math
import os
import pathlib
import sqlite3
import sys
import typing as _t

# ─────────────────────────────────────────────────────────────────────────────
# Optional third‑party imports
# ─────────────────────────────────────────────────────────────────────────────
_MISSING: list[str] = []
with contextlib.suppress(ImportError):
    import matplotlib.pyplot as _plt          # type: ignore
if '_plt' not in globals():
    _MISSING.append('matplotlib')

with contextlib.suppress(ImportError):
    import networkx as _nx                    # type: ignore
if '_nx' not in globals():
    _MISSING.append('networkx')

with contextlib.suppress(ImportError):
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.cluster import AgglomerativeClustering          # type: ignore
    from sklearn.metrics.pairwise import cosine_distances        # type: ignore
else:
    _SKLEARN_OK = True
if 'TfidfVectorizer' not in globals():
    _MISSING.append('scikit-learn')
    _SKLEARN_OK = False

# Optional stronger embeddings
with contextlib.suppress(ImportError):
    from sentence_transformers import SentenceTransformer         # type: ignore
    import numpy as _np                                           # type: ignore
    _SBERT_OK = True
else:
    _SBERT_OK = 'SentenceTransformer' in globals()

with contextlib.suppress(ImportError):
    import openai                                                 # type: ignore
    _OPENAI_OK = True
else:
    _OPENAI_OK = 'openai' in globals()

# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────
@_dc.dataclass(slots=True)
class Config:
    db_path:   pathlib.Path
    out_path:  pathlib.Path
    img_fmt:   str = 'png'
    backend:   str = 'auto'
    max_clusters: int = 8

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def _fatal(msg: str, exit_code: int = 1) -> None:  # pragma: no cover
    """Print error + exit (graceful for non‑tech users)."""
    print(f'[ERROR] {msg}')
    sys.exit(exit_code)


def _parse_args() -> Config:
    parser = argparse.ArgumentParser(
        prog='phylo_tree_smart_city',
        description='Visualise Smart‑City scenario “evolution” as a tree.',
    )
    parser.add_argument('--db',   help='SQLite ledger path',
                        default='omni_ledger.sqlite')
    parser.add_argument('--out',  help='Output image file',
                        default='phylo_tree.png')
    parser.add_argument('--format', choices=('png', 'svg'),
                        default='png', help='Image format (png|svg)')
    parser.add_argument('--backend', choices=('auto', 'tfidf', 'sbert', 'openai'),
                        default=os.getenv('OMNI_EMBEDDING', 'auto'),
                        help='Embedding backend override')
    parser.add_argument('--clusters', type=int, default=8,
                        help='Soft maximum number of clusters')
    args = parser.parse_args()

    return Config(
        db_path=pathlib.Path(args.db).expanduser(),
        out_path=pathlib.Path(args.out).with_suffix(f'.{args.format}'),
        img_fmt=args.format,
        backend=args.backend,
        max_clusters=args.clusters,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────
def _load_ledger(db_path: pathlib.Path) -> list[tuple[str, float]]:
    """Return list of (sentence, avg_reward) rows."""
    if not db_path.exists():
        # fallback sample scenarios
        return [
            ('Flash‑flood closes two bridges at rush hour', 42.0),
            ('Cyber‑attack on traffic‑light network',        55.0),
            ('Record heatwave threatens rolling brown‑outs', 65.0),
            ('Protest blocks downtown core',                 51.0),
        ]
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cur = conn.execute('SELECT scenario, avg_reward FROM ledger')
            rows = cur.fetchall()
        return rows
    except sqlite3.Error as exc:
        _fatal(f'SQLite error: {exc}')
    return []  # unreachable

# ─────────────────────────────────────────────────────────────────────────────
# Embedding backends
# ─────────────────────────────────────────────────────────────────────────────
def _embed_tfidf(sentences: list[str]) -> '_np.ndarray':
    if not _SKLEARN_OK:
        _fatal('scikit‑learn is required for TF‑IDF backend.')
    vec = TfidfVectorizer().fit_transform(sentences)
    return vec.toarray()


def _embed_sbert(sentences: list[str]) -> '_np.ndarray':
    if not _SBERT_OK:
        _fatal('sentence‑transformers not installed.')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)


def _embed_openai(sentences: list[str]) -> '_np.ndarray':
    if not _OPENAI_OK:
        _fatal('openai python package missing.')
    if not os.getenv('OPENAI_API_KEY'):
        _fatal('OPENAI_API_KEY not set for OpenAI backend.')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    embeddings = []
    for s in sentences:
        resp = openai.Embedding.create(model='text-embedding-3-small',
                                       input=s)
        embeddings.append(resp['data'][0]['embedding'])
    import numpy as _np
    return _np.array(embeddings, dtype='float32')


def _select_backend(cfg: Config, sentences: list[str]) -> '_np.ndarray':
    """Return sentence embeddings using the chosen/auto backend."""
    back = cfg.backend
    if back == 'auto':
        # prefer openai > sbert > tfidf
        back = ('openai' if _OPENAI_OK and os.getenv('OPENAI_API_KEY')
                else 'sbert' if _SBERT_OK
                else 'tfidf')
    if back == 'openai':
        return _embed_openai(sentences)
    if back == 'sbert':
        return _embed_sbert(sentences)
    return _embed_tfidf(sentences)

# ─────────────────────────────────────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────────────────────────────────────
def _cluster_vectors(vecs: '_np.ndarray', max_clusters: int) -> list[int]:
    """Hierarchical clustering → cluster labels."""
    if vecs.shape[0] <= 2:
        return [0] * vecs.shape[0]
    dist = cosine_distances(vecs)
    clust = AgglomerativeClustering(affinity='precomputed',
                                    linkage='average',
                                    n_clusters=None,
                                    distance_threshold=0.55)
    labels = clust.fit_predict(dist)
    # reduce tiny clusters > max_clusters
    uniq = sorted(set(labels))
    if len(uniq) > max_clusters:
        mapping = {u: i if i < max_clusters else max_clusters-1
                   for i, u in enumerate(uniq)}
        labels = [mapping[l] for l in labels]
    return labels

# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────
def _build_graph(rows: list[tuple[str, float]], labels: list[int]) -> '_nx.DiGraph':
    G = _nx.DiGraph()
    G.add_node('root', size=0.0)
    clusters: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(idx)
    for cid, mem in clusters.items():
        cname = f'cluster_{cid}'
        mean_reward = sum(rows[i][1] for i in mem) / len(mem)
        G.add_node(cname, size=mean_reward)
        G.add_edge('root', cname)
        for i in mem:
            desc, reward = rows[i]
            nid = f'leaf_{i}'
            G.add_node(nid, size=reward, label=desc)
            G.add_edge(cname, nid)
    return G

# ─────────────────────────────────────────────────────────────────────────────
# Drawing
# ─────────────────────────────────────────────────────────────────────────────
def _layout_graph(G: '_nx.DiGraph') -> dict[str, tuple[float, float]]:
    """Return node → (x,y) positions, prefer hierarchical layout."""
    # attempt graphviz:
    with contextlib.suppress(Exception):
        from networkx.drawing.nx_agraph import graphviz_layout
        return graphviz_layout(G, prog='dot')
    # deterministic spring layout
    return _nx.spring_layout(G, seed=42)  # type: ignore


def _draw(G: '_nx.DiGraph', out_path: pathlib.Path, fmt: str) -> None:
    pos = _layout_graph(G)
    sizes = [max(120, G.nodes[n]['size'] * 6) for n in G.nodes]
    labels = {n: G.nodes[n].get('label', n.replace('cluster_', 'C'))
              for n in G.nodes if n != 'root'}

    _plt.figure(figsize=(11, 8), dpi=200)
    _nx.draw_networkx_edges(G, pos, alpha=0.3, width=1.2)
    _nx.draw_networkx_nodes(G, pos, node_size=sizes,
                            node_color='#4e79ff', alpha=0.85)
    _nx.draw_networkx_labels(G, pos, labels,
                             font_size=8, font_family='sans-serif')
    _plt.title('Smart‑City Scenario Phylogenetic Tree  '
               f'({len([n for n in G.nodes if n.startswith("leaf")])} scenarios)',
               fontsize=10)
    _plt.axis('off')
    _plt.tight_layout()
    _plt.savefig(out_path, format=fmt, dpi=300)
    print(f'[OK] Tree saved → {out_path.resolve()}')
    _plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Main entry‑point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    if _MISSING:
        _fatal('Missing required packages: ' + ', '.join(_MISSING))
    cfg = _parse_args()
    rows = _load_ledger(cfg.db_path)
    if not rows:
        _fatal('No scenarios found – ledger is empty.')
    sentences = [r[0] for r in rows]

    vecs = _select_backend(cfg, sentences)
    labels = _cluster_vectors(vecs, cfg.max_clusters)
    G = _build_graph(rows, labels)
    _draw(G, cfg.out_path, cfg.img_fmt)


if __name__ == '__main__':
    main()
