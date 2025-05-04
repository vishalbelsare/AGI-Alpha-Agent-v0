#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""OMNI‑Factory · Smart‑City Scenario Phylogenetic Tree
════════════════════════════════════════════════════════════════════════════
Visualise the *evolutionary landscape* of disruption scenarios archived by
**Alpha‑Factory v1** as a phylogenetic‑style tree.

The tree helps non‑technical stakeholders quickly grasp how the platform’s
open‑ended learning has branched into diverse *scenario lineages* (clusters)
and where the most valuable "alpha" has been discovered so far.

Key features
────────────
• **Offline‑first** — runs with zero external writes or internet and *does not*
  require an API key.  If online and `OPENAI_API_KEY` is present, it can use
  OpenAI embeddings for higher‑fidelity semantics.
• **Auto‑adaptive embeddings** — chooses the strongest locally available
  backend in the order *OpenAI → sentence‑transformers (SBERT) → TF‑IDF*.
• **Radial *or* hierarchical layout** — leverages Graphviz when available
  (`dot` or `twopi`).  Falls back to deterministic NetworkX spring layout.
• **Rich CLI** — power‑user flags but sane defaults for one‑shot execution:

    $ python phylo_tree_smart_city.py                 # PNG, auto backend
    $ python phylo_tree_smart_city.py --format svg    # high‑res SVG
    $ python phylo_tree_smart_city.py --radial        # radial layout

• **Safe by design** — read‑only SQL, graceful error messages for
  non‑technical operators, small memory footprint (<100 MB).
• **PEP 517 compliant** — pure Python ≥3.9, portable across Windows/macOS/Linux.

Outputs
───────
Two high‑resolution images next to the script (unless overridden):
    • ``phylo_tree.png`` — bitmap (300 DPI)
    • ``phylo_tree.svg`` — vector graphics (ideal for slide‑decks)

The PNG is always written; SVG is optional via ``--format svg`` or detected if
Graphviz radial layout is selected.

Usage (quick‑start)
───────────────────
>>> pip install "matplotlib>=3.5" networkx scikit-learn
>>> # optional (stronger embeddings)
>>> pip install sentence-transformers openai numpy
>>> python phylo_tree_smart_city.py

Environment variable overrides
──────────────────────────────
OMNI_DB_PATH      path to SQLite ledger (default: ./omni_ledger.sqlite)
OMNI_OUT_DIR      directory for output images    (default: script dir)
OMNI_EMBEDDING    tfidf | sbert | openai         (default: auto)
OMNI_CLUSTER_MAX  positive int — max clusters    (default: 8)
OMNI_LAYOUT       radial | hier                  (default: auto)

"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses as _dc
import os
import pathlib
import sqlite3
import sys
import typing as _t
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────
# Optional third‑party imports (gracefully degraded if missing)
# ──────────────────────────────────────────────────────────────────────────
_MISSING: list[str] = []

try:
    import matplotlib.pyplot as _plt  # type: ignore
except ImportError:
    _MISSING.append("matplotlib")

try:
    import networkx as _nx  # type: ignore
except ImportError:
    _MISSING.append("networkx")

# sklearn is mandatory *unless* the user forces a non‑tfidf backend
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.cluster import AgglomerativeClustering  # type: ignore
    from sklearn.metrics.pairwise import cosine_distances  # type: ignore
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    _MISSING.append("scikit‑learn")

# optional backends
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    import numpy as _np  # type: ignore
    _SBERT_OK = True
except ImportError:
    _SBERT_OK = False

try:
    import openai  # type: ignore
    import numpy as _np  # ensured above if SBERT OK; else re‑import later
    _OPENAI_PKG = True
except ImportError:
    _OPENAI_PKG = False

# ──────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ──────────────────────────────────────────────────────────────────────────
@_dc.dataclass(slots=True)
class Config:
    db_path: pathlib.Path
    out_dir: pathlib.Path
    img_fmt: str
    backend: str  # auto|tfidf|sbert|openai
    max_clusters: int
    radial: bool

    @property
    def png_path(self) -> pathlib.Path:
        return self.out_dir / "phylo_tree.png"

    @property
    def svg_path(self) -> pathlib.Path:
        return self.out_dir / "phylo_tree.svg"


# ──────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────

def _fatal(msg: str, code: int = 1) -> None:  # pragma: no cover
    print(f"[ERROR] {msg}")
    sys.exit(code)


def _log(msg: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")


# ──────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────

def _parse_args() -> Config:
    p = argparse.ArgumentParser(
        prog="phylo_tree_smart_city",
        description="Visualise Smart‑City scenario evolution as a phylogenetic tree.",
    )
    p.add_argument("--db", dest="db", default=os.getenv("OMNI_DB_PATH", "omni_ledger.sqlite"),
                   help="SQLite ledger path (default: ./omni_ledger.sqlite)")
    p.add_argument("--outdir", dest="outdir", default=os.getenv("OMNI_OUT_DIR", "."),
                   help="Output directory for images (default: script dir)")
    p.add_argument("--format", dest="fmt", choices=("png", "svg"),
                   default="png", help="Primary image format (default: png)")
    p.add_argument("--backend", choices=("auto", "tfidf", "sbert", "openai"),
                   default=os.getenv("OMNI_EMBEDDING", "auto"),
                   help="Embedding backend override")
    p.add_argument("--clusters", type=int, default=int(os.getenv("OMNI_CLUSTER_MAX", "8")),
                   help="Soft max number of clusters (default: 8)")
    p.add_argument("--radial", action="store_true", default=os.getenv("OMNI_LAYOUT") == "radial",
                   help="Force radial layout (Graphviz twopi if available)")

    a = p.parse_args()
    return Config(
        db_path=pathlib.Path(a.db).expanduser().resolve(),
        out_dir=pathlib.Path(a.outdir).expanduser().resolve(),
        img_fmt=a.fmt,
        backend=a.backend,
        max_clusters=max(2, a.clusters),
        radial=a.radial,
    )


# ──────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────

def _load_ledger(db: pathlib.Path) -> list[tuple[str, float]]:
    """Return list of (scenario_sentence, avg_reward). Provide canned demo if missing."""
    if not db.exists():
        _log("Ledger not found — using demo data.")
        return [
            ("Flash‑flood closes two bridges at rush hour", 42.0),
            ("Cyber‑attack on traffic‑light network", 55.0),
            ("Record heatwave threatens rolling brown‑outs", 65.0),
            ("Protest blocks downtown core", 51.0),
        ]
    try:
        with sqlite3.connect(str(db)) as conn:
            cur = conn.execute("SELECT scenario, avg_reward FROM ledger")
            rows = cur.fetchall()
        return rows
    except sqlite3.Error as exc:
        _fatal(f"Unable to read SQLite ledger: {exc}")
        return []  # for static type‑checkers


# ──────────────────────────────────────────────────────────────────────────
# Embedding backends
# ──────────────────────────────────────────────────────────────────────────

def _embed_tfidf(sentences: list[str]):
    if not _SKLEARN_OK:
        _fatal("scikit‑learn is required for TF‑IDF backend.")
    vec = TfidfVectorizer().fit_transform(sentences)
    return vec.toarray()


def _embed_sbert(sentences: list[str]):
    if not _SBERT_OK:
        _fatal("sentence‑transformers not installed — try 'pip install sentence-transformers'.")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)


def _embed_openai(sentences: list[str]):
    if not _OPENAI_PKG:
        _fatal("openai package not installed — try 'pip install openai'.")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        _fatal("OPENAI_API_KEY not set for OpenAI backend.")
    openai.api_key = key
    _log("Fetching OpenAI embeddings … this may incur costs.")
    resp = openai.Embedding.create(model="text-embedding-3-small", input=sentences)
    # ensure consistent order
    embeddings = [d["embedding"] for d in sorted(resp["data"], key=lambda x: x["index"]) ]
    import numpy as _np  # local import if previously missing
    return _np.array(embeddings, dtype="float32")


def _select_backend(cfg: Config, sentences: list[str]):
    back = cfg.backend
    if back == "auto":
        back = (
            "openai" if _OPENAI_PKG and os.getenv("OPENAI_API_KEY") else
            "sbert"   if _SBERT_OK else
            "tfidf"
        )
    _log(f"Embedding backend → {back.upper()}")
    if back == "openai":
        return _embed_openai(sentences)
    if back == "sbert":
        return _embed_sbert(sentences)
    return _embed_tfidf(sentences)


# ──────────────────────────────────────────────────────────────────────────
# Clustering
# ──────────────────────────────────────────────────────────────────────────

def _cluster(vecs, k_max: int):
    if vecs.shape[0] <= 2:
        return [0] * vecs.shape[0]
    dist = cosine_distances(vecs)
    cl = AgglomerativeClustering(
        affinity="precomputed",
        linkage="average",
        n_clusters=None,
        distance_threshold=0.55,
    ).fit(dist)
    labels = cl.labels_.tolist()
    # merge if clusters > k_max
    uniq = sorted(set(labels))
    if len(uniq) > k_max:
        mapping = {u: (i if i < k_max else k_max - 1) for i, u in enumerate(uniq)}
        labels = [mapping[l] for l in labels]
    return labels


# ──────────────────────────────────────────────────────────────────────────
# Graph building & drawing
# ──────────────────────────────────────────────────────────────────────────

def _build_graph(rows: list[tuple[str, float]], labels) -> "_nx.DiGraph":
    G = _nx.DiGraph()
    G.add_node("root", size=0.0)
    clusters: dict[int, list[int]] = {}
    for idx, lab in enumerate(labels):
        clusters.setdefault(lab, []).append(idx)

    for cid, members in clusters.items():
        cname = f"cluster_{cid}"
        mean_reward = sum(rows[i][1] for i in members) / len(members)
        G.add_node(cname, size=mean_reward)
        G.add_edge("root", cname)
        for m in members:
            desc, reward = rows[m]
            lid = f"leaf_{m}"
            G.add_node(lid, size=reward, label=desc)
            G.add_edge(cname, lid)
    return G


def _layout_graph(G: "_nx.DiGraph", radial: bool):
    # prefer Graphviz via pygraphviz/pydot
    if radial:
        with contextlib.suppress(Exception):
            from networkx.drawing.nx_agraph import graphviz_layout  # type: ignore
            return graphviz_layout(G, prog="twopi", args="-Goverlap=false")
    else:
        with contextlib.suppress(Exception):
            from networkx.drawing.nx_agraph import graphviz_layout  # type: ignore
            return graphviz_layout(G, prog="dot")
    # fallback deterministic spring
    return _nx.spring_layout(G, seed=42)  # type: ignore


def _draw(G: "_nx.DiGraph", cfg: Config) -> None:
    pos = _layout_graph(G, cfg.radial)
    sizes = [max(120, G.nodes[n]["size"] * 6) for n in G.nodes]
    labels = {
        n: G.nodes[n].get("label", n.replace("cluster_", "C"))
        for n in G.nodes if n != "root"
    }
    _plt.figure(figsize=(11, 8), dpi=220)
    _nx.draw_networkx_edges(G, pos, alpha=0.35, width=1.2)
    _nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="#4e79ff", alpha=0.85)
    _nx.draw_networkx_labels(G, pos, labels, font_size=8, font_family="sans-serif")
    _plt.title(
        f"Smart‑City Scenario Phylogenetic Tree — {len(labels)} scenarios",
        fontsize=10,
    )
    _plt.axis("off")
    _plt.tight_layout()

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    _plt.savefig(cfg.png_path, format="png", dpi=300)
    _log(f"PNG saved → {cfg.png_path}")
    if cfg.img_fmt == "svg" or cfg.svg_path.suffix == ".svg":
        _plt.savefig(cfg.svg_path, format="svg")
        _log(f"SVG saved → {cfg.svg_path}")
    _plt.close()


# ──────────────────────────────────────────────────────────────────────────
# Main routine
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:  # pragma: no cover
    if missing := [m for m in _MISSING if m in {"matplotlib", "networkx"}]:
        _fatal("Missing required packages: " + ", ".join(missing))

    cfg = _parse_args()
    rows = _load_ledger(cfg.db_path)
    if not rows:
        _fatal("Ledger contains no scenarios.")

    sentences = [r[0] for r in rows]
    vecs = _select_backend(cfg, sentences)
    labels = _cluster(vecs, cfg.max_clusters)
    G = _build_graph(rows, labels)
    _draw(G, cfg)
    _log("Done.")


if __name__ == "__main__":
    main()
