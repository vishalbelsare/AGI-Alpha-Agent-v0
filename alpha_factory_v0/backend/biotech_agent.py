"""
BiotechAgent – mini RAG over a lightweight knowledge‑graph
=========================================================

*   Graph = RDF triples (stored in Turtle *.ttl*), loaded via `rdflib`.
*   Embedding store → FAISS for similarity search of node/edge docs.
*   Works **offline** (Sentence‑Transformers) or **online** (OpenAI embeddings).
*   Returns answers with **inline citations** back to the source URI.

Env‑vars
--------
BIOTECH_KG_FILE          defaults to ``data/biotech_graph.ttl``
OPENAI_API_KEY           enables OpenAI embeddings + GPT inference
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import List

import numpy as np
import rdflib

try:
    import faiss
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("faiss‑cpu missing — add to requirements.txt") from exc

try:  # online
    from openai import AsyncOpenAI
except ModuleNotFoundError:
    AsyncOpenAI = None  # type: ignore

from sentence_transformers import SentenceTransformer

_KG_FILE = Path(os.getenv("BIOTECH_KG_FILE", "data/biotech_graph.ttl"))
_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
_EMBED_DIM = 384
_SB_MODEL = SentenceTransformer("nomic-embed-text") if not _OPENAI else None

# --------------------------------------------------------------------- #
# Load KG & build index                                                 #
# --------------------------------------------------------------------- #
def _load_kg() -> tuple[faiss.IndexFlatIP, List[str]]:
    g = rdflib.Graph()
    g.parse(_KG_FILE)

    docs: List[str] = []
    for s, p, o in g:
        docs.append(f"{s} {p} {o}")

    if not docs:
        docs.append("Empty Bio‑KG")  # sentinel to avoid zero‑row index

    vecs = asyncio.run(_embed(docs))
    index = faiss.IndexFlatIP(_EMBED_DIM)
    index.add(vecs)
    return index, docs


async def _embed(texts: List[str]) -> np.ndarray:
    if _OPENAI:
        client = AsyncOpenAI()
        resp = await client.embeddings.create(
            input=texts,
            model="text-embedding-3-small",
            encoding_format="float",
        )
        vecs = np.array([d.embedding for d in resp.data], dtype="float32")
    else:
        loop = asyncio.get_event_loop()
        vecs = await loop.run_in_executor(None, _SB_MODEL.encode, texts)
        vecs = vecs.astype("float32")
    # L2 normalise for IP search
    faiss.normalize_L2(vecs)
    return vecs


_FAISS, _DOCS = _load_kg()

# --------------------------------------------------------------------- #
# Agent                                                                 #
# --------------------------------------------------------------------- #
class BiotechAgent:
    """Minimalist, embeddable biotech RAG agent."""

    def __init__(self, top_k: int = 5) -> None:
        self.k = top_k
        self._client = AsyncOpenAI() if _OPENAI else None

    # ------------------------------------------------------------------ #
    async def answer(self, query: str) -> str:
        hits = await self._search(query)
        context = "\n\n".join(f"[{i+1}] {h}" for i, h in enumerate(hits))

        if _OPENAI:
            chat = await self._client.chat.completions.create(  # type: ignore
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a biotech research assistant. "
                        "Answer with citations like [1] [2] …",
                    },
                    {"role": "user", "content": f"{query}\n\nContext:\n{context}"},
                ],
                temperature=0,
                max_tokens=700,
            )
            return chat.choices[0].message.content.strip()

        # offline fallback
        bullets = "\n".join(f"- {h[:120]}…" for h in hits)
        return f"**(offline)** Possible answer for *{query}*\n\n{bullets}"

    # ------------------------------------------------------------------ #
    async def _search(self, query: str) -> List[str]:
        q = await _embed([query])
        scores, idx = _FAISS.search(q, self.k)
        return [_DOCS[i] for i in idx[0] if i != -1]


# CLI convenience
if __name__ == "__main__":
    import sys, asyncio  # noqa: E402

    q = " ".join(sys.argv[1:]) or "Describe the role of p53 in DNA repair."
    print(asyncio.run(BiotechAgent().answer(q)))
