"""
PolicyAgent
===========

GPT‑driven Retrieval‑Augmented agent that answers questions about
statutes, regulations and policy documents.

Features
--------
* Vector‑store retrieval (FAISS, 384‑d *text‑embedding‑3‑small* by default)
* OpenAI Agents SDK **tool interface** so any planner can call it.
* Works **offline** (no OPENAI_API_KEY) via a local embedding model
  ``nomic‑embed‑text`` from *sentence‑transformers*.
* Async & streaming friendly.

Env‑vars
--------
STATUTE_CORPUS_DIR   path with .txt/.md docs (default: ./corpus/statutes)
STATUTE_INDEX_PATH   FAISS index file (default: ./corpus/index.faiss)
OPENAI_API_KEY       optional – enables OpenAI embeddings + GPT‑4o
OPENAI_MODEL         chat model (default gpt‑4o-preview)
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, List

import numpy as np

# ── optional heavy deps wrapped in try/except so the rest of the code loads ──
try:
    from openai.agents import Tool, Agent
    from openai import AsyncOpenAI
except ModuleNotFoundError:  # pragma: no cover
    Tool = Agent = AsyncOpenAI = object  # type: ignore

try:
    import faiss
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("✗ `faiss‑cpu` missing ‑ add it to requirements.txt") from exc

# ───────────────────────── configuration ────────────────────────────────
CORPUS_DIR = Path(os.getenv("STATUTE_CORPUS_DIR", "corpus/statutes"))
INDEX_PATH = Path(os.getenv("STATUTE_INDEX_PATH", "corpus/index.faiss"))
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-preview")
EMBED_DIM = 384

_HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

# ───────────────────────── embedding helpers ────────────────────────────
if _HAS_OPENAI:
    _oai = AsyncOpenAI()

    async def _embed(texts: List[str]) -> np.ndarray:
        resp = await _oai.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
            encoding_format="float",
        )
        return np.array([d.embedding for d in resp.data], dtype="float32")
else:  # offline SBERT
    from sentence_transformers import SentenceTransformer

    _sbert = SentenceTransformer("nomic-embed-text")

    async def _embed(texts: List[str]) -> np.ndarray:  # type: ignore
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sbert.encode, texts)

# ───────────────────────── FAISS index build/load ───────────────────────
def _load_index() -> tuple[faiss.IndexFlatIP, List[str]]:
    if INDEX_PATH.exists():
        idx = faiss.read_index(str(INDEX_PATH))
        docs = INDEX_PATH.with_suffix(".txt").read_text().split("\n\n###\n")
        return idx, docs

    docs: List[str] = [
        p.read_text()
        for p in CORPUS_DIR.rglob("*")
        if p.suffix.lower() in {".txt", ".md"}
    ]
    if not docs:
        raise RuntimeError(f"No policy docs found under {CORPUS_DIR.resolve()}")

    vecs = asyncio.run(_embed(docs))
    idx = faiss.IndexFlatIP(EMBED_DIM)
    idx.add(vecs)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(idx, str(INDEX_PATH))
    INDEX_PATH.with_suffix(".txt").write_text("\n\n###\n".join(docs))
    return idx, docs


_FAISS, _DOCS = _load_index()

# ───────────────────────── retrieval tool ───────────────────────────────
class StatuteSearch(Tool):
    name = "statute_search"
    description = (
        "Finds the most relevant legal / statutory passages for a query. "
        "Returns `[{text, score}, …]` sorted by score."
    )

    async def invoke(self, query: str, k: int = 5) -> List[dict[str, Any]]:  # type: ignore
        qv = await _embed([query])
        scores, idxs = _FAISS.search(qv, k)  # type: ignore
        hits = [
            {"text": _DOCS[i][:1_000], "score": float(s)}
            for s, i in zip(scores[0], idxs[0])
            if i != -1
        ]
        return hits


# ───────────────────────── policy agent wrapper ─────────────────────────
class PolicyAgent:
    """Simple wrapper that turns retrieval + LLM into one call."""

    def __init__(self) -> None:
        self._tool = StatuteSearch()
        prompt = (
            "You are **PolicyAgent**, a legal research assistant. "
            "Use `statute_search` to quote authoritative passages. "
            "Reply concisely and add bullet citations."
        )
        if _HAS_OPENAI:
            self._agent = Agent(model=CHAT_MODEL, tools=[self._tool], system=prompt)
        else:
            self._agent = None  # offline fallback uses tool directly

    # Public API ---------------------------------------------------------
    async def answer(self, q: str) -> str:
        if self._agent:
            msg = await self._agent.submit(q)
            return msg.content or "🛈 empty"
        hits = await self._tool.invoke(q, 3)
        bullets = "\n".join(f"- {h['text'][:120]}…" for h in hits)
        return f"*(offline)* Best‑effort answer for: **{q}**\n\n{bullets}"

    # Allow other planners to import us as a Tool
    class _QATool(Tool):
        name = "policy_qa"
        description = "Answers legal / policy questions using a RAG pipeline."

        def __init__(self, agent: "PolicyAgent"):
            self._agent = agent

        async def invoke(self, q: str) -> str:  # type: ignore
            return await self._agent.answer(q)

    def as_tool(self) -> Tool:
        return self._QATool(self)


# ───────────────────────── CLI helper (manual test) ─────────────────────
if __name__ == "__main__":
    import sys, asyncio  # noqa: E402

    question = " ".join(sys.argv[1:]) or "What is the GDPR article on data portability?"
    print(asyncio.run(PolicyAgent().answer(question)))
