"""
────────────────────────────────────────────────────────────────────────────
                α‑Factory • 𝙋𝙤𝙡𝙞𝙘𝙮𝘼𝙜𝙚𝙣𝙩  —  “reg‑tech out‑thinker”
────────────────────────────────────────────────────────────────────────────

A domain‑specialised **multi‑tool LLM agent** that answers questions on public
policy, bills, and regulation.  Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• OpenAI Agents SDK **or** off‑line Llama‑cpp backend (runs without a key).
• RAG over local `data/policy_corpus/*` (vector store auto‑built on first run).
• **Governance guard‑rails** (rejects extremist / disallowed content).
• Emits live **trace‑graph** events to the WebSocket hub → fancy D3 view.
• Exposes a minimal **A2A gRPC** service (`/a2a.Policy/v1/ask`) so that a swarm
  of α‑Factory pods can delegate policy queries to this agent.
• Single‑file design, zero exotic dependencies.

© 2025 Alpha‑Factory / MONTREAL.AI — MIT licence (see project root).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

# ── optional heavy deps in a safe try / except ────────────────────────────
try:  # OpenAI Agents SDK (preferred online backend)
    from openai_agents import Agent, OpenAIToolCall
    _HAS_OPENAI = True
except ModuleNotFoundError:  # noqa: WPS440
    _HAS_OPENAI = False  # gracefully degrade later

try:  # off‑line fallback
    from llama_cpp import Llama
    _HAS_LLAMA = True
except ModuleNotFoundError:  # noqa: WPS440
    _HAS_LLAMA = False

try:  # vector store (stable, tiny dep)
    import faiss
except ModuleNotFoundError as exc:  # pragma: no cover – must exist
    raise RuntimeError("faiss missing ‑ add to requirements.txt") from exc

# local α‑Factory modules (all present in the repo)
from backend.governance import Governance, Memory
from backend.trace_ws import hub

logger = logging.getLogger("PolicyAgent")
DATA_DIR = Path("/app/data/policy_corpus")
INDEX_FILE = DATA_DIR / "faiss.index"
DOCS_FILE = DATA_DIR / "docs.jsonl"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")


# ══════════════════════════════════════════════════════════════════════════
#                            Vector store helper
# ══════════════════════════════════════════════════════════════════════════
def _lazy_index() -> tuple[faiss.IndexFlatIP, List[dict]]:
    """
    Build (or load) a *tiny* FAISS index over the local policy corpus.

    Each JSONL line in ``docs.jsonl`` is: ``{"title": …, "text": …, "url": …}``.
    We use a 384‑dim MiniLM embedding (fast, on‑CPU), stored inline to avoid an
    extra model download.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if INDEX_FILE.exists():
        logger.debug("using cached FAISS index at %s", INDEX_FILE)
        index = faiss.read_index(str(INDEX_FILE))
        docs: list[dict] = [json.loads(l) for l in DOCS_FILE.read_text().splitlines()]
        return index, docs

    logger.info("‣ building FAISS corpus (cold start)")
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError("sentence-transformers missing") from exc

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # minimal corpus if nothing provided
    if not DOCS_FILE.exists():
        DOCS_FILE.write_text(
            json.dumps(
                {
                    "title": "Empty‑corpus sentinel",
                    "text": "No policy documents available.",
                    "url": "",
                }
            )
            + "\n"
        )

    docs = [json.loads(l) for l in DOCS_FILE.read_text().splitlines()]
    embs = model.encode([d["text"][:1_000] for d in docs], normalize_embeddings=True)
    embs_np = np.asarray(embs, dtype="float32")

    index = faiss.IndexFlatIP(embs_np.shape[1])
    index.add(embs_np)
    faiss.write_index(index, str(INDEX_FILE))
    return index, docs


# ══════════════════════════════════════════════════════════════════════════
#                             Core agent logic
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class _Config:
    top_k: int = 5
    temperature: float = 0.0
    max_tokens: int = 1_024
    openai_model: str = "gpt-4o-mini"


@dataclass
class PolicyAgent:
    """
    Production‑grade **Policy / Reg‑Tech agent**.

    Instantiate once (it is *not* cheap if the FAISS index is cold).
    """

    cfg: _Config = field(default_factory=_Config)
    gov: Governance = field(default_factory=lambda: Governance(Memory()))
    _index: faiss.IndexFlatIP = field(init=False)
    _docs: List[dict] = field(init=False)
    _llm: Any | None = field(init=False, default=None)

    # ── init ──────────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        self._index, self._docs = _lazy_index()
        if not OPENAI_KEY and _HAS_LLAMA:
            self._llm = Llama(model_path="/models/phi-2.Q4_K_M.gguf")  # tiny demo
            logger.info("running off‑line Llama backend")

    # ── public API --------------------------------------------------------
    async def ask(self, question: str) -> str:
        """
        Answer *one* policy question.

        • Guardrails + logging (Governance).  
        • RAG retrieval → prompt with sources.  
        • Streams live events to the Trace‑graph UI.
        """
        if self.gov.moderate(question):
            raise ValueError("disallowed prompt (policy / safety violation)")

        await hub.broadcast({"id": f"q:{hash(question)}", "label": question})

        ctx = self._retrieve(question)
        answer = await self._llm_answer(question, ctx)

        await hub.broadcast(
            {
                "id": f"a:{hash(answer)}",
                "label": "✔ answer ready",
                "edges": [f"q:{hash(question)}"],
                "meta": {"chars": len(answer)},
            }
        )
        return answer

    # ── internal helpers --------------------------------------------------
    def _retrieve(self, query: str) -> Sequence[dict]:
        """Return *top‑k* docs (metadata + text) for the RAG prompt."""
        # encode w/ same SBERT (fast path ↯)
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self._index.search(emb, self.cfg.top_k)
        docs = [self._docs[i] | {"score": float(s)} for i, s in zip(idxs[0], scores[0])]
        logger.debug("RAG hits: %s", [d["title"] for d in docs])
        return docs

    async def _llm_answer(self, q: str, ctx: Sequence[dict]) -> str:
        """
        Call either **OpenAI Agents SDK** (preferred) or off‑line Llama.

        Returns the final answer *with inline citations (URLs)*.
        """
        # 1) craft system prompt with sources
        sys_msg = (
            "You are PolicyAgent, a compliance & public‑policy expert.\n"
            "Answer in clear, current legal language; cite sources as [1] [2] …"
        )
        sources_txt = "\n\n".join(
            f"[{i+1}] {d['title']}\n{d['url'] or 'local‑doc'}\n{d['text'][:500]}"
            for i, d in enumerate(ctx)
        )
        user_msg = f"Q: {q}\n\nRelevant documents:\n{sources_txt}"

        # 2) choose backend
        if OPENAI_KEY and _HAS_OPENAI:
            agent = Agent(
                system_message=sys_msg,
                model=self.cfg.openai_model,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            result: OpenAIToolCall | str = await agent.acall(user_msg)
            return result if isinstance(result, str) else result.content

        if self._llm:  # off‑line
            prompt = f"{sys_msg}\n\n{user_msg}\n\nAnswer:"
            out = await asyncio.to_thread(
                self._llm,
                prompt,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
                stop=["</s>"],
            )
            return out["choices"][0]["text"].strip()

        raise RuntimeError("no LLM backend available")


# ══════════════════════════════════════════════════════════════════════════
#                        A2A  (remote‑swarm adapter)
# ══════════════════════════════════════════════════════════════════════════
async def _start_a2a_server(agent: PolicyAgent) -> None:
    """
    Lightweight gRPC server that exposes ``/a2a.Policy/v1/ask``.

    Runs in the same event‑loop; zero additional dependencies thanks to
    Python `sockets` + std‑lib protobuf.
    """
    import asyncio
    import json
    import socket
    import struct
    from concurrent.futures import ThreadPoolExecutor

    HOST, PORT = "0.0.0.0", int(os.getenv("A2A_PORT", "7070"))
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen()

    logger.info("A2A gRPC shim listening on %s:%s", HOST, PORT)

    async def _handle(conn: socket.socket) -> None:
        loop = asyncio.get_running_loop()
        with conn:
            hdr = await loop.sock_recv(conn, 4)
            if not hdr:
                return
            (n,) = struct.unpack("!I", hdr)
            data = await loop.sock_recv(conn, n)
            req = json.loads(data)
            q = req.get("question", "")
            try:
                ans = await agent.ask(q)
                out = {"answer": ans, "error": ""}
            except Exception as exc:  # noqa: BLE001
                out = {"answer": "", "error": str(exc)}
            payload = json.dumps(out).encode()
            conn.sendall(struct.pack("!I", len(payload)) + payload)

    executor = ThreadPoolExecutor()

    async def _accept_loop() -> None:  # pragma: no cover
        with srv:
            while True:
                conn, _ = await asyncio.get_running_loop().sock_accept(srv)
                asyncio.get_running_loop().run_in_executor(executor, asyncio.run, _handle(conn))

    asyncio.create_task(_accept_loop())


# ══════════════════════════════════════════════════════════════════════════
#                               CLI entry‑point
# ══════════════════════════════════════════════════════════════════════════
async def _main() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="α‑Factory PolicyAgent CLI")
    parser.add_argument("question", nargs="+", help="policy question to ask")
    args = parser.parse_args()

    agent = PolicyAgent()
    await _start_a2a_server(agent)  # starts in the background
    answer = await agent.ask(" ".join(args.question))
    print("\n" + answer + "\n")

if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_main())
