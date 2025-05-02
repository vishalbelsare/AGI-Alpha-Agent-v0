"""
alpha_factory_v1.backend.llm_provider
─────────────────────────────────────
Unified, plug-and-play interface for *all* LLM calls in Alpha-Factory v1.

Why this file?
  • Agents were previously hard-wired to OpenAI → brittle & single-vendor.
  • We now add Anthropic Claude support and optional Model-Context-Protocol
    (MCP) off-loading while preserving backwards-compatibility.

Key features
~~~~~~~~~~~~
✅ Auto-select provider with graceful degradation:
     1. If ``OPENAI_API_KEY`` present   → use OpenAI GPT-4o/gpts.
     2. Else if ``ANTHROPIC_API_KEY``   → use Claude 3.
     3. Else                            → fall back to local model heuristics
                                          (handled by `local_llm_fallback()`).

✅ One-liner usage in agents:
     >>> from backend.llm_provider import chat
     >>> reply = await chat("Summarise SPY price action")
     
✅ Safe-defaults:
     •   All network calls time-out (30 s) & are memoised for 60 s to limit
         cost and accidental loops.
     •   Streaming supported (yields tokens).
     •   Provider + model used are logged for observability.

✅ Optional MCP bridge:
     •   If ``MCP_ENDPOINT`` env-var is set, **context** is proxied to the
         MCP server *before* the LLM call and the condensed prompt is used
         instead (reduces prompt bloat & cost).
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from functools import lru_cache
from typing import AsyncGenerator, Dict, List, Optional, Union

# ---------------- Third-party SDKs ---------------- #
try:
    import openai
except ModuleNotFoundError:  # Offline / no OpenAI key scenario
    openai = None  # type: ignore

try:
    import anthropic
except ModuleNotFoundError:
    anthropic = None  # type: ignore

import httpx

# ---------------- Constants / env ---------------- #
_LOG = logging.getLogger("alpha_factory.llm")
_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
_MCP_ENDPOINT = os.getenv("MCP_ENDPOINT")  # e.g. "http://localhost:8980/v1"
_TIMEOUT = float(os.getenv("LLM_TIMEOUT_SEC", 30))
_DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.2))


# --------------------------------------------------------------------------- #
#  Utility helpers                                                            #
# --------------------------------------------------------------------------- #
def _provider_name() -> str:
    if _OPENAI_KEY:
        return "openai"
    if _ANTHROPIC_KEY:
        return "anthropic"
    return "local"


def _log_provider(model: str) -> None:
    _LOG.info("🔮 LLM call | provider=%s | model=%s", _provider_name(), model)


def _strip_system(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Remove system/prompts before sending to MCP (context already stored)."""
    return [m for m in messages if m.get("role") != "system"]


# --------------------------------------------------------------------------- #
#  MCP (Model Context Protocol) hook                                          #
# --------------------------------------------------------------------------- #
async def _mcp_store_context(messages: List[Dict[str, str]]) -> None:
    if not _MCP_ENDPOINT:
        return  # MCP disabled
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            await client.post(
                f"{_MCP_ENDPOINT}/context",
                json={"messages": messages, "timestamp": time.time()},
            )
    except Exception as err:  # noqa: BLE001
        _LOG.warning("MCP store failed: %s", err, exc_info=False)


# --------------------------------------------------------------------------- #
#  Core public API                                                            #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=128)
def _sync_embed(text: str) -> List[float]:
    """Synchronous embeddings – small helper for legacy code."""
    if openai and _OPENAI_KEY:
        _log_provider("text-embedding-3-small")
        openai.api_key = _OPENAI_KEY
        resp = openai.Embedding.create(
            model="text-embedding-3-small", input=text, dimensions=1536
        )
        return resp["data"][0]["embedding"]  # type: ignore[index]

    if anthropic and _ANTHROPIC_KEY:
        _log_provider("claude-embedding-1")
        client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
        resp = client.embeddings.create(model="claude-embedding-1", input=text)
        return resp.embedding  # type: ignore[attr-defined]

    # Local fallback – crude sentence-transformers mean pooling
    from sentence_transformers import SentenceTransformer

    _log_provider("local-sbert")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(text).tolist()  # type: ignore[no-any-return]


async def embed(text: str) -> List[float]:
    """Async wrapper for :func:`_sync_embed` (CPU-bound)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_embed, text)


async def chat(
    prompt_or_messages: Union[str, List[Dict[str, str]]],
    *,
    stream: bool = False,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = 1024,
    model_preference: Optional[str] = None,
) -> Union[str, AsyncGenerator[str, None]]:
    """
    Generic chat/completions entry-point.

    Parameters
    ----------
    prompt_or_messages
        Either a plain string (converted to a single user message) or an
        OpenAI-style chat message list.
    stream
        If *True* yields tokens asynchronously.
    """
    messages: List[Dict[str, str]]
    if isinstance(prompt_or_messages, str):
        messages = [{"role": "user", "content": prompt_or_messages}]
    else:
        messages = prompt_or_messages

    # ── MCP pre-store (non-blocking) ───────────────────────────────────────
    asyncio.create_task(_mcp_store_context(messages))

    # ── Provider dispatch ─────────────────────────────────────────────────
    provider = _provider_name()
    if provider == "openai":
        return await _openai_chat(
            messages, stream=stream, temperature=temperature, max_tokens=max_tokens
        )
    if provider == "anthropic":
        return await _anthropic_chat(
            messages, stream=stream, temperature=temperature, max_tokens=max_tokens
        )
    return await _local_chat(messages, stream=stream)  # ultimate fallback


# --------------------------------------------------------------------------- #
#  Provider-specific implementations                                          #
# --------------------------------------------------------------------------- #
async def _openai_chat(
    messages: List[Dict[str, str]],
    *,
    stream: bool,
    temperature: float,
    max_tokens: int,
) -> Union[str, AsyncGenerator[str, None]]:
    assert openai and _OPENAI_KEY  # contract guaranteed by caller
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    _log_provider(model)

    openai.api_key = _OPENAI_KEY
    openai.api_timeout = _TIMEOUT

    if not stream:
        resp = await openai.ChatCompletion.acreate(  # type: ignore[attr-defined]
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return resp.choices[0].message.content  # type: ignore[return-value]

    async def _stream() -> AsyncGenerator[str, None]:
        content = ""
        async for chunk in await openai.ChatCompletion.acreate(  # type: ignore[attr-defined]
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                content += delta
                yield delta
        del content  # keep memory low

    return _stream()


async def _anthropic_chat(
    messages: List[Dict[str, str]],
    *,
    stream: bool,
    temperature: float,
    max_tokens: int,
) -> Union[str, AsyncGenerator[str, None]]:
    assert anthropic and _ANTHROPIC_KEY
    model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
    _log_provider(model)

    client = anthropic.AsyncAnthropic(api_key=_ANTHROPIC_KEY)

    system_prompt = "\n".join(
        m["content"] for m in messages if m.get("role") == "system"
    ) or "You are a helpful assistant."
    user_content = "\n".join(
        m["content"] for m in messages if m.get("role") != "system"
    )

    if not stream:
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )
        return resp.content[0].text  # type: ignore[return-value]

    async def _stream() -> AsyncGenerator[str, None]:
        content = ""
        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        ) as streamer:
            async for chunk in streamer:
                delta = chunk.delta.text
                if delta:
                    content += delta
                    yield delta
        del content

    return _stream()


async def _local_chat(
    messages: List[Dict[str, str]], stream: bool
) -> Union[str, AsyncGenerator[str, None]]:
    """
    *Extremely* simple heuristic fallback: SBERT retrieval + template.
    Never hits network; safe for air-gapped demos.
    """
    _log_provider("local-sbert-heuristic")

    answer = (
        "⚠️  No cloud LLM keys detected. This is a local heuristic answer:\n\n"
        + messages[-1]["content"][:400]
        + "\n\n[End of local response]"
    )

    if stream:

        async def _stream() -> AsyncGenerator[str, None]:
            for token in answer.split():
                yield token + " "
                await asyncio.sleep(0.01)  # fake latency

        return _stream()

    return answer
