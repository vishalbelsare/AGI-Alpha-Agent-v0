"""
alpha_factory_v1.backend.utils.llm_provider
===========================================

Unified, battle-tested interface to **ANY** large-language-model back-end
(OpenAI, Anthropic, Mistral, Together, Ollama, llama-cpp, HF endpoints…)
with *automatic graceful-degradation* to fully-offline quantised GGUF
models when no API keys or Internet are available.

Key design goals
----------------
1. **Single call-site** for every agent:  ``llm.chat("prompt")``.
2. **Zero downtime**: if the “best” provider fails or rate-limits, we
   transparently fall back down the provider ladder.
3. **Secure & economical**: 
   • secrets only via env vars • automatic token budgeting /
   truncation • optional local caching of completions.
4. **Observability**: Prometheus counters for tokens & calls; structured
   logging including latency.
5. **Extensible**: add new providers in <10 LOC (see `_providers.py`).

Example
-------
>>> from alpha_factory_v1.backend.utils.llm_provider import LLMProvider
>>> llm = LLMProvider()
>>> print(llm.chat("Explain risk-parity like I'm five."))
“I have five different kinds of sweets …”

CLI smoke test
$ python -m alpha_factory_v1.backend.utils.llm_provider \
    --prompt "Summarise Alpha-Factory in one tweet."
"""

from __future__ import annotations

# ───────────────────────── stdlib ──────────────────────────
import functools
import json
import logging
import os
import pathlib
import queue
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Generator, List, Literal, Optional

# ──────────────────────── soft-imports ─────────────────────
try:  # OpenAI
    import openai  # type: ignore
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

try:  # Anthropic
    import anthropic  # type: ignore
    _HAS_ANTHROPIC = True
except Exception:
    _HAS_ANTHROPIC = False

try:  # Mistral
    import mistralai  # type: ignore
    _HAS_MISTRAL = True
except Exception:
    _HAS_MISTRAL = False

try:  # Together
    import together  # type: ignore
    _HAS_TOGETHER = True
except Exception:
    _HAS_TOGETHER = False

try:  # HuggingFace text-generation-inference client
    from text_generation import Client as TGIClient  # type: ignore
    _HAS_TGI = True
except Exception:
    _HAS_TGI = False

try:  # llama-cpp for local inference
    from llama_cpp import Llama  # type: ignore
    _HAS_LLAMA = True
except Exception:
    _HAS_LLAMA = False

try:  # Prometheus metrics
    from prometheus_client import Counter, Histogram  # type: ignore
    _HAS_PROM = True
except Exception:
    _HAS_PROM = False

try:  # Token counting (OpenAI tiktoken)
    import tiktoken  # type: ignore
    _HAS_TOKENS = True
except Exception:
    _HAS_TOKENS = False

# ───────────────────────── logging ─────────────────────────
logger = logging.getLogger("alpha_factory.llm_provider")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ────────────────────── Prometheus metrics ─────────────────
if _HAS_PROM:
    _REQ_CT = Counter(
        "af_llm_requests_total",
        "LLM chat completion requests",
        labelnames=["provider", "status"],
    )
    _TOK_CT = Counter(
        "af_llm_tokens_total",
        "LLM tokens (prompt+completion)",
        labelnames=["provider"],
    )
    _LAT_HIST = Histogram(
        "af_llm_latency_seconds",
        "LLM round-trip latency",
        labelnames=["provider"],
    )
else:  # pragma: no cover
    class _NoOp:  # noqa: D401
        def labels(self, *a, **k):  # noqa: D401
            return self
        def inc(self, *_a, **_k): ...
        def observe(self, *_a, **_k): ...

    _REQ_CT = _TOK_CT = _LAT_HIST = _NoOp()  # type: ignore

# ─────────────────── helper: token counting ────────────────
@functools.lru_cache(maxsize=4)
def _encoder(model: str = "cl100k_base"):
    if not _HAS_TOKENS:
        return None
    return tiktoken.get_encoding(model)

def _count_tokens(text: str) -> int:
    enc = _encoder()
    return len(enc.encode(text)) if enc else len(text.split())

# ─────────────────── provider registry ---------------------
@dataclass
class _ProviderCfg:
    name: str
    enabled: bool
    weight: int  # priority (lower is better)
    handler: "Provider"

class Provider:
    """Abstract provider base."""

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        stream: bool,
        stop: Optional[List[str]],
    ) -> str | Generator[str, None, None]:
        raise NotImplementedError

# ───────────────────────── providers ───────────────────────
class _OpenAI(Provider):
    def __init__(self) -> None:
        self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def chat(self, messages, *, temperature, max_tokens, stream, stop):
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        kwargs = dict(model=model,
                      messages=messages,
                      temperature=temperature,
                      max_tokens=max_tokens,
                      stop=stop or None,
                      stream=stream)
        tic = time.perf_counter()
        try:
            if stream:
                resp = self._client.chat.completions.create(**kwargs)
                for chunk in resp:
                    yield chunk.choices[0].delta.content or ""
            else:
                resp = self._client.chat.completions.create(**kwargs)
                out = resp.choices[0].message.content
                _TOK_CT.labels("openai").inc(resp.usage.total_tokens)
                return out
        finally:
            _LAT_HIST.labels("openai").observe(time.perf_counter() - tic)

class _Anthropic(Provider):
    def __init__(self) -> None:
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def chat(self, messages, *, temperature, max_tokens, stream, stop):
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-opus-20240229")
        # convert message format
        claude_messages = [
            {"role": m["role"], "content": m["content"]} for m in messages
        ]
        tic = time.perf_counter()
        try:
            if stream:
                resp = self._client.messages.create(
                    model=model,
                    messages=claude_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop or None,
                    stream=True,
                )
                for chunk in resp:
                    yield chunk.delta.text or ""
            else:
                resp = self._client.messages.create(
                    model=model,
                    messages=claude_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop_sequences=stop or None,
                )
                txt = resp.content[0].text
                _TOK_CT.labels("anthropic").inc(resp.usage.input_tokens +
                                                resp.usage.output_tokens)
                return txt
        finally:
            _LAT_HIST.labels("anthropic").observe(time.perf_counter() - tic)

class _Mistral(Provider):
    def __init__(self) -> None:
        self._client = mistralai.MistralClient(
            api_key=os.getenv("MISTRAL_API_KEY", "")
        )

    def chat(self, messages, *, temperature, max_tokens, stream, stop):
        model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        tic = time.perf_counter()
        try:
            resp = self._client.chat(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop or None,
                stream=stream,
            )
            if stream:
                for chunk in resp:
                    yield chunk.choices[0].delta.content or ""
            else:
                out = resp.choices[0].message.content
                return out
        finally:
            _LAT_HIST.labels("mistral").observe(time.perf_counter() - tic)

class _Together(Provider):
    def __init__(self) -> None:
        self._client = together.Together(api_key=os.getenv("TOGETHER_API_KEY", ""))

    def chat(self, messages, *, temperature, max_tokens, stream, stop):
        model = os.getenv("TOGETHER_MODEL", "mistralai/Mixtral-8x22B-Instruct-v0.1")
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop or None,
            stream=stream,
        )
        tic = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(**kwargs)
            if stream:
                for chunk in resp:
                    yield chunk.choices[0].delta.content or ""
            else:
                return resp.choices[0].message.content
        finally:
            _LAT_HIST.labels("together").observe(time.perf_counter() - tic)

class _TGI(Provider):
    """HuggingFace text-generation-inference server (self-hosted)."""

    def __init__(self) -> None:
        endpoint = os.getenv("TGI_ENDPOINT", "http://localhost:8080")
        self._client = TGIClient(endpoint)

    def chat(self, messages, *, temperature, max_tokens, stream, stop):
        prompt = "\n".join(m["content"] for m in messages)
        tic = time.perf_counter()
        resp = self._client.generate_stream(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop or [],
            stream=stream,
        )
        try:
            if stream:
                for chunk in resp:
                    yield chunk.token.text
            else:
                out = "".join(tok.token.text for tok in resp)
                return out
        finally:
            _LAT_HIST.labels("tgi").observe(time.perf_counter() - tic)

class _LlamaCPP(Provider):
    """Fully offline local model via llama-cpp-python."""

    def __init__(self) -> None:
        model_path = os.getenv(
            "LLAMA_MODEL_PATH",
            # sensible default tiny 3-B quant (auto-download)
            str(pathlib.Path.home() / ".cache" / "llama" /
                "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf"),
        )
        if not pathlib.Path(model_path).exists():
            logger.info("Downloading TinyLlama quant to %s …", model_path)
            import huggingface_hub as hf  # type: ignore
            fp = hf.hf_hub_download(
                repo_id="TheBloke/TinyLlama-1.1B-Chat-GGUF",
                filename=os.path.basename(model_path),
            )
            pathlib.Path(fp).rename(model_path)
        self._llm = Llama(
            model_path=model_path,
            n_ctx=int(os.getenv("LLAMA_N_CTX", 2048)),
            n_threads=max(1, os.cpu_count() // 2),
        )

    def chat(self, messages, *, temperature, max_tokens, stream, stop):
        prompt = "\n".join(m["content"] for m in messages)
        kwargs = dict(temperature=temperature,
                      max_tokens=max_tokens,
                      stop=stop)
        tic = time.perf_counter()
        out = self._llm(prompt, **kwargs)
        _LAT_HIST.labels("llama").observe(time.perf_counter() - tic)
        return out["choices"][0]["text"]

# ──────────────────── provider selection -------------------
_PROVIDERS: List[_ProviderCfg] = []

def _register(name: str, cond: bool, weight: int, cls):
    if cond:
        _PROVIDERS.append(_ProviderCfg(name, True, weight, cls()))

_register("openai", _HAS_OPENAI and bool(os.getenv("OPENAI_API_KEY")), 0, _OpenAI)
_register("anthropic", _HAS_ANTHROPIC and bool(os.getenv("ANTHROPIC_API_KEY")), 1,
          _Anthropic)
_register("mistral", _HAS_MISTRAL and bool(os.getenv("MISTRAL_API_KEY")), 2,
          _Mistral)
_register("together", _HAS_TOGETHER and bool(os.getenv("TOGETHER_API_KEY")), 3,
          _Together)
_register("tgi", _HAS_TGI and bool(os.getenv("TGI_ENDPOINT")), 4, _TGI)
_register("llama", _HAS_LLAMA, 5, _LlamaCPP)

# sort by priority
_PROVIDERS.sort(key=lambda p: p.weight)

if not _PROVIDERS:
    raise RuntimeError(
        "No LLM provider available (no API keys + no llama-cpp). "
        "Set e.g. OPENAI_API_KEY or install llama-cpp-python."
    )

# ─────────────────────── in-memory cache ───────────────────
_CACHE: Dict[str, Dict[str, str]] = {}  # {provider: {prompt_hash: completion}}

def _hash(messages: List[Dict[str, str]]) -> str:
    return str(hash(json.dumps(messages, sort_keys=True)))

# ─────────────────────── main interface ────────────────────
class LLMProvider:
    """Facade used by every agent (`llm = LLMProvider()`)."""

    def __init__(self, *, temperature: float = 0.7, max_tokens: int = 512) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ------------------------------------------------------
    def chat(
        self,
        prompt: str | List[Dict[str, str]],
        *,
        system_prompt: str | None = None,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
    ) -> str | Generator[str, None, None]:
        """Chat completion w/ provider cascade & caching.

        Parameters
        ----------
        prompt :
            Either raw user text or already-formatted messages list.
        system_prompt :
            Optional system instruction (prepended).
        stream :
            If *True*, returns a generator yielding tokens.
        """
        messages: List[Dict[str, str]]
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt

        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages

        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        # quick token sanity check
        prompt_toks = sum(_count_tokens(m["content"]) for m in messages)
        if prompt_toks > 12000:
            logger.warning("Prompt length %d tokens > 12k, truncating oldest msgs.",
                           prompt_toks)
            while prompt_toks > 12000 and len(messages) > 1:
                messages.pop(1)
                prompt_toks = sum(_count_tokens(m["content"]) for m in messages)

        for cfg in _PROVIDERS:
            cache_hit = None
            h = _hash(messages)
            if use_cache and not stream:
                cache_hit = _CACHE.get(cfg.name, {}).get(h)
            if cache_hit:
                _REQ_CT.labels(cfg.name, "cache").inc()
                return cache_hit

            try:
                _REQ_CT.labels(cfg.name, "attempt").inc()
                out = cfg.handler.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    stop=stop,
                )
                if not stream:
                    _TOK_CT.labels(cfg.name).inc(_count_tokens(out))  # type: ignore[arg-type]
                    if use_cache:
                        _CACHE.setdefault(cfg.name, {})[h] = out  # type: ignore[index]
                return out
            except Exception as exc:  # pragma: no cover
                logger.warning("Provider %s failed: %s", cfg.name, exc)

        raise RuntimeError("All LLM providers failed.")

# ────────────────────────── CLI demo ───────────────────────
if __name__ == "__main__":  # pragma: no cover
    import argparse, textwrap

    ap = argparse.ArgumentParser("llm_provider smoke-test")
    ap.add_argument("--prompt", required=True, help="User prompt text")
    ap.add_argument("--system", help="System prompt")
    ap.add_argument("--stream", action="store_true")
    ns = ap.parse_args()

    llm = LLMProvider()
    if ns.stream:
        print("Streaming:")
        for tok in llm.chat(ns.prompt, system_prompt=ns.system, stream=True):
            print(tok, end="", flush=True)
        print()
    else:
        out = llm.chat(ns.prompt, system_prompt=ns.system)
        print(textwrap.fill(out, width=100))
