"""
backend/model_provider.py
------------------------

Unified wrapper that lets every agent call `ModelProvider.complete(prompt, **kw)`
without caring which LLM backend is available:

Priority
1. OpenAI  (env OPENAI_API_KEY)
2. Anthropic (env ANTHROPIC_API_KEY)
3. Local  – LiteLLM routed to an Ollama model (OpenHermes‑13B by default)
4. Stub   – always returns a deterministic fallback string so agents never crash
"""

import logging
import os
from typing import Any, Dict, List, Tuple


class ModelProvider:
    def __init__(self):
        self.log = logging.getLogger("ModelProvider")
        self.backend: Tuple[str, Any] | None = None
        self.model_name = "unknown"
        self._init_backend()

    # ─────────────────── internal helpers ───────────────────
    def _init_backend(self) -> None:
        """Detect best available backend; fall back to stub mode."""
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        try:
            if openai_key:
                import openai

                openai.api_key = openai_key
                self.backend = ("openai", openai)
                self.model_name = "gpt-4o-mini"
                self.log.info("Using OpenAI backend (%s).", self.model_name)
                return

            if anthropic_key:
                import anthropic

                self.backend = ("anthropic", anthropic.Client(anthropic_key))
                self.model_name = "claude-3-haiku-20240307"
                self.log.info("Using Anthropic backend (%s).", self.model_name)
                return

            # —— try local LiteLLM routed to Ollama ——
            try:
                import litellm  # noqa: F401

                self.backend = ("local", litellm)
                self.model_name = "ollama/openhermes-13b"
                self.log.info("Using local LiteLLM backend (%s).", self.model_name)
                return
            except Exception as err:  # litellm import or runtime error
                self.log.warning("Local backend unavailable: %s", err)

        except Exception as err:
            self.log.warning("Remote backend init failed: %s", err)

        # stub fallback
        self.backend = ("stub", None)
        self.model_name = "stub"
        self.log.warning("No LLM backend available; running in stub mode.")

    # ─────────────────── public API ───────────────────
    def complete(
        self,
        prompt: str,
        tools: List[Dict[str, Any]] | None = None,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        """
        Return model completion text.
        Guarantees a usable string even if all real backends fail.
        """
        kind, client = self.backend or ("stub", None)
        try:
            if kind == "openai":
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful agent."},
                        {"role": "user", "content": prompt},
                    ],
                    tools=tools or [],
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return resp.choices[0].message.content.strip()

            if kind == "anthropic":
                resp = client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return resp.content[0].text.strip()

            if kind == "local":
                resp = client.completion(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return resp["choices"][0]["message"]["content"].strip()

        except Exception as err:
            self.log.error("LLM call failed (%s); falling back to stub.", err)

        # —— stub fallback (deterministic) ——
        return '{"agent":"FinanceAgent","reason":"stub"}'

