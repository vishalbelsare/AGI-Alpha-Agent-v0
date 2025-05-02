"""
Unified interface to OpenAI, Anthropic or local llama-cpp.

    from utils.llm_provider import LLMProvider
    llm = LLMProvider()
    answer = llm.chat("Explain risk-parity like I'm five.")
"""

from __future__ import annotations
import os
from typing import List, Dict

class _OfflineModel:
    """Very small local model using llama-cpp; auto-downloads ggml quant."""
    def __init__(self) -> None:
        from llama_cpp import Llama
        model_path = os.getenv("LLAMA_MODEL_PATH", "ggml-model-q4.bin")
        if not os.path.exists(model_path):
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                "TheBloke/Llama-3-8B-Instruct-GGUF",
                filename="llama-3-8b-instruct.Q4_K_M.gguf",
            )
        self._llm = Llama(model_path=model_path, n_ctx=2048)

    def chat(self, prompt: str, **kw) -> str:
        res = self._llm(prompt, temperature=0.7, max_tokens=256)
        return res["choices"][0]["text"]

class LLMProvider:
    def __init__(self) -> None:
        self.offline = False
        if os.getenv("OPENAI_API_KEY"):
            import openai
            self.provider = "openai"
            self._client = openai.OpenAI()
        elif os.getenv("ANTHROPIC_API_KEY"):
            import anthropic
            self.provider = "anthropic"
            self._client = anthropic.Anthropic()
        else:
            self.provider = "offline"
            self._client = _OfflineModel()
            self.offline = True

    # unified chat method
    def chat(self, prompt: str, **kwargs) -> str:
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=kwargs.get("model", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 256),
            )
            return resp.choices[0].message.content
        elif self.provider == "anthropic":
            resp = self._client.messages.create(
                model=kwargs.get("model", "claude-3-opus-20240229"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get("max_tokens", 256),
                temperature=kwargs.get("temperature", 0.7),
            )
            return resp.content[0].text
        else:  # offline
            return self._client.chat(prompt, **kwargs)
