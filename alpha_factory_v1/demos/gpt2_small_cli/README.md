[See docs/DISCLAIMER_SNIPPET.md](../../../docs/DISCLAIMER_SNIPPET.md)
This repository is a conceptual research prototype. References to "AGI" and "superintelligence" describe aspirational goals and do not indicate the presence of a real general intelligence. Use at your own risk. Nothing herein constitutes financial advice. MontrealAI and the maintainers accept no liability for losses incurred from using this software.

# GPT‑2 Small CLI Demo

This minimal example downloads the official OpenAI GPT‑2 124M checkpoint using
`scripts/download_openai_gpt2.py`. The weights are converted to the Hugging Face
format via `scripts/convert_openai_gpt2.py` on first run. If PyTorch is
unavailable, the demo falls back to the hosted `gpt2` model from the
`transformers` hub.

```bash
python -m alpha_factory_v1.demos.gpt2_small_cli --prompt "The future of AI" --max-length 50
```
