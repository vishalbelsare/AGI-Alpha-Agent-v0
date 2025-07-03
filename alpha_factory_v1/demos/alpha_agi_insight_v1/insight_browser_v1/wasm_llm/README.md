[See docs/DISCLAIMER_SNIPPET.md](../../../../../docs/DISCLAIMER_SNIPPET.md)
This repository is a conceptual research prototype. References to "AGI" and "superintelligence" describe aspirational goals and do not indicate the presence of a real general intelligence. Use at your own risk. Nothing herein constitutes financial advice. MontrealAI and the maintainers accept no liability for losses incurred from using this software.

# wasm-gpt2

This folder is intentionally empty. During development the lightweight `wasm-gpt2` model (~124 MB) should be placed here and pinned to IPFS. The build script copies the contents of this directory to `dist/wasm_llm/` so the demo can load the model offline.

To fetch the model automatically, run `npm run fetch-assets` or
`python ../../../../scripts/fetch_assets.py`.
The helper retrieves `wasm-gpt2.tar` from the official mirror. Set
`WASM_GPT2_URL` to override the default URL:

```bash
export WASM_GPT2_URL="https://cloudflare-ipfs.com/ipfs/bafybeihdwdcefgh4dqkjv67uzcmw7ojee6xedzdetojuzjevtenxquvyku?download=1"
npm run fetch-assets
```

The default URL points to the canonical IPFS CID. The script automatically falls
back to the configured IPFS gateway if the mirror is unavailable.
