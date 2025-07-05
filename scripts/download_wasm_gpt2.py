#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Download the GPT‑2 small weights used by the browser demo.

This helper mirrors :mod:`scripts.download_hf_gpt2` but defaults to the
``wasm_llm`` directory so the Insight browser can run offline. The files are
retrieved from the official Hugging Face repository unless
``HF_GPT2_BASE_URL`` points to a different mirror.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from scripts import download_hf_gpt2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dest",
        nargs="?",
        default="alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1/wasm_llm",
        type=Path,
        help="Destination directory",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=3,
        help="Number of download attempts",
    )
    args = parser.parse_args()

    try:
        download_hf_gpt2.download_hf_gpt2(args.dest, attempts=args.attempts)
    except Exception as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
