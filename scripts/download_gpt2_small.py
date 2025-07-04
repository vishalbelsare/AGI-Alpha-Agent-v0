#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# See docs/DISCLAIMER_SNIPPET.md
"""Retrieve the 124M GPT-2 checkpoint from OpenAI's public storage.

Files are fetched from ``https://openaipublic.blob.core.windows.net/gpt-2/models/124M/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from scripts import download_openai_gpt2


def download_model(dest: Path, model: str = "124M") -> None:
    """Download GPT-2 weights using the official OpenAI mirror."""
    download_openai_gpt2.download_openai_gpt2(model, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=Path, nargs="?", default=Path("models"), help="Target directory")
    parser.add_argument("--model", default="124M", help="GPT-2 model size")
    args = parser.parse_args()

    try:
        download_model(args.dest, args.model)
    except Exception as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
