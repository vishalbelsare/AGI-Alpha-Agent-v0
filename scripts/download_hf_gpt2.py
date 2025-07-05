#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# See docs/DISCLAIMER_SNIPPET.md
"""Download the Hugging Face GPT-2 checkpoint files.

Downloads files from ``https://huggingface.co/openai-community/gpt2/resolve/main/``
by default. The base URL can be overridden via ``HF_GPT2_BASE_URL``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import requests  # type: ignore[import-untyped]
from requests.adapters import HTTPAdapter, Retry  # type: ignore[import-untyped]
from tqdm import tqdm


def _session() -> requests.Session:
    """Return a session with basic retry logic."""
    retry = Retry(total=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


_FILES = [
    "pytorch_model.bin",
    "vocab.json",
    "merges.txt",
    "config.json",
]

CHECKSUMS = {
    "pytorch_model.bin": "7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421",
    "vocab.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    "merges.txt": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "config.json": "0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4",
}


def _base_url() -> str:
    return os.environ.get(
        "HF_GPT2_BASE_URL",
        "https://huggingface.co/openai-community/gpt2/resolve/main",
    ).rstrip("/")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with _session().get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        with open(dest, "wb") as fh, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))


def _verify(dest: Path) -> None:
    expected = CHECKSUMS.get(dest.name)
    if expected:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"Checksum mismatch for {dest.name}")


def download_hf_gpt2(dest: Path | str = "models/gpt2", attempts: int = 3) -> None:
    dest_dir = Path(dest)
    base = _base_url()
    last_exc: Exception | None = None
    for name in _FILES:
        url = f"{base}/{name}"
        target = dest_dir / name
        if target.exists():
            print(f"{target} already exists, skipping")
            continue
        for i in range(1, attempts + 1):
            try:
                print(f"Downloading {url} to {target} (attempt {i})")
                _download(url, target)
                _verify(target)
                break
            except Exception as exc:  # noqa: PERF203
                last_exc = exc
                if i < attempts:
                    print(f"Attempt {i} failed: {exc}, retrying...")
                else:
                    print(f"ERROR: could not download {url}: {exc}")
                    if target.exists():
                        try:
                            target.unlink()
                        except Exception:
                            pass
    if last_exc:
        raise last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dest", type=Path, nargs="?", default=Path("models/gpt2"), help="Target directory")
    args = parser.parse_args()
    try:
        download_hf_gpt2(args.dest)
    except Exception as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
