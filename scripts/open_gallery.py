#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Open the Alpha-Factory demo gallery in a web browser.

This helper mirrors ``open_gallery.sh`` but uses Python for portability.
It attempts to open the published GitHub Pages gallery and falls back to a
local build under ``site/`` when offline.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from urllib.request import Request, urlopen
import webbrowser


def _gallery_url() -> str:
    remote = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], text=True).strip()
    repo_path = remote.split("github.com")[-1].lstrip(":/")
    repo_path = repo_path.removesuffix(".git")
    org, repo = repo_path.split("/", 1)
    return f"https://{org}.github.io/{repo}/gallery.html"


def _remote_available(url: str) -> bool:
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=3) as resp:
            status = getattr(resp, "status", None)
        return bool(status and 200 <= int(status) < 300)
    except Exception:
        return False


def main() -> None:
    url = _gallery_url()
    if _remote_available(url):
        print(f"Opening {url}")
        webbrowser.open(url)
        return
    repo_root = Path(__file__).resolve().parents[1]
    local_page = repo_root / "site" / "gallery.html"
    if local_page.is_file():
        print(f"Remote gallery unavailable. Opening local copy at {local_page}", file=sys.stderr)
        webbrowser.open(local_page.as_uri())
    else:
        print("Gallery not found. Build it with ./scripts/build_gallery_site.sh", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
