#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Open the Alpha-Factory demo gallery in a web browser.

This helper mirrors ``open_gallery.sh`` but uses Python for portability.  It
attempts to open the published GitHub Pages gallery and falls back to a local
build under ``site/`` when offline.  If the local build is missing, the script
automatically runs ``scripts/build_gallery_site.sh`` to generate the site so
non‑technical users can access the demos with a single command.  When serving
the local copy, a lightweight HTTP server is spawned to preserve all
functionality such as service workers and relative assets.
"""
from __future__ import annotations

from pathlib import Path

try:
    from alpha_factory_v1.utils.disclaimer import DISCLAIMER
except Exception:  # pragma: no cover - fallback when package not installed
    _DOCS_PATH = Path(__file__).resolve().parents[1] / "docs" / "DISCLAIMER_SNIPPET.md"
    DISCLAIMER = _DOCS_PATH.read_text(encoding="utf-8").strip()

import os
import subprocess
import sys
from urllib.request import Request, urlopen
import webbrowser
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from functools import partial
import threading


def _build_local_site(repo_root: Path) -> bool:
    """Return ``True`` if the gallery was built successfully."""
    script = repo_root / "scripts" / "build_gallery_site.sh"
    if not script.is_file():
        return False
    try:
        subprocess.run([str(script)], check=True)
    except Exception:
        return False
    return True


def _gallery_url() -> str:
    env = os.environ.get("AF_GALLERY_URL")
    if env:
        return env.rstrip("/") + "/index.html"
    remote = subprocess.check_output(["git", "config", "--get", "remote.origin.url"], text=True).strip()
    repo_path = remote.split("github.com")[-1].lstrip(":/")
    repo_path = repo_path.removesuffix(".git")
    org, repo = repo_path.split("/", 1)
    return f"https://{org}.github.io/{repo}/index.html"


def _remote_available(url: str) -> bool:
    try:
        req = Request(url, method="HEAD")
        with urlopen(req, timeout=3) as resp:
            status = getattr(resp, "status", None)
        return bool(status and 200 <= int(status) < 300)
    except Exception:
        return False


def main(*, print_only: bool = False) -> None:
    """Open the demo gallery or print the URL if ``print_only`` is True."""
    print(DISCLAIMER, file=sys.stderr)
    url = _gallery_url()
    if _remote_available(url):
        if print_only:
            print(url)
            return
        print(f"Opening {url}")
        webbrowser.open(url)
        return
    repo_root = Path(__file__).resolve().parents[1]
    site_dir = repo_root / "site"
    local_page = site_dir / "index.html"
    if not local_page.is_file():
        print("Remote gallery unavailable. Building local copy...", file=sys.stderr)
        if not _build_local_site(repo_root) or not local_page.is_file():
            print(
                "Gallery not found. Build it with ./scripts/build_gallery_site.sh",
                file=sys.stderr,
            )
            sys.exit(1)

    handler = partial(SimpleHTTPRequestHandler, directory=str(site_dir))
    with ThreadingHTTPServer(("127.0.0.1", 0), handler) as httpd:
        port = httpd.server_address[1]
        url = f"http://127.0.0.1:{port}/index.html"
        print(
            f"Remote gallery unavailable. Serving local copy at {url}",
            file=sys.stderr,
        )

        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            if print_only:
                print(url)
            else:
                webbrowser.open(url)
            thread.join()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Open the demo gallery")
    parser.add_argument(
        "--print-url",
        action="store_true",
        help="Only print the gallery URL instead of launching a browser",
    )
    args = parser.parse_args()
    main(print_only=args.print_url)
