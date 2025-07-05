# SPDX-License-Identifier: Apache-2.0
"""Verify integrity attribute for the service worker registration script."""
from __future__ import annotations

from pathlib import Path
import hashlib
import base64
import re

BROWSER = Path(__file__).resolve().parents[1] / "alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1"


def sha384(path: Path) -> str:
    digest = hashlib.sha384(path.read_bytes()).digest()
    return "sha384-" + base64.b64encode(digest).decode()


def test_service_worker_integrity() -> None:
    dist = BROWSER / "dist"
    html = (dist / "index.html").read_text()
    match = re.search(r'<script[^>]*src=["\']service-worker.js["\'][^>]*>', html)
    assert match, "service-worker.js script tag missing"
    tag = match.group(0)
    integrity = re.search(r'integrity=["\']([^"\']+)["\']', tag)
    assert integrity, "integrity attribute missing"
    expected = sha384(dist / "sw.js")
    assert integrity.group(1) == expected
