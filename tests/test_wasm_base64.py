# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import pytest

pw = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402
from playwright._impl._errors import Error as PlaywrightError  # noqa: E402


def test_pyodide_base64_global() -> None:
    dist = Path(__file__).resolve().parents[1] / (
        "alpha_factory_v1/demos/alpha_agi_insight_v1/insight_browser_v1/dist/index.html"
    )
    url = dist.as_uri()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(url)
            page.wait_for_selector("#controls")
            val = page.evaluate("window.PYODIDE_WASM_BASE64")
            browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser not installed: {exc}")
    assert val, "PYODIDE_WASM_BASE64 not set"
