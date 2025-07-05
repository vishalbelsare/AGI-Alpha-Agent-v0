# SPDX-License-Identifier: Apache-2.0
import pytest
from pathlib import Path

pw = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright  # noqa: E402
from playwright._impl._errors import Error as PlaywrightError  # noqa: E402


def test_metrics_update_during_sim() -> None:
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
            page.wait_for_selector("#simulator-panel")
            page.evaluate("document.querySelector('#simulator-panel #sim-gen').value=2")
            page.evaluate("document.querySelector('#simulator-panel #sim-pop').value=3")
            page.click("#simulator-panel #sim-start")
            page.wait_for_function("document.querySelector('#worker-time').textContent.includes('ms')")
            page.wait_for_function("document.querySelector('#heap').textContent !== ''")
            page.wait_for_function("document.querySelector('#fps-value').textContent.includes('fps')")
            page.click("#simulator-panel #sim-cancel")
            browser.close()
    except PlaywrightError as exc:
        pytest.skip(f"Playwright browser not installed: {exc}")
