# SPDX-License-Identifier: Apache-2.0
import pytest
from pathlib import Path

pw = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright


def test_send_beacon_opt_in() -> None:
    dist = Path(__file__).resolve().parents[1] / "dist" / "index.html"
    url = dist.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.evaluate(
            "window.OTEL_ENDPOINT='https://example.com';"
            "window.confirm=() => true;"
            "navigator.sendBeacon=(...a)=>{window.beacon=a;return true;}"
        )
        page.reload()
        page.wait_for_selector("#controls")
        page.click("text=Share")
        page.evaluate("window.dispatchEvent(new Event('beforeunload'))")
        assert page.evaluate("Array.isArray(window.beacon)")
        browser.close()


def test_session_id_hashed() -> None:
    dist = Path(__file__).resolve().parents[1] / "dist" / "index.html"
    url = dist.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.evaluate(
            "window.OTEL_ENDPOINT='https://example.com';"
            "window.confirm=() => true;"
            "navigator.sendBeacon=(...a)=>{window.beacon=a;return true;}"
        )
        page.reload()
        page.wait_for_selector("#controls")
        page.click("text=Share")
        page.evaluate("window.dispatchEvent(new Event('beforeunload'))")
        payload = page.evaluate("window.beacon[1]")
        import json

        metrics = json.loads(payload)
        assert "session" in metrics
        assert isinstance(metrics["session"], str)
        assert len(metrics["session"]) == 64
        browser.close()


def test_session_id_deterministic() -> None:
    dist = Path(__file__).resolve().parents[1] / "dist" / "index.html"
    url = dist.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.evaluate(
            "window.OTEL_ENDPOINT='https://example.com';"
            "window.confirm=() => true;"
            "navigator.sendBeacon=(...a)=>{window.beacon=a;return true;}"
        )
        page.reload()
        page.wait_for_selector("#controls")
        page.click("text=Share")
        page.evaluate("window.dispatchEvent(new Event('beforeunload'))")
        first = page.evaluate("window.beacon[1]")
        page.reload()
        page.evaluate("navigator.sendBeacon=(...a)=>{window.beacon=a;return true;}")
        page.wait_for_selector("#controls")
        page.click("text=Share")
        page.evaluate("window.dispatchEvent(new Event('beforeunload'))")
        second = page.evaluate("window.beacon[1]")
        import json

        assert json.loads(first)["session"] == json.loads(second)["session"]
        browser.close()


def test_offline_queue_flushes_on_reconnect() -> None:
    dist = Path(__file__).resolve().parents[1] / "dist" / "index.html"
    url = dist.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.evaluate(
            "window.OTEL_ENDPOINT='https://example.com';"
            "window.confirm=() => true;"
            "Object.defineProperty(navigator,'onLine',{get:()=>false,configurable:true});"
            "navigator.sendBeacon=()=>false;"
        )
        page.reload()
        page.wait_for_selector("#controls")
        page.click("text=Share")
        page.evaluate("window.dispatchEvent(new Event('beforeunload'))")
        assert page.evaluate("JSON.parse(localStorage.getItem('telemetryQueue')).length > 0")
        page.evaluate(
            "navigator.sendBeacon=(...a)=>{(window.sent=window.sent||[]).push(a);return true;}"
            "Object.defineProperty(navigator,'onLine',{get:()=>true});"
            "window.dispatchEvent(new Event('online'));"
        )
        page.wait_for_function("window.sent && window.sent.length > 0")
        assert page.evaluate("localStorage.getItem('telemetryQueue')") == "[]"
        browser.close()


def test_queue_limit_and_fetch_fallback() -> None:
    dist = Path(__file__).resolve().parents[1] / "dist" / "index.html"
    url = dist.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.evaluate(
            "window.OTEL_ENDPOINT='https://example.com';"
            "window.confirm=() => true;"
            "navigator.sendBeacon=()=>false;"
            "Object.defineProperty(navigator,'onLine',{get:()=>false,configurable:true});"
            "localStorage.setItem('telemetryQueue',JSON.stringify(Array.from({length:100},()=>({}))))"
        )
        page.reload()
        page.wait_for_selector("#controls")
        page.click("text=Share")
        page.evaluate("window.dispatchEvent(new Event('beforeunload'))")
        assert page.evaluate("JSON.parse(localStorage.getItem('telemetryQueue')).length") == 100
        page.evaluate(
            "window.fetch=(...a)=>{window.fetchArgs=a;return Promise.resolve({status:200});};"
            "Object.defineProperty(navigator,'onLine',{get:()=>true});"
            "window.dispatchEvent(new Event('online'));"
        )
        page.wait_for_function("window.fetchArgs !== undefined")
        assert page.evaluate("localStorage.getItem('telemetryQueue')") == "[]"
        browser.close()


def test_queue_never_exceeds_cap() -> None:
    dist = Path(__file__).resolve().parents[1] / "dist" / "index.html"
    url = dist.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        page.evaluate(
            "window.OTEL_ENDPOINT='https://example.com';"
            "window.confirm=() => true;"
            "navigator.sendBeacon=()=>false;"
            "Object.defineProperty(navigator,'onLine',{get:()=>false,configurable:true});"
        )
        page.reload()
        page.wait_for_selector("#controls")
        for _ in range(105):
            page.click("text=Share")
            page.evaluate("window.dispatchEvent(new Event('beforeunload'))")
        assert page.evaluate("JSON.parse(localStorage.getItem('telemetryQueue')).length") == 100
        browser.close()


def test_localstorage_failure_disables_telemetry() -> None:
    dist = Path(__file__).resolve().parents[1] / "dist" / "index.html"
    url = dist.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto(url)
        page.evaluate(
            "window.OTEL_ENDPOINT='https://example.com';"
            "window.confirm=() => true;"
            "Object.defineProperty(localStorage,'setItem',{value:()=>{throw new Error('fail');},configurable:true});"
        )
        page.reload()
        page.wait_for_selector("#controls")
        page.evaluate("window.dispatchEvent(new Event('beforeunload'))")
        assert not errors
        browser.close()


def test_no_uncaught_error_on_setitem_failure() -> None:
    dist = Path(__file__).resolve().parents[1] / "dist" / "index.html"
    url = dist.as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        page.goto(url)
        page.evaluate(
            "window.OTEL_ENDPOINT='https://example.com';"
            "window.confirm=() => true;"
            "Object.defineProperty(localStorage,'setItem',{value:()=>{throw new Error('boom');},configurable:true});"
        )
        page.reload()
        page.wait_for_selector("#controls")
        page.evaluate("window.dispatchEvent(new Event('beforeunload'))")
        assert not errors
        browser.close()
