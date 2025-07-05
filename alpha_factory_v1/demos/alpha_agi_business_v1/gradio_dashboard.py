#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Simple Gradio dashboard for the Alpha‑AGI Business demo."""
from __future__ import annotations

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - offline shim
    from alpha_factory_v1 import af_requests as requests  # type: ignore

import gradio as gr

HOST = os.getenv("BUSINESS_HOST", "http://localhost:8000")
PORT = int(os.getenv("GRADIO_PORT", "7860"))
HEADERS: dict[str, str] = {}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alpha-AGI Business dashboard")
    parser.add_argument(
        "--token",
        help="REST API bearer token (defaults to API_TOKEN env var)",
    )
    return parser.parse_args(argv)


def _list_agents() -> list[str]:
    resp = requests.get(f"{HOST}/agents", headers=HEADERS, timeout=5)
    resp.raise_for_status()
    return resp.json()


def _trigger(agent: str) -> str:
    resp = requests.post(f"{HOST}/agent/{agent}/trigger", headers=HEADERS, timeout=5)
    resp.raise_for_status()
    return f"{agent} queued"


def _recent_alpha(limit: int = 5) -> list[dict]:
    resp = requests.get(f"{HOST}/memory/alpha_opportunity/recent", params={"n": limit}, timeout=5)
    resp.raise_for_status()
    return resp.json()


def _search_memory(query: str, limit: int = 5) -> list[dict]:
    resp = requests.get(f"{HOST}/memory/search", params={"q": query, "k": limit}, timeout=5)
    resp.raise_for_status()
    return resp.json()


def _fetch_logs() -> list[str]:
    resp = requests.get(f"{HOST}/api/logs", headers=HEADERS, timeout=5)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    args = _parse_args()
    global HEADERS
    token = args.token or os.getenv("API_TOKEN")
    HEADERS = {"Authorization": f"Bearer {token}"} if token else {}

    with gr.Blocks(title="Alpha‑AGI Business Dashboard") as ui:
        gr.Markdown("# Alpha‑AGI Business Dashboard")
        out = gr.JSON()
        with gr.Row():
            list_btn = gr.Button("List Agents")
            disc_btn = gr.Button("Trigger Discovery")
            opp_btn = gr.Button("Trigger Opportunity")
            exe_btn = gr.Button("Trigger Execution")
            risk_btn = gr.Button("Trigger Risk")
            comp_btn = gr.Button("Trigger Compliance")
            alpha_btn = gr.Button("Recent Alpha")
            logs_btn = gr.Button("Fetch Logs")
        with gr.Row():
            query_in = gr.Textbox(label="Search Memory")
            limit_in = gr.Slider(1, 20, value=5, step=1, label="Results")
            search_btn = gr.Button("Search")

        list_btn.click(_list_agents, outputs=out)
        disc_btn.click(lambda: _trigger("alpha_discovery"), outputs=out)
        opp_btn.click(lambda: _trigger("alpha_opportunity"), outputs=out)
        exe_btn.click(lambda: _trigger("alpha_execution"), outputs=out)
        risk_btn.click(lambda: _trigger("alpha_risk"), outputs=out)
        comp_btn.click(lambda: _trigger("alpha_compliance"), outputs=out)
        alpha_btn.click(_recent_alpha, outputs=out)
        logs_btn.click(_fetch_logs, outputs=out)
        search_btn.click(
            lambda q, k: _search_memory(q, int(k)),
            inputs=[query_in, limit_in],
            outputs=out,
        )

    ui.launch(server_name="0.0.0.0", server_port=PORT, share=False)


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
