#!/usr/bin/env python3
"""Minimal OpenAI Agents client for the business demo.

This helper queries the ``business_helper`` agent exposed by
``openai_agents_bridge.py``.  It works offline and upgrades
transparently when ``OPENAI_API_KEY`` is set.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    import requests  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - offline shim
    from alpha_factory_v1 import requests  # type: ignore


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call BusinessAgent via OpenAI Agents runtime"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("AGENTS_HOST", "http://localhost:5001"),
        help="Base URL for the Agents runtime (default: http://localhost:5001)",
    )
    parser.add_argument(
        "--action",
        default="recent_alpha",
        help="Action to invoke (default: recent_alpha)",
    )
    parser.add_argument(
        "--job",
        help="Optional JSON file with a custom job payload",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    headers = {}
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    url = f"{args.host}/v1/agents/business_helper/invoke"

    def _invoke(payload: dict[str, object]) -> None:
        resp = requests.post(url, json=payload, headers=headers, timeout=10)
        try:
            print(json.dumps(resp.json(), indent=2))
        except json.JSONDecodeError:
            print(resp.text)

    if args.interactive:
        print("Interactive mode – enter an action or 'quit' to exit")
        while True:
            try:
                action = input("action> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not action or action.lower() in {"quit", "exit"}:
                break
            payload = _construct_payload(action, args.job)
            _invoke(payload)
    else:
        payload = _construct_payload(args.action, args.job)
        _invoke(payload)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
