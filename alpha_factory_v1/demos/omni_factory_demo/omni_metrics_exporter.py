# SPDX-License-Identifier: Apache-2.0
"""
OMNI‑Factory • Prometheus Metrics Exporter
──────────────────────────────────────────
Simple *stand‑alone* HTTP endpoint that scrapes the OMNI ledger (sqlite)
and exposes live metrics in Prometheus 2.x exposition‑format.

Usage
─────
    python omni_metrics_exporter.py
    python omni_metrics_exporter.py --ledger ./custom.sqlite -p 9200

Then in Prometheus:
    scrape_configs:
      - job_name: 'omni'
        static_configs:
          - targets: ['localhost:9137']

No third‑party dependencies (pure python ≥3.9).

Metrics
───────
• omni_tokens_minted_total … counter  – all CityCoins ever minted
• omni_tokens_minted_last  … gauge    – CityCoins from most recent task
• omni_avg_reward_last     … gauge    – average reward of most recent task
• omni_tasks_total         … counter  – rows in ledger table
"""
from __future__ import annotations

import http.server
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import List

LEDGER = Path(os.getenv("OMNI_LEDGER", "./omni_ledger.sqlite")).expanduser().resolve()

_HELP_BLOCK = """# HELP omni_tokens_minted_total Total CityCoins ever minted
# TYPE omni_tokens_minted_total counter
# HELP omni_tokens_minted_last CityCoins minted by the most recent task
# TYPE omni_tokens_minted_last gauge
# HELP omni_avg_reward_last Average reward of the most recent task
# TYPE omni_avg_reward_last gauge
# HELP omni_tasks_total Total number of tasks processed
# TYPE omni_tasks_total counter
# HELP omni_export_timestamp Unix timestamp of this scrape
# TYPE omni_export_timestamp gauge
""".strip()

class _MetricsHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/metrics":
            self.send_error(404, "Only /metrics is available")
            return
        body = self._build_metrics().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # --------------------------------------------------------------------- #
    def log_message(self, format: str, *args):  # noqa: D401,E501 pylint: disable=invalid-name
        # Silence default noisy logging; uncomment for debug
        return

    @staticmethod
    def _build_metrics() -> str:
        if not LEDGER.exists():
            totals = last_tokens = last_reward = tasks = 0
        else:
            with sqlite3.connect(LEDGER) as conn:
                rows = conn.execute("SELECT tokens, avg_reward FROM ledger ORDER BY ts").fetchall()
            tasks = len(rows)
            if tasks:
                totals = sum(r[0] for r in rows)
                last_tokens, last_reward = rows[-1]
            else:
                totals = last_tokens = last_reward = 0
        lines: List[str] = [_HELP_BLOCK]
        lines.append(f"omni_tokens_minted_total {totals}")
        lines.append(f"omni_tokens_minted_last {last_tokens}")
        lines.append(f"omni_avg_reward_last {last_reward}")
        lines.append(f"omni_tasks_total {tasks}")
        lines.append(f"omni_export_timestamp {int(time.time())}")
        return "\n".join(lines) + "\n"

def _run(host: str = "0.0.0.0", port: int = 9137) -> None:
    with http.server.ThreadingHTTPServer((host, port), _MetricsHandler) as srv:
        print(f"Serving OMNI‑Factory metrics at http://{host}:{port}/metrics  (ledger: {LEDGER})")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nExporter stopped – goodbye.")

if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="OMNI‑Factory Prometheus metrics exporter")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("-p", "--port", type=int, default=9137, help="TCP port (default: 9137)")
    parser.add_argument("--ledger", help="Path to omni_ledger.sqlite (overrides env var)")
    args = parser.parse_args()

    global LEDGER
    if args.ledger:
        LEDGER = Path(args.ledger).expanduser().resolve()

    _run(args.host, args.port)
