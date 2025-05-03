
# SPDX-License-Identifier: Apache-2.0
"""
omni_dashboard.py
════════════════════════════════════════════════════════════════════
Light‑weight self‑hosted dashboard for the OMNI‑Factory demo.

 • Live chart of cumulative CityCoins
 • Table view of the last N ledger entries
 • Works fully offline; only requires `dash` + `pandas` (auto‑install prompt)

Launch:
    python omni_dashboard.py          # opens http://127.0.0.1:8050

The script will watch the sqlite ledger file produced by omni_factory_demo.py
(default: ./omni_ledger.sqlite).  Use the LEDGER_PATH env‑var to point to a
different location.

Designed to be *drop‑in* – no config changes needed in the rest of the system.
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import List

_LEDGER = Path(os.getenv("LEDGER_PATH", "./omni_ledger.sqlite")).expanduser()

# ── Soft‑deps with graceful fallback ─────────────────────────────────────────
try:
    import dash
    from dash import dcc, html, dash_table, Output, Input  # type: ignore
    import pandas as pd                                    # type: ignore
except ModuleNotFoundError as exc:                         # pragma: no cover
    pkg = str(exc).split("'")[1]
    print(f"[omni_dashboard] Missing python package: {pkg}\n"
          "Install with:\n    pip install dash pandas", file=sys.stderr)
    sys.exit(1)

_REFRESH_SEC = float(os.getenv("DASH_REFRESH", "2"))     # poll freq UI

# ── Helpers ─────────────────────────────────────────────────────────────────
def _fetch_ledger(n_rows: int | None = None) -> pd.DataFrame:
    if not _LEDGER.exists():
        return pd.DataFrame(columns=["ts", "scenario", "tokens", "avg_reward"])
    conn = sqlite3.connect(_LEDGER)
    query = "SELECT ts, scenario, tokens, avg_reward FROM ledger ORDER BY ts DESC"
    if n_rows:
        query += f" LIMIT {n_rows}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    return df

def _cumsum(df: pd.DataFrame) -> pd.DataFrame:
    df_sorted = df.sort_values("ts")
    df_sorted["cumulative_tokens"] = df_sorted["tokens"].cumsum()
    return df_sorted

# ── Dash layout ─────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="OMNI‑Factory Dashboard", suppress_callback_exceptions=True)

app.layout = html.Div(
    style={"fontFamily": "sans‑serif", "margin": "0 auto", "maxWidth": "1000px"},
    children=[
        html.H2("🏙️  OMNI‑Factory • Smart‑City Resilience Dashboard", style={"textAlign": "center"}),
        dcc.Interval(id="tick", interval=_REFRESH_SEC*1000, n_intervals=0),
        dcc.Graph(id="token‑chart"),
        html.H4("Latest events"),
        dash_table.DataTable(id="table", page_size=10,
                             style_cell={"textAlign": "left", "whiteSpace": "pre"},
                             style_table={"overflowX": "auto"}),
        html.Footer("Ledger: " + str(_LEDGER), style={"marginTop": "2rem", "fontSize": "0.8em"})
    ]
)

# ── Callbacks ───────────────────────────────────────────────────────────────
@app.callback(Output("token‑chart", "figure"),
              Output("table", "data"), Output("table", "columns"),
              Input("tick", "n_intervals"))
def update(_):
    df = _fetch_ledger()
    cols = [{"name": c, "id": c} for c in df.columns]
    if df.empty:
        return {"layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False},
                             "annotations": [{"text": "No data yet", "xref": "paper", "yref": "paper",
                                               "showarrow": False, "font": {"size": 20}}]}}, [], cols
    df_cum = _cumsum(df)
    fig = {
        "data": [
            {"x": df_cum["ts"], "y": df_cum["cumulative_tokens"],
             "type": "scatter", "mode": "lines+markers",
             "name": "Cumulative CityCoins"}
        ],
        "layout": {"margin": {"l": 40, "r": 20, "t": 30, "b": 40},
                    "yaxis": {"title": "Tokens"}, "xaxis": {"title": "Time"}}
    }
    return fig, df.to_dict("records"), cols

# ── Convenience auto‑open browser ───────────────────────────────────────────
def _open_browser():
    import webbrowser, socket
    # wait until server up
    time.sleep(1)
    try:
        webbrowser.open_new(f"http://127.0.0.1:8050")
    except (RuntimeError, webbrowser.Error):
        pass

if __name__ == "__main__":                                   # pragma: no cover
    threading.Thread(target=_open_browser, daemon=True).start()
    print("[omni_dashboard] Serving on http://127.0.0.1:8050  (Ctrl‑C to quit)")
    app.run_server(host="0.0.0.0", port=8050, debug=False)
