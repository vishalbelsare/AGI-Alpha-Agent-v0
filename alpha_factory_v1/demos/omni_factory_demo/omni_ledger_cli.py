#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
omni_ledger_cli.py
══════════════════
Command-line companion for the OMNI-Factory Smart-City demo ledger.

• Zero external dependencies – pure Python ≥ 3.9.  
• Works out-of-the-box on Windows, macOS, Linux.  
• Safe-by-default: read-only operations unless `--outfile` is provided.  
• Internationalisation-ready (UTF-8 throughout).  

===================================================================
USAGE EXAMPLES
-------------------------------------------------------------------
# View the twenty most-recent tasks (default)
python omni_ledger_cli.py list

# Tail the last 5 rows
python omni_ledger_cli.py list --tail 5

# Aggregate statistics (total CityCoins, average reward, etc.)
python omni_ledger_cli.py stats

# Export full ledger to CSV
python omni_ledger_cli.py export --outfile citycoin_ledger.csv

# Export to JSONL instead
python omni_ledger_cli.py export --outfile citycoin_ledger.jsonl
===================================================================
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# Keep in sync with omni_factory_demo.py
DEFAULT_LEDGER = Path("./omni_ledger.sqlite").resolve()

# ════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════
Row = Tuple[float, str, int, float]   # ts, scenario, tokens, avg_reward


def _open_db(path: Path) -> sqlite3.Connection:
    if not path.exists():
        sys.exit(f"Ledger not found: {path}")
    return sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)


def _stream_rows(conn: sqlite3.Connection) -> Iterable[Row]:
    yield from conn.execute(
        "SELECT ts, scenario, tokens, avg_reward FROM ledger ORDER BY ts"
    )


def _pretty_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _print_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    cols = list(zip(*([headers] + list(rows))))  # transpose
    widths = [max(len(str(x)) for x in col) for col in cols]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt.format(*row))


# ════════════════════════════════════════════════════════════════════
# Sub-commands
# ════════════════════════════════════════════════════════════════════
def cmd_list(args: argparse.Namespace) -> None:     # noqa: D401
    """Show recent ledger entries."""
    with _open_db(args.ledger) as conn:
        rows = list(_stream_rows(conn))
    if args.tail:
        rows = list(islice(rows, len(rows) - args.tail, None))
    display = [
        (_pretty_ts(ts), scenario[:60], f"{tokens:,}", f"{avg_reward:.3f}")
        for ts, scenario, tokens, avg_reward in rows
    ]
    _print_table(
        ("Timestamp", "Scenario", "Tokens", "Avg-Reward"),
        display or [("—", "ledger is empty", "—", "—")],
    )


def cmd_stats(args: argparse.Namespace) -> None:    # noqa: D401
    """Print aggregated ledger statistics."""
    with _open_db(args.ledger) as conn:
        rows = list(_stream_rows(conn))
    if not rows:
        sys.exit("Ledger is empty.")
    tokens_total = sum(r[2] for r in rows)
    reward_mean = sum(r[3] for r in rows) / len(rows)
    first_ts, last_ts = rows[0][0], rows[-1][0]
    days = max((last_ts - first_ts) / 86_400, 1e-6)
    tokens_per_day = tokens_total / days
    print(f"Entries        : {len(rows):,}")
    print(f"Total CityCoins: {tokens_total:,}")
    print(f"Average reward : {reward_mean:.3f}")
    print(f"Period         : {_pretty_ts(first_ts)}  →  {_pretty_ts(last_ts)}")
    print(f"CityCoins/day  : {tokens_per_day:,.1f}")


def _determine_format(outfile: Path) -> str:
    if outfile.suffix.lower() in (".csv", ".tsv"):
        return "csv"
    if outfile.suffix.lower() in (".jsonl", ".json"):
        return "jsonl"
    sys.exit("Unsupported export format – use .csv, .tsv, .jsonl, or .json")


def cmd_export(args: argparse.Namespace) -> None:   # noqa: D401
    """Dump ledger to CSV/TSV or JSONL."""
    out: Path = args.outfile.expanduser().resolve()
    if out.exists():
        sys.exit(f"Refusing to overwrite existing file: {out}")
    with _open_db(args.ledger) as conn:
        rows = list(_stream_rows(conn))
    fmt = _determine_format(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        delim = "\t" if out.suffix.lower() == ".tsv" else ","
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=delim)
            writer.writerow(("timestamp", "scenario", "tokens", "avg_reward"))
            writer.writerows(rows)
    else:  # jsonl
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(
                    json.dumps(
                        {
                            "timestamp": r[0],
                            "scenario": r[1],
                            "tokens": r[2],
                            "avg_reward": r[3],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(f"Exported {len(rows):,} rows → {out}")


# ════════════════════════════════════════════════════════════════════
# CLI plumbing
# ════════════════════════════════════════════════════════════════════
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="omni_ledger_cli",
        description="Inspect or export the OMNI-Factory CityCoin ledger.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="Path to omni_ledger.sqlite",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # list
    pl = sub.add_parser("list", help="List recent ledger rows")
    pl.add_argument("--tail", type=int, default=20, help="Lines to show")
    pl.set_defaults(func=cmd_list)

    # stats
    ps = sub.add_parser("stats", help="Aggregate statistics")
    ps.set_defaults(func=cmd_stats)

    # export
    pe = sub.add_parser("export", help="Export ledger to CSV/TSV/JSONL")
    pe.add_argument("--outfile", type=Path, required=True, help="Destination file")
    pe.set_defaults(func=cmd_export)
    return p


def main(argv: List[str] | None = None) -> None:    # pragma: no cover
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":                          # pragma: no cover
    main()
