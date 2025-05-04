#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
omni_ledger_cli.py • $AGIALPHA ledger companion
════════════════════════════════════════════════
A **safe-by-default**, cross-platform command-line tool to inspect or
export the OMNI-Factory ledger produced by *omni_factory_demo.py*.

Highlights
──────────
• Pure Python ≥3.9 – zero third-party deps, UTF-8 everywhere.  
• Read-only *unless* a mutating sub-command **and** `--force` are used.  
• Built-in integrity audit (sha256 checksums + supply-cap).  
• Colourised human-friendly tables (auto-disabled for pipes/files).  

──────────────────────────────────────────────────────────────────────
Usage examples  (`-h` / `--help` on any sub-command shows details)
──────────────────────────────────────────────────────────────────────
# Show the 20 most-recent tasks  
python omni_ledger_cli.py list

# Tail last 5 rows
python omni_ledger_cli.py list --tail 5

# Supply-cap, minted-per-day, average reward, etc.
python omni_ledger_cli.py stats

# Histogram of daily $AGIALPHA minted
python omni_ledger_cli.py histogram --bins 30

# Full consistency check (timestamps ↑, checksums ✓, hard-cap ✓)
python omni_ledger_cli.py verify

# Export full ledger to CSV or JSONL (read-only)
python omni_ledger_cli.py export --outfile omni_ledger.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

# Keep in sync with omni_factory_demo.py
DEFAULT_LEDGER = Path("./omni_ledger.sqlite").resolve()
LEDGER_SCHEMA_VERSION = 1  # increment if schema evolves
TOKEN_SYMBOL = "$AGIALPHA"
HARD_CAP_ENV = "OMNI_AGIALPHA_SUPPLY"


# ════════════════════════════════════════════════════════════════════
# Helpers (I/O, colour, pretty-printing)
# ════════════════════════════════════════════════════════════════════
def _isatty() -> bool:
    return sys.stdout.isatty()


class _Colour:
    _ENABLED = _isatty() and os.getenv("NO_COLOR", "") == ""

    @staticmethod
    def _wrap(code: str, txt: str) -> str:
        return f"\x1b[{code}m{txt}\x1b[0m" if _Colour._ENABLED else txt

    @classmethod
    def bold(cls, txt: str) -> str:
        return cls._wrap("1", txt)

    @classmethod
    def green(cls, txt: str) -> str:
        return cls._wrap("32", txt)

    @classmethod
    def red(cls, txt: str) -> str:
        return cls._wrap("31", txt)

    @classmethod
    def yellow(cls, txt: str) -> str:
        return cls._wrap("33", txt)


def _pretty_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _print_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    rows = list(rows)
    cols = list(zip(*([headers] + rows))) or [[]]
    widths = [min(max(len(str(x)) for x in col), 120) for col in cols]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(_Colour.bold(fmt.format(*headers)))
    print("─" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt.format(*row))


def _open_db(path: Path) -> sqlite3.Connection:
    if not path.exists():
        sys.exit(_Colour.red(f"Ledger not found: {path}"))
    return sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)


# Basic row type - matches omni_factory_demo schema
Row = Tuple[float, str, int, float, str]  # ts, scenario, tokens, avg_reward, checksum


def _stream_rows(conn: sqlite3.Connection) -> Iterable[Row]:
    yield from conn.execute(
        "SELECT ts, scenario, tokens, avg_reward, checksum FROM ledger ORDER BY ts"
    )


def _load_hard_cap(conn: sqlite3.Connection) -> int:
    # Prefer env var override, else fall back to value stored in cfg-table if present
    env = os.getenv(HARD_CAP_ENV)
    if env:
        return int(env)
    with closing(
        conn.execute(
            "SELECT value FROM pragma_table_info('ledger') WHERE name='tokens'"
        )
    ) as cur:
        # legacy fallback if no explicit cap stored; demo hard coded to 10 000 000
        return 10_000_000


# ════════════════════════════════════════════════════════════════════
# Sub-command implementations
# ════════════════════════════════════════════════════════════════════
def cmd_list(args: argparse.Namespace) -> None:
    with _open_db(args.ledger) as conn:
        rows = list(_stream_rows(conn))
    if args.tail:
        rows = list(islice(rows, len(rows) - args.tail, None))
    display = [
        (
            _pretty_ts(ts),
            scenario[:60],
            f"{tokens:,}",
            f"{avg_reward:.3f}",
        )
        for ts, scenario, tokens, avg_reward, _ in rows
    ]
    _print_table(
        ("Timestamp", "Scenario", f"{TOKEN_SYMBOL}", "AvgReward"),
        display or [("—", "ledger is empty", "—", "—")],
    )


def cmd_stats(args: argparse.Namespace) -> None:
    with _open_db(args.ledger) as conn:
        rows = list(_stream_rows(conn))
        hard_cap = _load_hard_cap(conn)
    if not rows:
        sys.exit("Ledger is empty.")
    tokens_total = sum(r[2] for r in rows)
    reward_mean = sum(r[3] for r in rows) / len(rows)
    first_ts, last_ts = rows[0][0], rows[-1][0]
    days = max((last_ts - first_ts) / 86_400, 1e-6)
    tokens_per_day = tokens_total / days
    utilisation = tokens_total / hard_cap * 100

    print(_Colour.bold("Ledger statistics"))
    print(f"Entries          : {len(rows):,}")
    print(f"Total {TOKEN_SYMBOL:8}: {tokens_total:,} ({utilisation:.2f}% of hard-cap)")
    print(f"Average reward   : {reward_mean:.3f}")
    print(f"First entry      : {_pretty_ts(first_ts)}")
    print(f"Last entry       : {_pretty_ts(last_ts)}")
    print(f"{TOKEN_SYMBOL}/day     : {tokens_per_day:,.1f}")


def _determine_format(outfile: Path) -> str:
    ext = outfile.suffix.lower()
    if ext in (".csv", ".tsv"):
        return "csv"
    if ext in (".jsonl", ".json"):
        return "jsonl"
    sys.exit("Unsupported export format – use .csv, .tsv, .jsonl, or .json")


def cmd_export(args: argparse.Namespace) -> None:
    out: Path = args.outfile.expanduser().resolve()
    if out.exists():
        sys.exit(_Colour.red(f"Refusing to overwrite existing file: {out}"))
    with _open_db(args.ledger) as conn:
        rows = list(_stream_rows(conn))
    fmt = _determine_format(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        delim = "\t" if out.suffix.lower() == ".tsv" else ","
        with out.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=delim)
            writer.writerow(("timestamp", "scenario", "tokens", "avg_reward"))
            for ts, scenario, tokens, avg_reward, _ in rows:
                writer.writerow((ts, scenario, tokens, avg_reward))
    else:  # jsonl / json
        with out.open("w", encoding="utf-8") as fh:
            for ts, scenario, tokens, avg_reward, _ in rows:
                fh.write(
                    json.dumps(
                        {
                            "timestamp": ts,
                            "scenario": scenario,
                            "tokens": tokens,
                            "avg_reward": avg_reward,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(_Colour.green(f"Exported {len(rows):,} rows → {out}"))


def cmd_histogram(args: argparse.Namespace) -> None:
    with _open_db(args.ledger) as conn:
        rows = list(_stream_rows(conn))
    if not rows:
        sys.exit("Ledger is empty.")
    # group by day
    buckets = defaultdict(int)
    for ts, _, tokens, *_ in rows:
        day = int(ts // 86_400)
        buckets[day] += tokens
    if args.bins:
        # merge to approx N bins
        min_day, max_day = min(buckets), max(buckets)
        span = max_day - min_day + 1
        bin_size = max(1, span // args.bins)
        merged = Counter()
        for day, tok in buckets.items():
            merged[(day - min_day) // bin_size] += tok
        buckets = merged
    # render simple ascii bar-chart
    max_tok = max(buckets.values())
    width = 40
    for i in sorted(buckets):
        tok = buckets[i]
        bar = "█" * max(1, int(tok / max_tok * width))
        start = datetime.fromtimestamp((min(buckets) + i) * 86_400, tz=timezone.utc)
        label = start.strftime("%Y-%m-%d")
        print(f"{label} {bar} {tok:,}")


def cmd_supply(args: argparse.Namespace) -> None:
    with _open_db(args.ledger) as conn:
        total = conn.execute("SELECT SUM(tokens) FROM ledger").fetchone()[0] or 0
        hard_cap = _load_hard_cap(conn)
    pct = total / hard_cap * 100
    print(f"{TOKEN_SYMBOL} minted: {total:,} / {hard_cap:,} ({pct:.2f}%)")


def _checksum(ts: float, scenario: str, tokens: int, avg_reward: float) -> str:
    import hashlib

    return hashlib.sha256(f"{ts}{scenario}{tokens}{avg_reward}".encode()).hexdigest()[:16]


def cmd_verify(args: argparse.Namespace) -> None:
    with _open_db(args.ledger) as conn:
        rows = list(_stream_rows(conn))
        hard_cap = _load_hard_cap(conn)
    ok = True
    prev_ts = -math.inf
    minted = 0
    for i, (ts, scenario, tokens, avg_reward, chksum) in enumerate(rows, 1):
        if ts < prev_ts:
            print(_Colour.red(f"[{i}] Timestamp order error: {ts} < {prev_ts}"))
            ok = False
        prev_ts = ts
        if chksum != _checksum(ts, scenario, tokens, avg_reward):
            print(_Colour.red(f"[{i}] Checksum mismatch for row dated {_pretty_ts(ts)}"))
            ok = False
        minted += tokens
    if minted > hard_cap:
        print(_Colour.red(f"Hard-cap exceeded ({minted:,} > {hard_cap:,})"))
        ok = False
    if ok:
        print(_Colour.green("Ledger verification ✓ – all checks passed."))
    else:
        sys.exit(2)


# ════════════════════════════════════════════════════════════════════
# CLI plumbing
# ════════════════════════════════════════════════════════════════════
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="omni_ledger_cli",
        description=f"Inspect, verify or export the {TOKEN_SYMBOL} ledger.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="Path to omni_ledger.sqlite",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # dynamically register sub-commands
    def _add(name: str, func, help_: str, extra: List[Tuple[str, dict]] | None = None):
        sp = sub.add_parser(name, help=help_)
        if extra:
            for flag, kwargs in extra:
                sp.add_argument(flag, **kwargs)
        sp.set_defaults(func=func)

    _add("list", cmd_list, "List recent ledger rows",
         [("--tail", dict(type=int, default=20, help="Lines to show (0=all)"))])
    _add("stats", cmd_stats, "Aggregate statistics")
    _add("histogram", cmd_histogram, "ASCII histogram of minted tokens",
         [("--bins", dict(type=int, default=60, help="Approximate number of bars"))])
    _add("supply", cmd_supply, f"Show total minted {TOKEN_SYMBOL}")
    _add("verify", cmd_verify, "Audit ledger integrity & hard-cap")
    _add("export", cmd_export, "Export ledger",
         [("--outfile", dict(type=Path, required=True, help="Destination file"))])
    return p


def main(argv: List[str] | None = None) -> None:  # pragma: no cover
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
