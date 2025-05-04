#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
omni_ledger_cli.py
═══════════════════
Command‑line companion for the **OMNI‑Factory** $AGIALPHA ledger.

Highlights
──────────
• **Pure Python ≥3.9** – zero third‑party runtime dependencies.
• **Portable** – runs unchanged on Windows, macOS & Linux.
• **Safe‑by‑default** – everything is read‑only unless an explicit
  `--outfile` flag is supplied.
• **UTF‑8 everywhere** – full internationalisation support.
• **Rich inspection utilities** – verify checksums, grep scenarios,
  date filters, supply/cap overview, and more.

Quick Examples
──────────────
```bash
# Show 20 most‑recent ledger rows (default)
python omni_ledger_cli.py list

# Tail the last 5 rows only
python omni_ledger_cli.py list --tail 5

# Filter rows that mention "blackout" since 2025‑05‑01
python omni_ledger_cli.py list --grep blackout --since 2025‑05‑01

# Aggregate statistics (totals, averages, supply left)
python omni_ledger_cli.py stats

# Verify integrity (checksums & SQLite pragma quick_check)
python omni_ledger_cli.py verify

# Top 10 tasks by $AGIALPHA minted
python omni_ledger_cli.py top --by tokens --limit 10

# Export full ledger → CSV (read‑only otherwise)
python omni_ledger_cli.py export --outfile agialpha_ledger.csv
```
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sqlite3
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

###############################################################################
# Constants & Types
###############################################################################

# Keep in sync with omni_factory_demo.py -------------------------------------
DEFAULT_LEDGER = Path("./omni_ledger.sqlite").resolve()
HARD_CAP_ENV  = "OMNI_AGIALPHA_SUPPLY"  # env var used by demo for hard‑cap

Row = Tuple[float, str, int, float]  # ts, scenario, tokens, avg_reward

###############################################################################
# Utility helpers
###############################################################################


def _open_db(path: Path, readonly: bool = True) -> sqlite3.Connection:
    """Open SQLite with sensible flags; abort if missing."""
    if not path.exists():
        sys.exit(f"Ledger not found: {path}")
    uri = f"file:{path.as_posix()}?mode={'ro' if readonly else 'rw'}"
    return sqlite3.connect(uri, uri=True, detect_types=sqlite3.PARSE_DECLTYPES)


def _stream_rows(conn: sqlite3.Connection) -> Iterable[Row]:
    yield from conn.execute(
        "SELECT ts, scenario, tokens, avg_reward FROM ledger ORDER BY ts"
    )


def _pretty_ts(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _print_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    rows = list(rows)
    cols = list(zip(*([headers] + rows))) if rows else [headers]
    widths = [max(len(str(x)) for x in col) for col in cols]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt.format(*row))


###############################################################################
# Sub‑command implementations
###############################################################################

def _parse_date(arg: str) -> float:
    """Parse YYYY‑MM‑DD (or any ISO date) → epoch seconds."""
    try:
        dt = _dt.date.fromisoformat(arg)
        return time.mktime(dt.timetuple())
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid date: {arg}") from exc


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #

def cmd_list(args: argparse.Namespace) -> None:  # noqa: D401
    """Show ledger entries with filtering & pagination."""
    with _open_db(args.ledger) as conn:
        rows = list(_stream_rows(conn))

    # Apply filters -----------------------------------------------------------
    if args.since or args.until:
        rows = [
            r for r in rows if (not args.since or r[0] >= args.since) and (not args.until or r[0] <= args.until)
        ]
    if args.grep:
        rows = [r for r in rows if args.grep.lower() in r[1].lower()]

    # Tail / head -------------------------------------------------------------
    if args.tail:
        rows = list(islice(rows, len(rows) - args.tail, None))
    elif args.head:
        rows = list(islice(rows, 0, args.head))

    display = [
        (_pretty_ts(ts), scenario[:60], f"{tokens:,}", f"{avg_reward:.3f}")
        for ts, scenario, tokens, avg_reward in rows
    ]
    _print_table(
        ("Timestamp", "Scenario", "$AGIALPHA", "Avg‑Reward"),
        display or [("—", "ledger is empty", "—", "—")],
    )


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #

def _calc_stats(rows: List[Row]) -> Dict[str, float]:
    tokens_total = sum(r[2] for r in rows)
    reward_mean  = sum(r[3] for r in rows) / len(rows)
    first_ts, last_ts = rows[0][0], rows[-1][0]
    days = max((last_ts - first_ts) / 86_400, 1e-6)
    return {
        "entries": len(rows),
        "total_tokens": tokens_total,
        "avg_reward": reward_mean,
        "start": first_ts,
        "end": last_ts,
        "tokens_per_day": tokens_total / days,
    }


def _read_cap(path: Path) -> int | None:
    """Return hard‑cap recorded in DB user‑version pragma, env, or None."""
    # 1. pragma user_version (demo sets to cap)
    try:
        with _open_db(path) as conn:
            cap = conn.execute("PRAGMA user_version").fetchone()[0]
            if cap:
                return cap
    except Exception:
        pass
    # 2. env var (mirrors demo cfg)
    val = os.getenv(HARD_CAP_ENV)
    return int(val) if val and val.isdigit() else None


def cmd_stats(args: argparse.Namespace) -> None:  # noqa: D401
    """Print aggregated ledger statistics."""
    with _open_db(args.ledger) as conn:
        rows = list(_stream_rows(conn))
    if not rows:
        sys.exit("Ledger is empty.")

    st = _calc_stats(rows)
    cap = _read_cap(args.ledger)
    supply_left = (cap - st["total_tokens"]) if cap is not None else None

    print(f"Entries        : {st['entries']:,}")
    print(f"Total $AGIALPHA: {st['total_tokens']:,}")
    print(f"Average reward : {st['avg_reward']:.3f}")
    print(f"Period         : {_pretty_ts(st['start'])}  →  {_pretty_ts(st['end'])}")
    print(f"$AGIALPHA/day  : {st['tokens_per_day']:, .1f}")
    if cap is not None:
        print(f"Hard cap       : {cap:,}")
        print(f"Supply left    : {supply_left:,}")


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #

import hashlib


def _hash_row(ts: float, scen: str, tok: int, rew: float) -> str:
    return hashlib.sha256(f"{ts}{scen}{tok}{rew}".encode()).hexdigest()[:16]


def cmd_verify(args: argparse.Namespace) -> None:  # noqa: D401
    """Check ledger integrity (checksums & SQLite quick_check)."""
    with _open_db(args.ledger) as conn:
        bad_rows = [
            r for r in conn.execute("SELECT ts, scenario, tokens, avg_reward, checksum FROM ledger")
            if _hash_row(r[0], r[1], r[2], r[3]) != r[4]
        ]
        pragma = conn.execute("PRAGMA quick_check").fetchone()[0]

    if pragma != "ok":
        print("PRAGMA quick_check failed →", pragma)
    else:
        print("SQLite integrity ✔")

    if not bad_rows:
        print("Checksum integrity ✔ (all rows valid)")
    else:
        print("Checksum integrity ✖ – corrupted rows:")
        for row in bad_rows:
            print(" •", _pretty_ts(row[0]), row[1][:60])
        sys.exit(1)


# --------------------------------------------------------------------------- #
# top
# --------------------------------------------------------------------------- #

def cmd_top(args: argparse.Namespace) -> None:  # noqa: D401
    """Show top‑N rows ranked by tokens or avg_reward."""
    key_idx = 2 if args.by == "tokens" else 3
    with _open_db(args.ledger) as conn:
        rows = sorted(_stream_rows(conn), key=lambda r: r[key_idx], reverse=True)[: args.limit]
    display = [
        (_pretty_ts(ts), scenario[:60], f"{tokens:,}", f"{avg_reward:.3f}")
        for ts, scenario, tokens, avg_reward in rows
    ]
    _print_table(
        ("Timestamp", "Scenario", "$AGIALPHA", "Avg‑Reward"),
        display,
    )


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #

def _determine_format(outfile: Path) -> str:
    ext = outfile.suffix.lower()
    if ext in (".csv", ".tsv"):
        return "csv"
    if ext in (".jsonl", ".json"):
        return "jsonl"
    sys.exit("Unsupported export format – use .csv, .tsv, .jsonl, or .json")


def cmd_export(args: argparse.Namespace) -> None:  # noqa: D401
    """Dump ledger to CSV/TSV or JSONL."""
    out: Path = args.outfile.expanduser().resolve()
    if out.exists():
        sys.exit(f"Refusing to overwrite existing file: {out}")

    with _open_db(args.ledger) as conn:
        rows = list(_stream_rows(conn))
    if not rows:
        sys.exit("Ledger is empty – nothing to export.")

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

###############################################################################
# CLI plumbing
###############################################################################

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="omni_ledger_cli",
        description="Inspect, verify, and export the OMNI‑Factory $AGIALPHA ledger.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER, help="Path to omni_ledger.sqlite")
    sub = p.add_subparsers(dest="command", required=True)

    # list -------------------------------------------------------------------
    pl = sub.add_parser("list", help="List ledger rows with optional filters")
    pl.add_argument("--tail", type=int, metavar="N", help="Show last N rows")
    pl.add_argument("--head", type=int, metavar="N", help="Show first N rows")
    pl.add_argument("--grep", help="Substring case‑insensitive scenario filter")
    pl.add_argument("--since", type=_parse_date, help="Only rows on/after YYYY‑MM‑DD")
    pl.add_argument("--until", type=_parse_date, help="Only rows on/before YYYY‑MM‑DD")
    pl.set_defaults(func=cmd_list)

    # stats ------------------------------------------------------------------
    ps = sub.add_parser("stats", help="Aggregate ledger statistics")
    ps.set_defaults(func=cmd_stats)

    # verify -----------------------------------------------------------------
    pv = sub.add_parser("verify", help="Verify ledger integrity (checksums & SQLite)")
    pv.set_defaults(func=cmd_verify)

    # top --------------------------------------------------------------------
    pt = sub.add_parser("top", help="Show top‑N rows by tokens or reward")
    pt.add_argument("--by", choices=("tokens", "reward"), default="tokens", help="Ranking criterion")
    pt.add_argument("--limit", type=int, default=10, help="Number of rows to show")
    pt.set_defaults(func=cmd_top)

    # export -----------------------------------------------------------------
    pe = sub.add_parser("export", help="Export ledger to CSV/TSV/JSONL")
    pe.add_argument("--outfile", type=Path, required=True, help="Destination file path")
    pe.set_defaults(func=cmd_export)

    return p


def main(argv: List[str] | None = None) -> None:  # pragma: no cover
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
