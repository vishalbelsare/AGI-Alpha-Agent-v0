#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
omni_ledger_cli.py • $AGIALPHA ledger companion
════════════════════════════════════════════════
A **safe-by-default**, cross-platform command-line tool to inspect,
verify or export the OMNI-Factory ledger produced by
*omni_factory_demo.py*.

Highlights
──────────
• Pure Python ≥ 3.9 – zero third-party deps, UTF-8 throughout.  
• Read-only unless a mutating sub-command **and** `--force` are used.  
• Built-in integrity audit (sha256 checksums + hard-cap).  
• Colourised human-friendly tables (auto-disabled for pipes/files).

Full help: `python omni_ledger_cli.py -h`
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

# Keep in sync with omni_factory_demo.py ────────────────────────────
DEFAULT_LEDGER = Path("./omni_ledger.sqlite").resolve()
LEDGER_SCHEMA_VERSION = 1          # bump if schema evolves
TOKEN_SYMBOL = "$AGIALPHA"
HARD_CAP_ENV = "OMNI_AGIALPHA_SUPPLY"

# ════════════════════════════════════════════════════════════════════
# Helpers (colour, pretty-printing, DB access)
# ════════════════════════════════════════════════════════════════════
def _isatty() -> bool:
    """Return True iff stdout is an interactive TTY and NO_COLOR not set."""
    return sys.stdout.isatty() and os.getenv("NO_COLOR", "") == ""

class _Colour:
    """Minimal ANSI colour helper (auto-disabled if output not a TTY)."""

    _ENABLED = _isatty()

    @staticmethod
    def _wrap(code: str, txt: str) -> str:
        return f"\x1b[{code}m{txt}\x1b[0m" if _Colour._ENABLED else txt

    @classmethod
    def bold(cls, txt: str) -> str:      return cls._wrap("1", txt)
    @classmethod
    def green(cls, txt: str) -> str:     return cls._wrap("32", txt)
    @classmethod
    def red(cls, txt: str) -> str:       return cls._wrap("31", txt)
    @classmethod
    def yellow(cls, txt: str) -> str:    return cls._wrap("33", txt)

def _pretty_ts(ts: float) -> str:
    """Return human-readable UTC timestamp."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def _print_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    """Nicely aligned, colourised table for human eyes."""
    rows = list(rows)
    cols = list(zip(*([headers] + rows))) if rows else [[]]
    widths = [min(max(len(str(x)) for x in col), 120) for col in cols]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(_Colour.bold(fmt.format(*headers)))
    print("─" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        print(fmt.format(*row))

def _open_db(path: Path) -> sqlite3.Connection:
    """Open ledger DB or exit with error."""
    if not path.exists():
        sys.exit(_Colour.red(f"Ledger not found: {path}"))
    return sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)

# Basic row type ─ matches omni_factory_demo schema
Row = Tuple[float, str, int, float, str]      # ts, scenario, tokens, avg_reward, checksum

def _stream_rows(conn: sqlite3.Connection) -> Iterable[Row]:
    yield from conn.execute(
        "SELECT ts, scenario, tokens, avg_reward, checksum "
        "FROM ledger ORDER BY ts"
    )

def _load_hard_cap(conn: sqlite3.Connection) -> int:
    """Return hard-cap from env var or fallback default (10 000 000)."""
    env = os.getenv(HARD_CAP_ENV)
    if env:
        return int(env)
    # legacy fallback if env not set – demo hard-codes 10 M
    return 10_000_000

# ════════════════════════════════════════════════════════════════════
# Sub-command: list
# ════════════════════════════════════════════════════════════════════
def cmd_list(args: argparse.Namespace) -> None:
    """Show most-recent ledger rows (tail N)."""
    with _open_db(args.ledger) as conn:
        rows = list(_stream_rows(conn))
    if args.tail:
        rows = list(islice(rows, len(rows) - args.tail, None))
    display = [
        (_pretty_ts(ts), scenario[:60],
         f"{tokens:,}", f"{avg_reward:.3f}")
        for ts, scenario, tokens, avg_reward, _ in rows
    ]
    _print_table(
        ("Timestamp", "Scenario", TOKEN_SYMBOL, "AvgReward"),
        display or [("—", "ledger is empty", "—", "—")],
    )

# ════════════════════════════════════════════════════════════════════
# Sub-command: stats
# ════════════════════════════════════════════════════════════════════
def cmd_stats(args: argparse.Namespace) -> None:
    """Aggregate ledger statistics (total minted, reward mean, utilisation…)."""
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
    print(f"Total {TOKEN_SYMBOL:10}: {tokens_total:,}  "
          f"({_Colour.yellow(f'{utilisation:.2f}%')} of hard-cap)")
    print(f"Average reward   : {reward_mean:.3f}")
    print(f"First entry      : {_pretty_ts(first_ts)}")
    print(f"Last entry       : {_pretty_ts(last_ts)}")
    print(f"{TOKEN_SYMBOL}/day     : {tokens_per_day:,.1f}")

# ════════════════════════════════════════════════════════════════════
# CLI plumbing
# ════════════════════════════════════════════════════════════════════
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="omni_ledger_cli",
        description=f"Inspect, verify or export the {TOKEN_SYMBOL} ledger.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER,
                   help="Path to omni_ledger.sqlite")
    sub = p.add_subparsers(dest="command", required=True)

    # list
    pl = sub.add_parser("list", help="List recent ledger rows")
    pl.add_argument("--tail", type=int, default=20,
                    help="Lines to show (0=all)")
    pl.set_defaults(func=cmd_list)

    # stats
    ps = sub.add_parser("stats", help="Aggregate statistics")
    ps.set_defaults(func=cmd_stats)

    # The following are stub-registered here; implementations arrive in Part 2/3
    sub.add_parser("histogram", help="ASCII histogram of minted tokens")
    sub.add_parser("supply",    help=f"Show total minted {TOKEN_SYMBOL}")
    sub.add_parser("verify",    help="Audit ledger integrity & hard-cap")
    exp = sub.add_parser("export", help="Export ledger to CSV/TSV/JSONL")
    exp.add_argument("--outfile", type=Path, required=True, help="Destination file")
    return p

def main(argv: List[str] | None = None) -> None:    # pragma: no cover
    args = _build_parser().parse_args(argv)
    # Dispatch to the fully-wired function; will error if user calls un-implemented cmd
    args.func(args)

if __name__ == "__main__":                          # pragma: no cover
    main()

# ────────────────────────────── Part 2 starts here ──────────────────────────────
# Extra helpers ─────────────────────────────────────────────────────────────────
def _determine_format(outfile: Path) -> str:
    """Return 'csv' or 'jsonl' based on extension; abort on unsupported."""
    ext = outfile.suffix.lower()
    if ext in (".csv", ".tsv"):
        return "csv"
    if ext in (".jsonl", ".json"):
        return "jsonl"
    sys.exit(_Colour.red("Unsupported export format – use .csv, .tsv, .jsonl, or .json"))

# Sub-command: export ───────────────────────────────────────────────────────────
def cmd_export(args: argparse.Namespace) -> None:
    """Dump ledger to CSV/TSV or JSONL (read-only)."""
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
            for ts, scen, tok, rew, _ in rows:
                writer.writerow((ts, scen, tok, rew))
    else:                                              # JSON Lines
        with out.open("w", encoding="utf-8") as fh:
            for ts, scen, tok, rew, _ in rows:
                fh.write(json.dumps(
                    {"timestamp": ts, "scenario": scen,
                     "tokens": tok, "avg_reward": rew},
                    ensure_ascii=False) + "\n")

    print(_Colour.green(f"Exported {len(rows):,} rows → {out}"))

# Sub-command: histogram ────────────────────────────────────────────────────────
def cmd_histogram(args: argparse.Namespace) -> None:
    """ASCII bar-chart of daily $AGIALPHA minted."""
    with _open_db(args.ledger) as conn:
        rows = list(_stream_rows(conn))
    if not rows:
        sys.exit("Ledger is empty.")

    # group by day (UTC)
    day_totals = defaultdict(int)
    for ts, _, tok, *_ in rows:
        day_totals[int(ts // 86_400)] += tok

    # binning (optional)
    if args.bins:
        min_day, max_day = min(day_totals), max(day_totals)
        span = max_day - min_day + 1
        bin_size = max(1, span // args.bins)
        merged = Counter()
        for day, tok in day_totals.items():
            merged[(day - min_day) // bin_size] += tok
        day_totals = merged

    max_tok = max(day_totals.values())
    width   = 40
    for idx in sorted(day_totals):
        tok  = day_totals[idx]
        bar  = "█" * max(1, int(tok / max_tok * width))
        date = datetime.fromtimestamp((min(day_totals) + idx) * 86_400,
                                      tz=timezone.utc).strftime("%Y-%m-%d")
        print(f"{date} {bar} {tok:,}")

# Sub-command: supply ───────────────────────────────────────────────────────────
def cmd_supply(args: argparse.Namespace) -> None:
    """Print total minted vs hard-cap."""
    with _open_db(args.ledger) as conn:
        total = conn.execute("SELECT SUM(tokens) FROM ledger").fetchone()[0] or 0
        cap   = _load_hard_cap(conn)
    pct = total / cap * 100
    print(f"{TOKEN_SYMBOL} minted: {total:,} / {cap:,} ({pct:.2f}%)")

# Re-define _build_parser so the new commands are wired in ──────────────────────
def _build_parser() -> argparse.ArgumentParser:                       # type: ignore[override]
    p = argparse.ArgumentParser(
        prog="omni_ledger_cli",
        description=f"Inspect, verify or export the {TOKEN_SYMBOL} ledger.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER,
                   help="Path to omni_ledger.sqlite")
    sub = p.add_subparsers(dest="command", required=True)

    # list / stats already implemented
    pl = sub.add_parser("list", help="List recent ledger rows")
    pl.add_argument("--tail", type=int, default=20,
                    help="Lines to show (0=all)")
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("stats", help="Aggregate statistics")
    ps.set_defaults(func=cmd_stats)

    # newly-implemented commands
    ph = sub.add_parser("histogram", help="ASCII histogram of minted tokens")
    ph.add_argument("--bins", type=int, default=60,
                    help="Approximate number of bars")
    ph.set_defaults(func=cmd_histogram)

    psup = sub.add_parser("supply", help=f"Show total minted {TOKEN_SYMBOL}")
    psup.set_defaults(func=cmd_supply)

    pv = sub.add_parser("verify", help="Audit ledger integrity & hard-cap")
    # func attached in Part 3

    pe = sub.add_parser("export", help="Export ledger to CSV/TSV/JSONL")
    pe.add_argument("--outfile", type=Path, required=True,
                    help="Destination file")
    pe.set_defaults(func=cmd_export)
    return p
# ─────────────────────────────── Part 2 ends here ──────────────────────────────


# ────────────────────────────── Part 3 starts here ──────────────────────────────
# Integrity verifier ────────────────────────────────────────────────────────────
def cmd_verify(args: argparse.Namespace) -> None:
    """
    Audit the ledger for:
      • strictly ascending timestamps
      • correct per-row sha256 checksum
      • non-negative token amounts
      • global supply-cap compliance
    Exits with code 2 on failure.
    """
    with _open_db(args.ledger) as conn:
        cur = conn.execute(
            "SELECT ts, scenario, tokens, avg_reward, checksum FROM ledger ORDER BY ts"
        )
        hard_cap = _load_hard_cap(conn)

        ok           = True
        total_tokens = 0
        prev_ts      = -math.inf

        for idx, (ts, scen, tok, rew, chksum) in enumerate(cur, 1):
            # ◼︎ Time monotonicity
            if ts < prev_ts:
                print(_Colour.red(f"[row {idx}] timestamp disorder: {ts} < {prev_ts}"))
                ok = False
            prev_ts = ts

            # ◼︎ Checksum correctness
            if chksum != _checksum(ts, scen, tok, rew):
                print(_Colour.red(f"[row {idx}] checksum mismatch  ({_pretty_ts(ts)})"))
                ok = False

            # ◼︎ Non-negative tokens
            if tok < 0:
                print(_Colour.red(f"[row {idx}] negative token amount: {tok}"))
                ok = False

            total_tokens += tok

        # ◼︎ Hard-cap
        if total_tokens > hard_cap:
            print(
                _Colour.red(
                    f"Hard-cap exceeded: {total_tokens:,} > {hard_cap:,} {TOKEN_SYMBOL}"
                )
            )
            ok = False

    if ok:
        print(_Colour.green("Ledger verification ✓ – all checks passed."))
    else:
        sys.exit(2)

# Hidden self-test harness ──────────────────────────────────────────────────────
def _cmd_selftest(_: argparse.Namespace) -> None:
    """
    Create a temporary in-memory ledger → run verify → corrupt a row → expect failure.
    Prints ✓/✗ and exits 0/1 accordingly.
    """
    import tempfile
    tmp = Path(tempfile.mktemp(suffix=".sqlite"))
    try:
        with sqlite3.connect(tmp) as conn:
            conn.execute(
                "CREATE TABLE ledger(ts REAL, scenario TEXT, tokens INT, "
                "avg_reward REAL, checksum TEXT)"
            )
            ts       = time.time()
            scenario = "unit-test ok"
            tokens   = 100
            reward   = 0.99
            chksum   = _checksum(ts, scenario, tokens, reward)
            conn.execute(
                "INSERT INTO ledger VALUES (?,?,?,?,?)",
                (ts, scenario, tokens, reward, chksum),
            )

        # 1. should pass
        class _Args:  # simple stand-in for argparse namespace
            ledger = tmp
        print(_Colour.yellow("• Running positive verification …"))
        cmd_verify(_Args)  # should not raise/exit

        # 2. corrupt checksum → expect failure exit-code 2
        with sqlite3.connect(tmp) as conn:
            conn.execute("UPDATE ledger SET checksum = 'badhash' WHERE rowid = 1")

        print(_Colour.yellow("• Running negative verification …"))
        try:
            cmd_verify(_Args)
        except SystemExit as exc:
            if exc.code == 2:
                print(_Colour.green("Self-test ✓"))
                return
            raise
        print(_Colour.red("Self-test ✗ — corruption went undetected"))
        sys.exit(1)
    finally:
        tmp.unlink(missing_ok=True)

# Final parser override to wire `verify` & hidden `selftest` ────────────────────
def _build_parser() -> argparse.ArgumentParser:                       # type: ignore[override]
    p = argparse.ArgumentParser(
        prog="omni_ledger_cli",
        description=f"Inspect, verify or export the {TOKEN_SYMBOL} ledger.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER,
                   help="Path to omni_ledger.sqlite")
    sub = p.add_subparsers(dest="command", required=True)

    # list
    pl = sub.add_parser("list", help="List recent ledger rows")
    pl.add_argument("--tail", type=int, default=20,
                    help="Lines to show (0=all)")
    pl.set_defaults(func=cmd_list)

    # stats
    ps = sub.add_parser("stats", help="Aggregate statistics")
    ps.set_defaults(func=cmd_stats)

    # histogram
    ph = sub.add_parser("histogram", help="ASCII histogram of minted tokens")
    ph.add_argument("--bins", type=int, default=60,
                    help="Approximate number of bars")
    ph.set_defaults(func=cmd_histogram)

    # supply
    psup = sub.add_parser("supply", help=f"Show total minted {TOKEN_SYMBOL}")
    psup.set_defaults(func=cmd_supply)

    # export
    pe = sub.add_parser("export", help="Export ledger to CSV/TSV/JSONL")
    pe.add_argument("--outfile", type=Path, required=True,
                    help="Destination file")
    pe.set_defaults(func=cmd_export)

    # verify
    pv = sub.add_parser("verify", help="Audit ledger integrity & hard-cap")
    pv.set_defaults(func=cmd_verify)

    # hidden self-test
    ptest = sub.add_parser("selftest", help=argparse.SUPPRESS)
    ptest.set_defaults(func=_cmd_selftest)

    return p
# ─────────────────────────────── Part 3 ends here ───────────────────────────────
