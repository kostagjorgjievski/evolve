"""Command-line entry point for EvolveTS."""

from __future__ import annotations

import argparse

from evolvets import __version__
from evolvets.ledger.store import LedgerLockedError, LedgerReader, LedgerWriter
from evolvets.ledger.validate import validate_run


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="evolvets", description=__doc__)
    parser.add_argument("--version", action="version", version=f"evolvets {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    ledger = subparsers.add_parser("ledger", help="ledger inspection and recovery")
    ledger_sub = ledger.add_subparsers(dest="ledger_command", metavar="LEDGER-COMMAND")
    verify = ledger_sub.add_parser("verify", help="verify chain, blobs and ordering of one run")
    verify.add_argument("root", help="ledger root directory (contains <run_id>/)")
    verify.add_argument("run_id", help="the run to verify")
    unlock = ledger_sub.add_parser(
        "unlock", help="break a stale stream lock (dead pid, or empty and old)"
    )
    unlock.add_argument("root", help="ledger root directory (contains <run_id>/)")
    unlock.add_argument("run_id", help="the run whose stream is locked")
    unlock.add_argument("stream", choices=["main", "sealed"], help="the locked stream")
    return parser


def _cmd_ledger_verify(root: str, run_id: str) -> int:
    """Run chain, blob and ordering checks; non-zero on any violation."""
    reader = LedgerReader(root, run_id)
    chain = reader.verify(check_blobs=True)
    if not chain.ok:
        print(f"FAIL {run_id}: {chain.stream} stream, position {chain.seq}: {chain.reason}")
        return 1
    violations = validate_run(list(reader.records()))
    if violations:
        for violation in violations:
            print(f"FAIL {run_id}: {violation}")
        return 1
    print(f"OK {run_id}: {chain.records} records; chain, blobs and ordering verified")
    return 0


def _cmd_ledger_unlock(root: str, run_id: str, stream: str) -> int:
    """Break a stale lock; live, foreign or fresh-empty locks are refused."""
    try:
        LedgerWriter.break_stale_lock(root, run_id, stream)
    except LedgerLockedError as exc:
        print(f"FAIL {run_id}/{stream}: {exc}")
        return 1
    print(f"OK {run_id}/{stream}: stale lock broken")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the EvolveTS CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ledger":
        if args.ledger_command == "verify":
            return _cmd_ledger_verify(args.root, args.run_id)
        if args.ledger_command == "unlock":
            return _cmd_ledger_unlock(args.root, args.run_id, args.stream)
        parser.print_help()
        return 0
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
