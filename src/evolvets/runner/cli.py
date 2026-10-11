"""Command-line entry point for EvolveTS."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

from evolvets import __version__
from evolvets.arena.realdata import sources
from evolvets.arena.realdata.fetch import FetchError, fetch, verify_raw
from evolvets.arena.realdata.manifest import ManifestError, RawManifest
from evolvets.ledger.store import LedgerLockedError, LedgerReader, LedgerWriter
from evolvets.ledger.validate import validate_run

_DATASET_ORDER = ("etth1", "ecl")


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

    data = subparsers.add_parser("data", help="real-data download and verification")
    data_sub = data.add_subparsers(dest="data_command", metavar="DATA-COMMAND")
    for name, help_text in (
        ("fetch", "download and pin the raw data files"),
        ("verify", "verify raw data files against their manifests"),
    ):
        command = data_sub.add_parser(name, help=help_text)
        command.add_argument(
            "dataset", choices=[*_DATASET_ORDER, "all"], help="which dataset (or 'all')"
        )
        command.add_argument("--data-root", default="data", help="data root directory")
        command.add_argument(
            "--manifest-dir", default="data/manifests", help="directory holding the manifests"
        )
    return parser


def _selected_datasets(dataset: str) -> Iterator[str]:
    """Expand the CLI dataset argument to dataset names in pinned order."""
    if dataset == "all":
        yield from _DATASET_ORDER
    else:
        yield dataset


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


def _cmd_data(action: str, dataset: str, data_root: str, manifest_dir: str) -> int:
    """Fetch or verify each requested dataset, one OK/FAIL line per dataset."""
    failed = False
    for name in _selected_datasets(dataset):
        try:
            source = sources.SOURCES[name]
            if action == "fetch":
                manifest: RawManifest = fetch(source, Path(data_root), Path(manifest_dir))
            else:
                manifest = verify_raw(source, Path(data_root), Path(manifest_dir))
        except (FetchError, ManifestError, OSError) as exc:
            failed = True
            print(f"FAIL {name}: {exc}")
        else:
            print(f"OK {name}: {manifest.sha256} {manifest.size_bytes} bytes")
    return 1 if failed else 0


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
    if args.command == "data":
        if args.data_command in ("fetch", "verify"):
            return _cmd_data(args.data_command, args.dataset, args.data_root, args.manifest_dir)
        parser.print_help()
        return 0
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
