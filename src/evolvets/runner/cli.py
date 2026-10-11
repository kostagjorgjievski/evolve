"""Command-line entry point for EvolveTS."""

from __future__ import annotations

import argparse

from evolvets import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(prog="evolvets", description=__doc__)
    parser.add_argument("--version", action="version", version=f"evolvets {__version__}")
    parser.add_subparsers(dest="command", metavar="COMMAND")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the EvolveTS CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
