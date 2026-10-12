"""Command-line entry point for EvolveTS."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from evolvets import __version__
from evolvets.arena.errors import ArenaError
from evolvets.arena.realdata import sources
from evolvets.arena.realdata.fetch import FetchError, fetch, file_sha256, verify_raw
from evolvets.arena.realdata.manifest import ManifestError, RawManifest
from evolvets.arena.realdata.prepare import (
    PrepareError,
    prepare,
    read_split_manifest,
    splits_manifest_path,
)
from evolvets.arena.sealed.names import SEALED_DATASETS
from evolvets.arena.sealed.pools import (
    build_sealed_cache,
    init_salt,
    synthetic_manifest_path,
)
from evolvets.cache.datasets import CACHE_DATASETS, public_inputs
from evolvets.cache.store import (
    DEFAULT_HORIZON,
    CacheError,
    build_cache,
    cache_id,
    verify_cache,
)
from evolvets.foundation.errors import FoundationError
from evolvets.foundation.fake import FAKE_SPEC, FakeAdapter
from evolvets.foundation.interfaces import Adapter
from evolvets.foundation.models import load_adapter, require_adapter_module
from evolvets.foundation.registry import MODELS, ModelSpec, get_spec
from evolvets.foundation.smoke import run_smoke
from evolvets.foundation.store import (
    build_foundation_cache,
    compare_rebuild,
    foundation_cache_id,
    verify_foundation_cache,
)
from evolvets.ledger.store import LedgerLockedError, LedgerReader, LedgerWriter
from evolvets.ledger.validate import validate_run

_DATASET_ORDER = ("etth1", "ecl")
#: ``--model`` choices: the test/CI fake, then the registry's model order.
_FOUNDATION_MODELS = ("fake", *MODELS)


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
        ("verify", "verify raw (and processed) files against their manifests"),
        ("prepare", "build processed files and split manifests from the raw files"),
    ):
        command = data_sub.add_parser(name, help=help_text)
        command.add_argument(
            "dataset", choices=[*_DATASET_ORDER, "all"], help="which dataset (or 'all')"
        )
        command.add_argument("--data-root", default="data", help="data root directory")
        command.add_argument(
            "--manifest-dir", default="data/manifests", help="directory holding the manifests"
        )

    cache = subparsers.add_parser("cache", help="expert-forecast cache build and verification")
    cache_sub = cache.add_subparsers(dest="cache_command", metavar="CACHE-COMMAND")
    for name, help_text in (
        ("build", "compute the expert forecasts and install them as a cache"),
        ("verify", "check an existing cache against its manifest"),
    ):
        command = cache_sub.add_parser(name, help=help_text)
        command.add_argument(
            "dataset", choices=[*CACHE_DATASETS, "all"], help="which dataset (or 'all')"
        )
        command.add_argument("--cache-root", default="cache_store", help="cache root directory")
        command.add_argument("--data-root", default="data", help="data root directory")
        command.add_argument(
            "--manifest-dir", default="data/manifests", help="directory holding the manifests"
        )
        if name == "build":
            command.add_argument(
                "--horizon",
                type=int,
                default=DEFAULT_HORIZON,
                help="forecast horizon to cache (default: 24)",
            )
            command.add_argument(
                "--per-family",
                type=int,
                help="synthetic tasks per family; required for synthetic-dev (and 'all')",
            )

    foundation = subparsers.add_parser(
        "foundation", help="foundation-model forecast cache build and verification"
    )
    foundation_sub = foundation.add_subparsers(
        dest="foundation_command", metavar="FOUNDATION-COMMAND"
    )
    foundation_build = foundation_sub.add_parser(
        "build", help="compute one model's forecasts and install them as a cache"
    )
    foundation_build.add_argument(
        "--dataset",
        choices=CACHE_DATASETS,
        required=True,
        help="which dataset (one model per build)",
    )
    foundation_build.add_argument(
        "--model",
        choices=_FOUNDATION_MODELS,
        required=True,
        help="which model ('fake' for tests and CI; real models resolve through their adapters)",
    )
    foundation_build.add_argument(
        "--root", default="foundation_cache", help="foundation cache root directory"
    )
    foundation_build.add_argument(
        "--cache-root", default="cache_store", help="simple cache root directory"
    )
    foundation_build.add_argument("--data-root", default="data", help="data root directory")
    foundation_build.add_argument(
        "--manifest-dir", default="data/manifests", help="directory holding the manifests"
    )
    foundation_build.add_argument(
        "--batch-size", type=int, default=32, help="contexts per adapter call (default: 32)"
    )
    foundation_build.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="adapter device; auto uses CUDA when available",
    )
    foundation_build.add_argument(
        "--per-family", type=int, help="synthetic tasks per family; required for synthetic-dev"
    )
    foundation_verify = foundation_sub.add_parser(
        "verify", help="check an existing cache against its manifest"
    )
    foundation_verify.add_argument(
        "--dataset", choices=CACHE_DATASETS, required=True, help="which dataset"
    )
    foundation_verify.add_argument(
        "--model",
        choices=_FOUNDATION_MODELS,
        required=True,
        help="which model ('fake' for tests and CI; real models resolve through their adapters)",
    )
    foundation_verify.add_argument(
        "--root", default="foundation_cache", help="foundation cache root directory"
    )
    foundation_compare = foundation_sub.add_parser(
        "compare",
        help="recompute a seeded sample of origins and report the drift (diagnostic, read-only)",
    )
    foundation_compare.add_argument(
        "--dataset", choices=CACHE_DATASETS, required=True, help="which dataset"
    )
    foundation_compare.add_argument(
        "--model",
        choices=_FOUNDATION_MODELS,
        required=True,
        help="which model ('fake' for tests and CI; real models resolve through their adapters)",
    )
    foundation_compare.add_argument(
        "--root", default="foundation_cache", help="foundation cache root directory"
    )
    foundation_compare.add_argument("--data-root", default="data", help="data root directory")
    foundation_compare.add_argument(
        "--manifest-dir", default="data/manifests", help="directory holding the manifests"
    )
    foundation_compare.add_argument(
        "--sample", type=int, required=True, help="how many origins to recompute"
    )
    foundation_compare.add_argument(
        "--seed", type=int, required=True, help="seed for the origin sample"
    )
    foundation_smoke = foundation_sub.add_parser(
        "smoke",
        help="forecast a fixed series at 20 origins and check batching and accuracy",
    )
    foundation_smoke.add_argument(
        "--model",
        choices=_FOUNDATION_MODELS,
        required=True,
        help="which model ('fake' runs torch-free; real models need their fm-* group)",
    )
    foundation_smoke.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="adapter device; auto uses CUDA when available",
    )

    sealed = subparsers.add_parser(
        "sealed", help="sealed salt creation and sealed cache build (trusted, human-started)"
    )
    sealed_sub = sealed.add_subparsers(dest="sealed_command", metavar="SEALED-COMMAND")
    init_salt_cmd = sealed_sub.add_parser(
        "init-salt", help="create the sealed synthetic salt and its manifest, once"
    )
    init_salt_cmd.add_argument(
        "--per-family", type=int, required=True, help="sealed synthetic worlds per family"
    )
    init_salt_cmd.add_argument("--data-root", default="data", help="data root directory")
    init_salt_cmd.add_argument(
        "--manifest-dir", default="data/manifests", help="directory holding the manifests"
    )
    build_sealed = sealed_sub.add_parser(
        "build-cache", help="compute the sealed pools' forecasts and install them as caches"
    )
    build_sealed.add_argument(
        "dataset", choices=[*SEALED_DATASETS, "all"], help="which dataset (or 'all')"
    )
    build_sealed.add_argument("--data-root", default="data", help="data root directory")
    build_sealed.add_argument(
        "--manifest-dir", default="data/manifests", help="directory holding the manifests"
    )
    build_sealed.add_argument(
        "--sealed-cache-root",
        default="data/sealed_cache",
        help="sealed cache root directory (never the public cache root)",
    )
    build_sealed.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="forecast horizon to cache (default: 24)",
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


def _verify_processed(name: str, data_root: Path, manifest_dir: Path) -> None:
    """Check both processed files against the split manifest, if one exists."""
    if not splits_manifest_path(manifest_dir, name).exists():
        return
    spec = read_split_manifest(manifest_dir, name)
    for label, expected in (("public", spec.public_sha256), ("sealed", spec.sealed_sha256)):
        processed = data_root / "processed" / f"{name}.{label}.json"
        if not processed.exists():
            raise PrepareError(f"{processed} is missing; run 'evolvets data prepare {name}'")
        digest, _ = file_sha256(processed)
        if digest != expected:
            raise PrepareError(
                f"{processed} does not match the split manifest: file sha256 {digest}, "
                f"manifest sha256 {expected}"
            )


def _cmd_data(action: str, dataset: str, data_root: str, manifest_dir: str) -> int:
    """Fetch, prepare or verify each requested dataset, one OK/FAIL line per dataset."""
    failed = False
    root, manifests = Path(data_root), Path(manifest_dir)
    for name in _selected_datasets(dataset):
        try:
            if action == "prepare":
                prepare(name, root, manifests)
                spec = read_split_manifest(manifests, name)
                print(f"OK {name}: {len(spec.series)} series, processed {spec.public_sha256}")
                continue
            source = sources.SOURCES[name]
            if action == "fetch":
                manifest: RawManifest = fetch(source, root, manifests)
            else:
                manifest = verify_raw(source, root, manifests)
                _verify_processed(name, root, manifests)
        except (FetchError, ManifestError, PrepareError, ValueError, OSError) as exc:
            failed = True
            print(f"FAIL {name}: {exc}")
        else:
            print(f"OK {name}: {manifest.sha256} {manifest.size_bytes} bytes")
    return 1 if failed else 0


def _cmd_cache(
    action: str,
    dataset: str,
    cache_root: str,
    data_root: str,
    manifest_dir: str,
    horizon: int = DEFAULT_HORIZON,
    per_family: int | None = None,
) -> int:
    """Build or verify each requested cache, one OK/FAIL line per dataset."""
    failed = False
    root = Path(cache_root)
    for name in CACHE_DATASETS if dataset == "all" else (dataset,):
        try:
            if action == "build":
                source, series = public_inputs(
                    name, Path(data_root), Path(manifest_dir), per_family
                )
                manifest = build_cache(name, source, series, root, horizon=horizon)
            else:
                manifest = verify_cache(root, name)
        except (CacheError, PrepareError, ValueError, OSError) as exc:
            failed = True
            print(f"FAIL {name}: {exc}")
        else:
            origins = sum(entry.n_origins for entry in manifest.series)
            print(
                f"OK {name}: {len(manifest.series)} series, {origins} origins, "
                f"cache {cache_id(root, name)}"
            )
    return 1 if failed else 0


def _foundation_adapter(model: str, device: str) -> tuple[Adapter, ModelSpec]:
    """Resolve ``--model`` to an adapter and its spec; ``fake`` stays torch-free."""
    if model == "fake":
        if device == "cuda":
            raise FoundationError("--device cuda needs a real model; the fake adapter runs on cpu")
        return FakeAdapter(), FAKE_SPEC
    return load_adapter(model, device), get_spec(model)


def _cmd_foundation_build(
    dataset: str,
    model: str,
    root: str,
    cache_root: str,
    data_root: str,
    manifest_dir: str,
    batch_size: int,
    device: str,
    per_family: int | None,
) -> int:
    """Build one (dataset, model) foundation cache; one FAIL line on any refusal."""
    try:
        adapter, spec = _foundation_adapter(model, device)
        manifest = build_foundation_cache(
            dataset,
            adapter,
            spec,
            Path(root),
            Path(cache_root),
            Path(data_root),
            Path(manifest_dir),
            batch_size,
            per_family=per_family,
        )
        ident = foundation_cache_id(Path(root), dataset, model)
    except (FoundationError, CacheError, PrepareError, ValueError, OSError) as exc:
        print(f"FAIL {dataset}/{model}: {exc}")
        return 1
    origins = sum(entry.n_origins for entry in manifest.series)
    print(f"OK {dataset}/{model}: {len(manifest.series)} series, {origins} origins, cache {ident}")
    return 0


def _cmd_foundation_verify(dataset: str, model: str, root: str) -> int:
    """Verify one (dataset, model) cache against its manifest; FAIL names the fault."""
    try:
        if model != "fake":
            require_adapter_module(model)  # no heavy load: verify only reads bytes
        manifest = verify_foundation_cache(Path(root), dataset, model)
        ident = foundation_cache_id(Path(root), dataset, model)
    except (FoundationError, CacheError, ValueError, OSError) as exc:
        print(f"FAIL {dataset}/{model}: {exc}")
        return 1
    origins = sum(entry.n_origins for entry in manifest.series)
    print(f"OK {dataset}/{model}: {len(manifest.series)} series, {origins} origins, cache {ident}")
    return 0


def _cmd_foundation_compare(
    dataset: str,
    model: str,
    root: str,
    data_root: str,
    manifest_dir: str,
    sample: int,
    seed: int,
) -> int:
    """Recompute a sample of origins and report the drift; a diagnostic, so 0 unless it fails."""
    try:
        adapter, _spec = _foundation_adapter(model, "auto")
        report = compare_rebuild(
            Path(root),
            dataset,
            model,
            adapter,
            sample,
            seed,
            data_root=Path(data_root),
            manifest_dir=Path(manifest_dir),
        )
    except (FoundationError, CacheError, PrepareError, ValueError, OSError) as exc:
        print(f"FAIL {dataset}/{model}: {exc}")
        return 1
    print(
        f"OK {dataset}/{model}: compared {report.n_compared}/{report.n_total} origins, "
        f"max_abs {report.max_abs_diff:.3e}, max_rel {report.max_rel_diff:.3e}, "
        f"{report.n_nonfinite} non-finite values"
    )
    return 0


def _cmd_foundation_smoke(model: str, device: str) -> int:
    """Run the smoke check on one model; one JSON line, nonzero when a check fails."""
    try:
        adapter, spec = _foundation_adapter(model, device)
        report = run_smoke(spec, adapter)
    except FoundationError as exc:
        print(f"FAIL smoke/{model}: {exc}")
        return 1
    print(json.dumps(asdict(report)))
    return 0 if report.ok else 1


def _cmd_sealed_init(per_family: int, data_root: str, manifest_dir: str) -> int:
    """Create (or confirm) the sealed salt; print its hash, never the salt itself."""
    try:
        digest = init_salt(Path(data_root), Path(manifest_dir), per_family)
    except (ArenaError, ValueError, OSError) as exc:
        print(f"FAIL init-salt: {exc}")
        return 1
    manifest_file = synthetic_manifest_path(Path(manifest_dir))
    print(f"OK init-salt: salt sha256 {digest}; manifest {manifest_file}")
    return 0


def _cmd_sealed_build(
    dataset: str, data_root: str, manifest_dir: str, sealed_cache_root: str, horizon: int
) -> int:
    """Build each requested sealed cache, one OK/FAIL line per dataset."""
    failed = False
    root, data, manifests = Path(sealed_cache_root), Path(data_root), Path(manifest_dir)
    for name in SEALED_DATASETS if dataset == "all" else (dataset,):
        try:
            manifest = build_sealed_cache(name, data, manifests, root, horizon=horizon)
            ident = cache_id(root, name)
        except (ArenaError, CacheError, ValueError, OSError) as exc:
            failed = True
            print(f"FAIL {name}: {exc}")
        else:
            print(f"OK {name}: {len(manifest.series)} series, cache {ident}")
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
        if args.data_command in ("fetch", "verify", "prepare"):
            return _cmd_data(args.data_command, args.dataset, args.data_root, args.manifest_dir)
        parser.print_help()
        return 0
    if args.command == "cache":
        if args.cache_command == "build":
            return _cmd_cache(
                "build",
                args.dataset,
                args.cache_root,
                args.data_root,
                args.manifest_dir,
                horizon=args.horizon,
                per_family=args.per_family,
            )
        if args.cache_command == "verify":
            return _cmd_cache(
                "verify", args.dataset, args.cache_root, args.data_root, args.manifest_dir
            )
        parser.print_help()
        return 0
    if args.command == "foundation":
        if args.foundation_command == "build":
            return _cmd_foundation_build(
                args.dataset,
                args.model,
                args.root,
                args.cache_root,
                args.data_root,
                args.manifest_dir,
                args.batch_size,
                args.device,
                args.per_family,
            )
        if args.foundation_command == "verify":
            return _cmd_foundation_verify(args.dataset, args.model, args.root)
        if args.foundation_command == "smoke":
            return _cmd_foundation_smoke(args.model, args.device)
        if args.foundation_command == "compare":
            return _cmd_foundation_compare(
                args.dataset,
                args.model,
                args.root,
                args.data_root,
                args.manifest_dir,
                args.sample,
                args.seed,
            )
        parser.print_help()
        return 0
    if args.command == "sealed":
        if args.sealed_command == "init-salt":
            return _cmd_sealed_init(args.per_family, args.data_root, args.manifest_dir)
        if args.sealed_command == "build-cache":
            return _cmd_sealed_build(
                args.dataset,
                args.data_root,
                args.manifest_dir,
                args.sealed_cache_root,
                horizon=args.horizon,
            )
        parser.print_help()
        return 0
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
