"""Command-line entry point for EvolveTS."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import cast

from evolvets import __version__
from evolvets.arena.dev import DevPool, load_dev_pool
from evolvets.arena.errors import ArenaError
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.arena.oracles import oracle_dataset, oracle_to_obj
from evolvets.arena.pool import POOL_SPECS
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
    build_sealed_foundation_cache,
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
from evolvets.ledger.canonical import canonical_json
from evolvets.ledger.payloads import RunEnd
from evolvets.ledger.store import LedgerLockedError, LedgerReader, LedgerWriter
from evolvets.ledger.validate import validate_run
from evolvets.runner.gate1_grids import GATE1_GRIDS
from evolvets.runner.gate1_report import ReportError, build_report
from evolvets.runner.gate1_tune import DevPaths, TuneSpec, tune
from evolvets.runner.runs import RunError, current_git_commit, new_run_id
from evolvets.updater.baselines import BASELINE_IDS, BASELINES, DEFAULT_CONFIGS
from evolvets.updater.interfaces import ConfigValue

_DATASET_ORDER = ("etth1", "ecl")
#: ``--model`` choices: the test/CI fake, then the registry's model order.
_FOUNDATION_MODELS = ("fake", *MODELS)
#: Where ``evolvets foundation build`` installs its caches; ``arena replay``
#: joins the pooled spec's foundation caches from the same root.
_FOUNDATION_ROOT = "foundation_cache"


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
        "--root", default=_FOUNDATION_ROOT, help="foundation cache root directory"
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
        "--root", default=_FOUNDATION_ROOT, help="foundation cache root directory"
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
        "--root", default=_FOUNDATION_ROOT, help="foundation cache root directory"
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
    build_sealed_fm = sealed_sub.add_parser(
        "build-foundation-cache",
        help="compute one model's forecasts for a sealed pool, under the sealed cache root",
    )
    build_sealed_fm.add_argument(
        "dataset",
        choices=SEALED_DATASETS,
        help="which sealed pool (one model per build; the sealed simple cache must exist)",
    )
    build_sealed_fm.add_argument(
        "--model",
        choices=_FOUNDATION_MODELS,
        required=True,
        help="which model ('fake' for tests and CI; real models resolve through their adapters)",
    )
    build_sealed_fm.add_argument("--data-root", default="data", help="data root directory")
    build_sealed_fm.add_argument(
        "--manifest-dir", default="data/manifests", help="directory holding the manifests"
    )
    build_sealed_fm.add_argument(
        "--sealed-cache-root",
        default="data/sealed_cache",
        help="sealed cache root directory (never a public cache root)",
    )
    build_sealed_fm.add_argument(
        "--batch-size", type=int, default=32, help="contexts per adapter call (default: 32)"
    )
    build_sealed_fm.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="adapter device; auto uses CUDA when available",
    )

    arena = subparsers.add_parser(
        "arena", help="dev replays and hindsight oracles (no sealed data, no ledger)"
    )
    arena_sub = arena.add_subparsers(dest="arena_command", metavar="ARENA-COMMAND")
    arena_replay = arena_sub.add_parser(
        "replay", help="replay one baseline updater over a dev pool"
    )
    arena_oracle = arena_sub.add_parser("oracle", help="print the hindsight oracles of a dev pool")
    for command in (arena_replay, arena_oracle):
        command.add_argument(
            "dataset",
            choices=CACHE_DATASETS,
            help="which dev dataset (sealed pools are served by the sealed server)",
        )
        command.add_argument(
            "--pool-spec",
            choices=POOL_SPECS,
            default="simple",
            help="which experts the pool replays (default: simple)",
        )
        command.add_argument(
            "--foundation-model",
            nargs="+",
            metavar="NAME",
            help="foundation models to join (simple+foundation only; default: the registry)",
        )
        command.add_argument("--cache-root", default="cache_store", help="cache root directory")
        command.add_argument(
            "--foundation-root",
            default=_FOUNDATION_ROOT,
            help="foundation cache root directory (as 'evolvets foundation build')",
        )
        command.add_argument("--data-root", default="data", help="data root directory")
        command.add_argument(
            "--manifest-dir", default="data/manifests", help="directory holding the manifests"
        )
    arena_replay.add_argument(
        "--updater",
        choices=BASELINE_IDS,
        required=True,
        help="which baseline updater to replay",
    )
    arena_replay.add_argument(
        "--config",
        type=_json_object,
        metavar="JSON",
        help="updater config as a JSON object (default: the baseline's DEFAULT_CONFIGS entry)",
    )
    arena_replay.add_argument("--seed", type=int, default=0, help="updater seed (default: 0)")

    gate1 = subparsers.add_parser(
        "gate1", help="Gate 1: recorded dev tuning in the ledger and the report from it"
    )
    gate1_sub = gate1.add_subparsers(dest="gate1_command", metavar="GATE1-COMMAND")
    gate1_tune_cmd = gate1_sub.add_parser(
        "tune", help="replay every tuning grid over dev pools inside one recorded run"
    )
    gate1_tune_cmd.add_argument(
        "--dataset",
        choices=[*CACHE_DATASETS, "all"],
        action="append",
        required=True,
        help="which dev datasets to tune; repeatable, 'all' expands to every cache dataset",
    )
    gate1_tune_cmd.add_argument(
        "--pool-spec",
        choices=POOL_SPECS,
        action="append",
        required=True,
        help="which pool specs to replay; repeatable",
    )
    gate1_tune_cmd.add_argument(
        "--foundation-model",
        nargs="+",
        metavar="NAME",
        help="foundation models to join (simple+foundation only; default: the registry)",
    )
    gate1_tune_cmd.add_argument("--seed", type=int, default=0, help="updater seed (default: 0)")
    gate1_tune_cmd.add_argument(
        "--max-wall-seconds",
        type=float,
        default=172800.0,
        help="wall-clock budget of the run in seconds (default: 172800, two days)",
    )
    gate1_tune_cmd.add_argument("--ledger-root", default="ledgers", help="ledger root directory")
    gate1_tune_cmd.add_argument(
        "--run-id", metavar="ID", help="run id (default: a fresh gate1-tune id)"
    )
    gate1_tune_cmd.add_argument("--cache-root", default="cache_store", help="cache root directory")
    gate1_tune_cmd.add_argument(
        "--foundation-root",
        default=_FOUNDATION_ROOT,
        help="foundation cache root directory (as 'evolvets foundation build')",
    )
    gate1_tune_cmd.add_argument("--data-root", default="data", help="data root directory")
    gate1_tune_cmd.add_argument(
        "--manifest-dir", default="data/manifests", help="directory holding the manifests"
    )
    gate1_report_cmd = gate1_sub.add_parser(
        "report", help="regenerate the deterministic Markdown report of named dev runs"
    )
    gate1_report_cmd.add_argument(
        "--ledger-root", default="ledgers", help="ledger root directory (contains <run_id>/)"
    )
    gate1_report_cmd.add_argument(
        "--run",
        metavar="ID",
        action="append",
        required=True,
        help="run id to report; repeatable, in report order",
    )
    gate1_report_cmd.add_argument(
        "--bootstrap-seed", type=int, default=0, help="seed of the gaps' paired bootstrap"
    )
    gate1_report_output = gate1_report_cmd.add_mutually_exclusive_group(required=True)
    gate1_report_output.add_argument("--out", metavar="PATH", help="write the report to PATH")
    gate1_report_output.add_argument(
        "--check", metavar="PATH", help="compare the regenerated report against PATH"
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


def _cmd_sealed_foundation_build(
    dataset: str,
    model: str,
    data_root: str,
    manifest_dir: str,
    sealed_cache_root: str,
    batch_size: int,
    device: str,
) -> int:
    """Build one model's sealed foundation cache; one FAIL line on any refusal."""
    try:
        adapter, spec = _foundation_adapter(model, device)
        manifest = build_sealed_foundation_cache(
            dataset,
            adapter,
            spec,
            Path(data_root),
            Path(manifest_dir),
            Path(sealed_cache_root),
            batch_size,
        )
        ident = foundation_cache_id(Path(sealed_cache_root), dataset, model)
    except (ArenaError, FoundationError, CacheError, ValueError, OSError) as exc:
        print(f"FAIL {dataset}/{model}: {exc}")
        return 1
    origins = sum(entry.n_origins for entry in manifest.series)
    print(f"OK {dataset}/{model}: {len(manifest.series)} series, {origins} origins, cache {ident}")
    return 0


def _json_object(text: str) -> dict[str, ConfigValue]:
    """Parse ``--config`` as a JSON object (an argparse type)."""
    try:
        obj: object = json.loads(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--config must be JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise argparse.ArgumentTypeError(f"--config must be a JSON object, got {text!r}")
    return cast("dict[str, ConfigValue]", obj)


def _arena_pool(args: argparse.Namespace) -> DevPool:
    """Load the dev pool an arena command replays or oracles over."""
    models = tuple(args.foundation_model) if args.foundation_model else None
    return load_dev_pool(
        args.dataset,
        Path(args.cache_root),
        Path(args.data_root),
        Path(args.manifest_dir),
        pool_spec=args.pool_spec,
        foundation_root=(
            Path(args.foundation_root) if args.pool_spec == "simple+foundation" else None
        ),
        foundation_models=models,
    )


def _cmd_arena_replay(args: argparse.Namespace) -> int:
    """Replay one baseline over a dev pool; one JSON line, FAIL on an ArenaError."""
    config: Mapping[str, ConfigValue] = (
        args.config if args.config is not None else DEFAULT_CONFIGS[args.updater]
    )
    try:
        pool = _arena_pool(args)
        started = time.monotonic()
        result = replay_dataset(BASELINES[args.updater], config, args.seed, pool)
    except ArenaError as exc:
        print(f"FAIL {args.dataset}: {exc}")
        return 1
    obj = result_to_obj(result)
    del obj["series"]
    print(canonical_json({"result": obj, "wall_seconds": time.monotonic() - started}))
    return 0  # a failed updater is a result, not an error


def _cmd_arena_oracle(args: argparse.Namespace) -> int:
    """Print one dev pool's hindsight oracles; one JSON line, FAIL on an ArenaError."""
    try:
        pool = _arena_pool(args)
        started = time.monotonic()
        result = oracle_dataset(pool)
    except ArenaError as exc:
        print(f"FAIL {args.dataset}: {exc}")
        return 1
    obj = oracle_to_obj(result)
    del obj["series"]
    print(canonical_json({"result": obj, "wall_seconds": time.monotonic() - started}))
    return 0


def _gate1_tune_spec(args: argparse.Namespace) -> TuneSpec:
    """Build the TuneSpec from the CLI arguments, first-seen order, duplicates dropped."""
    datasets: list[str] = []
    for name in args.dataset:
        for dataset in CACHE_DATASETS if name == "all" else (name,):
            if dataset not in datasets:
                datasets.append(dataset)
    pool_specs: list[str] = []
    for pool_spec in args.pool_spec:
        if pool_spec not in pool_specs:
            pool_specs.append(pool_spec)
    return TuneSpec(
        datasets=tuple(datasets),
        pool_specs=tuple(pool_specs),
        foundation_models=tuple(args.foundation_model) if args.foundation_model else None,
        seed=args.seed,
        max_wall_seconds=args.max_wall_seconds,
    )


def _cmd_gate1_tune(args: argparse.Namespace) -> int:
    """Run the dev tuning sweep in one recorded run; OK plus the ledger's own summary."""
    try:
        git_commit = current_git_commit(Path.cwd())
    except RunError as exc:
        print(f"FAIL {exc}")
        return 1
    spec = _gate1_tune_spec(args)
    paths = DevPaths(
        cache_root=Path(args.cache_root),
        foundation_root=Path(args.foundation_root),
        data_root=Path(args.data_root),
        manifest_dir=Path(args.manifest_dir),
    )
    run_id = args.run_id if args.run_id is not None else new_run_id("gate1-tune")
    ledger_root = Path(args.ledger_root)
    main_stream = ledger_root / run_id / "main.jsonl"
    reused = main_stream.exists()
    try:
        status = tune(
            spec,
            paths,
            ledger_root=ledger_root,
            run_id=run_id,
            git_commit=git_commit,
            grids=GATE1_GRIDS,
        )
    except Exception as exc:
        # A run this invocation started and aborted is reported by run id; a
        # refusal before start (bad spec, unloadable pool, reused run id) is not.
        if main_stream.exists() and not reused:
            print(f"FAIL {run_id} aborted: {exc}")
        else:
            print(f"FAIL {exc}")
        return 1
    records = list(LedgerReader(ledger_root, run_id).records())
    run_end = records[-1].payload
    summary = run_end.summary if isinstance(run_end, RunEnd) else {}
    print(f"OK {run_id} {status}")
    print(canonical_json(summary))
    return 0 if status == "completed" else 2


def _cmd_gate1_report(args: argparse.Namespace) -> int:
    """Build the dev report from the ledger, then write it or check a copy.

    The report is built before anything is written, so a refusal leaves
    ``--out`` unwritten; ``--check`` reports the first differing line.
    """
    try:
        report = build_report(Path(args.ledger_root), args.run, bootstrap_seed=args.bootstrap_seed)
    except ReportError as exc:
        print(f"FAIL {exc}")
        return 1
    if args.out is not None:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"OK wrote {args.out}")
        return 0
    return _check_report(args.check, report)


def _check_report(path: str, report: str) -> int:
    """Compare the built report with the file at ``path``; FAIL names the first difference."""
    try:
        existing = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"FAIL {exc}")
        return 1
    built = report.splitlines()
    written = existing.splitlines()
    for number, (left, right) in enumerate(zip(built, written, strict=False), start=1):
        if left != right:
            print(f"FAIL {path}: differs at line {number}")
            return 1
    if len(built) != len(written):
        print(f"FAIL {path}: differs at line {min(len(built), len(written)) + 1}")
        return 1
    print(f"OK {path} matches ({len(built)} lines)")
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
        if args.sealed_command == "build-foundation-cache":
            return _cmd_sealed_foundation_build(
                args.dataset,
                args.model,
                args.data_root,
                args.manifest_dir,
                args.sealed_cache_root,
                args.batch_size,
                args.device,
            )
        parser.print_help()
        return 0
    if args.command == "arena":
        if args.arena_command == "replay":
            return _cmd_arena_replay(args)
        if args.arena_command == "oracle":
            return _cmd_arena_oracle(args)
        parser.print_help()
        return 0
    if args.command == "gate1":
        if args.gate1_command == "tune":
            return _cmd_gate1_tune(args)
        if args.gate1_command == "report":
            return _cmd_gate1_report(args)
        parser.print_help()
        return 0
    if args.command is None:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
