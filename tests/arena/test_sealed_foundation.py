"""Sealed foundation-pool tests (TASK_009 PR D2: PR D's test 5, sealed half).

The sealed foundation build refuses public roots and writes only under the
sealed cache root, joining the sealed simple cache origin for origin; the
sealed server replays a fake-built sealed foundation cache end to end under
pool spec ``simple+foundation``, and its blob equals an in-process replay's.
The foundation expert is the torch-free fake adapter, so every test runs in
plain CI.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.errors import ArenaError
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.arena.pool import pool_expert_names
from evolvets.arena.realdata.fetch import fetch
from evolvets.arena.realdata.prepare import prepare
from evolvets.arena.sealed.client import SealedError, SealedPaths, SealedRequest, evaluate_sealed
from evolvets.arena.sealed.pools import (
    build_sealed_cache,
    build_sealed_foundation_cache,
    init_salt,
    load_sealed_pool,
)
from evolvets.cache.store import cache_id, read_manifest
from evolvets.experts.pool import EXPERT_NAMES
from evolvets.foundation.fake import FAKE_SPEC, FakeAdapter
from evolvets.foundation.registry import ModelSpec
from evolvets.foundation.store import (
    foundation_cache_id,
    read_foundation_manifest,
    verify_foundation_cache,
)
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import Baseline
from evolvets.ledger.store import LedgerReader
from evolvets.runner import cli
from tests.arena.sealed_toys import EqualWeights
from tests.arena.test_realdata_prepare import _build_etth1_fixture, _install
from tests.arena.test_sealed_server import _begin_run

TOYS = "tests.arena.sealed_toys"
ROWS = 1200
_START = 287  # first_origin(24)
_SYNTHETIC_ORIGINS = 6 * (2000 - _START)  # six worlds, 2000 rows each
_OK_LINE = re.compile(
    r"^OK synthetic-sealed/fake: (\d+) series, (\d+) origins, cache ([0-9a-f]{64})$", re.MULTILINE
)


def _build_fake(synthetic: bool, data_root: Path, manifest_dir: Path, sealed_root: Path) -> None:
    """Build the fake foundation cache for one sealed pool under ``sealed_root``."""
    build_sealed_foundation_cache(
        "synthetic-sealed" if synthetic else "etth1",
        FakeAdapter(),
        FAKE_SPEC,
        data_root,
        manifest_dir,
        sealed_root,
        batch_size=32,
    )


def _custom_spec(name: str, etth1: str) -> ModelSpec:
    """A test spec whose only difference from the fake is its etth1 overlap."""
    return replace(FAKE_SPEC, name=name, pretraining_overlap={"etth1": etth1, "ecl": "no"})


@pytest.fixture(scope="module")
def synthetic_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A salted synthetic sealed pool with one world per family and its caches built."""
    base = tmp_path_factory.mktemp("sealed-foundation-synthetic")
    data_root, manifest_dir = base / "data", base / "manifests"
    init_salt(data_root, manifest_dir, per_family=1)
    sealed_root = base / "sealed_cache"
    build_sealed_cache("synthetic-sealed", data_root, manifest_dir, sealed_root)
    _build_fake(True, data_root, manifest_dir, sealed_root)
    return {
        "base": base,
        "data_root": data_root,
        "manifest_dir": manifest_dir,
        "sealed_root": sealed_root,
    }


@pytest.fixture(scope="module")
def real_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The fixture ETTh1, prepared, with its sealed caches built."""
    base = tmp_path_factory.mktemp("sealed-foundation-real")
    etth1 = _build_etth1_fixture(base / "fixtures", rows=ROWS)
    monkeypatch = pytest.MonkeyPatch()
    try:
        _install(monkeypatch, etth1=etth1)
        data_root, manifest_dir = base / "data", base / "manifests"
        fetch(etth1, data_root, manifest_dir, today=date(2026, 10, 11))
        prepare("etth1", data_root, manifest_dir)
        sealed_root = base / "sealed_cache"
        build_sealed_cache("etth1", data_root, manifest_dir, sealed_root)
        _build_fake(False, data_root, manifest_dir, sealed_root)
        return {"data_root": data_root, "manifest_dir": manifest_dir, "sealed_root": sealed_root}
    finally:
        monkeypatch.undo()


def _load_pooled(env: dict[str, Path], dataset: str, models: tuple[str, ...] | None = None) -> Any:
    """Load a sealed pool with the foundation models joined."""
    return load_sealed_pool(
        dataset,
        env["data_root"],
        env["manifest_dir"],
        env["sealed_root"],
        pool_spec="simple+foundation",
        foundation_models=models,
    )


def _baseline(ledger_root: Path, run_id: str) -> Baseline:
    """The last baseline of the run's sealed stream."""
    records = list(LedgerReader(ledger_root, run_id).records(["sealed"]))
    assert records, "expected at least one sealed-stream record"
    assert isinstance(records[-1].payload, Baseline)
    return records[-1].payload


def _blob_result(ledger_root: Path, run_id: str, baseline: Baseline) -> dict[str, Any]:
    """The parsed per-task blob's result object of ``baseline``."""
    assert baseline.per_task_blob is not None
    raw = BlobStore(ledger_root / run_id / "blobs").get(baseline.per_task_blob)
    obj: Any = json.loads(raw.decode("utf-8"))
    return obj["result"]


# --- Test 5a: the build writes only under the sealed cache root ----------------------------------


def test_build_writes_only_under_the_sealed_root(synthetic_env: dict[str, Path]) -> None:
    """The fake cache lands beside the sealed simple cache and nowhere else."""
    env = synthetic_env
    manifest = read_foundation_manifest(env["sealed_root"], "synthetic-sealed", "fake")
    assert manifest.simple_cache_id == cache_id(env["sealed_root"], "synthetic-sealed")
    assert manifest.horizon == 24
    # the sealed root is the only thing the build created or touched: the
    # base directory still holds exactly the salted data, its manifests and
    # the sealed root, and every file under the root belongs to a cache.
    assert sorted(path.name for path in env["base"].iterdir()) == [
        "data",
        "manifests",
        "sealed_cache",
    ]
    files = sorted(
        str(path.relative_to(env["sealed_root"]))
        for path in env["sealed_root"].rglob("*")
        if path.is_file()
    )
    assert files
    assert all(path.startswith("synthetic-sealed/") for path in files)
    top = sorted(
        path.name for path in (env["sealed_root"] / "synthetic-sealed").iterdir() if path.is_dir()
    )
    assert top == ["fake", "series"]  # the foundation cache beside the simple cache's series
    assert verify_foundation_cache(env["sealed_root"], "synthetic-sealed", "fake").dataset == (
        "synthetic-sealed"
    )


def test_real_build_joins_the_sealed_simple_cache(real_env: dict[str, Path]) -> None:
    """On ETTh1 the foundation series equal the sealed simple cache's row for row."""
    env = real_env
    manifest = read_foundation_manifest(env["sealed_root"], "etth1", "fake")
    simple = read_manifest(env["sealed_root"], "etth1")
    assert manifest.simple_cache_id == cache_id(env["sealed_root"], "etth1")
    assert [entry.series_id for entry in manifest.series] == [
        entry.series_id for entry in simple.series
    ]
    for entry, cached in zip(manifest.series, simple.series, strict=True):
        assert (
            entry.length,
            entry.period,
            entry.first_origin,
            entry.n_origins,
            entry.values_sha256,
        ) == (
            cached.length,
            cached.period,
            cached.first_origin,
            cached.n_origins,
            cached.values_sha256,
        )
    assert simple.series[0].length == 926 + 24  # sealed.end + horizon: no reserve value forecast


def test_build_requires_the_sealed_simple_cache(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """Without a sealed simple cache to join, the build refuses before forecasting."""
    env = synthetic_env
    empty = tmp_path / "sealed_cache"
    with pytest.raises(ArenaError, match="simple cache"):
        build_sealed_foundation_cache(
            "synthetic-sealed",
            FakeAdapter(),
            FAKE_SPEC,
            env["data_root"],
            env["manifest_dir"],
            empty,
            8,
        )
    assert not empty.exists()


def test_build_refuses_public_roots(synthetic_env: dict[str, Path], tmp_path: Path) -> None:
    """A root that is or sits under cache_store/ or foundation_cache/ is refused."""
    env = synthetic_env
    bad_roots = [
        tmp_path / "cache_store",
        tmp_path / "foundation_cache",
        tmp_path / "nested" / "cache_store" / "sealed",
        tmp_path / "foundation_cache" / "sealed",
    ]
    for bad in bad_roots:
        with pytest.raises(ArenaError, match="public cache root"):
            build_sealed_foundation_cache(
                "synthetic-sealed",
                FakeAdapter(),
                FAKE_SPEC,
                env["data_root"],
                env["manifest_dir"],
                bad,
                8,
            )
        assert not bad.exists()
    with pytest.raises(ArenaError, match="public cache root"):
        load_sealed_pool(
            "synthetic-sealed",
            env["data_root"],
            env["manifest_dir"],
            tmp_path / "foundation_cache",
            pool_spec="simple+foundation",
        )


# --- Test 5b: the sealed pool with foundation experts --------------------------------------------


def test_sealed_pool_with_foundation_spec(synthetic_env: dict[str, Path]) -> None:
    """The joined pool replays six experts and records its foundation cache id."""
    env = synthetic_env
    pool = _load_pooled(env, "synthetic-sealed", ("fake",))
    assert pool.pool_spec == "simple+foundation"
    assert pool_expert_names(pool) == (*EXPERT_NAMES, "fake")
    result = replay_dataset(EqualWeights, {}, 3, pool)
    assert result.status == "ok"
    assert result.foundation_cache_ids == (
        foundation_cache_id(env["sealed_root"], "synthetic-sealed", "fake"),
    )
    assert result.leakage_risk is False  # synthetic data is no model's pretraining data


def test_sealed_pool_spec_refusals(synthetic_env: dict[str, Path]) -> None:
    """Bad specs, a missing cache and a cache that does not join are refused."""
    env = synthetic_env
    args = (env["data_root"], env["manifest_dir"], env["sealed_root"])
    with pytest.raises(ArenaError, match="takes no foundation"):
        load_sealed_pool("synthetic-sealed", *args, pool_spec="simple", foundation_models=("fake",))
    with pytest.raises(ArenaError, match="pool_spec"):
        load_sealed_pool("synthetic-sealed", *args, pool_spec="foundation")
    with pytest.raises(ArenaError, match="is missing"):
        load_sealed_pool(
            "synthetic-sealed", *args, pool_spec="simple+foundation", foundation_models=("nope",)
        )

    manifest_file = env["sealed_root"] / "synthetic-sealed" / "fake" / "manifest.json"
    original = manifest_file.read_bytes()
    tampered = json.loads(original.decode("utf-8"))
    tampered["simple_cache_id"] = "0" * 64
    try:
        manifest_file.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(ArenaError, match="was built against simple cache"):
            _load_pooled(env, "synthetic-sealed", ("fake",))
    finally:  # the module fixture keeps serving later tests
        manifest_file.write_bytes(original)


def test_sealed_leakage_flag_on_real_data(real_env: dict[str, Path]) -> None:
    """A pooled model with yes/unknown etth1 overlap flags the sealed result too."""
    env = real_env
    build_sealed_foundation_cache(
        "etth1",
        FakeAdapter("leaky"),
        _custom_spec("leaky", "yes"),
        env["data_root"],
        env["manifest_dir"],
        env["sealed_root"],
        64,
    )
    clean = _load_pooled(env, "etth1", ("fake",))
    assert replay_dataset(EqualWeights, {}, 0, clean).leakage_risk is False
    leaky = _load_pooled(env, "etth1", ("leaky",))
    result = replay_dataset(EqualWeights, {}, 0, leaky)
    assert result.status == "ok"
    assert result.leakage_risk is True


# --- Test 5c: the sealed server replays the foundation pool end to end ---------------------------


def test_server_replays_the_foundation_pool_end_to_end(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """A pooled sealed evaluation through the server equals the in-process replay."""
    env = synthetic_env
    ledger_root = tmp_path / "ledgers"
    _begin_run(ledger_root, "run-1")
    spy = tmp_path / "spy.jsonl"
    receipt = evaluate_sealed(
        SealedRequest(
            dataset="synthetic-sealed",
            updater_id="toy",
            updater_ref=f"{TOYS}:Spy",
            config={"spy_path": str(spy)},
            seed=3,
            pool_spec="simple+foundation",
            foundation_models=("fake",),
        ),
        ledger_root=ledger_root,
        run_id="run-1",
        paths=SealedPaths(
            data_root=env["data_root"],
            manifest_dir=env["manifest_dir"],
            sealed_cache_root=env["sealed_root"],
        ),
    )
    assert receipt.status == "ok"
    baseline = _baseline(ledger_root, "run-1")
    result = _blob_result(ledger_root, "run-1", baseline)
    expected = replay_dataset(EqualWeights, {}, 3, _load_pooled(env, "synthetic-sealed", ("fake",)))
    assert result == result_to_obj(expected)
    assert result["pool_spec"] == "simple+foundation"
    assert result["foundation_cache_ids"] == [
        foundation_cache_id(env["sealed_root"], "synthetic-sealed", "fake")
    ]
    assert result["leakage_risk"] is False

    # the pool reached the updater: one fresh worker per series saw six experts
    lines = [json.loads(line) for line in spy.read_text(encoding="utf-8").splitlines()]
    inits = [line["init"] for line in lines if "init" in line]
    assert len(inits) == len(expected.series)
    assert {init[0] for init in inits} == {len(EXPERT_NAMES) + 1}
    first_update = next(line["update"] for line in lines if "update" in line)
    assert first_update["context"]["expert_names"] == [*EXPERT_NAMES, "fake"]


def test_server_refuses_a_pool_it_cannot_serve(tmp_path: Path) -> None:
    """A pooled request without the foundation cache fails naming the log, not silently."""
    base = tmp_path
    data_root, manifest_dir = base / "data", base / "manifests"
    init_salt(data_root, manifest_dir, per_family=1)
    sealed_root = base / "sealed_cache"
    build_sealed_cache("synthetic-sealed", data_root, manifest_dir, sealed_root)
    ledger_root = base / "ledgers"
    _begin_run(ledger_root, "run-1")
    with pytest.raises(SealedError) as excinfo:
        evaluate_sealed(
            SealedRequest(
                dataset="synthetic-sealed",
                updater_id="toy",
                updater_ref=f"{TOYS}:Spy",
                config={},
                seed=3,
                pool_spec="simple+foundation",
                foundation_models=("fake",),
            ),
            ledger_root=ledger_root,
            run_id="run-1",
            paths=SealedPaths(
                data_root=data_root, manifest_dir=manifest_dir, sealed_cache_root=sealed_root
            ),
        )
    assert "server.log" in str(excinfo.value)
    log = (data_root / "sealed" / "server.log").read_text(encoding="utf-8")
    assert "ArenaError" in log and "fake" in log
    assert list(LedgerReader(ledger_root, "run-1").records(["sealed"])) == []


# --- the CLI ------------------------------------------------------------------------------------


def test_cli_build_foundation_cache(
    synthetic_env: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The sealed CLI builds the fake cache, and refuses a public root."""
    env = synthetic_env
    sealed_root = tmp_path / "sealed_cache"
    common = [
        "--data-root",
        str(env["data_root"]),
        "--manifest-dir",
        str(env["manifest_dir"]),
        "--sealed-cache-root",
        str(sealed_root),
    ]
    assert cli.main(["sealed", "build-cache", "synthetic-sealed", *common]) == 0
    argv = [
        "sealed",
        "build-foundation-cache",
        "synthetic-sealed",
        "--model",
        "fake",
        *common,
    ]
    assert cli.main(argv) == 0
    match = _OK_LINE.search(capsys.readouterr().out)
    assert match is not None
    assert match.group(1, 2, 3) == (
        "6",
        str(_SYNTHETIC_ORIGINS),
        foundation_cache_id(sealed_root, "synthetic-sealed", "fake"),
    )

    public = tmp_path / "foundation_cache"
    assert cli.main([*argv[:-1], str(public)]) == 1
    assert "public cache root" in capsys.readouterr().out
    assert not public.exists()
