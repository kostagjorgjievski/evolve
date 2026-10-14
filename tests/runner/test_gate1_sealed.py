"""Gate 1 on sealed: the plan derived from a completed dev tune run (TASK_012 PR A).

The fixture pairs a synthetic-dev tune environment with a salted
synthetic-sealed one (``per_family = 1``, the fake foundation model built on
both sides), so ``plan_from_tune_run`` maps one completed tune run onto
oracles plus dev-selected baselines and skips the rest. Grids stay at one or
two baselines with one configuration each. The recorded run that executes
the plan is the next PR of this stack; these tests plan only.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.errors import ArenaError
from evolvets.arena.sealed.client import SealedPaths
from evolvets.arena.sealed.pools import (
    build_sealed_cache,
    build_sealed_foundation_cache,
    init_salt,
)
from evolvets.arena.versions import protocol_versions
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.foundation.fake import FAKE_SPEC, FakeAdapter
from evolvets.foundation.store import build_foundation_cache
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.canonical import canonical_bytes
from evolvets.ledger.payloads import Baseline, RunEnd, RunStart
from evolvets.ledger.schema import Budget, Record
from evolvets.ledger.store import LedgerReader, LedgerWriter
from evolvets.runner import gate1_tune
from evolvets.runner.gate1_sealed import SEALED_CONDITION, plan_from_tune_run
from evolvets.runner.gate1_tune import TUNE_CONDITION, DevPaths, TuneSpec, tune
from evolvets.runner.runs import RunError

COMMIT = "c" * 40
SMALL_GRIDS: dict[str, list[dict[str, Any]]] = {"equal": [{}], "hedge": [{"eta": 0.5}]}


@pytest.fixture(scope="module")
def env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A synthetic-dev tune environment and its synthetic-sealed counterpart."""
    base = tmp_path_factory.mktemp("gate1-sealed-dev")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-dev", base, base, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    foundation_root = base / "foundation_cache"
    build_foundation_cache(
        "synthetic-dev",
        FakeAdapter(FAKE_SPEC.name),
        FAKE_SPEC,
        foundation_root,
        cache_root,
        base,
        base,
        7,
        per_family=1,
    )
    sealed_base = tmp_path_factory.mktemp("gate1-sealed-pools")
    data_root, manifest_dir = sealed_base / "data", sealed_base / "manifests"
    init_salt(data_root, manifest_dir, per_family=1)
    sealed_root = sealed_base / "sealed_cache"
    build_sealed_cache("synthetic-sealed", data_root, manifest_dir, sealed_root)
    build_sealed_foundation_cache(
        "synthetic-sealed",
        FakeAdapter(FAKE_SPEC.name),
        FAKE_SPEC,
        data_root,
        manifest_dir,
        sealed_root,
        32,
    )
    return {
        "dev_paths": DevPaths(
            cache_root=cache_root,
            foundation_root=foundation_root,
            data_root=base,
            manifest_dir=base,
        ),
        "sealed_paths": SealedPaths(
            data_root=data_root, manifest_dir=manifest_dir, sealed_cache_root=sealed_root
        ),
        "data_root": data_root,
        "sealed_root": sealed_root,
    }


def tune_spec(**overrides: Any) -> TuneSpec:
    """A TuneSpec over both pool specs, overridable per test."""
    values: dict[str, Any] = {
        "datasets": ("synthetic-dev",),
        "pool_specs": ("simple", "simple+foundation"),
        "foundation_models": ("fake",),
        "seed": 7,
        "max_wall_seconds": 10000.0,
    }
    values.update(overrides)
    return TuneSpec(**values)


def run_tune(
    env: dict[str, Any],
    ledger_root: Path,
    *,
    run_id: str,
    grids: dict[str, list[dict[str, Any]]] | None = None,
    spec: TuneSpec | None = None,
) -> str:
    """Run one tuning sweep against the fixture; returns the run_end status."""
    return tune(
        spec if spec is not None else tune_spec(),
        env["dev_paths"],
        ledger_root=ledger_root,
        run_id=run_id,
        git_commit=COMMIT,
        grids=SMALL_GRIDS if grids is None else grids,
    )


def read_records(
    ledger_root: Path, run_id: str, streams: tuple[str, ...] = ("main",)
) -> list[Record]:
    """The records of one run, in reader order."""
    return list(LedgerReader(ledger_root, run_id).records(streams))


def selection_of(ledger_root: Path, run_id: str) -> Any:
    """The run_end summary's 'selected' of one completed tune run."""
    records = read_records(ledger_root, run_id)
    payload = records[-1].payload
    assert isinstance(payload, RunEnd)
    return payload.summary["selected"]


@pytest.fixture(scope="module")
def tune_run(env: dict[str, Any], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One completed tune run over both pool specs in a shared ledger root."""
    ledger_root = tmp_path_factory.mktemp("gate1-sealed-ledgers") / "ledgers"
    assert run_tune(env, ledger_root, run_id="tune-1") == "completed"
    return {"env": env, "ledger_root": ledger_root, "tune_run_id": "tune-1"}


# --- Test 1: the plan -----------------------------------------------------------------------------


def test_plan_items_and_skipped(env: dict[str, Any], tmp_path: Path) -> None:
    """Oracles first per pool, then the selected baselines in BASELINE_IDS order."""
    ledger = tmp_path / "ledgers"
    run_tune(env, ledger, run_id="tune-plan")
    plan = plan_from_tune_run(ledger, "tune-plan")
    selection = selection_of(ledger, "tune-plan")
    assert [(item.dataset, item.pool_spec, item.updater_id) for item in plan.items] == [
        ("synthetic-sealed", "simple", "oracle"),
        ("synthetic-sealed", "simple+foundation", "oracle"),
        ("synthetic-sealed", "simple", "equal"),
        ("synthetic-sealed", "simple", "hedge"),
        ("synthetic-sealed", "simple+foundation", "equal"),
        ("synthetic-sealed", "simple+foundation", "hedge"),
    ]
    for item in plan.items:
        assert item.foundation_models == (
            ("fake",) if item.pool_spec == "simple+foundation" else None
        )
        if item.updater_id == "oracle":
            assert item.config == {}
        else:
            assert (
                item.config == selection["synthetic-dev"][item.pool_spec][item.updater_id]["config"]
            )
    assert plan.skipped == tuple(
        (dataset, pool_spec, baseline_id)
        for dataset in ("synthetic-sealed",)
        for pool_spec in ("simple", "simple+foundation")
        for baseline_id in ("ftl", "fixed_share", "adahedge")
    )
    assert plan.tune_run_id == "tune-plan"
    assert plan.tune_run_end_record_id == read_records(ledger, "tune-plan")[-1].record_id
    assert plan.tune_git_commit == COMMIT
    assert plan.seed == 7
    assert plan.previous_sealed_runs == ()


def test_plan_skips_a_baseline_with_no_ok_grid_point(env: dict[str, Any], tmp_path: Path) -> None:
    """A baseline whose every grid point failed is skipped, not planned."""
    ledger = tmp_path / "ledgers"
    run_tune(
        env,
        ledger,
        run_id="tune-failed",
        grids={"hedge": [{"eta": 0.0}]},
        spec=tune_spec(pool_specs=("simple",)),
    )
    plan = plan_from_tune_run(ledger, "tune-failed")
    assert [entry for entry in plan.skipped if entry[2] == "hedge"] == [
        ("synthetic-sealed", "simple", "hedge")
    ]
    assert [item.updater_id for item in plan.items] == ["oracle"]
    assert all(item.updater_id != "hedge" for item in plan.items)


# --- Test 2: plan refusals ------------------------------------------------------------------------


def _hand_tune_run(
    ledger_root: Path,
    run_id: str,
    config_bytes: bytes,
    *,
    condition: str = TUNE_CONDITION,
    versions: dict[str, str] | None = None,
    run_end_status: str | None = "completed",
    selected: Any = None,
) -> None:
    """Write a hand-built tune-like main stream; only what plan_from_tune_run reads."""
    blobs = BlobStore(ledger_root / run_id / "blobs")
    with LedgerWriter(ledger_root, run_id, "main") as writer:
        writer.append(
            "run_start",
            RunStart(
                condition=condition,
                seed=7,
                requires_predictions=False,
                config_blob=blobs.put(config_bytes),
                git_commit=COMMIT,
                protocol_versions=versions if versions is not None else protocol_versions(),
                budget_limits=Budget(wall_seconds=1.0, evaluations=1),
            ),
        )
        if run_end_status is not None:
            writer.append("run_end", RunEnd(status=run_end_status, summary={"selected": selected}))


def _copy_run(ledger_root: Path, run_id: str, dest_root: Path) -> Path:
    """Copy one run's directory into another ledger root."""
    dest = dest_root / run_id
    shutil.copytree(ledger_root / run_id, dest)
    return dest


def test_plan_refusals(
    tune_run: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every refusal is a RunError raised before anything is written."""
    env, ledger_root = tune_run["env"], tune_run["ledger_root"]
    first = read_records(ledger_root, "tune-1")[0].payload
    assert isinstance(first, RunStart)
    config_bytes = BlobStore(ledger_root / "tune-1" / "blobs").get(first.config_blob)
    real_selection = selection_of(ledger_root, "tune-1")

    with pytest.raises(RunError, match="does not verify"):
        plan_from_tune_run(tmp_path / "empty-ledgers", "missing")

    tampered_root = tmp_path / "tampered"
    _copy_run(ledger_root, "tune-1", tampered_root)
    stream = tampered_root / "tune-1" / "main.jsonl"
    stream.write_text(
        stream.read_text(encoding="utf-8").replace("gate1-dev-tune", "gate1-dev-tunX"),
        encoding="utf-8",
    )
    with pytest.raises(RunError, match="does not verify"):
        plan_from_tune_run(tampered_root, "tune-1")

    with_sealed = tmp_path / "with-sealed"
    _copy_run(ledger_root, "tune-1", with_sealed)
    with LedgerWriter(with_sealed, "tune-1", "sealed") as writer:
        writer.append(
            "baseline",
            Baseline(
                updater_id="toy",
                hyperparams={},
                pool="sealed",
                status="ok",
                metrics={},
                per_task_blob=None,
                error=None,
            ),
        )
    with pytest.raises(RunError, match="sealed stream"):
        plan_from_tune_run(with_sealed, "tune-1")

    exhausted = tmp_path / "exhausted"
    run_tune(env, exhausted, run_id="tune-x", spec=tune_spec(max_wall_seconds=1e-9))
    with pytest.raises(RunError, match="budget_exhausted"):
        plan_from_tune_run(exhausted, "tune-x")

    aborted = tmp_path / "aborted"
    real_oracle = gate1_tune.oracle_dataset

    def failing_oracle(pool: object) -> object:
        raise ArenaError("no oracles today")

    monkeypatch.setattr(gate1_tune, "oracle_dataset", failing_oracle)
    with pytest.raises(ArenaError):
        run_tune(env, aborted, run_id="tune-a")
    monkeypatch.setattr(gate1_tune, "oracle_dataset", real_oracle)
    with pytest.raises(RunError, match="aborted"):
        plan_from_tune_run(aborted, "tune-a")

    hand = tmp_path / "hand"
    _hand_tune_run(hand, "tune-noend", config_bytes, run_end_status=None, selected={})
    with pytest.raises(RunError, match="no run_end"):
        plan_from_tune_run(hand, "tune-noend")

    _hand_tune_run(hand, "tune-wrong", config_bytes, condition="gate1-dev-tune-x", selected={})
    with pytest.raises(RunError, match="condition"):
        plan_from_tune_run(hand, "tune-wrong")

    _hand_tune_run(
        hand,
        "tune-versions",
        config_bytes,
        versions={
            **protocol_versions(),
            "arena": "0.0",
        },
        selected={},
    )
    with pytest.raises(RunError, match="protocol versions"):
        plan_from_tune_run(hand, "tune-versions")

    _hand_tune_run(hand, "tune-selection", config_bytes, selected=real_selection)
    with pytest.raises(RunError, match="selection differs"):
        plan_from_tune_run(hand, "tune-selection")


# --- Test 8 (plan part): once per tune run --------------------------------------------------------


def _hand_sealed_run(
    ledger_root: Path, tune_run_id: str, run_id: str, *, status: str | None = "completed"
) -> None:
    """Hand-build a sealed run of ``tune_run_id``; only what the plan scan reads."""
    blobs = BlobStore(ledger_root / run_id / "blobs")
    with LedgerWriter(ledger_root, run_id, "main") as writer:
        writer.append(
            "run_start",
            RunStart(
                condition=SEALED_CONDITION,
                seed=7,
                requires_predictions=False,
                config_blob=blobs.put(canonical_bytes({"tune_run_id": tune_run_id})),
                git_commit=COMMIT,
                protocol_versions=protocol_versions(),
                budget_limits=Budget(wall_seconds=1.0),
            ),
        )
        if status is not None:
            writer.append("run_end", RunEnd(status=status, summary={}))


def test_a_completed_sealed_run_is_refused_and_other_runs_are_skipped_or_remembered(
    env: dict[str, Any], tmp_path: Path
) -> None:
    """A completed sealed run refuses re-planning; aborted ones are remembered, not refused."""
    ledger = tmp_path / "ledgers"
    run_tune(env, ledger, run_id="tune-8")
    plan = plan_from_tune_run(ledger, "tune-8")
    assert plan.previous_sealed_runs == ()

    # a sealed run of another tune run is ignored, whatever its status
    _hand_sealed_run(ledger, "tune-9", "sealed-other", status="completed")

    # a run whose main stream fails to read is skipped, not fatal
    _hand_sealed_run(ledger, "tune-8", "sealed-corrupt", status="completed")
    stream = ledger / "sealed-corrupt" / "main.jsonl"
    stream.write_text(
        stream.read_text(encoding="utf-8").replace("gate1-sealed", "gate1-sealeX"),
        encoding="utf-8",
    )
    plan = plan_from_tune_run(ledger, "tune-8")
    assert plan.previous_sealed_runs == ()

    # aborted and never-ended sealed runs are remembered, never refused
    _hand_sealed_run(ledger, "tune-8", "sealed-aborted", status="aborted")
    _hand_sealed_run(ledger, "tune-8", "sealed-open", status=None)
    plan = plan_from_tune_run(ledger, "tune-8")
    assert plan.previous_sealed_runs == ("sealed-aborted", "sealed-open")

    # a completed one refuses: a configuration is evaluated on sealed once
    _hand_sealed_run(ledger, "tune-8", "sealed-done")
    with pytest.raises(RunError, match="already has a completed sealed run sealed-done"):
        plan_from_tune_run(ledger, "tune-8")
