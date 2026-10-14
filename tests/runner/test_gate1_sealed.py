"""Gate 1 on sealed: the plan from a dev tune run, and its recorded execution.

The fixture pairs a synthetic-dev tune environment with a salted
synthetic-sealed one (``per_family = 1``, the fake foundation model built on
both sides), so ``plan_from_tune_run`` maps one completed tune run onto
oracles plus dev-selected baselines and skips the rest, and ``sealed_run``
executes that plan through the sealed client, one spawned server per item.
A second sealed environment holds the simple cache only, so a pooled request
fails there. Grids stay at one or two baselines with one configuration each.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.errors import ArenaError
from evolvets.arena.sealed.client import (
    SealedError,
    SealedOracleRequest,
    SealedPaths,
    SealedRequest,
    evaluate_sealed,
    evaluate_sealed_oracle,
)
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
from evolvets.runner import cli, gate1_sealed, gate1_tune
from evolvets.runner.gate1_sealed import (
    SEALED_CONDITION,
    SealedPlan,
    SealedPlanItem,
    plan_from_tune_run,
    sealed_run,
)
from evolvets.runner.gate1_tune import TUNE_CONDITION, DevPaths, TuneSpec, tune
from evolvets.runner.runs import RunError
from evolvets.updater.baselines import BASELINE_IDS
from tests.arena.test_realdata_prepare import _record_opens

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


@pytest.fixture(scope="module")
def simple_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A sealed environment whose cache root holds the simple cache only.

    The pooled requests of a plan over both pool specs fail here: the server
    cannot join a foundation cache that was never built.
    """
    base = tmp_path_factory.mktemp("gate1-sealed-simple")
    data_root, manifest_dir = base / "data", base / "manifests"
    init_salt(data_root, manifest_dir, per_family=1)
    sealed_root = base / "sealed_cache"
    build_sealed_cache("synthetic-sealed", data_root, manifest_dir, sealed_root)
    return {
        "sealed_paths": SealedPaths(
            data_root=data_root, manifest_dir=manifest_dir, sealed_cache_root=sealed_root
        )
    }


@pytest.fixture(scope="module")
def pair(env: dict[str, Any], tune_run: dict[str, Any]) -> dict[str, Any]:
    """One completed sealed run of the shared tune run's plan, with the plan."""
    ledger_root = tune_run["ledger_root"]
    plan = plan_from_tune_run(ledger_root, tune_run["tune_run_id"])
    status = sealed_run(
        plan,
        env["sealed_paths"],
        ledger_root=ledger_root,
        run_id="sealed-1",
        git_commit=COMMIT,
        max_wall_seconds=10000.0,
    )
    assert status == "completed"
    return {**tune_run, "sealed_run_id": "sealed-1", "plan": plan}


def run_sealed(
    env: dict[str, Any],
    ledger_root: Path,
    tune_run_id: str,
    *,
    run_id: str,
    paths: SealedPaths | None = None,
    max_wall_seconds: float = 10000.0,
) -> str:
    """Plan one tune run and execute the plan against a fixture environment."""
    plan = plan_from_tune_run(ledger_root, tune_run_id)
    sealed_paths = env["sealed_paths"] if paths is None else paths
    return sealed_run(
        plan,
        sealed_paths,
        ledger_root=ledger_root,
        run_id=run_id,
        git_commit=COMMIT,
        max_wall_seconds=max_wall_seconds,
    )


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

    (hand / "tune-empty").mkdir()
    (hand / "tune-empty" / "main.jsonl").touch()
    with pytest.raises(RunError, match="no run_start"):
        plan_from_tune_run(hand, "tune-empty")

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
    ledger_root: Path,
    tune_run_id: str,
    run_id: str,
    *,
    status: str | None = "completed",
    config_bytes: bytes | None = None,
) -> None:
    """Hand-build a sealed run of ``tune_run_id``; only what the plan scan reads."""
    blobs = BlobStore(ledger_root / run_id / "blobs")
    if config_bytes is None:
        config_bytes = canonical_bytes({"tune_run_id": tune_run_id})
    with LedgerWriter(ledger_root, run_id, "main") as writer:
        writer.append(
            "run_start",
            RunStart(
                condition=SEALED_CONDITION,
                seed=7,
                requires_predictions=False,
                config_blob=blobs.put(config_bytes),
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

    # a run whose config blob is missing is skipped, not fatal, even completed
    _hand_sealed_run(ledger, "tune-8", "sealed-noblobs")
    shutil.rmtree(ledger / "sealed-noblobs" / "blobs")
    plan = plan_from_tune_run(ledger, "tune-8")
    assert plan.previous_sealed_runs == ()

    # a run whose config blob is not a JSON object is skipped, not fatal
    _hand_sealed_run(ledger, "tune-8", "sealed-notobject", config_bytes=canonical_bytes([1, 2]))
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


# --- Test 3: the record sequence ------------------------------------------------------------------


def test_record_sequence(pair: dict[str, Any]) -> None:
    """Main holds exactly run_start and run_end completed; sealed mirrors the plan."""
    ledger = pair["ledger_root"]
    main = read_records(ledger, pair["sealed_run_id"], ("main",))
    assert [record.kind for record in main] == ["run_start", "run_end"]
    end = main[-1].payload
    assert isinstance(end, RunEnd)
    assert end.status == "completed"
    sealed = read_records(ledger, pair["sealed_run_id"], ("sealed",))
    plan = pair["plan"]
    assert len(sealed) == len(plan.items)
    for record, item in zip(sealed, plan.items, strict=True):
        assert record.kind == "baseline"
        payload = record.payload
        assert isinstance(payload, Baseline)
        assert payload.updater_id == item.updater_id
        assert payload.hyperparams == dict(item.config)
        assert payload.pool == "sealed"
    assert cli.main(["ledger", "verify", str(ledger), pair["sealed_run_id"]]) == 0


# --- Test 4: the same numbers as the client -------------------------------------------------------


def _begin_plain_run(ledger_root: Path, run_id: str) -> None:
    """Hand-start one run: a minimal main-stream run_start the server accepts."""
    blobs = BlobStore(ledger_root / run_id / "blobs")
    with LedgerWriter(ledger_root, run_id, "main") as writer:
        writer.append(
            "run_start",
            RunStart(
                condition="sealed-direct",
                seed=7,
                requires_predictions=False,
                config_blob=blobs.put(b"direct"),
                git_commit=COMMIT,
                protocol_versions=protocol_versions(),
                budget_limits=Budget(wall_seconds=1e9, evaluations=100),
            ),
        )


def test_same_numbers_as_the_client(pair: dict[str, Any]) -> None:
    """Each sealed record's metrics equal a direct client call for the same request."""
    ledger, plan = pair["ledger_root"], pair["plan"]
    paths = pair["env"]["sealed_paths"]
    _begin_plain_run(ledger, "sealed-direct")
    direct: list[tuple[str, dict[str, float]]] = []
    for item in plan.items:
        if item.updater_id == "oracle":
            receipt = evaluate_sealed_oracle(
                SealedOracleRequest(
                    dataset=item.dataset,
                    pool_spec=item.pool_spec,
                    foundation_models=item.foundation_models,
                ),
                ledger_root=ledger,
                run_id="sealed-direct",
                paths=paths,
            )
        else:
            receipt = evaluate_sealed(
                SealedRequest(
                    dataset=item.dataset,
                    updater_id=item.updater_id,
                    updater_ref=f"evolvets.updater.baselines:{item.updater_id}",
                    config=dict(item.config),
                    seed=plan.seed,
                    pool_spec=item.pool_spec,
                    foundation_models=item.foundation_models,
                ),
                ledger_root=ledger,
                run_id="sealed-direct",
                paths=paths,
            )
        last = read_records(ledger, "sealed-direct", ("sealed",))[-1].payload
        assert isinstance(last, Baseline)
        direct.append((receipt.status, dict(last.metrics)))
    sealed = read_records(ledger, pair["sealed_run_id"], ("sealed",))
    assert len(sealed) == len(direct)
    for record, (status, metrics) in zip(sealed, direct, strict=True):
        payload = record.payload
        assert isinstance(payload, Baseline)
        assert payload.status == status
        assert payload.metrics == metrics


# --- Test 5: the run_start config -----------------------------------------------------------------


def test_run_start_config(pair: dict[str, Any]) -> None:
    """The config names exactly the listed keys, dev axes remapped to sealed ones."""
    ledger, plan = pair["ledger_root"], pair["plan"]
    first = read_records(ledger, pair["sealed_run_id"], ("main",))[0].payload
    assert isinstance(first, RunStart)
    config: Any = json.loads(
        BlobStore(ledger / pair["sealed_run_id"] / "blobs").get(first.config_blob)
    )
    assert set(config) == {
        "experiment",
        "stage",
        "tune_run_id",
        "tune_run_end_record_id",
        "tune_git_commit",
        "datasets",
        "dev_datasets",
        "pool_specs",
        "foundation_models",
        "selected",
        "updater_refs",
        "seed",
        "max_wall_seconds",
        "series_timeout_seconds",
        "previous_sealed_runs",
    }
    selection = selection_of(ledger, pair["tune_run_id"])

    def config_of(pool_spec: str, baseline_id: str) -> dict[str, Any] | None:
        """The tune selection's config; absent and None selections are both None."""
        # a baseline with no grid point has no selection entry at all
        pick = selection["synthetic-dev"][pool_spec].get(baseline_id)
        return None if pick is None else pick["config"]

    expected_selected = {
        "synthetic-sealed": {
            pool_spec: {
                baseline_id: config_of(pool_spec, baseline_id) for baseline_id in BASELINE_IDS
            }
            for pool_spec in ("simple", "simple+foundation")
        }
    }
    assert config["experiment"] == "gate1"
    assert config["stage"] == "sealed"
    assert config["tune_run_id"] == pair["tune_run_id"]
    assert config["tune_run_end_record_id"] == plan.tune_run_end_record_id
    tune_end = read_records(ledger, pair["tune_run_id"])[-1]
    assert config["tune_run_end_record_id"] == tune_end.record_id
    assert config["tune_git_commit"] == COMMIT
    assert config["datasets"] == ["synthetic-sealed"]
    assert config["dev_datasets"] == ["synthetic-dev"]
    assert config["pool_specs"] == ["simple", "simple+foundation"]
    assert config["foundation_models"] == {
        "synthetic-sealed": {"simple": [], "simple+foundation": ["fake"]}
    }
    assert config["selected"] == expected_selected
    assert config["updater_refs"] == {
        baseline_id: f"evolvets.updater.baselines:{baseline_id}" for baseline_id in BASELINE_IDS
    }
    assert config["seed"] == 7
    assert config["max_wall_seconds"] == 10000.0
    assert config["series_timeout_seconds"] == 600.0
    assert config["previous_sealed_runs"] == []
    assert first.condition == SEALED_CONDITION == "gate1-sealed"
    assert first.seed == 7
    assert first.budget_limits == Budget(wall_seconds=10000.0, evaluations=4)
    assert first.git_commit == COMMIT


# --- Test 6: the summary --------------------------------------------------------------------------


def test_summary(pair: dict[str, Any]) -> None:
    """Receipts index the sealed stream; the summary holds no metric at any depth."""
    ledger, plan = pair["ledger_root"], pair["plan"]
    end = read_records(ledger, pair["sealed_run_id"], ("main",))[-1].payload
    assert isinstance(end, RunEnd)
    assert end.status == "completed"
    summary = end.summary
    assert set(summary) == {"receipts", "skipped", "oracles", "evaluations", "statuses"}
    sealed = read_records(ledger, pair["sealed_run_id"], ("sealed",))
    assert summary["receipts"] == [
        {
            "dataset": item.dataset,
            "pool_spec": item.pool_spec,
            "updater_id": item.updater_id,
            "record_id": record.record_id,
            "status": record.payload.status,
        }
        for item, record in zip(plan.items, sealed, strict=True)
    ]
    assert summary["skipped"] == [
        {"dataset": dataset, "pool_spec": pool_spec, "updater_id": baseline_id}
        for dataset, pool_spec, baseline_id in plan.skipped
    ]
    assert summary["oracles"] == 2
    assert summary["evaluations"] == 4
    assert summary["statuses"] == {"ok": 4}

    def has_metric(obj: object) -> bool:
        """Whether any dict below obj carries a 'metrics' or 'mase' key."""
        if isinstance(obj, dict):
            if "metrics" in obj or "mase" in obj:
                return True
            return any(has_metric(value) for value in obj.values())
        if isinstance(obj, list):
            return any(has_metric(entry) for entry in obj)
        return False

    assert not has_metric(summary)


# --- Test 7: caller blindness ---------------------------------------------------------------------


def test_sealed_run_opens_no_sealed_path(
    env: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """During sealed_run the caller opens no sealed file, cache or stream (components)."""
    ledger = tmp_path / "ledgers"
    run_tune(env, ledger, run_id="tune-blind")
    opened: list[str] = []
    _record_opens(monkeypatch, opened)
    assert run_sealed(env, ledger, "tune-blind", run_id="sealed-blind") == "completed"
    assert opened  # the run did open its own ledger and blobs
    sealed_dir = env["sealed_paths"].data_root / "sealed"
    cache_root = env["sealed_paths"].sealed_cache_root
    for path in opened:
        entry = Path(path)
        assert sealed_dir not in entry.parents, path
        assert cache_root not in entry.parents, path
        assert not any(
            part == "sealed.jsonl" or part.endswith(".sealed.json") for part in entry.parts
        ), path


def test_gate1_sealed_imports_only_the_client() -> None:
    """gate1_sealed imports from evolvets.arena.sealed only client and names."""
    tree = ast.parse(Path(gate1_sealed.__file__).read_text(encoding="utf-8"))
    allowed = {"evolvets.arena.sealed.client", "evolvets.arena.sealed.names"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "evolvets.arena.sealed"
        ):
            assert node.module in allowed, ast.dump(node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("evolvets.arena.sealed"):
                    assert alias.name in allowed, ast.dump(node)


# --- Test 8 (run part): once per tune run ---------------------------------------------------------


def test_once_per_tune_run(
    pair: dict[str, Any], env: dict[str, Any], simple_env: dict[str, Any], tmp_path: Path
) -> None:
    """A completed sealed run refuses re-planning; an aborted one is remembered."""
    with pytest.raises(RunError, match="already has a completed sealed run sealed-1"):
        plan_from_tune_run(pair["ledger_root"], pair["tune_run_id"])
    ledger = tmp_path / "ledgers"
    run_tune(env, ledger, run_id="tune-8")
    with pytest.raises(SealedError):
        run_sealed(
            env,
            ledger,
            "tune-8",
            run_id="sealed-8a",
            paths=simple_env["sealed_paths"],
        )
    plan = plan_from_tune_run(ledger, "tune-8")
    assert plan.previous_sealed_runs == ("sealed-8a",)
    assert (
        run_sealed(env, ledger, "tune-8", run_id="sealed-8b", max_wall_seconds=1e-9)
        == "budget_exhausted"
    )
    first = read_records(ledger, "sealed-8b", ("main",))[0].payload
    assert isinstance(first, RunStart)
    config: Any = json.loads(BlobStore(ledger / "sealed-8b" / "blobs").get(first.config_blob))
    assert config["previous_sealed_runs"] == ["sealed-8a"]


# --- Test 9: the budget ---------------------------------------------------------------------------


def test_the_budget_stops_after_an_item_and_a_large_one_completes(
    env: dict[str, Any], tmp_path: Path
) -> None:
    """A tiny budget still runs the first item; a large one runs every item."""
    ledger = tmp_path / "ledgers"
    run_tune(env, ledger, run_id="tune-9")
    assert run_sealed(env, ledger, "tune-9", run_id="sealed-9a", max_wall_seconds=1e-9) == (
        "budget_exhausted"
    )
    end = read_records(ledger, "sealed-9a", ("main",))[-1].payload
    assert isinstance(end, RunEnd)
    assert end.status == "budget_exhausted"
    summary = end.summary
    assert len(summary["receipts"]) == 1  # type: ignore[arg-type]
    assert summary["oracles"] == 1
    assert summary["evaluations"] == 0
    assert len(read_records(ledger, "sealed-9a", ("sealed",))) == 1
    plan = plan_from_tune_run(ledger, "tune-9")  # sealed-9a is not completed, so allowed
    assert plan.previous_sealed_runs == ("sealed-9a",)
    assert run_sealed(env, ledger, "tune-9", run_id="sealed-9b", max_wall_seconds=1e9) == (
        "completed"
    )
    end = read_records(ledger, "sealed-9b", ("main",))[-1].payload
    assert isinstance(end, RunEnd)
    receipts = end.summary["receipts"]
    assert isinstance(receipts, list)
    assert len(receipts) == len(plan.items)


# --- Test 10: a SealedError aborts ----------------------------------------------------------------


def test_a_sealed_error_aborts_and_reraises(
    env: dict[str, Any], simple_env: dict[str, Any], tmp_path: Path
) -> None:
    """A missing pooled foundation cache aborts the run; earlier records stay intact."""
    ledger = tmp_path / "ledgers"
    run_tune(env, ledger, run_id="tune-10")
    with pytest.raises(SealedError):
        run_sealed(env, ledger, "tune-10", run_id="sealed-10", paths=simple_env["sealed_paths"])
    main = read_records(ledger, "sealed-10", ("main",))
    assert [record.kind for record in main] == ["run_start", "run_end"]
    end = main[-1].payload
    assert isinstance(end, RunEnd)
    assert end.status == "aborted"
    assert "SealedError" in end.summary["error"]
    sealed = read_records(ledger, "sealed-10", ("sealed",))
    assert len(sealed) == 1  # the simple oracle ran; the pooled one refused
    assert sealed[0].payload.updater_id == "oracle"
    assert LedgerReader(ledger, "sealed-10").verify(check_blobs=True).ok


# --- Test 11: an updater failure is a result ------------------------------------------------------


def test_an_updater_failure_is_a_result(env: dict[str, Any], tmp_path: Path) -> None:
    """A config the worker refuses gives a failed receipt; the run still completes."""
    ledger = tmp_path / "ledgers"
    run_tune(env, ledger, run_id="tune-11", spec=tune_spec(pool_specs=("simple",)))
    plan = plan_from_tune_run(ledger, "tune-11")
    oracle = plan.items[0]
    assert oracle.updater_id == "oracle"
    hand = dataclasses.replace(
        plan,
        items=(
            oracle,
            SealedPlanItem(
                dataset="synthetic-sealed",
                pool_spec="simple",
                updater_id="hedge",
                config={"eta": 0.0},
                foundation_models=None,
            ),
        ),
        skipped=tuple(
            ("synthetic-sealed", "simple", baseline_id)
            for baseline_id in ("equal", "ftl", "fixed_share", "adahedge")
        ),
    )
    assert isinstance(hand, SealedPlan)
    assert (
        sealed_run(
            hand,
            env["sealed_paths"],
            ledger_root=ledger,
            run_id="sealed-11",
            git_commit=COMMIT,
            max_wall_seconds=10000.0,
        )
        == "completed"
    )
    end = read_records(ledger, "sealed-11", ("main",))[-1].payload
    assert isinstance(end, RunEnd)
    receipts = end.summary["receipts"]
    assert isinstance(receipts, list) and len(receipts) == 2
    assert receipts[1]["updater_id"] == "hedge"
    assert receipts[1]["status"] == "failed"
    assert end.summary["statuses"] == {"failed": 1}


# --- Test 12: the CLI -----------------------------------------------------------------------------


def _cli_args(env: dict[str, Any], ledger: Path, run_id: str) -> list[str]:
    """A gate1 sealed command line pointed at a fixture environment."""
    paths = env["sealed_paths"]
    return [
        "gate1",
        "sealed",
        "--tune-run",
        "tune-cli",
        "--ledger-root",
        str(ledger),
        "--run-id",
        run_id,
        "--data-root",
        str(paths.data_root),
        "--manifest-dir",
        str(paths.manifest_dir),
        "--sealed-cache-root",
        str(paths.sealed_cache_root),
    ]


def test_cli_sealed_records_a_verifiable_run(
    env: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OK, the receipt summary, exit 0, and a run the ledger verifies."""
    monkeypatch.setattr(cli, "current_git_commit", lambda _cwd: COMMIT)
    ledger = tmp_path / "ledgers"
    run_tune(env, ledger, run_id="tune-cli")
    assert cli.main(_cli_args(env, ledger, "sealed-cli")) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "OK sealed-cli completed"
    summary = json.loads(out[1])
    assert set(summary) == {"receipts", "skipped", "oracles", "evaluations", "statuses"}
    assert summary["oracles"] == 2
    assert cli.main(["ledger", "verify", str(ledger), "sealed-cli"]) == 0
    capsys.readouterr()  # drain the verify line before the next capture
    # a second call of the same tune run is refused before anything is written
    assert cli.main(_cli_args(env, ledger, "sealed-cli-2")) == 1
    assert capsys.readouterr().out.splitlines() == [
        "FAIL tune run tune-cli already has a completed sealed run sealed-cli"
    ]
    assert sorted(path.name for path in ledger.iterdir()) == ["sealed-cli", "tune-cli"]


def test_cli_sealed_refuses_a_dirty_tree(
    env: dict[str, Any],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RunError from current_git_commit is one FAIL line, exit 1, nothing written."""

    def dirty(_cwd: Path) -> str:
        raise RunError("working tree has uncommitted changes; commit before starting a run")

    monkeypatch.setattr(cli, "current_git_commit", dirty)
    ledger = tmp_path / "ledgers"
    run_tune(env, ledger, run_id="tune-dirty")
    assert cli.main(_cli_args(env, ledger, "sealed-dirty")) == 1
    out = capsys.readouterr().out.splitlines()
    assert out == ["FAIL working tree has uncommitted changes; commit before starting a run"]
    assert not (ledger / "sealed-dirty").exists()
