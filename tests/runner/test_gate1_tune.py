"""Gate 1 tuning: the recorded run and dev-configuration selection (TASK_011 PR B, part 2).

Tests 2 to 15 of the PR B list (the grids and the selection rule on
hand-built ledgers are ``test_gate1_grids.py``). ``tune`` is exercised end
to end against the synthetic-dev fixture (as ``tests/arena/test_arena_cli.py``)
and the pooled fixture with the fake foundation model (as
``tests/updater/test_baselines_replay.py``); small grids keep every run
quick. The numbers in the records must equal in-process ``replay_dataset`` /
``oracle_dataset`` calls exactly, and a tune run never opens a sealed path.
The TASK_015 additions at the end pin the scan datasets' place in the sweep.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.dev import load_dev_pool
from evolvets.arena.errors import ArenaError
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.arena.oracles import oracle_dataset, oracle_to_obj
from evolvets.arena.pool import pool_leakage_risk
from evolvets.cache.datasets import CACHE_DATASETS, public_inputs
from evolvets.cache.store import build_cache, cache_id
from evolvets.foundation.fake import FAKE_SPEC, FakeAdapter
from evolvets.foundation.store import build_foundation_cache, foundation_cache_id
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import Baseline, RunStart
from evolvets.ledger.schema import Budget, Record
from evolvets.ledger.store import LedgerReader
from evolvets.ledger.validate import validate_run
from evolvets.runner import cli, gate1_tune
from evolvets.runner.gate1_grids import select_configs
from evolvets.runner.gate1_tune import (
    TUNE_CONDITION,
    DevPaths,
    TuneSpec,
    planned_replays,
    tune,
)
from evolvets.runner.runs import Run, RunError
from evolvets.updater.baselines import BASELINES, hedge
from evolvets.updater.interfaces import ConfigValue
from evolvets.updater.learned import LEARNED
from tests.arena.test_realdata_prepare import _record_opens

COMMIT = "b" * 40
SMALL_GRIDS: dict[str, list[dict[str, ConfigValue]]] = {
    "equal": [{}],
    "hedge": [{"eta": 0.5}, {"eta": 1.0}],
}
SMALL_ORDER = [("equal", {}), ("hedge", {"eta": 0.5}), ("hedge", {"eta": 1.0})]


@pytest.fixture(scope="module")
def synthetic_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A synthetic-dev cache with one series per family, and the paths to load it."""
    base = tmp_path_factory.mktemp("gate1-synthetic")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-dev", base, base, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    return {
        "base": base,
        "cache_root": cache_root,
        "foundation_root": base / "foundation_cache",
        "data_root": base,
        "manifest_dir": base,
    }


@pytest.fixture(scope="module")
def pooled_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """That cache plus a fake foundation cache built against it (TASK_009 helpers)."""
    base = tmp_path_factory.mktemp("gate1-pooled")
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
    return {
        "base": base,
        "cache_root": cache_root,
        "foundation_root": foundation_root,
        "data_root": base,
        "manifest_dir": base,
    }


def tune_spec(**overrides: Any) -> TuneSpec:
    """A TuneSpec with test defaults, overridable per test."""
    values: dict[str, Any] = {
        "datasets": ("synthetic-dev",),
        "pool_specs": ("simple",),
        "foundation_models": None,
        "seed": 7,
        "max_wall_seconds": 10000.0,
    }
    values.update(overrides)
    return TuneSpec(**values)


def dev_paths(env: dict[str, Path]) -> DevPaths:
    """The DevPaths of one fixture environment."""
    return DevPaths(
        cache_root=env["cache_root"],
        foundation_root=env["foundation_root"],
        data_root=env["data_root"],
        manifest_dir=env["manifest_dir"],
    )


def run_tune(
    env: dict[str, Path],
    ledger_root: Path,
    *,
    run_id: str = "run-1",
    grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]] = SMALL_GRIDS,
    spec: TuneSpec | None = None,
) -> str:
    """Run one tuning sweep against a fixture environment; returns the run_end status."""
    return tune(
        spec if spec is not None else tune_spec(),
        dev_paths(env),
        ledger_root=ledger_root,
        run_id=run_id,
        git_commit=COMMIT,
        grids=grids,
    )


def read_records(ledger_root: Path, run_id: str) -> list[Record]:
    """Every record of one run, both streams, in reader order."""
    return list(LedgerReader(ledger_root, run_id).records())


def blob_reader(ledger_root: Path, run_id: str) -> Callable[[str], object]:
    """A ``read_blob`` over one run's blob store, as ``select_configs`` takes it."""
    blobs = BlobStore(ledger_root / run_id / "blobs")

    def read(ref: str) -> object:
        return json.loads(blobs.get(ref).decode("utf-8"))

    return read


def dev_pool_of(env: dict[str, Path], **kwargs: Any) -> Any:
    """The dev pool of a fixture environment, with optional loader overrides."""
    return load_dev_pool(
        "synthetic-dev",
        env["cache_root"],
        env["data_root"],
        env["manifest_dir"],
        **kwargs,
    )


# --- Test 2: the record sequence ------------------------------------------------------------------


def test_record_sequence(synthetic_env: dict[str, Path], tmp_path: Path) -> None:
    """Main holds run_start, the oracle, the updater baselines in grid order, run_end."""
    ledger = tmp_path / "ledgers"
    assert run_tune(synthetic_env, ledger) == "completed"
    records = read_records(ledger, "run-1")
    assert [record.kind for record in records] == [
        "run_start",
        "baseline",
        "baseline",
        "baseline",
        "baseline",
        "run_end",
    ]
    oracle = records[1].payload
    assert isinstance(oracle, Baseline)
    assert (oracle.updater_id, oracle.hyperparams, oracle.pool) == ("oracle", {}, "dev")
    updaters = [record.payload for record in records[2:5]]
    assert all(isinstance(payload, Baseline) for payload in updaters)
    assert [(payload.updater_id, payload.hyperparams) for payload in updaters] == SMALL_ORDER
    assert all(payload.pool == "dev" for payload in updaters)
    assert records[-1].payload.status == "completed"
    assert not (ledger / "run-1" / "sealed.jsonl").exists()
    assert validate_run(records) == []


# --- Test 3: the numbers equal in-process calls ---------------------------------------------------


def test_records_equal_the_in_process_numbers(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """Every record's metrics and blob equal a direct replay_dataset / oracle_dataset."""
    ledger = tmp_path / "ledgers"
    run_tune(synthetic_env, ledger)
    records = read_records(ledger, "run-1")
    read = blob_reader(ledger, "run-1")
    pool = dev_pool_of(synthetic_env)
    oracle_payload = records[1].payload
    assert isinstance(oracle_payload, Baseline)
    expected_oracle = oracle_dataset(pool)
    assert oracle_payload.metrics == expected_oracle.metrics
    assert read(oracle_payload.per_task_blob) == {"result": oracle_to_obj(expected_oracle)}
    assert records[1].budget.evaluations == 0
    assert records[1].budget.wall_seconds > 0
    for record, (baseline_id, config) in zip(records[2:5], SMALL_ORDER, strict=True):
        payload = record.payload
        assert isinstance(payload, Baseline)
        expected = replay_dataset(BASELINES[baseline_id], config, 7, pool)
        assert payload.status == expected.status
        assert payload.metrics == expected.metrics
        assert payload.error == expected.error
        assert payload.hyperparams == config
        blob = read(payload.per_task_blob)
        assert isinstance(blob, dict)
        assert blob["updater_ref"] == f"evolvets.updater.baselines:{baseline_id}"
        assert blob["result"] == result_to_obj(expected)
        assert record.budget.evaluations == 1
        assert record.budget.wall_seconds > 0


# --- Test 4: per-series wall-clock ----------------------------------------------------------------


class _RefusesInit:
    """An updater whose init always fails; replay records it as a failed series."""

    def init(self, n_experts: int, horizon: int, config: object, seed: int) -> object:
        raise ValueError("this instance refuses to init")

    def update(self, state: object, feedback: object, context: object) -> object:
        raise AssertionError("update is never reached when init failed")


def test_series_wall_clock(
    synthetic_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One entry per factory call; the loop-sum stays inside the record's charge."""
    ledger = tmp_path / "ledgers"
    run_tune(synthetic_env, ledger)
    records = read_records(ledger, "run-1")
    read = blob_reader(ledger, "run-1")
    n_series = len(dev_pool_of(synthetic_env).series)
    for record in records[2:5]:
        payload = record.payload
        assert isinstance(payload, Baseline)
        blob = read(payload.per_task_blob)
        assert isinstance(blob, dict)
        walls = blob["series_wall_seconds"]
        assert len(walls) == n_series
        total = 0.0
        for entry in walls:
            assert entry >= 0
            total = total + entry
        assert total <= record.budget.wall_seconds
    # A result failing at series k (0-based) carries k + 1 entries: the third
    # series' instance refuses to init, so the factory was called three times.
    made = 0

    def flaky_hedge() -> object:
        nonlocal made
        made += 1
        return _RefusesInit() if made == 3 else hedge()

    monkeypatch.setattr(gate1_tune, "BASELINES", {"hedge": flaky_hedge})
    failed_ledger = tmp_path / "ledgers-failed"
    run_tune(synthetic_env, failed_ledger, run_id="run-f", grids={"hedge": [{"eta": 1.0}]})
    failed = read_records(failed_ledger, "run-f")[2].payload
    assert isinstance(failed, Baseline)
    assert failed.status == "failed"
    blob = blob_reader(failed_ledger, "run-f")(failed.per_task_blob)
    assert isinstance(blob, dict)
    assert len(blob["series_wall_seconds"]) == 3


# --- Test 5: the run_start config -----------------------------------------------------------------


def test_run_start_config_blob(synthetic_env: dict[str, Path], tmp_path: Path) -> None:
    """The config blob names exactly the planned keys, pinning pools, caches and grids."""
    ledger = tmp_path / "ledgers"
    spec = tune_spec()
    run_tune(synthetic_env, ledger, spec=spec)
    records = read_records(ledger, "run-1")
    payload = records[0].payload
    assert isinstance(payload, RunStart)
    config: Any = json.loads(BlobStore(ledger / "run-1" / "blobs").get(payload.config_blob))
    assert set(config) == {
        "experiment",
        "stage",
        "datasets",
        "pool_specs",
        "simple_cache_ids",
        "foundation_cache_ids",
        "foundation_models",
        "grids",
        "selection_metric",
        "seed",
        "max_wall_seconds",
        "updater_refs",
    }
    assert config["experiment"] == "gate1"
    assert config["stage"] == "dev-tune"
    assert config["datasets"] == ["synthetic-dev"]
    assert config["pool_specs"] == ["simple"]
    assert config["simple_cache_ids"] == {"synthetic-dev": dev_pool_of(synthetic_env).cache_id}
    assert config["foundation_cache_ids"] == {"synthetic-dev": {"simple": []}}
    assert config["foundation_models"] == {"synthetic-dev": {"simple": []}}
    assert config["grids"] == {"equal": [{}], "hedge": [{"eta": 0.5}, {"eta": 1.0}]}
    assert config["selection_metric"] == "mase"
    assert config["seed"] == 7
    assert config["max_wall_seconds"] == 10000.0
    assert config["updater_refs"] == {
        "equal": "evolvets.updater.baselines:equal",
        "hedge": "evolvets.updater.baselines:hedge",
    }
    assert planned_replays(spec, SMALL_GRIDS) == 3
    assert payload.condition == TUNE_CONDITION == "gate1-dev-tune"
    assert payload.seed == 7
    assert payload.budget_limits == Budget(wall_seconds=10000.0, evaluations=3)
    assert payload.git_commit == COMMIT
    assert payload.requires_predictions is False


# --- Test 6: the pooled pool specs ----------------------------------------------------------------


def test_pooled_run_pins_its_caches(pooled_env: dict[str, Path], tmp_path: Path) -> None:
    """Both pool specs run; config and blobs name the fake cache and model per spec."""
    ledger = tmp_path / "ledgers"
    spec = tune_spec(pool_specs=("simple", "simple+foundation"), foundation_models=("fake",))
    assert run_tune(pooled_env, ledger, grids={"equal": [{}]}, spec=spec) == "completed"
    records = read_records(ledger, "run-1")
    assert [record.kind for record in records] == [
        "run_start",
        "baseline",  # oracle, simple
        "baseline",  # equal, simple
        "baseline",  # oracle, simple+foundation
        "baseline",  # equal, simple+foundation
        "run_end",
    ]
    payload = records[0].payload
    assert isinstance(payload, RunStart)
    config: Any = json.loads(BlobStore(ledger / "run-1" / "blobs").get(payload.config_blob))
    fake_id = foundation_cache_id(pooled_env["foundation_root"], "synthetic-dev", "fake")
    assert config["foundation_cache_ids"] == {
        "synthetic-dev": {"simple": [], "simple+foundation": [fake_id]}
    }
    assert config["foundation_models"] == {
        "synthetic-dev": {"simple": [], "simple+foundation": ["fake"]}
    }
    read = blob_reader(ledger, "run-1")
    pooled = dev_pool_of(
        pooled_env,
        pool_spec="simple+foundation",
        foundation_root=pooled_env["foundation_root"],
        foundation_models=("fake",),
    )
    for record in records[1:5]:
        p = record.payload
        assert isinstance(p, Baseline)
        result = read(p.per_task_blob)["result"]
        assert isinstance(result, dict)
        if result["pool_spec"] == "simple+foundation":
            assert result["foundation_cache_ids"] == [fake_id]
            assert result["leakage_risk"] == pool_leakage_risk(pooled.dataset, pooled.foundation)
        else:
            assert result["foundation_cache_ids"] == []


# --- Test 7: a failed grid point is a result ------------------------------------------------------


def test_a_failed_grid_point_is_a_result(synthetic_env: dict[str, Path], tmp_path: Path) -> None:
    """A config the updater refuses is recorded as failed and the run still completes."""
    ledger = tmp_path / "ledgers"
    assert run_tune(synthetic_env, ledger, grids={"hedge": [{"eta": 0.0}]}) == "completed"
    records = read_records(ledger, "run-1")
    payload = records[2].payload
    assert isinstance(payload, Baseline)
    assert payload.status == "failed"
    assert payload.error
    assert payload.metrics == {}
    summary = records[-1].payload.summary
    assert summary["replays"] == 1
    assert summary["ok"] == 0
    assert summary["failed"] == 1
    assert summary["selected"]["synthetic-dev"]["simple"]["hedge"] is None
    assert validate_run(records) == []


# --- Test 8: the selection rule on the run from test 2 --------------------------------------------


def test_selected_summary_matches_select_configs(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """The run_end summary's 'selected' equals select_configs over the run's own records."""
    ledger = tmp_path / "ledgers"
    run_tune(synthetic_env, ledger)
    records = read_records(ledger, "run-1")
    assert records[-1].payload.summary["selected"] == select_configs(
        records, blob_reader(ledger, "run-1")
    )


# --- Test 9: budget exhaustion --------------------------------------------------------------------


def test_budget_exhausted_stops_and_reports_the_overrun(
    synthetic_env: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A tiny wall-clock budget stops the loop after the oracle, unclipped and reported."""
    ledger = tmp_path / "ledgers"
    status = run_tune(synthetic_env, ledger, spec=tune_spec(max_wall_seconds=1e-9))
    assert status == "budget_exhausted"
    records = read_records(ledger, "run-1")
    assert [record.kind for record in records] == ["run_start", "baseline", "run_end"]
    oracle = records[1].payload
    assert isinstance(oracle, Baseline)
    assert oracle.updater_id == "oracle"
    assert records[1].budget.wall_seconds >= 1e-9  # recorded as measured, never clipped
    summary = records[-1].payload.summary
    assert summary["replays"] == 0
    assert summary["oracles"] == 1
    assert cli.main(["ledger", "verify", str(ledger), "run-1"]) == 1
    assert "wall_seconds" in capsys.readouterr().out


# --- Test 10: an ArenaError mid-run aborts --------------------------------------------------------


def test_an_arena_error_mid_run_aborts_and_reraises(
    pooled_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing second oracle aborts the run through the Run context and re-raises."""
    real_oracle = gate1_tune.oracle_dataset
    state = {"calls": 0}

    def flaky_oracle(pool: object) -> object:
        state["calls"] += 1
        if state["calls"] == 2:
            raise ArenaError("the cache was tampered with")
        return real_oracle(pool)

    monkeypatch.setattr(gate1_tune, "oracle_dataset", flaky_oracle)
    ledger = tmp_path / "ledgers"
    spec = tune_spec(pool_specs=("simple", "simple+foundation"), foundation_models=("fake",))
    with pytest.raises(ArenaError, match="tampered"):
        run_tune(pooled_env, ledger, grids={"equal": [{}]}, spec=spec)
    records = read_records(ledger, "run-1")
    assert [record.kind for record in records] == [
        "run_start",
        "baseline",  # oracle, simple
        "baseline",  # equal, simple
        "run_end",
    ]
    assert records[-1].payload.status == "aborted"
    assert records[-1].payload.summary["error"] == "ArenaError: the cache was tampered with"
    assert LedgerReader(ledger, "run-1").verify(check_blobs=True).ok


# --- Test 11: nothing before start ----------------------------------------------------------------


def test_a_pool_that_cannot_load_writes_nothing(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """The second dataset's missing cache aborts before any run is started."""
    ledger = tmp_path / "ledgers"
    with pytest.raises(ArenaError):
        run_tune(synthetic_env, ledger, spec=tune_spec(datasets=("synthetic-dev", "etth1")))
    assert not ledger.exists()


# --- Test 12: sealed blindness --------------------------------------------------------------------


def test_tune_opens_no_sealed_path(
    synthetic_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every path opened during a tune run is a public path (components, not substrings)."""
    ledger = tmp_path / "ledgers"
    opened: list[str] = []
    _record_opens(monkeypatch, opened)
    run_tune(synthetic_env, ledger, grids={"equal": [{}]})
    assert opened  # the run did open its caches and its ledger
    for path in opened:
        parts = Path(path).parts
        assert "sealed" not in parts, path
        assert "synthetic.salt" not in parts, path
        assert not any(part.endswith(".sealed.json") for part in parts), path


def test_gate1_tune_has_no_sealed_import() -> None:
    """An AST walk finds no import of evolvets.arena.sealed in runner/gate1_tune.py."""
    tree = ast.parse(Path(gate1_tune.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("evolvets.arena.sealed"), ast.dump(node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("evolvets.arena.sealed"), ast.dump(node)


# --- Test 13: determinism -------------------------------------------------------------------------


def test_two_runs_with_the_same_inputs_agree(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """Same inputs, different run ids: equal metrics, blob results and selections."""
    first, second = tmp_path / "ledgers-a", tmp_path / "ledgers-b"
    run_tune(synthetic_env, first, run_id="run-1")
    run_tune(synthetic_env, second, run_id="run-2")
    records_a = read_records(first, "run-1")
    records_b = read_records(second, "run-2")
    assert len(records_a) == len(records_b)
    read_a, read_b = blob_reader(first, "run-1"), blob_reader(second, "run-2")
    for one, two in zip(records_a, records_b, strict=True):
        assert one.kind == two.kind
        if one.kind != "baseline":
            continue
        assert one.payload.metrics == two.payload.metrics
        assert one.payload.status == two.payload.status
        assert one.payload.error == two.payload.error
        result_a = read_a(one.payload.per_task_blob)
        result_b = read_b(two.payload.per_task_blob)
        assert isinstance(result_a, dict) and isinstance(result_b, dict)
        assert result_a["result"] == result_b["result"]

    def numbers(selection: object) -> object:
        """A selection with the per-record UUIDs stripped; the picks' numbers stay."""
        assert isinstance(selection, dict)
        return {
            dataset: {
                pool_spec: {
                    baseline_id: None if pick is None else _without_record_id(pick)
                    for baseline_id, pick in by_baseline.items()
                }
                for pool_spec, by_baseline in by_pool.items()
            }
            for dataset, by_pool in selection.items()
        }

    def _without_record_id(pick: dict[str, object]) -> dict[str, object]:
        return {key: value for key, value in pick.items() if key != "record_id"}

    # record ids are fresh UUIDs per record, so only the picks' numbers can agree
    assert numbers(records_a[-1].payload.summary["selected"]) == numbers(
        records_b[-1].payload.summary["selected"]
    )


# --- Test 14: the seed ----------------------------------------------------------------------------


def test_the_seed_reaches_every_replay(
    synthetic_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """replay_dataset receives spec.seed on every call; a bad seed refuses before writing."""
    real = gate1_tune.replay_dataset
    seeds: list[int] = []

    def spy(factory: object, config: object, seed: int, pool: object) -> object:
        seeds.append(seed)
        return real(factory, config, seed, pool)

    monkeypatch.setattr(gate1_tune, "replay_dataset", spy)
    run_tune(synthetic_env, tmp_path / "ledgers", spec=tune_spec(seed=7))
    assert seeds == [7, 7, 7]

    bad = tmp_path / "ledgers-bad"
    with pytest.raises(ValueError, match="seed"):
        run_tune(synthetic_env, bad, spec=tune_spec(seed=-1))
    assert not bad.exists()


# --- Test 15: the CLI -----------------------------------------------------------------------------


def _cli_args(env: dict[str, Path], ledger: Path) -> list[str]:
    """A gate1 tune command line pointed at a fixture environment."""
    return [
        "gate1",
        "tune",
        "--dataset",
        "synthetic-dev",
        "--pool-spec",
        "simple",
        "--ledger-root",
        str(ledger),
        "--run-id",
        "run-cli",
        "--cache-root",
        str(env["cache_root"]),
        "--data-root",
        str(env["data_root"]),
        "--manifest-dir",
        str(env["manifest_dir"]),
    ]


def test_cli_tune_records_a_verifiable_run(
    synthetic_env: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OK, the ledger's summary, exit 0, and a run the ledger verifies."""
    monkeypatch.setattr(cli, "current_git_commit", lambda _cwd: COMMIT)
    monkeypatch.setattr(cli, "GATE1_GRIDS", SMALL_GRIDS)
    ledger = tmp_path / "ledgers"
    assert cli.main(_cli_args(synthetic_env, ledger)) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "OK run-cli completed"
    summary = json.loads(out[1])
    assert summary["replays"] == 3
    assert "selected" in summary
    assert cli.main(["ledger", "verify", str(ledger), "run-cli"]) == 0


def test_cli_tune_refuses_a_dirty_tree(
    synthetic_env: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A RunError from current_git_commit is one FAIL line, exit 1, nothing written."""

    def dirty(_cwd: Path) -> str:
        raise RunError("working tree has uncommitted changes; commit before starting a run")

    monkeypatch.setattr(cli, "current_git_commit", dirty)
    ledger = tmp_path / "ledgers"
    assert cli.main(_cli_args(synthetic_env, ledger)) == 1
    out = capsys.readouterr().out.splitlines()
    assert out == ["FAIL working tree has uncommitted changes; commit before starting a run"]
    assert not ledger.exists()


def test_cli_tune_rejects_an_unknown_dataset() -> None:
    """argparse rejects a dataset outside DEV_DATASETS + 'all'."""
    with pytest.raises(SystemExit):
        cli.main(["gate1", "tune", "--dataset", "nope", "--pool-spec", "simple"])


# --- TASK_014 PR C refactor: the replay recorder serves the learned updaters too ------------------


def test_record_replay_factories_and_ref_prefix(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """_record_replay records any factory mapping and ref prefix, not just the baselines'."""
    ledger = tmp_path / "ledgers"
    pool = dev_pool_of(synthetic_env)
    config: dict[str, ConfigValue] = {"half_life": 48.0, "shrink": 0.1, "refit_every": 24}
    counts = {"replays": 0, "ok": 0, "failed": 0, "invalid_weights": 0}
    with Run.start(
        ledger,
        "replay-run",
        condition=TUNE_CONDITION,
        seed=3,
        config={},
        budget_limits=Budget(wall_seconds=100.0, evaluations=1),
        git_commit=COMMIT,
    ) as run:
        gate1_tune._record_replay(
            run,
            pool,
            "ridge_stack",
            config,
            3,
            counts,
            factories=LEARNED,
            ref_prefix="evolvets.updater.learned:",
        )
        run.end("completed", {})
    records = read_records(ledger, "replay-run")
    payload = records[1].payload
    assert isinstance(payload, Baseline)
    expected = replay_dataset(LEARNED["ridge_stack"], config, 3, pool)
    assert (payload.updater_id, payload.hyperparams, payload.status) == (
        "ridge_stack",
        dict(config),
        expected.status,
    )
    assert payload.metrics == expected.metrics
    blob = blob_reader(ledger, "replay-run")(payload.per_task_blob)
    assert isinstance(blob, dict)
    assert blob["updater_ref"] == "evolvets.updater.learned:ridge_stack"
    assert blob["result"] == result_to_obj(expected)
    assert counts["replays"] == 1
    assert counts[expected.status] == 1


# --- TASK_015 PR A: the scan datasets through the sweep (tests 12 and 13) -------------------------


@pytest.fixture(scope="module")
def shift1_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A synthetic-shift1 scan cache with one series per family, and its paths."""
    base = tmp_path_factory.mktemp("gate1-shift1")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-shift1", base, base, per_family=1)
    build_cache("synthetic-shift1", source, series, cache_root)
    return {
        "base": base,
        "cache_root": cache_root,
        "foundation_root": base / "foundation_cache",
        "data_root": base,
        "manifest_dir": base,
    }


def test_gate1_axes_all_expands_to_the_cache_datasets_only() -> None:
    """'all' names exactly the cache datasets; a scan dataset stays only if named."""
    args = argparse.Namespace(
        dataset=["all", "synthetic-wide", "all", "synthetic-dev"],
        pool_spec=["simple", "simple"],
    )
    datasets, pool_specs = cli._gate1_axes(args)
    assert datasets == [*CACHE_DATASETS, "synthetic-wide"]  # first-seen order, no duplicates
    assert pool_specs == ["simple"]


def test_tune_over_a_scan_dataset(shift1_env: dict[str, Path], tmp_path: Path) -> None:
    """A sweep over synthetic-shift1 completes and carries the oracle metrics (test 13)."""
    ledger = tmp_path / "ledgers"
    assert (
        run_tune(shift1_env, ledger, spec=tune_spec(datasets=("synthetic-shift1",))) == "completed"
    )
    records = read_records(ledger, "run-1")
    payload = records[0].payload
    assert isinstance(payload, RunStart)
    config: Any = json.loads(BlobStore(ledger / "run-1" / "blobs").get(payload.config_blob))
    assert config["datasets"] == ["synthetic-shift1"]
    assert config["simple_cache_ids"] == {
        "synthetic-shift1": cache_id(shift1_env["cache_root"], "synthetic-shift1")
    }
    for record in records[1:-1]:
        entry = record.payload
        assert isinstance(entry, Baseline)
        if entry.status == "ok" and entry.updater_id != "oracle":
            assert "oracle_mase" in entry.metrics  # the hindsight oracle rides on replays
