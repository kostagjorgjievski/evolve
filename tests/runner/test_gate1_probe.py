"""Gate 1 probe: the learned updaters' recorded dev run (TASK_014 PR C).

Tests 1 to 10 of the PR C list (test 11 — the tune module's tests pass
unchanged — is the suite itself). Each test first makes a tune run with
small grids in ``tmp_path``, then probes with small learned grids; the
fixtures are those of ``test_gate1_tune.py``. The probe's records must equal
in-process ``replay_dataset`` calls exactly, its run must be pinned to the
tune run it names, and every refusal before start must leave no run
directory behind.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.dev import load_dev_pool
from evolvets.arena.errors import ArenaError
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.arena.synthetic.holdout import HOLDOUT_DATASETS
from evolvets.arena.versions import protocol_versions
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.foundation.fake import FAKE_SPEC, FakeAdapter
from evolvets.foundation.store import build_foundation_cache
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.canonical import canonical_json
from evolvets.ledger.payloads import Baseline, RunStart
from evolvets.ledger.schema import Budget, Record
from evolvets.ledger.store import LedgerReader
from evolvets.ledger.validate import validate_run
from evolvets.runner import cli, gate1_tune, runs
from evolvets.runner.gate1_grids import select_configs
from evolvets.runner.gate1_probe import (
    PROBE_CONDITION,
    PROBE_GRIDS,
    ProbeError,
    ProbeSpec,
    probe,
)
from evolvets.runner.gate1_tune import DevPaths, TuneSpec, tune
from evolvets.runner.runs import RunError
from evolvets.updater.interfaces import ConfigValue
from evolvets.updater.learned import LEARNED, LEARNED_DEFAULT_CONFIGS, LEARNED_IDS

COMMIT = "c" * 40
SMALL_GRIDS: dict[str, list[dict[str, ConfigValue]]] = {
    "equal": [{}],
    "hedge": [{"eta": 0.5}, {"eta": 1.0}],
}
SMALL_LEARNED: dict[str, list[dict[str, ConfigValue]]] = {
    "ridge_stack": [
        {"half_life": 48.0, "shrink": 1.0, "refit_every": 24},
        {"half_life": 336.0, "shrink": 0.1, "refit_every": 24},
    ],
    "loss_gate": [{"beta": 4.0, "half_life": 336.0, "ridge": 0.01, "refit_every": 24}],
}
SMALL_LEARNED_ORDER = [
    ("ridge_stack", {"half_life": 48.0, "shrink": 1.0, "refit_every": 24}),
    ("ridge_stack", {"half_life": 336.0, "shrink": 0.1, "refit_every": 24}),
    ("loss_gate", {"beta": 4.0, "half_life": 336.0, "ridge": 0.01, "refit_every": 24}),
]


@pytest.fixture(scope="module")
def synthetic_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A synthetic-dev cache with one series per family, and the paths to load it."""
    base = tmp_path_factory.mktemp("probe-synthetic")
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
    base = tmp_path_factory.mktemp("probe-pooled")
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


def probe_spec(tune_run_id: str = "tune-1", **overrides: Any) -> ProbeSpec:
    """A ProbeSpec with test defaults, overridable per test."""
    values: dict[str, Any] = {
        "tune_run_id": tune_run_id,
        "datasets": ("synthetic-dev",),
        "pool_specs": ("simple",),
        "foundation_models": None,
        "seed": 3,
        "max_wall_seconds": 10000.0,
    }
    values.update(overrides)
    return ProbeSpec(**values)


def dev_paths(env: dict[str, Path]) -> DevPaths:
    """The DevPaths of one fixture environment."""
    return DevPaths(
        cache_root=env["cache_root"],
        foundation_root=env["foundation_root"],
        data_root=env["data_root"],
        manifest_dir=env["manifest_dir"],
    )


def tune_once(
    env: dict[str, Path],
    ledger_root: Path,
    *,
    run_id: str = "tune-1",
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


def run_probe(
    env: dict[str, Path],
    ledger_root: Path,
    *,
    tune_run_id: str = "tune-1",
    run_id: str = "probe-1",
    grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]] = SMALL_LEARNED,
    spec: ProbeSpec | None = None,
) -> str:
    """Run one probe against a fixture environment; returns the run_end status."""
    return probe(
        spec if spec is not None else probe_spec(tune_run_id),
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


# --- Test 1: the grids ----------------------------------------------------------------------------


def test_probe_grids() -> None:
    """The grids cover each learned id, in LEARNED_IDS order, without duplicates."""
    assert tuple(PROBE_GRIDS) == LEARNED_IDS
    assert [len(PROBE_GRIDS[name]) for name in LEARNED_IDS] == [6, 6]
    assert PROBE_GRIDS["ridge_stack"] == tuple(
        {"half_life": half_life, "shrink": shrink, "refit_every": 24}
        for half_life in (48.0, 336.0, 2016.0)
        for shrink in (0.1, 1.0)
    )
    assert PROBE_GRIDS["loss_gate"] == tuple(
        {"beta": beta, "half_life": half_life, "ridge": 0.01, "refit_every": 24}
        for beta in (4.0, 16.0, 64.0)
        for half_life in (336.0, 2016.0)
    )
    for name in LEARNED_IDS:
        grid = PROBE_GRIDS[name]
        assert len({canonical_json(config) for config in grid}) == len(grid)
        for config in grid:
            LEARNED[name]().init(8, 24, config, 0)
        assert any(config == dict(LEARNED_DEFAULT_CONFIGS[name]) for config in grid)


# --- Test 2: the record sequence ------------------------------------------------------------------


def test_record_sequence(synthetic_env: dict[str, Path], tmp_path: Path) -> None:
    """Main holds run_start, one baseline per planned replay in the stated order, run_end."""
    ledger = tmp_path / "ledgers"
    assert tune_once(synthetic_env, ledger) == "completed"
    assert run_probe(synthetic_env, ledger) == "completed"
    records = read_records(ledger, "probe-1")
    assert [record.kind for record in records] == [
        "run_start",
        "baseline",
        "baseline",
        "baseline",
        "run_end",
    ]
    payloads = [record.payload for record in records[1:4]]
    assert all(isinstance(payload, Baseline) for payload in payloads)
    order = [(payload.updater_id, payload.hyperparams) for payload in payloads]
    assert order == SMALL_LEARNED_ORDER
    assert all(payload.pool == "dev" for payload in payloads)
    assert all(payload.updater_id in LEARNED_IDS for payload in payloads)  # no oracle record
    assert records[-1].payload.status == "completed"
    assert not (ledger / "probe-1" / "sealed.jsonl").exists()
    assert validate_run(records) == []


# --- Test 3: the numbers equal in-process calls ---------------------------------------------------


def test_records_equal_the_in_process_numbers(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """Every record's metrics and blob equal a direct replay_dataset of the learned updater."""
    ledger = tmp_path / "ledgers"
    tune_once(synthetic_env, ledger)
    run_probe(synthetic_env, ledger)
    records = read_records(ledger, "probe-1")
    read = blob_reader(ledger, "probe-1")
    pool = dev_pool_of(synthetic_env)
    for record, (learned_id, config) in zip(records[1:4], SMALL_LEARNED_ORDER, strict=True):
        payload = record.payload
        assert isinstance(payload, Baseline)
        expected = replay_dataset(LEARNED[learned_id], config, 3, pool)
        assert payload.status == expected.status
        assert payload.metrics == expected.metrics
        assert payload.error == expected.error
        assert payload.hyperparams == config
        blob = read(payload.per_task_blob)
        assert isinstance(blob, dict)
        assert blob["updater_ref"] == f"evolvets.updater.learned:{learned_id}"
        assert blob["result"] == result_to_obj(expected)
        assert record.budget.evaluations == 1
        assert record.budget.wall_seconds > 0


# --- Test 4: the run_start config -----------------------------------------------------------------


def test_run_start_config_blob(synthetic_env: dict[str, Path], tmp_path: Path) -> None:
    """The config blob names exactly the planned keys, pinning the tune run and its caches."""
    ledger = tmp_path / "ledgers"
    spec = probe_spec(max_wall_seconds=5000.0)
    tune_once(synthetic_env, ledger)
    run_probe(synthetic_env, ledger, spec=spec)
    tune_records = read_records(ledger, "tune-1")
    records = read_records(ledger, "probe-1")
    payload = records[0].payload
    assert isinstance(payload, RunStart)
    config: Any = json.loads(BlobStore(ledger / "probe-1" / "blobs").get(payload.config_blob))
    assert set(config) == {
        "experiment",
        "stage",
        "tune_run_id",
        "tune_run_end_record_id",
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
    assert config["stage"] == "dev-probe"
    assert config["tune_run_id"] == "tune-1"
    assert config["tune_run_end_record_id"] == tune_records[-1].record_id
    assert config["datasets"] == ["synthetic-dev"]
    assert config["pool_specs"] == ["simple"]
    assert config["simple_cache_ids"] == {"synthetic-dev": dev_pool_of(synthetic_env).cache_id}
    assert config["foundation_cache_ids"] == {"synthetic-dev": {"simple": []}}
    assert config["foundation_models"] == {"synthetic-dev": {"simple": []}}
    assert config["grids"] == {name: list(grid) for name, grid in SMALL_LEARNED.items()}
    assert config["selection_metric"] == "mase"
    assert config["seed"] == 3
    assert config["max_wall_seconds"] == 5000.0
    assert config["updater_refs"] == {
        name: f"evolvets.updater.learned:{name}" for name in SMALL_LEARNED
    }
    assert payload.condition == PROBE_CONDITION == "gate1-dev-probe"
    assert payload.seed == 3
    assert payload.budget_limits == Budget(wall_seconds=5000.0, evaluations=3)
    assert payload.git_commit == COMMIT
    assert payload.requires_predictions is False


# --- Test 5: refusals before start ----------------------------------------------------------------


def test_refuses_a_missing_tune_run(synthetic_env: dict[str, Path], tmp_path: Path) -> None:
    """A tune run id with no ledger directory is refused before anything is written."""
    ledger = tmp_path / "ledgers"
    tune_once(synthetic_env, ledger)
    with pytest.raises(ProbeError):
        run_probe(synthetic_env, ledger, tune_run_id="nope", run_id="probe-none")
    assert not (ledger / "probe-none").exists()


def test_refuses_a_tampered_tune_run(synthetic_env: dict[str, Path], tmp_path: Path) -> None:
    """A tune run whose chain fails verify is refused before anything is written."""
    ledger = tmp_path / "ledgers"
    tune_once(synthetic_env, ledger)
    main = ledger / "tune-1" / "main.jsonl"
    raw = main.read_bytes()
    tampered = raw.replace(b"updater_id", b"updater_ie", 1)
    assert tampered != raw
    main.write_bytes(tampered)
    with pytest.raises(ProbeError):
        run_probe(synthetic_env, ledger, run_id="probe-none")
    assert not (ledger / "probe-none").exists()


def test_refuses_a_tune_run_that_did_not_complete(
    synthetic_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Aborted, budget_exhausted and run_end-less tune runs are all refused."""
    ledger = tmp_path / "ledgers"
    monkeypatch.setattr(
        gate1_tune, "oracle_dataset", lambda _pool: (_ for _ in ()).throw(ArenaError("boom"))
    )
    with pytest.raises(ArenaError):
        tune_once(synthetic_env, ledger, run_id="tune-aborted")
    monkeypatch.undo()
    assert read_records(ledger, "tune-aborted")[-1].payload.status == "aborted"
    with pytest.raises(ProbeError):
        run_probe(synthetic_env, ledger, tune_run_id="tune-aborted", run_id="probe-none")
    assert not (ledger / "probe-none").exists()

    budget_spec = tune_spec(max_wall_seconds=1e-9)
    assert (
        tune_once(synthetic_env, ledger, run_id="tune-budget", spec=budget_spec)
        == "budget_exhausted"
    )
    with pytest.raises(ProbeError):
        run_probe(synthetic_env, ledger, tune_run_id="tune-budget", run_id="probe-none")
    assert not (ledger / "probe-none").exists()

    main = ledger / "tune-budget" / "main.jsonl"
    lines = main.read_bytes().splitlines(keepends=True)
    main.write_bytes(b"".join(lines[:2]))  # run_start + oracle; no run_end
    with pytest.raises(ProbeError):
        run_probe(synthetic_env, ledger, tune_run_id="tune-budget", run_id="probe-none")
    assert not (ledger / "probe-none").exists()


def test_refuses_a_probe_run_as_the_tune_run(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """A run of the probe condition does not pass as a tune run."""
    ledger = tmp_path / "ledgers"
    tune_once(synthetic_env, ledger)
    run_probe(synthetic_env, ledger, run_id="probe-first")
    with pytest.raises(ProbeError):
        run_probe(synthetic_env, ledger, tune_run_id="probe-first", run_id="probe-none")
    assert not (ledger / "probe-none").exists()


def test_refuses_different_protocol_versions(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """A tune run pinned to other protocol versions is refused (AGENTS.md 2.4)."""
    ledger = tmp_path / "ledgers"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(runs, "protocol_versions", lambda: {**protocol_versions(), "arena": "999"})
        tune_once(synthetic_env, ledger, run_id="tune-old")
    with pytest.raises(ProbeError):
        run_probe(synthetic_env, ledger, tune_run_id="tune-old", run_id="probe-none")
    assert not (ledger / "probe-none").exists()


def test_refuses_a_pool_the_tune_run_did_not_cover(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """A requested pool spec outside the tune run's config is refused before loading it."""
    ledger = tmp_path / "ledgers"
    tune_once(synthetic_env, ledger, spec=tune_spec(pool_specs=("simple",)))
    with pytest.raises(ProbeError):
        run_probe(
            synthetic_env,
            ledger,
            spec=probe_spec(pool_specs=("simple", "simple+foundation")),
            run_id="probe-none",
        )
    assert not (ledger / "probe-none").exists()


def test_refuses_a_pool_from_different_caches(
    synthetic_env: dict[str, Path], tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    """A probe over caches the tune run never replayed is refused before start."""
    other = tmp_path_factory.mktemp("probe-other-cache")
    cache_root = other / "cache_store"
    source, series = public_inputs("synthetic-dev", other, other, per_family=2)
    build_cache("synthetic-dev", source, series, cache_root)
    other_env = {
        "base": other,
        "cache_root": cache_root,
        "foundation_root": other / "foundation_cache",
        "data_root": other,
        "manifest_dir": other,
    }
    ledger = tmp_path / "ledgers"
    tune_once(other_env, ledger, grids={"equal": [{}]})
    with pytest.raises(ProbeError):
        run_probe(synthetic_env, ledger, run_id="probe-none")
    assert not (ledger / "probe-none").exists()


def test_refuses_a_non_learned_grid_key(synthetic_env: dict[str, Path], tmp_path: Path) -> None:
    """A grid keyed by a baseline id is a ValueError before anything is read."""
    ledger = tmp_path / "ledgers"
    tune_once(synthetic_env, ledger)
    with pytest.raises(ValueError, match="learned"):
        run_probe(synthetic_env, ledger, grids={"hedge": [{"eta": 0.5}]}, run_id="probe-none")
    assert not (ledger / "probe-none").exists()


# --- Test 6: a failed grid point is a result ------------------------------------------------------


def test_a_failed_grid_point_is_a_result(synthetic_env: dict[str, Path], tmp_path: Path) -> None:
    """A config the updater refuses is recorded as failed and the run still completes."""
    ledger = tmp_path / "ledgers"
    tune_once(synthetic_env, ledger)
    grids: dict[str, list[dict[str, ConfigValue]]] = {
        "ridge_stack": [
            {"half_life": 0.0, "shrink": 1.0, "refit_every": 24},  # not > 0: init refuses
            {"half_life": 336.0, "shrink": 1.0, "refit_every": 24},
        ],
        "loss_gate": [{"beta": 4.0, "half_life": 336.0, "ridge": 0.01, "refit_every": 24}],
    }
    assert run_probe(synthetic_env, ledger, grids=grids) == "completed"
    records = read_records(ledger, "probe-1")
    payload = records[1].payload
    assert isinstance(payload, Baseline)
    assert payload.status == "failed"
    assert payload.error
    summary = records[-1].payload.summary
    assert summary["replays"] == 3
    assert summary["failed"] == 1
    assert summary["ok"] == 2
    assert validate_run(records) == []


# --- Test 7: budget exhaustion --------------------------------------------------------------------


def test_budget_exhausted_before_the_second_replay(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """A tiny wall-clock budget runs the first replay only and ends budget_exhausted."""
    ledger = tmp_path / "ledgers"
    tune_once(synthetic_env, ledger)
    status = run_probe(synthetic_env, ledger, spec=probe_spec(max_wall_seconds=1e-9))
    assert status == "budget_exhausted"
    records = read_records(ledger, "probe-1")
    assert [record.kind for record in records] == ["run_start", "baseline", "run_end"]
    summary = records[-1].payload.summary
    assert summary["replays"] == 1
    assert records[1].budget.wall_seconds >= 1e-9  # recorded as measured, never clipped


# --- Test 8: the selection rule -------------------------------------------------------------------


def test_selection_over_the_probe_records(synthetic_env: dict[str, Path], tmp_path: Path) -> None:
    """The summary's selection is select_configs over learned ids; filters exclude the other run."""
    ledger = tmp_path / "ledgers"
    tune_once(synthetic_env, ledger)
    run_probe(synthetic_env, ledger)
    records = read_records(ledger, "probe-1")
    read = blob_reader(ledger, "probe-1")
    summary = records[-1].payload.summary
    assert summary["selected"] == select_configs(records, read, updater_ids=LEARNED_IDS)
    selected = summary["selected"]["synthetic-dev"]["simple"]
    assert set(selected) == {"ridge_stack", "loss_gate"}
    assert all(pick is not None and isinstance(pick["mase"], float) for pick in selected.values())

    tune_records = read_records(ledger, "tune-1")
    tune_read = blob_reader(ledger, "tune-1")
    assert select_configs(tune_records, tune_read, updater_ids=LEARNED_IDS) == {}
    assert select_configs(records, read) == {}


# --- Test 9: disjoint pieces over one tune run ----------------------------------------------------


def test_disjoint_pieces_over_one_tune_run(pooled_env: dict[str, Path], tmp_path: Path) -> None:
    """Two probes of one tune run, one per pool spec, both complete with distinct run ids."""
    ledger = tmp_path / "ledgers"
    spec = tune_spec(pool_specs=("simple", "simple+foundation"), foundation_models=("fake",))
    tune_once(pooled_env, ledger, spec=spec, grids={"equal": [{}]})
    assert run_probe(pooled_env, ledger, run_id="probe-simple") == "completed"
    assert (
        run_probe(
            pooled_env,
            ledger,
            run_id="probe-pooled",
            spec=probe_spec(pool_specs=("simple+foundation",), foundation_models=("fake",)),
        )
        == "completed"
    )
    for run_id, pool_spec in (("probe-simple", "simple"), ("probe-pooled", "simple+foundation")):
        read = blob_reader(ledger, run_id)
        for record in read_records(ledger, run_id):
            if record.kind != "baseline":
                continue
            result = read(record.payload.per_task_blob)["result"]
            assert isinstance(result, dict)
            assert result["pool_spec"] == pool_spec
        assert cli.main(["ledger", "verify", str(ledger), run_id]) == 0


# --- Test 10: the CLI -----------------------------------------------------------------------------


def _tune_cli_args(env: dict[str, Path], ledger: Path) -> list[str]:
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
        "tune-cli",
        "--cache-root",
        str(env["cache_root"]),
        "--data-root",
        str(env["data_root"]),
        "--manifest-dir",
        str(env["manifest_dir"]),
    ]


def _probe_cli_args(
    env: dict[str, Path], ledger: Path, tune_run: str, run_id: str, *extra: str
) -> list[str]:
    """A gate1 probe command line pointed at a fixture environment."""
    return [
        "gate1",
        "probe",
        "--tune-run",
        tune_run,
        "--dataset",
        "synthetic-dev",
        "--pool-spec",
        "simple",
        "--ledger-root",
        str(ledger),
        "--run-id",
        run_id,
        "--cache-root",
        str(env["cache_root"]),
        "--data-root",
        str(env["data_root"]),
        "--manifest-dir",
        str(env["manifest_dir"]),
        *extra,
    ]


def test_cli_probe_records_a_verifiable_run(
    synthetic_env: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OK, the ledger's summary, exit 0, and a run the ledger verifies."""
    monkeypatch.setattr(cli, "current_git_commit", lambda _cwd: COMMIT)
    monkeypatch.setattr(cli, "GATE1_GRIDS", SMALL_GRIDS)
    monkeypatch.setattr(cli, "PROBE_GRIDS", SMALL_LEARNED)
    ledger = tmp_path / "ledgers"
    assert cli.main(_tune_cli_args(synthetic_env, ledger)) == 0
    capsys.readouterr()
    assert cli.main(_probe_cli_args(synthetic_env, ledger, "tune-cli", "probe-cli")) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "OK probe-cli completed"
    summary = json.loads(out[1])
    assert summary["replays"] == 3
    assert "selected" in summary
    assert cli.main(["ledger", "verify", str(ledger), "probe-cli"]) == 0


def test_cli_probe_budget_exhausted_exits_two(
    synthetic_env: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A budget-exhausted probe prints OK with its status and exits 2."""
    monkeypatch.setattr(cli, "current_git_commit", lambda _cwd: COMMIT)
    monkeypatch.setattr(cli, "GATE1_GRIDS", SMALL_GRIDS)
    monkeypatch.setattr(cli, "PROBE_GRIDS", SMALL_LEARNED)
    ledger = tmp_path / "ledgers"
    assert cli.main(_tune_cli_args(synthetic_env, ledger)) == 0
    capsys.readouterr()
    args = _probe_cli_args(
        synthetic_env, ledger, "tune-cli", "probe-budget", "--max-wall-seconds", "1e-9"
    )
    assert cli.main(args) == 2
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "OK probe-budget budget_exhausted"
    assert json.loads(out[1])["replays"] == 1


def test_cli_probe_fails_a_missing_tune_run(
    synthetic_env: dict[str, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing tune run is one FAIL line, exit 1, and nothing is written."""
    monkeypatch.setattr(cli, "current_git_commit", lambda _cwd: COMMIT)
    monkeypatch.setattr(cli, "GATE1_GRIDS", SMALL_GRIDS)
    monkeypatch.setattr(cli, "PROBE_GRIDS", SMALL_LEARNED)
    ledger = tmp_path / "ledgers"
    assert cli.main(_tune_cli_args(synthetic_env, ledger)) == 0
    capsys.readouterr()
    assert cli.main(_probe_cli_args(synthetic_env, ledger, "nope", "probe-none")) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1
    assert out[0].startswith("FAIL ")
    assert not (ledger / "probe-none").exists()


def test_cli_probe_refuses_a_dirty_tree(
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
    assert cli.main(_probe_cli_args(synthetic_env, ledger, "any", "probe-none")) == 1
    out = capsys.readouterr().out.splitlines()
    assert out == ["FAIL working tree has uncommitted changes; commit before starting a run"]
    assert not ledger.exists()


# --- TASK_016 PR C: holdouts are never probed (test 10) -------------------------------------------


@pytest.mark.parametrize("dataset", HOLDOUT_DATASETS)
def test_probe_refuses_a_holdout_dataset(
    dataset: str, synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """A holdout dataset is refused before anything is read or written."""
    ledger = tmp_path / "ledgers"
    with pytest.raises(ValueError, match="is a holdout dataset: holdouts score accepted"):
        run_probe(synthetic_env, ledger, spec=probe_spec(datasets=(dataset,)))
    assert not (ledger / "probe-none").exists()


def test_cli_probe_rejects_a_holdout_dataset() -> None:
    """argparse rejects a holdout dataset: holdouts score accepted edits only."""
    with pytest.raises(SystemExit):
        cli.main(
            [
                "gate1",
                "probe",
                "--tune-run",
                "tune-1",
                "--dataset",
                "synthetic-shift3-holdout",
                "--pool-spec",
                "simple",
            ]
        )
