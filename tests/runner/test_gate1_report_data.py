"""Gate 1 report data: refusals, traced numbers, gaps and bootstrap (TASK_011 PR C1).

The data-level half of the PR C test list: numbers trace to records (2),
gaps and bootstrap intervals (3), failed grid points (6), incomplete runs
(7), every refusal but the CLI ones (8), listed violations (9), and the
AST import guard of (13). Rendering, the CLI and the subprocess import
check are PR C2's. Real numbers come from tune runs on the synthetic
fixture and on the ETTh1 fixture with a leaking pooled model; the refusal
cases come from hand-built ledgers in ``tmp_path``.
"""

from __future__ import annotations

import ast
import json
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.oracles import ORACLE_SWITCHES
from evolvets.arena.synthetic.grammar import FAMILIES
from evolvets.arena.synthetic.presets import SCAN_PRESETS
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
from evolvets.runner import gate1_report_data
from evolvets.runner.gate1_grids import select_configs
from evolvets.runner.gate1_report_data import (
    BOOTSTRAP_RESAMPLES,
    ReportError,
    build_report_data,
)
from evolvets.runner.gate1_tune import DevPaths, TuneSpec, tune
from tests.arena.test_dev import build_real_env

COMMIT = "d" * 40
SMALL_GRIDS: dict[str, list[dict[str, float]]] = {
    "equal": [{}],
    "hedge": [{"eta": 0.5}, {"eta": 1.0}],
}


@pytest.fixture(scope="module")
def synthetic_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A synthetic-dev cache with one series per family, and the paths to load it."""
    base = tmp_path_factory.mktemp("report-synthetic")
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


def _build_etth1_env(base: Path) -> dict[str, Path]:
    """The fixture ETTh1 prepared and cached, plus a leaking pooled model's cache.

    The ``leaky`` model records ETTh1 overlap ``"yes"``, so results replayed
    over ``simple+foundation`` carry ``leakage_risk`` true while the simple
    ones stay false (as ``real_env`` in tests/arena/test_pool.py). Shared
    with ``test_gate1_headroom`` (TASK_015 PR B), which wraps it in its own
    module-scoped fixture.
    """
    monkeypatch = pytest.MonkeyPatch()
    try:
        data_root, manifest_dir, cache_root = build_real_env(base, monkeypatch)
    finally:
        monkeypatch.undo()
    foundation_root = base / "foundation_cache"
    leaky = replace(FAKE_SPEC, name="leaky", pretraining_overlap={"etth1": "yes", "ecl": "no"})
    build_foundation_cache(
        "etth1",
        FakeAdapter(leaky.name),
        leaky,
        foundation_root,
        cache_root,
        data_root,
        manifest_dir,
        64,
    )
    return {
        "base": base,
        "cache_root": cache_root,
        "foundation_root": foundation_root,
        "data_root": data_root,
        "manifest_dir": manifest_dir,
    }


@pytest.fixture(scope="module")
def etth1_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The ETTh1 fixture environment of ``_build_etth1_env``, in this module's own directory."""
    return _build_etth1_env(tmp_path_factory.mktemp("report-etth1"))


def _dev_paths(env: dict[str, Path]) -> DevPaths:
    """The DevPaths of one fixture environment."""
    return DevPaths(
        cache_root=env["cache_root"],
        foundation_root=env["foundation_root"],
        data_root=env["data_root"],
        manifest_dir=env["manifest_dir"],
    )


def _tune(
    env: dict[str, Path],
    ledger_root: Path,
    *,
    run_id: str,
    grids: dict[str, list[dict[str, float]]] | None = None,
    spec: TuneSpec | None = None,
) -> str:
    """Run one tuning sweep against a fixture environment; returns the run_end status."""
    return tune(
        spec
        if spec is not None
        else TuneSpec(
            datasets=("synthetic-dev",),
            pool_specs=("simple",),
            foundation_models=None,
            seed=7,
            max_wall_seconds=10000.0,
        ),
        _dev_paths(env),
        ledger_root=ledger_root,
        run_id=run_id,
        git_commit=COMMIT,
        grids=grids if grids is not None else SMALL_GRIDS,
    )


def _records(ledger_root: Path, run_id: str) -> list[Record]:
    """Every record of one run, both streams, in reader order."""
    return list(LedgerReader(ledger_root, run_id).records())


def _read_blob(ledger_root: Path, run_id: str) -> Any:
    """A blob reader over one run's store."""
    blobs = BlobStore(ledger_root / run_id / "blobs")
    return lambda ref: json.loads(blobs.get(ref).decode("utf-8"))


def _baselines(records: list[Record]) -> list[tuple[Record, Baseline]]:
    """The main-stream baseline records with their payloads, in order."""
    out = []
    for record in records:
        if record.stream == "main" and record.kind == "baseline":
            payload = record.payload
            if isinstance(payload, Baseline):
                out.append((record, payload))
    return out


# --- Test 2: the numbers trace to records ---------------------------------------------------------


def test_tuned_rows_and_oracles_trace_to_the_records(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """Every dataclass field equals the record or loop-sum it comes from."""
    ledger = tmp_path / "ledgers"
    assert _tune(synthetic_env, ledger, run_id="run-1") == "completed"
    data = build_report_data(ledger, ["run-1"], bootstrap_seed=0)
    records = _records(ledger, "run-1")
    read = _read_blob(ledger, "run-1")
    assert [row.baseline for row in data.sections[0].tuned] == ["equal", "hedge"]
    by_id: dict[str, list[tuple[Record, Baseline]]] = {}
    for record, payload in _baselines(records):
        by_id.setdefault(payload.updater_id, []).append((record, payload))
    for row in data.sections[0].tuned:
        entries = by_id[row.baseline]
        assert row.tried == len(entries)
        failed = 0
        evaluations = 0.0
        wall = 0.0
        for record, payload in entries:
            if payload.status != "ok":
                failed += 1
            evaluations = evaluations + record.budget.evaluations
            wall = wall + record.budget.wall_seconds
        assert row.failed == failed
        assert row.evaluations == evaluations
        assert row.wall_seconds == wall
        best = min(
            (payload for _r, payload in entries if payload.status == "ok"),
            key=lambda payload: payload.metrics["mase"],
        )
        assert row.config == dict(best.hyperparams)
        assert row.mase == best.metrics["mase"]
    oracle_payload = _baselines(records)[0][1]
    assert oracle_payload.updater_id == "oracle"
    oracles = data.sections[0].oracles
    assert oracles.global_expert_mase == oracle_payload.metrics["global_expert_mase"]
    assert oracles.best_expert_mase == oracle_payload.metrics["best_expert_mase"]
    assert oracles.select_mase == oracle_payload.metrics["select_mase"]
    assert oracles.switch_mase == tuple(
        oracle_payload.metrics[f"switch_mase_m{m}"] for m in ORACLE_SWITCHES
    )
    # the true-best forecast is a property of the pool: every ok replay carries it
    equal_ok = next(payload for _r, payload in by_id["equal"] if payload.status == "ok")
    hedge_ok = next(payload for _r, payload in by_id["hedge"] if payload.status == "ok")
    assert oracles.true_best_mase == equal_ok.metrics["oracle_mase"]
    assert equal_ok.metrics["oracle_mase"] == hedge_ok.metrics["oracle_mase"]
    largest = 0.0
    for _record, payload in _baselines(records):
        if payload.updater_id == "oracle":
            continue
        for entry in read(payload.per_task_blob)["series_wall_seconds"]:
            if entry > largest:
                largest = entry
    assert data.sections[0].max_series_wall_seconds == largest


# --- Test 3: gaps and bootstrap -------------------------------------------------------------------


def _table_bounds(
    labeled: list[tuple[str, list[float]]], seed: int
) -> list[tuple[float | None, float | None]]:
    """One Gaps table's intervals, recomputed independently of the implementation.

    One ``random.Random(seed)`` serves the whole table in row order — a
    single-series row draws nothing — so row k's bounds reproduce only after
    the rows before it have consumed their draws.
    """
    rng = random.Random(seed)
    bounds = []
    for _label, differences in labeled:
        n = len(differences)
        if n == 1:
            bounds.append((None, None))
            continue
        means: list[float] = []
        for _resample in range(BOOTSTRAP_RESAMPLES):
            total = 0.0
            for _draw in range(n):
                total = total + differences[rng.randrange(n)]
            means.append(total / n)
        means.sort()
        bounds.append((means[50], means[1949]))
    return bounds


def _loop_mean(values: list[float]) -> float:
    """The left-to-right loop mean, recomputed independently."""
    total = 0.0
    for value in values:
        total = total + value
    return total / len(values)


def test_gap_means_and_intervals_recompute(synthetic_env: dict[str, Path], tmp_path: Path) -> None:
    """Each gap's mean and interval equal an independent recomputation, tables fresh-seeded."""
    ledger = tmp_path / "ledgers"
    assert _tune(synthetic_env, ledger, run_id="run-1") == "completed"
    data = build_report_data(ledger, ["run-1"], bootstrap_seed=11)
    records = _records(ledger, "run-1")
    read = _read_blob(ledger, "run-1")
    picks = select_configs(records, read)["synthetic-dev"]["simple"]
    best_id = min(picks, key=lambda baseline: picks[baseline]["mase"])
    best_payload = next(
        record.payload
        for record, payload in _baselines(records)
        if payload.updater_id == best_id and record.record_id == picks[best_id]["record_id"]
    )
    assert isinstance(best_payload, Baseline)
    best_series = read(best_payload.per_task_blob)["result"]["series"]
    baseline_mase = [entry["mase"] for entry in best_series]
    oracle_result = read(_baselines(records)[0][1].per_task_blob)["result"]
    expert_rows = [entry["expert_mase"] for entry in oracle_result["series"]]
    switch_rows = [entry["switch_mase"] for entry in oracle_result["series"]]
    select_values = [entry["select_mase"] for entry in oracle_result["series"]]
    global_index = oracle_result["expert_names"].index(oracle_result["global_expert"])
    n = len(baseline_mase)
    expected: list[tuple[str, list[float]]] = [
        ("true-best forecast", [s["mase"] - s["oracle_mase"] for s in best_series]),
        (
            f"best fixed expert ({oracle_result['global_expert']})",
            [baseline_mase[i] - expert_rows[i][global_index] for i in range(n)],
        ),
        ("best expert per series", [baseline_mase[i] - min(expert_rows[i]) for i in range(n)]),
    ]
    for index, m in enumerate(ORACLE_SWITCHES):
        expected.append(
            (
                f"switch_mase_m{m}",
                [baseline_mase[i] - switch_rows[i][index] for i in range(n)],
            )
        )
    expected.append(("select_mase", [baseline_mase[i] - select_values[i] for i in range(n)]))
    gaps = data.sections[0].gaps
    assert gaps is not None
    assert [row.label for row in gaps] == [label for label, _d in expected]
    bounds = _table_bounds(expected, 11)
    for row, (_label, differences), (lower, upper) in zip(gaps, expected, bounds, strict=True):
        assert row.mean == _loop_mean(differences)
        assert (row.lower, row.upper) == (lower, upper)
    # the family tables recompute with the same seed, over that family's series only
    assert [table.family for table in data.sections[0].family_gaps] == list(FAMILIES)
    series_ids = [entry["series_id"] for entry in best_series]
    for table in data.sections[0].family_gaps:
        indexes = [
            i
            for i, series_id in enumerate(series_ids)
            if series_id.split("/", 1)[0] == table.family
        ]
        family_labeled = [
            (label, [differences[i] for i in indexes]) for label, differences in expected
        ]
        family_bounds = _table_bounds(family_labeled, 11)
        assert table.rows[-1].lower is None  # one series per family: no interval
        for row, (label, differences), (lower, upper) in zip(
            table.rows, family_labeled, family_bounds, strict=True
        ):
            assert row.label == label
            assert row.mean == _loop_mean(differences)
            assert (row.lower, row.upper) == (lower, upper)
    # same inputs, same data; a different seed moves the intervals, not the means
    assert build_report_data(ledger, ["run-1"], bootstrap_seed=11) == data
    reseeded = build_report_data(ledger, ["run-1"], bootstrap_seed=12)
    regaps = reseeded.sections[0].gaps
    assert regaps is not None
    for row, again in zip(gaps, regaps, strict=True):
        assert again.mean == row.mean
        # one bound can coincide on a small pool; the interval as a pair moves
        assert (again.lower, again.upper) != (row.lower, row.upper)


def test_constant_differences_give_a_degenerate_interval(tmp_path: Path) -> None:
    """Every series differing by the same amount: interval bounds equal the mean."""
    ledger = tmp_path / "ledgers"
    _write_run(
        ledger,
        "run-c",
        baseline_series=[
            {"series_id": f"etth1/{name}", "mase": 1.25, "oracle_mase": None, "recovery": []}
            for name in ("HUFL", "HULL", "MUFL")
        ],
        constant_gap=0.25,
    )
    data = build_report_data(ledger, ["run-c"], bootstrap_seed=3)
    gaps = data.sections[0].gaps
    assert gaps is not None
    assert len(gaps) > 1
    for row in gaps:
        assert row.mean == 0.25
        assert row.lower == 0.25
        assert row.upper == 0.25


# --- TASK_015 PR B, test 12: family gaps on scan pools -------------------------------------------


@pytest.fixture(scope="module")
def shift1_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A synthetic-shift1 cache with one series per preset family, and the paths to load it."""
    base = tmp_path_factory.mktemp("report-shift1")
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


def test_a_scan_pool_carries_family_gaps_per_preset_family(
    shift1_env: dict[str, Path], tmp_path: Path
) -> None:
    """A synthetic-shift1 section has one FamilyGaps per preset family, in FAMILIES order."""
    ledger = tmp_path / "ledgers"
    assert (
        _tune(
            shift1_env,
            ledger,
            run_id="run-shift",
            spec=TuneSpec(
                datasets=("synthetic-shift1",),
                pool_specs=("simple",),
                foundation_models=None,
                seed=7,
                max_wall_seconds=10000.0,
            ),
        )
        == "completed"
    )
    data = build_report_data(ledger, ["run-shift"], bootstrap_seed=0)
    (section,) = data.sections
    assert section.dataset == "synthetic-shift1"
    preset = SCAN_PRESETS["synthetic-shift1"]
    assert [table.family for table in section.family_gaps] == [
        family for family in FAMILIES if family in preset.families
    ]
    # each family table holds the same oracle rows as the pool's, over that family only
    gaps = section.gaps
    assert gaps is not None
    for table in section.family_gaps:
        assert [row.label for row in table.rows] == [row.label for row in gaps]
        assert all(row.lower is None for row in table.rows)  # one series per family here


# --- Test 6: failed grid points -------------------------------------------------------------------


def test_a_failed_grid_point_counts_and_a_whole_failed_grid_gives_no_gaps(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """A failing point next to an ok one still selects; an all-failed grid gives None rows."""
    mixed = tmp_path / "ledgers-mixed"
    assert (
        _tune(synthetic_env, mixed, run_id="run-m", grids={"hedge": [{"eta": 0.0}, {"eta": 1.0}]})
        == "completed"
    )
    data = build_report_data(mixed, ["run-m"], bootstrap_seed=0)
    hedge = next(row for row in data.sections[0].tuned if row.baseline == "hedge")
    assert (hedge.tried, hedge.failed) == (2, 1)
    assert hedge.config == {"eta": 1.0}
    assert hedge.mase is not None
    assert data.sections[0].gaps is not None

    all_failed = tmp_path / "ledgers-failed"
    assert (
        _tune(synthetic_env, all_failed, run_id="run-f", grids={"hedge": [{"eta": 0.0}]})
        == "completed"
    )
    data = build_report_data(all_failed, ["run-f"], bootstrap_seed=0)
    hedge = data.sections[0].tuned[0]
    assert hedge.baseline == "hedge"
    assert hedge.config is None
    assert hedge.mase is None
    assert (hedge.tried, hedge.failed) == (1, 1)
    assert hedge.evaluations == 1.0
    assert hedge.wall_seconds > 0.0
    assert data.sections[0].gaps is None
    assert data.sections[0].family_gaps == ()
    assert data.sections[0].oracles.best_expert_mase > 0.0  # the oracles still report


# --- Test 7: incomplete runs are listed, never tabled ---------------------------------------------


def test_incomplete_runs_are_listed_and_never_tabled(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """Aborted, budget-exhausted and run_end-less runs appear as rows, contribute nothing."""
    ledger = tmp_path / "ledgers"
    assert _tune(synthetic_env, ledger, run_id="run-ok") == "completed"
    assert (
        _tune(
            synthetic_env,
            ledger,
            run_id="run-budget",
            spec=TuneSpec(
                datasets=("synthetic-dev",),
                pool_specs=("simple",),
                foundation_models=None,
                seed=7,
                max_wall_seconds=1e-9,
            ),
        )
        == "budget_exhausted"
    )
    _write_run(ledger, "run-aborted", end_status="aborted")
    _write_run(ledger, "run-noend", run_end=False)
    data = build_report_data(
        ledger, ["run-ok", "run-budget", "run-aborted", "run-noend"], bootstrap_seed=0
    )
    assert [row.run_id for row in data.run_rows] == [
        "run-ok",
        "run-budget",
        "run-aborted",
        "run-noend",
    ]
    by_id = {row.run_id: row for row in data.run_rows}
    assert by_id["run-ok"].status == "completed"
    assert by_id["run-budget"].status == "budget_exhausted"
    assert by_id["run-aborted"].status == "aborted"
    assert by_id["run-noend"].status == "incomplete"
    assert by_id["run-ok"].records == len(_records(ledger, "run-ok"))
    # only the completed run's numbers are tabled, and each pair appears once
    assert [section.dataset for section in data.sections] == ["synthetic-dev"]
    assert data.sections[0].tuned[0].tried == 1  # run-ok replayed equal once
    assert data.leakage_sections == ()


# --- Test 9: violations are listed ----------------------------------------------------------------


def test_violations_of_an_incomplete_run_are_listed(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """The budget-exhausted run's wall-seconds overrun shows in its row."""
    ledger = tmp_path / "ledgers"
    _tune(
        synthetic_env,
        ledger,
        run_id="run-budget",
        spec=TuneSpec(
            datasets=("synthetic-dev",),
            pool_specs=("simple",),
            foundation_models=None,
            seed=7,
            max_wall_seconds=1e-9,
        ),
    )
    data = build_report_data(ledger, ["run-budget"], bootstrap_seed=0)
    assert data.run_rows[0].status == "budget_exhausted"
    assert any("wall_seconds" in violation for violation in data.run_rows[0].violations)
    assert data.sections == ()  # nothing is tabled from it


# --- The leakage split, as dataclass fields (the rendered split is PR C2's) -----------------------


def test_leakage_flag_splits_the_sections(etth1_env: dict[str, Path], tmp_path: Path) -> None:
    """The pooled ETTh1 results carry the flag; simple ones and the sections stay apart."""
    ledger = tmp_path / "ledgers"
    status = tune(
        TuneSpec(
            datasets=("etth1",),
            pool_specs=("simple", "simple+foundation"),
            foundation_models=("leaky",),
            seed=7,
            max_wall_seconds=10000.0,
        ),
        _dev_paths(etth1_env),
        ledger_root=ledger,
        run_id="run-leak",
        git_commit=COMMIT,
        grids={"equal": [{}]},
    )
    assert status == "completed"
    data = build_report_data(ledger, ["run-leak"], bootstrap_seed=0)
    assert [(s.dataset, s.pool_spec, s.leakage_risk) for s in data.sections] == [
        ("etth1", "simple", False)
    ]
    assert [(s.dataset, s.pool_spec, s.leakage_risk) for s in data.leakage_sections] == [
        ("etth1", "simple+foundation", True)
    ]
    for section in (*data.sections, *data.leakage_sections):
        assert section.oracles.true_best_mase is None  # a real pool has no true-best forecast
        assert section.family_gaps == ()  # and no per-family rows
        gaps = section.gaps
        assert gaps is not None
        assert all(row.label != "true-best forecast" for row in gaps)


# --- Test 8: refusals -----------------------------------------------------------------------------


def test_a_duplicate_run_id_refuses(tmp_path: Path) -> None:
    """Naming one run twice refuses."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-1")
    with pytest.raises(ReportError, match="duplicate"):
        build_report_data(ledger, ["run-1", "run-1"], bootstrap_seed=0)


def test_a_missing_run_refuses(tmp_path: Path) -> None:
    """A run id with no ledger refuses."""
    with pytest.raises(ReportError, match="run-missing"):
        build_report_data(tmp_path, ["run-missing"], bootstrap_seed=0)


def test_protocol_versions_must_agree(tmp_path: Path) -> None:
    """Two runs pinned to different protocol versions refuse, both named."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-a")
    _write_run(ledger, "run-b", versions=dict(protocol_versions(), arena="999"))
    with pytest.raises(ReportError, match=r"run-a.*run-b|run-b.*run-a"):
        build_report_data(ledger, ["run-a", "run-b"], bootstrap_seed=0)


def test_a_tampered_record_refuses(tmp_path: Path) -> None:
    """A record whose text no longer hashes to its chain refuses, at its position."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-1")
    main = ledger / "run-1" / "main.jsonl"
    lines = main.read_text(encoding="utf-8").splitlines()
    lines[1] = lines[1].replace("wall_seconds", "wall_secunds")
    main.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ReportError, match="run-1"):
        build_report_data(ledger, ["run-1"], bootstrap_seed=0)


def test_a_deleted_blob_refuses(tmp_path: Path) -> None:
    """A blob a record references but the store no longer holds refuses."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-1")
    ref = _baselines(_records(ledger, "run-1"))[1][1].per_task_blob
    assert ref is not None
    digest = ref[len("sha256:") :]
    (ledger / "run-1" / "blobs" / digest[:2] / digest).unlink()
    with pytest.raises(ReportError, match="run-1"):
        build_report_data(ledger, ["run-1"], bootstrap_seed=0)


def test_a_cache_id_that_disagrees_with_the_config_refuses(tmp_path: Path) -> None:
    """A result replayed from another cache than the config pins refuses."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-1", cache_id="other-cache")
    with pytest.raises(ReportError, match="cache_id"):
        build_report_data(ledger, ["run-1"], bootstrap_seed=0)


def test_mismatched_series_ids_refuse(tmp_path: Path) -> None:
    """An ok result whose series differ from the oracle's refuses; failures never match."""
    ledger = tmp_path / "ledgers"
    _write_run(
        ledger,
        "run-1",
        baseline_series=[
            {"series_id": "etth1/HUFL", "mase": 1.0, "oracle_mase": None, "recovery": []},
            {"series_id": "etth1/WRONG", "mase": 1.1, "oracle_mase": None, "recovery": []},
        ],
        oracle_series_ids=["etth1/HUFL", "etth1/HULL"],
    )
    with pytest.raises(ReportError, match="series ids differ"):
        build_report_data(ledger, ["run-1"], bootstrap_seed=0)


def test_a_selection_the_records_do_not_reproduce_refuses(tmp_path: Path) -> None:
    """A run_end selection that select_configs does not recompute refuses."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-1", selected={"etth1": {"simple": {"equal": None}}})
    with pytest.raises(ReportError, match="selected"):
        build_report_data(ledger, ["run-1"], bootstrap_seed=0)


def test_two_complete_runs_over_one_pool_refuse(tmp_path: Path) -> None:
    """Two completed runs covering the same (dataset, pool spec) refuse, both named."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-a")
    _write_run(ledger, "run-b")
    with pytest.raises(ReportError, match=r"run-a.*run-b|run-b.*run-a"):
        build_report_data(ledger, ["run-a", "run-b"], bootstrap_seed=0)


def test_a_sealed_stream_record_refuses(tmp_path: Path) -> None:
    """A sealed-stream record in a named run refuses."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-1")
    with LedgerWriter(ledger, "run-1", "sealed") as writer:
        writer.append(
            "baseline",
            Baseline(
                updater_id="oracle",
                hyperparams={},
                pool="sealed",
                status="ok",
                metrics={},
                per_task_blob=None,
                error=None,
            ),
        )
    with pytest.raises(ReportError, match="sealed results are reported by the sealed Gate 1 task"):
        build_report_data(ledger, ["run-1"], bootstrap_seed=0)


def test_a_config_that_is_not_gate1_refuses(tmp_path: Path) -> None:
    """A run of another experiment refuses."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-1", experiment="gate2")
    with pytest.raises(ReportError, match="not 'gate1'"):
        build_report_data(ledger, ["run-1"], bootstrap_seed=0)


# --- Test 13 (AST part): sealed imports -----------------------------------------------------------


def test_gate1_report_data_imports_only_the_sealed_client() -> None:
    """An AST walk finds no sealed import beyond evolvets.arena.sealed.client."""
    tree = ast.parse(Path(gate1_report_data.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert (
                not module.startswith("evolvets.arena.sealed")
                or module == "evolvets.arena.sealed.client"
            ), ast.dump(node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("evolvets.arena.sealed"), ast.dump(node)


# --- the hand-built ledger helper -----------------------------------------------------------------


def _hand_config(*, experiment: str = "gate1") -> dict[str, Any]:
    """A run_start config blob in the shape gate1 tune writes (cache ``cafe``)."""
    return {
        "experiment": experiment,
        "stage": "dev-tune",
        "datasets": ["etth1"],
        "pool_specs": ["simple"],
        "simple_cache_ids": {"etth1": "cafe"},
        "foundation_cache_ids": {"etth1": {"simple": []}},
        "foundation_models": {"etth1": {"simple": []}},
        "grids": {"equal": [{}]},
        "selection_metric": "mase",
        "seed": 0,
        "max_wall_seconds": 1000.0,
        "updater_refs": {"equal": "evolvets.updater.baselines:equal"},
    }


def _write_run(
    ledger_root: Path,
    run_id: str,
    *,
    baseline_series: list[dict[str, Any]] | None = None,
    oracle_series_ids: list[str] | None = None,
    constant_gap: float | None = None,
    cache_id: str = "cafe",
    experiment: str = "gate1",
    versions: dict[str, str] | None = None,
    end_status: str = "completed",
    selected: Any | None = None,
    run_end: bool = True,
) -> None:
    """Write one small, well-formed gate1 tune run with hand-chosen numbers.

    The baseline replays ``equal`` over three ETTh1 series; ``constant_gap``
    shifts every oracle per-series value to ``mase - constant_gap`` so every
    gap row sees one constant difference. ``oracle_series_ids`` overrides the
    oracle's series ids and ``cache_id`` the results' cache pin (the config
    keeps ``cafe``), each to force the mismatch its test refuses.
    """
    series = baseline_series or [
        {"series_id": f"etth1/{name}", "mase": 1.0, "oracle_mase": None, "recovery": []}
        for name in ("HUFL", "HULL", "MULL")
    ]
    n = len(series)
    if constant_gap is not None:
        oracle_value = [entry["mase"] - constant_gap for entry in series]
    else:
        oracle_value = [0.7 + 0.05 * i for i in range(n)]
    expert_rows = [[value, value + 0.1] for value in oracle_value]
    switch_rows = [[value for _ in ORACLE_SWITCHES] for value in oracle_value]
    metrics: dict[str, float] = {"n_series": float(n)}
    totals = [0.0, 0.0]
    for row in expert_rows:
        totals[0] = totals[0] + row[0]
        totals[1] = totals[1] + row[1]
    metrics["expert_mase/e0"] = totals[0] / n
    metrics["expert_mase/e1"] = totals[1] / n
    metrics["global_expert_mase"] = totals[0] / n
    metrics["best_expert_mase"] = totals[0] / n
    for index, m in enumerate(ORACLE_SWITCHES):
        total = 0.0
        for row in switch_rows:
            total = total + row[index]
        metrics[f"switch_mase_m{m}"] = total / n
    total = 0.0
    for value in oracle_value:
        total = total + value
    metrics["select_mase"] = total / n
    oracle_ids = oracle_series_ids or [entry["series_id"] for entry in series]
    oracle_blob = {
        "result": {
            "dataset": "etth1",
            "cache_id": cache_id,
            "arena_version": protocol_versions()["arena"],
            "pool_spec": "simple",
            "foundation_cache_ids": [],
            "leakage_risk": False,
            "expert_names": ["e0", "e1"],
            "switches": list(ORACLE_SWITCHES),
            "global_expert": "e0",
            "metrics": metrics,
            "series": [
                {
                    "series_id": oracle_ids[i],
                    "expert_mase": expert_rows[i],
                    "switch_mase": switch_rows[i],
                    "select_mase": oracle_value[i],
                }
                for i in range(n)
            ],
        }
    }
    baseline_blob = {
        "updater_ref": "evolvets.updater.baselines:equal",
        "series_wall_seconds": [0.25] * n,
        "result": {
            "dataset": "etth1",
            "cache_id": cache_id,
            "arena_version": protocol_versions()["arena"],
            "pool_spec": "simple",
            "foundation_cache_ids": [],
            "leakage_risk": False,
            "status": "ok",
            "error": None,
            "failed_series": None,
            "failed_step": None,
            "metrics": {"n_series": float(n), "mase": 1.0},
            "series": series,
        },
    }
    writer = LedgerWriter(ledger_root, run_id, "main")
    blobs = BlobStore(ledger_root / run_id / "blobs")
    with writer:
        writer.append(
            "run_start",
            RunStart(
                condition="gate1-dev-tune",
                seed=0,
                requires_predictions=False,
                config_blob=blobs.put(canonical_bytes(_hand_config(experiment=experiment))),
                git_commit=COMMIT,
                protocol_versions=versions if versions is not None else protocol_versions(),
                budget_limits=Budget(wall_seconds=1000.0, evaluations=100),
            ),
        )
        writer.append(
            "baseline",
            Baseline(
                updater_id="oracle",
                hyperparams={},
                pool="dev",
                status="ok",
                metrics=dict(metrics),
                per_task_blob=blobs.put(canonical_bytes(oracle_blob)),
                error=None,
            ),
            budget=Budget(wall_seconds=0.5),
        )
        writer.append(
            "baseline",
            Baseline(
                updater_id="equal",
                hyperparams={},
                pool="dev",
                status="ok",
                metrics=dict(baseline_blob["result"]["metrics"]),
                per_task_blob=blobs.put(canonical_bytes(baseline_blob)),
                error=None,
            ),
            budget=Budget(wall_seconds=1.0, evaluations=1),
        )
        if run_end:
            chosen = selected
            if chosen is None:
                records = list(LedgerReader(ledger_root, run_id).records())
                chosen = select_configs(records, _read_blob(ledger_root, run_id))
            writer.append("run_end", RunEnd(status=end_status, summary={"selected": chosen}))
