"""Gate 1 headroom data: numbers, bootstrap, ranking and refusals (TASK_015 PR B).

The data-level half of the PR B test list (tests 2 to 9 and 12, the last one
in ``test_gate1_report_data.py``; the unchanged-report test is the suite
itself): the headroom, target MASE and gap trace to the selected and oracle
blobs and the gap equals the report's GapRow.mean (2), the bootstrap bounds
recompute draw for draw (3), a synthetic pool has four rows and a real one
three (4), a zero baseline refuses, in a resample too (5), the ranking is
mechanical (6), the report's refusals pass through with their messages (7),
incomplete runs are listed and never sectioned (8), and the cost columns
trace to the records' budgets (9). Real numbers come from tune runs on the
``synthetic-shift1``, ``synthetic-dev`` and ETTh1 fixtures; the refusal,
zero-baseline and single-series cases come from hand-built ledgers in
``tmp_path``, whose runs reuse ``test_gate1_report_data``'s and
``test_gate1_probe_data``'s writers.

The rendering half (PR B's second PR) adds tests 10 and 11 at the end: the
fixed layout with its headings, table headers, ``n/a`` cells and Notes (10),
and the ``gate1 headroom`` CLI surface (11), both over the same fixture
environments.
"""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.oracles import ORACLE_SWITCHES
from evolvets.arena.versions import protocol_versions
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import Baseline
from evolvets.ledger.schema import Record
from evolvets.ledger.store import LedgerReader
from evolvets.runner import cli
from evolvets.runner.gate1_grids import select_configs
from evolvets.runner.gate1_headroom import (
    HEADROOM_TARGETS,
    RANK_TARGET,
    HeadroomRow,
    HeadroomSection,
    ReportError,
    _rank,
    build_headroom_data,
    render_headroom,
)
from evolvets.runner.gate1_report_data import (
    BOOTSTRAP_RESAMPLES,
    build_report_data,
)
from evolvets.runner.gate1_tune import DevPaths, TuneSpec, tune
from tests.runner.test_gate1_probe_data import _write_probe_run
from tests.runner.test_gate1_report_data import _build_etth1_env, _write_run

COMMIT = "f" * 40
SMALL_GRIDS: dict[str, list[dict[str, float]]] = {
    "equal": [{}],
    "hedge": [{"eta": 0.5}, {"eta": 1.0}],
}


@pytest.fixture(scope="module")
def synthetic_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A synthetic-dev cache with one series per family, and the paths to load it."""
    base = tmp_path_factory.mktemp("headroom-synthetic")
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
def shift1_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A synthetic-shift1 cache with one series per preset family, and the paths to load it."""
    base = tmp_path_factory.mktemp("headroom-shift1")
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


@pytest.fixture(scope="module")
def etth1_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The ETTh1 fixture environment, in this module's own directory."""
    return _build_etth1_env(tmp_path_factory.mktemp("headroom-etth1"))


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
    dataset: str = "synthetic-dev",
    pool_specs: tuple[str, ...] = ("simple",),
    foundation_models: tuple[str, ...] | None = None,
    grids: dict[str, list[dict[str, float]]] | None = None,
) -> str:
    """Run one tuning sweep against a fixture environment; returns the run_end status."""
    return tune(
        TuneSpec(
            datasets=(dataset,),
            pool_specs=pool_specs,
            foundation_models=foundation_models,
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


def _loop_sum(values: list[float]) -> float:
    """The left-to-right loop sum, recomputed independently."""
    total = 0.0
    for value in values:
        total = total + value
    return total


def _loop_mean(values: list[float]) -> float:
    """The left-to-right loop mean, recomputed independently."""
    return _loop_sum(values) / len(values)


def _selected_blob(
    records: list[Record], read: Any, updater_id: str, record_id: str
) -> dict[str, Any]:
    """The blob of one selected record, found by updater id and record id."""
    for record, payload in _baselines(records):
        if payload.updater_id == updater_id and record.record_id == record_id:
            blob = read(payload.per_task_blob)
            assert isinstance(blob, dict)
            return blob
    raise AssertionError(f"no {updater_id} record {record_id} among {len(records)} records")


def _entry(name: str, mase: float) -> dict[str, Any]:
    """One series entry with a hand-chosen MASE, as the hand-built blobs carry them."""
    return {"series_id": f"etth1/{name}", "mase": mase, "oracle_mase": None, "recovery": []}


def _targets_of(
    best_series: list[dict[str, Any]], oracle_result: dict[str, Any]
) -> dict[str, list[float]]:
    """The per-series target values ``t_s`` of every always-present target, by label."""
    switch_rows = [entry["switch_mase"] for entry in oracle_result["series"]]
    expert_rows = [entry["expert_mase"] for entry in oracle_result["series"]]
    return {
        "switch_mase_m8": [row[ORACLE_SWITCHES.index(8)] for row in switch_rows],
        "switch_mase_m32": [row[ORACLE_SWITCHES.index(32)] for row in switch_rows],
        "best expert per series": [min(row) for row in expert_rows],
    }


# --- Test 2: the numbers trace to the selected and oracle blobs ----------------------------------


def test_headroom_gap_and_target_mase_trace_to_the_blobs(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """Each row's three numbers equal hand computations; the gap is the report's GapRow.mean."""
    ledger = tmp_path / "ledgers"
    assert _tune(synthetic_env, ledger, run_id="run-1") == "completed"
    data = build_headroom_data(ledger, ["run-1"], bootstrap_seed=11)
    (section,) = data.ranked
    assert (section.run_id, section.dataset, section.pool_spec) == (
        "run-1",
        "synthetic-dev",
        "simple",
    )
    # the best baseline is the report's, and the gap equals its matching GapRow.mean
    report = build_report_data(ledger, ["run-1"], bootstrap_seed=11)
    report_section = report.sections[0]
    best_tuned = min(
        (row for row in report_section.tuned if row.mase is not None),
        key=lambda row: row.mase,
    )
    assert section.best_baseline == best_tuned.baseline
    assert section.best_baseline_mase == best_tuned.mase
    gaps = report_section.gaps
    assert gaps is not None
    by_label = {gap.label: gap.mean for gap in gaps}
    # hand computations from the selected blob and the oracle blob
    records = _records(ledger, "run-1")
    read = _read_blob(ledger, "run-1")
    picks = select_configs(records, read)["synthetic-dev"]["simple"]
    assert section.best_baseline is not None
    best_series = _selected_blob(
        records, read, section.best_baseline, picks[section.best_baseline]["record_id"]
    )["result"]["series"]
    baseline_mase = [entry["mase"] for entry in best_series]
    oracle_payload = _baselines(records)[0][1]
    assert oracle_payload.updater_id == "oracle"
    oracle_result = read(oracle_payload.per_task_blob)["result"]
    targets = _targets_of(best_series, oracle_result)
    targets["true-best forecast"] = [entry["oracle_mase"] for entry in best_series]
    n = len(baseline_mase)
    assert section.series == n
    assert [row.target for row in section.rows] == list(HEADROOM_TARGETS)
    for row in section.rows:
        t_s = targets[row.target]
        assert row.target_mase == _loop_mean(t_s)
        assert row.gap == _loop_mean([baseline_mase[i] - t_s[i] for i in range(n)])
        assert row.gap == by_label[row.target]
        assert row.headroom == (_loop_sum(baseline_mase) - _loop_sum(t_s)) / _loop_sum(
            baseline_mase
        )


# --- Test 3: the bootstrap bounds recompute ------------------------------------------------------


def _section_bounds(baseline: list[float], targets: dict[str, list[float]], seed: int) -> Any:
    """One section's intervals, recomputed independently of the implementation.

    One ``random.Random(seed)`` serves the whole section in row order, so a
    row's bounds reproduce only after the rows before it consumed their
    draws; a single-series section draws nothing.
    """
    rng = random.Random(seed)
    n = len(baseline)
    bounds: dict[str, tuple[float | None, float | None]] = {}
    for target in HEADROOM_TARGETS:
        if target not in targets:
            continue
        t_s = targets[target]
        if n == 1:
            bounds[target] = (None, None)
            continue
        values: list[float] = []
        for _resample in range(BOOTSTRAP_RESAMPLES):
            baseline_total = 0.0
            target_total = 0.0
            for _draw in range(n):
                index = rng.randrange(n)
                baseline_total = baseline_total + baseline[index]
                target_total = target_total + t_s[index]
            values.append((baseline_total - target_total) / baseline_total)
        values.sort()
        bounds[target] = (values[50], values[1949])
    return bounds


def test_bounds_recompute_and_the_seed_moves_only_the_intervals(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """The bounds equal an independent re-implementation; a reseed keeps every headroom."""
    ledger = tmp_path / "ledgers"
    assert _tune(synthetic_env, ledger, run_id="run-1") == "completed"
    data = build_headroom_data(ledger, ["run-1"], bootstrap_seed=11)
    (section,) = data.ranked
    records = _records(ledger, "run-1")
    read = _read_blob(ledger, "run-1")
    picks = select_configs(records, read)["synthetic-dev"]["simple"]
    assert section.best_baseline is not None
    best_series = _selected_blob(
        records, read, section.best_baseline, picks[section.best_baseline]["record_id"]
    )["result"]["series"]
    baseline_mase = [entry["mase"] for entry in best_series]
    oracle_payload = _baselines(records)[0][1]
    targets = _targets_of(best_series, read(oracle_payload.per_task_blob)["result"])
    targets["true-best forecast"] = [entry["oracle_mase"] for entry in best_series]
    bounds = _section_bounds(baseline_mase, targets, 11)
    for row in section.rows:
        assert (row.lower, row.upper) == bounds[row.target]
    # the same seed gives identical data; a different one moves the intervals only
    assert build_headroom_data(ledger, ["run-1"], bootstrap_seed=11) == data
    reseeded = build_headroom_data(ledger, ["run-1"], bootstrap_seed=12)
    (moved_section,) = reseeded.ranked
    moved = False
    for row, other in zip(section.rows, moved_section.rows, strict=True):
        assert other.headroom == row.headroom
        assert other.target_mase == row.target_mase
        moved = moved or (other.lower, other.upper) != (row.lower, row.upper)
    assert moved


def test_a_single_series_pool_has_no_intervals(tmp_path: Path) -> None:
    """A one-series pool leaves both bounds None: there is nothing to resample."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-one", baseline_series=[_entry("HUFL", 1.0)])
    data = build_headroom_data(ledger, ["run-one"], bootstrap_seed=0)
    (section,) = data.ranked
    assert section.series == 1
    assert [row.target for row in section.rows] == [
        "switch_mase_m8",
        "switch_mase_m32",
        "best expert per series",
    ]
    for row in section.rows:
        assert row.lower is None
        assert row.upper is None
        assert row.headroom != 0.0  # the point estimate stands without an interval


# --- Test 4: the rows a pool carries -------------------------------------------------------------


def test_a_synthetic_pool_has_four_rows_and_a_real_one_three(
    shift1_env: dict[str, Path], etth1_env: dict[str, Path], tmp_path: Path
) -> None:
    """A scan pool carries the true-best row; an ETTh1 pool does not."""
    ledger = tmp_path / "ledgers"
    assert _tune(shift1_env, ledger, run_id="run-shift", dataset="synthetic-shift1") == "completed"
    data = build_headroom_data(ledger, ["run-shift"], bootstrap_seed=0)
    (section,) = data.ranked
    assert section.dataset == "synthetic-shift1"
    assert section.leakage_risk is False
    assert [row.target for row in section.rows] == list(HEADROOM_TARGETS)
    real = tmp_path / "ledgers-real"
    assert (
        _tune(etth1_env, real, run_id="run-etth1", dataset="etth1", grids={"equal": [{}]})
        == "completed"
    )
    data = build_headroom_data(real, ["run-etth1"], bootstrap_seed=0)
    (section,) = data.ranked
    assert section.dataset == "etth1"
    assert [row.target for row in section.rows] == [
        "switch_mase_m8",
        "switch_mase_m32",
        "best expert per series",
    ]


# --- Test 5: a zero baseline refuses -------------------------------------------------------------


def test_a_zero_baseline_refuses_point_estimate_and_resample_alike(tmp_path: Path) -> None:
    """A zero baseline sum refuses, and so does one resample that drew only zero series."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-zero", baseline_series=[_entry(name, 0.0) for name in ("A", "B", "C")])
    with pytest.raises(ReportError, match="headroom undefined: zero baseline MASE"):
        build_headroom_data(ledger, ["run-zero"], bootstrap_seed=0)
    # the baseline sum is 2.0, but a resample that draws A three times sums to zero
    _write_run(
        ledger, "run-mixed", baseline_series=[_entry("A", 0.0), _entry("B", 1.0), _entry("C", 1.0)]
    )
    with pytest.raises(ReportError, match="headroom undefined: zero baseline MASE"):
        build_headroom_data(ledger, ["run-mixed"], bootstrap_seed=0)


# --- Test 6: the ranking is mechanical -----------------------------------------------------------


def _hand_section(
    name: str, *, lower: float | None, headroom: float, leakage: bool = False
) -> HeadroomSection:
    """A minimal section whose switch-8 row carries the given bounds."""
    return HeadroomSection(
        run_id="run-x",
        dataset=name,
        pool_spec="simple",
        leakage_risk=leakage,
        series=3,
        best_baseline="equal",
        best_baseline_mase=1.0,
        replay_wall_seconds=1.0,
        tune_wall_seconds=9.0,
        rows=(
            HeadroomRow(
                target=RANK_TARGET,
                target_mase=0.5,
                gap=0.25,
                headroom=headroom,
                lower=lower,
                upper=None if lower is None else lower + 0.1,
            ),
        ),
    )


def test_the_ranking_sorts_by_lower_then_headroom_and_keeps_ties() -> None:
    """Descending lower bound; a tie on it breaks by headroom; a full tie keeps input order."""
    first = _hand_section("first", lower=0.01, headroom=0.02)
    second = _hand_section("second", lower=0.05, headroom=0.06)
    third = _hand_section("third", lower=0.03, headroom=0.04)
    assert _rank([first, second, third]) == (second, third, first)
    low = _hand_section("low", lower=0.03, headroom=0.031)
    high = _hand_section("high", lower=0.03, headroom=0.039)
    assert _rank([low, high]) == (high, low)
    a = _hand_section("a", lower=0.03, headroom=0.04)
    b = _hand_section("b", lower=0.03, headroom=0.04)
    assert _rank([a, b]) == (a, b)


def test_a_single_series_section_ranks_by_headroom() -> None:
    """Without an interval the headroom itself is the ranking key."""
    weak = _hand_section("weak", lower=None, headroom=0.02)
    strong = _hand_section("strong", lower=None, headroom=0.07)
    assert _rank([weak, strong]) == (strong, weak)


def test_leakage_sections_rank_separately_and_failed_pools_never_rank(
    synthetic_env: dict[str, Path], etth1_env: dict[str, Path], tmp_path: Path
) -> None:
    """The leakage split ranks its own list; an all-failed pool lands among the unranked."""
    ledger = tmp_path / "ledgers-leak"
    assert (
        _tune(
            etth1_env,
            ledger,
            run_id="run-leak",
            dataset="etth1",
            pool_specs=("simple", "simple+foundation"),
            foundation_models=("leaky",),
            grids={"equal": [{}]},
        )
        == "completed"
    )
    data = build_headroom_data(ledger, ["run-leak"], bootstrap_seed=0)
    assert [section.dataset for section in data.ranked] == ["etth1"]
    assert data.ranked[0].pool_spec == "simple"
    assert [section.pool_spec for section in data.leakage_ranked] == ["simple+foundation"]
    assert data.unranked == ()
    failed = tmp_path / "ledgers-failed"
    assert (
        _tune(
            synthetic_env,
            failed,
            run_id="run-failed",
            grids={"hedge": [{"eta": 0.0}]},
        )
        == "completed"
    )
    data = build_headroom_data(failed, ["run-failed"], bootstrap_seed=0)
    assert data.ranked == ()
    assert data.leakage_ranked == ()
    (section,) = data.unranked
    assert (section.dataset, section.pool_spec) == ("synthetic-dev", "simple")
    assert section.best_baseline is None
    assert section.best_baseline_mase is None
    assert section.replay_wall_seconds is None
    assert section.rows == ()
    assert section.tune_wall_seconds > 0.0  # the grid still ran and is still charged


# --- Test 7: the report's refusals pass through --------------------------------------------------


def test_the_report_refusals_pass_through_with_their_messages(tmp_path: Path) -> None:
    """Every refused input refuses with exactly build_report_data's message."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-a")
    _write_run(ledger, "run-b")
    _write_run(ledger, "run-c", versions=dict(protocol_versions(), arena="999"))
    _write_probe_run(ledger, "probe-1", tune_run_id="run-a")
    for run_ids in (
        ["run-a", "run-a"],
        ["run-a", "run-c"],
        ["probe-1"],
        ["run-a", "run-b"],
    ):
        with pytest.raises(ReportError) as headroom_error:
            build_headroom_data(ledger, run_ids, bootstrap_seed=0)
        with pytest.raises(ReportError) as report_error:
            build_report_data(ledger, run_ids, bootstrap_seed=0)
        assert str(headroom_error.value) == str(report_error.value)


# --- Test 8: incomplete runs are listed, never sectioned -----------------------------------------


def test_an_aborted_run_is_listed_and_never_sectioned(tmp_path: Path) -> None:
    """An aborted tune run appears in the rows and in no section list."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-aborted", end_status="aborted")
    data = build_headroom_data(ledger, ["run-aborted"], bootstrap_seed=0)
    (row,) = data.run_rows
    assert row.run_id == "run-aborted"
    assert row.status == "aborted"
    assert data.ranked == ()
    assert data.leakage_ranked == ()
    assert data.unranked == ()


# --- Test 9: the cost columns trace to the records' budgets --------------------------------------


def test_the_cost_columns_trace_to_the_records(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """Replay seconds is the selected record's wall-clock; tune seconds the whole grid's."""
    ledger = tmp_path / "ledgers"
    assert _tune(synthetic_env, ledger, run_id="run-1") == "completed"
    data = build_headroom_data(ledger, ["run-1"], bootstrap_seed=0)
    (section,) = data.ranked
    records = _records(ledger, "run-1")
    read = _read_blob(ledger, "run-1")
    picks = select_configs(records, read)["synthetic-dev"]["simple"]
    assert section.best_baseline is not None
    selected = next(
        record
        for record, payload in _baselines(records)
        if payload.updater_id == section.best_baseline
        and record.record_id == picks[section.best_baseline]["record_id"]
    )
    assert section.replay_wall_seconds == selected.budget.wall_seconds
    grid_seconds = 0.0
    for record, payload in _baselines(records):
        if payload.updater_id == "oracle":
            continue  # the oracle record is not part of the tuning grid
        grid_seconds = grid_seconds + record.budget.wall_seconds
    assert section.tune_wall_seconds == grid_seconds


# --- Test 10: the rendering -----------------------------------------------------------------------


#: The ranked tables' header line, as TASK_015 specifies it (both carry the same).
_RANKED_HEADER_LINE = (
    "| rank | dataset | pool | series | best tuned baseline | baseline MASE | "
    "switch-8 headroom | 95% interval | switch-32 headroom | best-per-series headroom | "
    "true-best headroom | replay seconds | tune seconds |"
)

#: The Notes lines, pinned from TASK_015: the rendering must carry them
#: verbatim. The first line's minus sign is U+2212, as the task writes it
#: (escaped for RUF001).
_TASK_NOTES = (
    "- Headroom is (best tuned baseline MASE \u2212 target MASE) / best tuned baseline MASE "
    "over the pool's series; positive means the target is better. Intervals are paired "
    "bootstrap intervals over series (2000 resamples) of the same ratio.",
    "- The baselines were tuned, and the targets computed, on these same dev series.",
    "- The switching oracles and the best expert per series select single experts in hindsight: "
    "no online updater can reach them, and they do not bound weighted combinations.",
    "- Replay seconds is the wall-clock of one evaluation (one replay of the selected "
    "configuration over every series of the pool); tune seconds is the whole grid.",
    "- Pools are ranked by the lower end of the switch-8 headroom interval; there is no "
    "multiple-comparison correction.",
)


@pytest.fixture(scope="module")
def scan_ledger(
    shift1_env: dict[str, Path],
    etth1_env: dict[str, Path],
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """A ledger with a shift1 run and an ETTh1 run over simple and the leaking pooled spec.

    Two clean sections (synthetic-shift1 and etth1 on simple) and one
    leakage-risk section (etth1 pooled with ``leaky``), so the rendering has
    one of everything the layout conditionally shows.
    """
    ledger = tmp_path_factory.mktemp("headroom-render") / "ledgers"
    assert _tune(shift1_env, ledger, run_id="run-shift", dataset="synthetic-shift1") == "completed"
    assert (
        _tune(
            etth1_env,
            ledger,
            run_id="run-etth1",
            dataset="etth1",
            pool_specs=("simple", "simple+foundation"),
            foundation_models=("leaky",),
            grids={"equal": [{}]},
        )
        == "completed"
    )
    return ledger


def _render(ledger: Path, run_ids: Sequence[str], *, bootstrap_seed: int = 5) -> str:
    """Render the headroom scan of some runs, the way the CLI does."""
    return render_headroom(
        build_headroom_data(ledger, run_ids, bootstrap_seed=bootstrap_seed),
        run_ids,
        bootstrap_seed=bootstrap_seed,
    )


def _cells(line: str) -> list[str]:
    """One table row's cells, stripped, as the rendered line carries them."""
    return [cell.strip() for cell in line.strip("|").split("|")]


def _table_rows(lines: list[str], heading: str) -> list[list[str]]:
    """The data rows of the table under a heading: header and rule dropped."""
    start = lines.index(heading)
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("#"))
    table = [line for line in lines[start + 2 : end] if line.startswith("|")]
    return [_cells(row) for row in table[2:]]


def test_the_layout_headings_and_table_headers_are_exact(scan_ledger: Path) -> None:
    """The sections come in the fixed order with exactly the specified headings."""
    lines = _render(scan_ledger, ["run-shift", "run-etth1"]).splitlines()
    # the document's own headings only: "# " and "## ", not the "###" detail subsections
    assert [line for line in lines if line.startswith("# ")] == ["# Headroom scan (dev)"]
    assert [line for line in lines if line.startswith("## ")] == [
        "## Runs",
        "## Protocol versions",
        "## Ranked headroom",
        "## Ranked headroom, leakage-risk pools",
        "## Per-pool detail",
        "## Notes",
    ]
    assert (
        lines[2] == "generated by `evolvets gate1 headroom --run run-shift --run run-etth1 "
        "--bootstrap-seed 5`"
    )
    runs_at = lines.index("## Runs")
    assert lines[runs_at + 2] == (
        "| run id | condition | status | git commit | records | validate_run |"
    )
    versions_at = lines.index("## Protocol versions")
    assert lines[versions_at + 2] == "| protocol | version |"
    assert lines[lines.index("## Ranked headroom") + 2] == _RANKED_HEADER_LINE
    assert lines[lines.index("## Ranked headroom, leakage-risk pools") + 2] == _RANKED_HEADER_LINE
    detail_at = lines.index("## Per-pool detail")
    assert lines.count("| target | target MASE | gap | headroom | 95% interval |") == 3
    assert lines[detail_at + 2].startswith("### ")
    # the detail lists the ranked sections first, then the leakage-risk ones, each marked
    shift1_at = lines.index("### synthetic-shift1, pool simple")
    etth1_at = lines.index("### etth1, pool simple")
    leakage_at = lines.index("### etth1, pool simple+foundation (leakage risk)")
    assert max(shift1_at, etth1_at) < leakage_at


def test_the_ranked_and_detail_rows_trace_to_the_data(scan_ledger: Path) -> None:
    """Every cell of the ranked rows and one detail subsection equals its section's numbers."""
    data = build_headroom_data(scan_ledger, ["run-shift", "run-etth1"], bootstrap_seed=5)
    lines = _render(scan_ledger, ["run-shift", "run-etth1"]).splitlines()
    clean = _table_rows(lines, "## Ranked headroom")
    leakage = _table_rows(lines, "## Ranked headroom, leakage-risk pools")
    assert len(clean) == len(data.ranked) == 2
    assert len(leakage) == len(data.leakage_ranked) == 1
    for cells, section, rank in (
        *zip(clean, data.ranked, range(1, 3), strict=True),
        *zip(leakage, data.leakage_ranked, range(1, 2), strict=True),
    ):
        by_target = {row.target: row for row in section.rows}
        switch8 = by_target["switch_mase_m8"]
        true_best = by_target.get("true-best forecast")
        assert section.best_baseline is not None
        assert section.best_baseline_mase is not None
        assert section.replay_wall_seconds is not None
        assert cells == [
            str(rank),
            section.dataset,
            section.pool_spec,
            str(section.series),
            section.best_baseline,
            f"{section.best_baseline_mase:.4f}",
            f"{switch8.headroom:.4f}",
            f"[{switch8.lower:.4f}, {switch8.upper:.4f}]",
            f"{by_target['switch_mase_m32'].headroom:.4f}",
            f"{by_target['best expert per series'].headroom:.4f}",
            "n/a" if true_best is None else f"{true_best.headroom:.4f}",
            f"{section.replay_wall_seconds:.4f}",
            f"{section.tune_wall_seconds:.4f}",
        ]
    # one detail subsection: the best-baseline sentence, then one row per target
    section = data.ranked[0]
    assert section.best_baseline is not None
    assert section.best_baseline_mase is not None
    at = lines.index(f"### {section.dataset}, pool {section.pool_spec}")
    assert lines[at + 2] == (
        f"Best tuned baseline: {section.best_baseline} "
        f"(dev MASE {section.best_baseline_mase:.4f}), run {section.run_id}"
    )
    assert lines[at + 4] == "| target | target MASE | gap | headroom | 95% interval |"
    rows = lines[at + 6 : at + 6 + len(section.rows)]  # +5 is the table rule
    for line, row in zip(rows, section.rows, strict=True):
        interval = "n/a" if row.lower is None else f"[{row.lower:.4f}, {row.upper:.4f}]"
        assert _cells(line) == [
            row.target,
            f"{row.target_mase:.4f}",
            f"{row.gap:.4f}",
            f"{row.headroom:.4f}",
            interval,
        ]


def test_missing_intervals_and_rows_render_as_na(tmp_path: Path) -> None:
    """A single-series pool renders n/a intervals; a real pool an n/a true-best cell."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "run-one", baseline_series=[_entry("HUFL", 1.0)])
    lines = _render(ledger, ["run-one"], bootstrap_seed=0).splitlines()
    (ranked_row,) = _table_rows(lines, "## Ranked headroom")
    assert ranked_row[7] == "n/a"  # the 95% interval: a single series has nothing to draw
    assert ranked_row[10] == "n/a"  # true-best headroom: a real pool carries no such row
    detail_rows = [line for line in lines if line.startswith("| switch_mase_m")]
    assert len(detail_rows) == 2
    per_series = next(line for line in lines if line.startswith("| best expert per series |"))
    for line in (*detail_rows, per_series):
        assert _cells(line)[-1] == "n/a"


def test_a_pool_without_a_tuned_baseline_is_a_bullet_never_a_section(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """An all-failed pool lands in the bullet list, in no table and no detail subsection."""
    ledger = tmp_path / "ledgers"
    assert (
        _tune(synthetic_env, ledger, run_id="run-failed", grids={"hedge": [{"eta": 0.0}]})
        == "completed"
    )
    lines = _render(ledger, ["run-failed"], bootstrap_seed=0).splitlines()
    at = lines.index("## Pools without a tuned baseline")
    assert lines[at + 2] == "- synthetic-dev, pool simple (run run-failed)"
    # the ranked heading and header stay, with no data row; nothing is detailed
    assert _table_rows(lines, "## Ranked headroom") == []
    assert not [line for line in lines if line.startswith("### ")]
    assert "## Ranked headroom, leakage-risk pools" not in lines


def test_the_leakage_heading_appears_only_with_leakage_sections(scan_ledger: Path) -> None:
    """Naming only the clean run leaves the leakage heading and marker out entirely."""
    clean = _render(scan_ledger, ["run-shift"])
    assert "## Ranked headroom, leakage-risk pools" not in clean
    assert "leakage" not in clean
    assert "## Pools without a tuned baseline" not in clean
    with_leakage = _render(scan_ledger, ["run-shift", "run-etth1"])
    assert "## Ranked headroom, leakage-risk pools" in with_leakage
    assert "(leakage risk)" in with_leakage


def test_the_notes_are_exactly_the_task_sentences(scan_ledger: Path) -> None:
    """The Notes section holds exactly TASK_015's five lines, verbatim."""
    report = _render(scan_ledger, ["run-shift", "run-etth1"])
    notes = report[report.index("## Notes") :]
    assert [line for line in notes.splitlines() if line.startswith("- ")] == list(_TASK_NOTES)


def test_the_generating_line_lists_every_run_and_renders_are_byte_identical(
    scan_ledger: Path, tmp_path: Path
) -> None:
    """Every --run appears in the generating line; two renders give the same bytes."""
    report = _render(scan_ledger, ["run-shift", "run-etth1"])
    assert report.endswith("\n")
    assert (
        "generated by `evolvets gate1 headroom --run run-shift --run run-etth1 "
        "--bootstrap-seed 5`" in report
    )
    assert _render(scan_ledger, ["run-shift", "run-etth1"]) == report
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(scan_ledger, elsewhere)
    assert _render(elsewhere, ["run-shift", "run-etth1"]) == report


# --- Test 11: the CLI -----------------------------------------------------------------------------


def _headroom_args(ledger: Path, *extra: str) -> list[str]:
    """A gate1 headroom invocation over both scan runs, plus the output arguments."""
    return [
        "gate1",
        "headroom",
        "--ledger-root",
        str(ledger),
        "--run",
        "run-shift",
        "--run",
        "run-etth1",
        *extra,
    ]


def test_cli_headroom_run_is_repeatable_and_out_writes(
    scan_ledger: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two --run flags name both runs, and --out carries their scan."""
    out = tmp_path / "headroom.md"
    args = _headroom_args(scan_ledger, "--bootstrap-seed", "3", "--out", str(out))
    assert cli.main(args) == 0
    expected = _render(scan_ledger, ["run-shift", "run-etth1"], bootstrap_seed=3)
    assert out.read_text(encoding="utf-8") == expected
    assert capsys.readouterr().out.startswith("OK ")


def test_cli_headroom_check_accepts_the_written_file_and_rejects_a_change(
    scan_ledger: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--check accepts the freshly written scan and fails on a one-character change."""
    out = tmp_path / "headroom.md"
    assert cli.main(_headroom_args(scan_ledger, "--out", str(out))) == 0
    assert cli.main(_headroom_args(scan_ledger, "--check", str(out))) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    target = next(i for i, line in enumerate(lines) if line.startswith("| run-shift |"))
    lines[target] = lines[target].replace("completed", "completee", 1)
    tampered = tmp_path / "tampered.md"
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert cli.main(_headroom_args(scan_ledger, "--check", str(tampered))) == 1
    assert f"differs at line {target + 1}" in capsys.readouterr().out


def test_cli_headroom_refusal_writes_nothing(
    scan_ledger: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ReportError is one FAIL line, exit 1, and --out stays unwritten."""
    out = tmp_path / "headroom.md"
    args = [
        "gate1",
        "headroom",
        "--ledger-root",
        str(scan_ledger),
        "--run",
        "run-shift",
        "--run",
        "run-shift",
        "--out",
        str(out),
    ]
    assert cli.main(args) == 1
    assert capsys.readouterr().out.startswith("FAIL ")
    assert not out.exists()


def test_cli_headroom_requires_its_run_and_output_flags(scan_ledger: Path, tmp_path: Path) -> None:
    """Without --run, or without --out/--check, argparse rejects the call."""
    with pytest.raises(SystemExit):
        cli.main(
            [
                "gate1",
                "headroom",
                "--ledger-root",
                str(scan_ledger),
                "--out",
                str(tmp_path / "headroom.md"),
            ]
        )
    with pytest.raises(SystemExit):
        cli.main(["gate1", "headroom", "--ledger-root", str(scan_ledger), "--run", "run-shift"])
