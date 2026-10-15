"""Gate 1 probe data: numbers, verdicts and refusals (TASK_014 PR D).

The data-level half of the PR D test list (tests 2 to 7; the unchanged-
report, leakage rendering and CLI tests are the second half's, in
``test_gate1_report.py``): the selections and gains trace to the selected
blobs (2), the verdicts are mechanical (3), every refusal has a test (4),
incomplete probe runs are listed and never tabled (5), a pool without an
ok baseline configuration has no gains (6) and an all-failed learned
updater keeps its row but loses its gain (7). Real numbers come from a
tune run plus probe runs on the synthetic fixture (PR C's ``probe``); the
refusal and verdict cases come from hand-built ledgers in ``tmp_path``,
whose tune runs reuse ``test_gate1_report_data``'s writer.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.errors import ArenaError
from evolvets.arena.versions import protocol_versions
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.canonical import canonical_bytes
from evolvets.ledger.payloads import Baseline, RunEnd, RunStart
from evolvets.ledger.schema import Budget, Record
from evolvets.ledger.store import LedgerReader, LedgerWriter
from evolvets.runner import gate1_tune
from evolvets.runner.gate1_grids import select_configs
from evolvets.runner.gate1_probe import ProbeSpec, probe
from evolvets.runner.gate1_probe_data import build_probe_data
from evolvets.runner.gate1_report_data import ReportError, _gap_rows, build_report_data
from evolvets.runner.gate1_tune import DevPaths, TuneSpec, tune
from evolvets.updater.interfaces import ConfigValue
from evolvets.updater.learned import LEARNED_IDS
from tests.runner.test_gate1_report_data import _write_run

COMMIT = "e" * 40
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


@pytest.fixture(scope="module")
def synthetic_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A synthetic-dev cache with one series per family, and the paths to load it."""
    base = tmp_path_factory.mktemp("probe-data-synthetic")
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


def tune_once(
    env: dict[str, Path],
    ledger_root: Path,
    *,
    run_id: str = "tune-1",
    grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]] = SMALL_GRIDS,
) -> str:
    """Run one tuning sweep against a fixture environment; returns the run_end status."""
    return tune(
        TuneSpec(
            datasets=("synthetic-dev",),
            pool_specs=("simple",),
            foundation_models=None,
            seed=7,
            max_wall_seconds=10000.0,
        ),
        DevPaths(
            cache_root=env["cache_root"],
            foundation_root=env["foundation_root"],
            data_root=env["data_root"],
            manifest_dir=env["manifest_dir"],
        ),
        ledger_root=ledger_root,
        run_id=run_id,
        git_commit=COMMIT,
        grids=grids,
    )


def run_probe(
    env: dict[str, Path],
    ledger_root: Path,
    *,
    run_id: str = "probe-1",
    grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]] = SMALL_LEARNED,
) -> str:
    """Run one probe against a fixture environment; returns the run_end status."""
    return probe(
        ProbeSpec(
            tune_run_id="tune-1",
            datasets=("synthetic-dev",),
            pool_specs=("simple",),
            foundation_models=None,
            seed=3,
            max_wall_seconds=10000.0,
        ),
        DevPaths(
            cache_root=env["cache_root"],
            foundation_root=env["foundation_root"],
            data_root=env["data_root"],
            manifest_dir=env["manifest_dir"],
        ),
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


def _selected_blob(
    records: list[Record], read: Callable[[str], object], updater_id: str, record_id: str
) -> Any:
    """The blob of one selected record, found by updater id and record id."""
    for record in records:
        payload = record.payload
        if (
            record.kind == "baseline"
            and isinstance(payload, Baseline)
            and payload.updater_id == updater_id
            and record.record_id == record_id
        ):
            return read(payload.per_task_blob)
    raise AssertionError(f"no {updater_id} record {record_id} among {len(records)} records")


# --- Test 2: the numbers trace to the selected blobs ----------------------------------------------


def test_selections_and_gains_trace_to_the_selected_blobs(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """The learned rows carry the probe's selection; each gain is the loop-mean and bounds."""
    ledger = tmp_path / "ledgers"
    assert tune_once(synthetic_env, ledger) == "completed"
    assert run_probe(synthetic_env, ledger) == "completed"
    data = build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=11)
    section = data.sections[0]
    assert (section.dataset, section.pool_spec, section.leakage_risk) == (
        "synthetic-dev",
        "simple",
        False,
    )
    picks = read_records(ledger, "probe-1")[-1].payload.summary["selected"]["synthetic-dev"][
        "simple"
    ]
    assert [row.baseline for row in section.learned] == list(LEARNED_IDS)
    for row in section.learned:
        assert row.config == picks[row.baseline]["config"]
        assert row.mase == picks[row.baseline]["mase"]
    # the baseline side is the tune run's best, its per-series MASE from the selected blob
    tune_records = read_records(ledger, "tune-1")
    tune_read = blob_reader(ledger, "tune-1")
    tune_picks = select_configs(tune_records, tune_read)["synthetic-dev"]["simple"]
    best_id = min(tune_picks, key=lambda baseline: tune_picks[baseline]["mase"])
    assert section.best_baseline == best_id
    assert section.best_baseline_mase == tune_picks[best_id]["mase"]
    best_series = _selected_blob(
        tune_records, tune_read, best_id, tune_picks[best_id]["record_id"]
    )["result"]["series"]
    best_mase = [entry["mase"] for entry in best_series]
    # each gain row is the loop-mean of d_s and the _gap_rows bounds on the same input
    probe_records = read_records(ledger, "probe-1")
    probe_read = blob_reader(ledger, "probe-1")
    labeled: list[tuple[str, list[float]]] = []
    for learned_id in LEARNED_IDS:
        learned_series = _selected_blob(
            probe_records, probe_read, learned_id, picks[learned_id]["record_id"]
        )["result"]["series"]
        learned_mase = [entry["mase"] for entry in learned_series]
        labeled.append(
            (learned_id, [best_mase[i] - learned_mase[i] for i in range(len(best_mase))])
        )
    expected = _gap_rows(labeled, 11)
    assert [row.updater for row in section.gains] == list(LEARNED_IDS)
    for row, (_label, differences), gap in zip(section.gains, labeled, expected, strict=True):
        total = 0.0
        for value in differences:
            total = total + value
        assert row.mean == total / len(differences)
        assert (row.lower, row.upper) == (gap.lower, gap.upper)
    # the same seed gives identical data; a different seed moves the bounds, never the means
    again = build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=11)
    assert again == data
    reseeded = build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=12)
    assert [row.updater for row in reseeded.sections[0].gains] == list(LEARNED_IDS)
    moved = False
    for row, other in zip(section.gains, reseeded.sections[0].gains, strict=True):
        assert other.mean == row.mean
        moved = moved or (other.lower, other.upper) != (row.lower, row.upper)
    assert moved


def test_run_rows_list_every_probe_run(synthetic_env: dict[str, Path], tmp_path: Path) -> None:
    """Each row carries its run's status, commit, tune run, record count and violations."""
    ledger = tmp_path / "ledgers"
    assert tune_once(synthetic_env, ledger) == "completed"
    assert run_probe(synthetic_env, ledger) == "completed"
    data = build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)
    (row,) = data.run_rows
    assert row.run_id == "probe-1"
    assert row.status == "completed"
    assert row.git_commit == COMMIT
    assert row.tune_run_id == "tune-1"
    assert row.records == len(read_records(ledger, "probe-1"))
    assert row.violations == ()


# --- Test 3: the verdicts are mechanical ----------------------------------------------------------


def _entry(series_id: str, mase: float) -> dict[str, Any]:
    """One series entry with a hand-chosen MASE, as the hand-built blobs carry them."""
    return {"series_id": series_id, "mase": mase, "oracle_mase": None, "recovery": []}


def test_verdicts_are_mechanical(tmp_path: Path) -> None:
    """All-positive differences beat, all-negative lose, mixed stay inside, one series is n/a."""
    ledger = tmp_path / "ledgers"
    ids = ["etth1/HUFL", "etth1/HULL", "etth1/MULL"]
    cases = [
        ((1.0, 1.0, 1.0), (0.5, 0.6, 0.7), "beats"),
        ((0.5, 0.6, 0.7), (1.0, 1.0, 1.0), "worse"),
        ((1.0, 1.0, 1.0), (0.4, 1.6, 1.0), "inside"),
    ]
    for index, (baseline, learned, verdict) in enumerate(cases):
        tune_id, probe_id = f"tune-{index}", f"probe-{index}"
        _write_run(
            ledger,
            tune_id,
            baseline_series=[_entry(ids[i], baseline[i]) for i in range(3)],
        )
        _write_probe_run(
            ledger,
            probe_id,
            tune_run_id=tune_id,
            learned={"ridge_stack": [_entry(ids[i], learned[i]) for i in range(3)]},
        )
        section = build_probe_data(ledger, [tune_id], [probe_id], bootstrap_seed=0).sections[0]
        assert section.best_baseline == "equal"
        (gain,) = section.gains
        assert gain.updater == "ridge_stack"
        assert gain.verdict == verdict
    # a single series has no interval to draw
    _write_run(ledger, "tune-one", baseline_series=[_entry("etth1/HUFL", 1.0)])
    _write_probe_run(
        ledger,
        "probe-one",
        tune_run_id="tune-one",
        learned={"ridge_stack": [_entry("etth1/HUFL", 0.5)]},
    )
    section = build_probe_data(ledger, ["tune-one"], ["probe-one"], bootstrap_seed=0).sections[0]
    (gain,) = section.gains
    assert gain.verdict == "n/a"
    assert gain.lower is None
    assert gain.upper is None
    assert gain.mean == 0.5


# --- Test 4: refusals -----------------------------------------------------------------------------


def _hand_ledger(tmp_path: Path) -> Path:
    """A tune run and a well-formed probe run pinned to it, both hand-built."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "tune-1")
    _write_probe_run(ledger, "probe-1")
    return ledger


def test_empty_duplicate_and_overlapping_probe_ids_refuse(tmp_path: Path) -> None:
    """No probe ids, a repeated one, or one also named as a tune run each refuse."""
    ledger = _hand_ledger(tmp_path)
    with pytest.raises(ReportError, match="at least one probe run id"):
        build_probe_data(ledger, ["tune-1"], [], bootstrap_seed=0)
    with pytest.raises(ReportError, match="duplicate probe run id"):
        build_probe_data(ledger, ["tune-1"], ["probe-1", "probe-1"], bootstrap_seed=0)
    with pytest.raises(ReportError, match="both a tune run and a probe run"):
        build_probe_data(ledger, ["tune-1"], ["tune-1"], bootstrap_seed=0)


def test_a_tampered_probe_run_refuses(tmp_path: Path) -> None:
    """A probe run whose chain fails verify refuses."""
    ledger = _hand_ledger(tmp_path)
    main = ledger / "probe-1" / "main.jsonl"
    raw = main.read_bytes()
    tampered = raw.replace(b"updater_id", b"updater_ie", 1)
    assert tampered != raw
    main.write_bytes(tampered)
    with pytest.raises(ReportError, match="probe-1"):
        build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)


def test_a_sealed_stream_record_in_a_probe_run_refuses(tmp_path: Path) -> None:
    """A sealed-stream record inside a probe run refuses."""
    ledger = _hand_ledger(tmp_path)
    with LedgerWriter(ledger, "probe-1", "sealed") as writer:
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
    with pytest.raises(ReportError, match="sealed results are reported"):
        build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)


def test_a_run_of_another_condition_refuses_as_a_probe_run(tmp_path: Path) -> None:
    """A tune run does not pass as a probe run."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "tune-1")
    _write_run(ledger, "tune-2")
    with pytest.raises(ReportError, match="gate1-dev-probe"):
        build_probe_data(ledger, ["tune-1"], ["tune-2"], bootstrap_seed=0)


def test_probe_protocol_versions_must_match_the_tune_runs(tmp_path: Path) -> None:
    """A probe run pinned to other protocol versions refuses (AGENTS.md 2.4)."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "tune-1")
    _write_probe_run(ledger, "probe-1", versions=dict(protocol_versions(), arena="999"))
    with pytest.raises(ReportError, match="protocol versions differ"):
        build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)


def test_a_config_that_pins_an_unnamed_tune_run_refuses(tmp_path: Path) -> None:
    """A probe config naming a tune run outside the reported ones refuses."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "tune-1")
    # the pinned end record is tune-1's real one; the named tune run is not reported
    _write_probe_run(
        ledger,
        "probe-1",
        tune_run_id="tune-other",
        tune_run_end_record_id=read_records(ledger, "tune-1")[-1].record_id,
    )
    with pytest.raises(ReportError, match="not one of the reported tune runs"):
        build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)


def test_a_config_that_pins_another_run_end_refuses(tmp_path: Path) -> None:
    """A probe config whose tune_run_end_record_id is not the tune run's run_end refuses."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "tune-1")
    _write_probe_run(ledger, "probe-1", tune_run_end_record_id="0" * 32)
    with pytest.raises(ReportError, match="tune_run_end_record_id"):
        build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)


def test_a_tune_run_that_did_not_complete_refuses(tmp_path: Path) -> None:
    """An aborted tune run cannot be a probe's reference."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "tune-a", end_status="aborted")
    with pytest.raises(ReportError, match="not completed"):
        build_probe_data(ledger, ["tune-a"], ["probe-any"], bootstrap_seed=0)


def test_a_result_cache_that_disagrees_with_the_probe_config_refuses(tmp_path: Path) -> None:
    """A result replayed from another cache than the probe config pins refuses."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "tune-1")
    _write_probe_run(ledger, "probe-1", cache_id="other-cache")
    with pytest.raises(ReportError, match="cache_id"):
        build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)


def test_a_probe_config_that_disagrees_with_the_tune_config_refuses(tmp_path: Path) -> None:
    """Probe caches that agree with the results but not with the tune run's refuse."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "tune-1")
    _write_probe_run(ledger, "probe-1", cache_id="other-cache", config_cache_id="other-cache")
    with pytest.raises(ReportError, match="simple cache"):
        build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)
    _write_probe_run(ledger, "probe-fm", foundation=["fm-x"], config_foundation=["fm-x"])
    with pytest.raises(ReportError, match="foundation caches"):
        build_probe_data(ledger, ["tune-1"], ["probe-fm"], bootstrap_seed=0)


def test_a_pool_the_tune_run_did_not_cover_refuses(tmp_path: Path) -> None:
    """A probe result over a pool spec the tune run never replayed refuses."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "tune-1")
    _write_probe_run(
        ledger, "probe-1", pool_spec="simple+foundation", config_pool_specs=["simple+foundation"]
    )
    with pytest.raises(ReportError, match="does not cover"):
        build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)


def test_series_ids_that_differ_from_the_tune_oracle_refuse(tmp_path: Path) -> None:
    """An ok probe result whose series differ from the tune run's oracle refuses."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "tune-1")
    _write_probe_run(
        ledger,
        "probe-1",
        learned={
            "ridge_stack": [
                _entry("etth1/HUFL", 0.9),
                _entry("etth1/HULL", 0.8),
                _entry("etth1/WRONG", 0.7),
            ]
        },
    )
    with pytest.raises(ReportError, match="series ids differ"):
        build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)


def test_a_selection_the_probe_records_do_not_reproduce_refuses(tmp_path: Path) -> None:
    """A probe run_end selection that select_configs does not recompute refuses."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "tune-1")
    _write_probe_run(ledger, "probe-1", selected={})
    with pytest.raises(ReportError, match="selected"):
        build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)


def test_two_complete_probe_runs_over_one_pool_refuse(tmp_path: Path) -> None:
    """Two completed probe runs covering the same (dataset, pool spec) refuse, both named."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "tune-1")
    _write_probe_run(ledger, "probe-a")
    _write_probe_run(ledger, "probe-b")
    with pytest.raises(ReportError, match="two complete probe runs"):
        build_probe_data(ledger, ["tune-1"], ["probe-a", "probe-b"], bootstrap_seed=0)


def test_a_record_that_is_not_a_learned_updaters_refuses(tmp_path: Path) -> None:
    """A probe record whose updater id is not a learned id refuses."""
    ledger = tmp_path / "ledgers"
    _write_run(ledger, "tune-1")
    _write_probe_run(ledger, "probe-1", learned={"hedge": None})
    with pytest.raises(ReportError, match="learned ids"):
        build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)


def test_the_report_refuses_a_probe_run_passed_as_a_run(tmp_path: Path) -> None:
    """A probe run named with --run, not --probe-run, refuses."""
    ledger = _hand_ledger(tmp_path)
    with pytest.raises(ReportError, match="probe-run"):
        build_report_data(ledger, ["probe-1"], bootstrap_seed=0)


# --- Test 5: incomplete probe runs are listed, never tabled ---------------------------------------


def test_incomplete_probe_runs_are_listed_and_never_tabled(
    synthetic_env: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An aborted probe run and one without run_end appear as rows and contribute nothing."""
    ledger = tmp_path / "ledgers"
    assert tune_once(synthetic_env, ledger) == "completed"
    monkeypatch.setattr(
        gate1_tune,
        "replay_dataset",
        lambda *_args: (_ for _ in ()).throw(ArenaError("boom")),
    )
    with pytest.raises(ArenaError):
        run_probe(synthetic_env, ledger, run_id="probe-aborted")
    monkeypatch.undo()
    assert run_probe(synthetic_env, ledger, run_id="probe-ok") == "completed"
    _write_run(ledger, "tune-hand")
    _write_probe_run(ledger, "probe-noend", tune_run_id="tune-hand", run_end=False)
    data = build_probe_data(
        ledger,
        ["tune-1", "tune-hand"],
        ["probe-aborted", "probe-ok", "probe-noend"],
        bootstrap_seed=0,
    )
    by_id = {row.run_id: row for row in data.run_rows}
    assert by_id["probe-aborted"].status == "aborted"
    assert by_id["probe-ok"].status == "completed"
    assert by_id["probe-noend"].status == "incomplete"
    assert by_id["probe-noend"].tune_run_id == "tune-hand"
    assert by_id["probe-ok"].records == len(read_records(ledger, "probe-ok"))
    # only the completed probe's pool is tabled, once, from its own tune run
    assert [(section.dataset, section.pool_spec) for section in data.sections] == [
        ("synthetic-dev", "simple")
    ]
    assert data.leakage_sections == ()


# --- Test 6: a pool without an ok baseline --------------------------------------------------------


def test_a_pool_without_an_ok_baseline_has_no_gains(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """A tune pool where every baseline failed leaves the probe nothing to gain against."""
    ledger = tmp_path / "ledgers"
    assert tune_once(synthetic_env, ledger, grids={"hedge": [{"eta": 0.0}]}) == "completed"
    assert run_probe(synthetic_env, ledger) == "completed"
    data = build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)
    section = data.sections[0]
    assert section.best_baseline is None
    assert section.best_baseline_mase is None
    assert section.gains == ()
    assert [row.baseline for row in section.learned] == list(LEARNED_IDS)  # the probe replayed


# --- Test 7: an all-failed learned updater --------------------------------------------------------


def test_an_all_failed_learned_updater_keeps_its_row_and_loses_its_gain(
    synthetic_env: dict[str, Path], tmp_path: Path
) -> None:
    """A learned updater whose grid all failed has a TunedRow with config None and no GainRow."""
    ledger = tmp_path / "ledgers"
    assert tune_once(synthetic_env, ledger) == "completed"
    grids: dict[str, list[dict[str, ConfigValue]]] = {
        "ridge_stack": [
            {"half_life": 0.0, "shrink": 1.0, "refit_every": 24},  # not > 0: init refuses
            {"half_life": -1.0, "shrink": 1.0, "refit_every": 24},
        ],
        "loss_gate": [{"beta": 4.0, "half_life": 336.0, "ridge": 0.01, "refit_every": 24}],
    }
    assert run_probe(synthetic_env, ledger, grids=grids) == "completed"
    data = build_probe_data(ledger, ["tune-1"], ["probe-1"], bootstrap_seed=0)
    section = data.sections[0]
    assert section.best_baseline is not None  # the tune run's baselines are fine
    by_id = {row.baseline: row for row in section.learned}
    assert by_id["ridge_stack"].config is None
    assert by_id["ridge_stack"].mase is None
    assert (by_id["ridge_stack"].tried, by_id["ridge_stack"].failed) == (2, 2)
    assert by_id["ridge_stack"].evaluations == 2.0
    assert by_id["ridge_stack"].wall_seconds > 0.0
    assert by_id["loss_gate"].config is not None
    assert [row.updater for row in section.gains] == ["loss_gate"]


# --- the hand-built probe run helper --------------------------------------------------------------


def _series(entries: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Three ETTh1 series entries with descending MASE, unless a test chooses its own."""
    if entries is not None:
        return entries
    names = ("HUFL", "HULL", "MULL")
    return [_entry(f"etth1/{names[i]}", 0.9 - 0.1 * i) for i in range(3)]


def _write_probe_run(
    ledger_root: Path,
    run_id: str,
    *,
    tune_run_id: str = "tune-1",
    tune_run_end_record_id: str | None = None,
    learned: dict[str, list[dict[str, Any]] | None] | None = None,
    dataset: str = "etth1",
    pool_spec: str = "simple",
    config_pool_specs: list[str] | None = None,
    cache_id: str = "cafe",
    config_cache_id: str | None = None,
    foundation: list[str] | None = None,
    config_foundation: list[str] | None = None,
    versions: dict[str, str] | None = None,
    selected: Any = None,
    run_end: bool = True,
) -> None:
    """Write one small, well-formed probe run with hand-chosen numbers.

    The tune run it pins is ``_write_run``'s (cache ``cafe``, three ETTh1
    series); ``learned`` maps each updater id to its result's series
    entries (``None`` meaning the default three). The ``config_*`` and
    result-side overrides skew one side away from the other — or away from
    the tune run's config, which keeps cache ``cafe`` and pool spec
    ``simple`` — each to force the mismatch its test refuses.
    """
    entries = {"ridge_stack": None} if learned is None else learned
    end_id = tune_run_end_record_id
    if end_id is None:
        end_id = read_records(ledger_root, tune_run_id)[-1].record_id
    config: dict[str, Any] = {
        "experiment": "gate1",
        "stage": "dev-probe",
        "tune_run_id": tune_run_id,
        "tune_run_end_record_id": end_id,
        "datasets": [dataset],
        "pool_specs": config_pool_specs if config_pool_specs is not None else [pool_spec],
        "simple_cache_ids": {dataset: config_cache_id if config_cache_id is not None else "cafe"},
        "foundation_cache_ids": {
            dataset: {pool_spec: config_foundation if config_foundation is not None else []}
        },
        "foundation_models": {dataset: {pool_spec: []}},
        "grids": {learned_id: [{}] for learned_id in entries},
        "selection_metric": "mase",
        "seed": 3,
        "max_wall_seconds": 1000.0,
        "updater_refs": {
            learned_id: f"evolvets.updater.learned:{learned_id}" for learned_id in entries
        },
    }
    writer = LedgerWriter(ledger_root, run_id, "main")
    blobs = BlobStore(ledger_root / run_id / "blobs")
    with writer:
        writer.append(
            "run_start",
            RunStart(
                condition="gate1-dev-probe",
                seed=3,
                requires_predictions=False,
                config_blob=blobs.put(canonical_bytes(config)),
                git_commit=COMMIT,
                protocol_versions=versions if versions is not None else protocol_versions(),
                budget_limits=Budget(wall_seconds=1000.0, evaluations=100),
            ),
        )
        for learned_id, series_entries in entries.items():
            series = _series(series_entries)
            n = len(series)
            total = 0.0
            for entry in series:
                total = total + entry["mase"]
            blob = {
                "updater_ref": f"evolvets.updater.learned:{learned_id}",
                "series_wall_seconds": [0.25] * n,
                "result": {
                    "dataset": dataset,
                    "cache_id": cache_id,
                    "arena_version": protocol_versions()["arena"],
                    "pool_spec": pool_spec,
                    "foundation_cache_ids": foundation if foundation is not None else [],
                    "leakage_risk": False,
                    "status": "ok",
                    "error": None,
                    "failed_series": None,
                    "failed_step": None,
                    "metrics": {"n_series": float(n), "mase": total / n},
                    "series": series,
                },
            }
            writer.append(
                "baseline",
                Baseline(
                    updater_id=learned_id,
                    hyperparams={},
                    pool="dev",
                    status="ok",
                    metrics=dict(blob["result"]["metrics"]),
                    per_task_blob=blobs.put(canonical_bytes(blob)),
                    error=None,
                ),
                budget=Budget(wall_seconds=1.0, evaluations=1),
            )
        if run_end:
            chosen = selected
            if chosen is None:
                records = list(LedgerReader(ledger_root, run_id).records())
                chosen = select_configs(
                    records, blob_reader(ledger_root, run_id), updater_ids=LEARNED_IDS
                )
            writer.append("run_end", RunEnd(status="completed", summary={"selected": chosen}))
