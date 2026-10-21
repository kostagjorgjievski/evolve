"""The Phase 1 report's builder (TASK_019 PR A, split by the size rule).

The reading branch carried tests 1, 2, 9 and 10-16; this file's numbers
branch adds tests 3-8 and 17. The module fixture builds one small pilot
ledger once — five conditions' lineages over the shift3 fixture pool, each
with its holdout run, one replay of the treatment and one aborted lineage,
every evaluation stubbed exactly as ``test_phase1`` stubs them (its ok
outcomes carry the per-series detail the real evaluator reports) — and
every test then reads that ledger through ``build_phase1_report_data``
alone. Refusals run on hand-built copies of a fixture run: the copy helper
preserves record ids, references and blobs, so the one edited thing is the
only thing that differs.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import random
import shutil
import uuid
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from pytest import MonkeyPatch

from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.improver.artifacts import (
    REFERENCE_ARTIFACT_SOURCE,
    WEIGHTS_REFERENCE_ARTIFACT_SOURCE,
)
from evolvets.improver.interfaces import LLMResponse
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.canonical import canonical_bytes, canonical_json
from evolvets.ledger.payloads import (
    Acceptance,
    Baseline,
    Evaluation,
    Payload,
    Prediction,
    RunEnd,
    RunStart,
)
from evolvets.ledger.schema import ZERO_HASH, Budget, Record, make_record
from evolvets.ledger.store import LedgerReader
from evolvets.runner import phase1, phase1_holdout, phase1_report_data
from evolvets.runner.gate1_report_data import GapRow
from evolvets.runner.phase1_eval import EvalOutcome
from evolvets.runner.phase1_report_data import (
    CalibrationRow,
    Phase1ReportError,
    build_phase1_report_data,
)
from tests.arena.test_realdata_prepare import _record_opens
from tests.improver.fake import ScriptedClient
from tests.runner.test_phase1 import (
    WEIGHTS_CHILD,
    StubEvaluator,
    child,
    failed,
    lineage_run,
    lineage_spec,
    reply,
)
from tests.runner.test_phase1_holdout import WEIGHTS_FLAT_CHILD, holdout_run

#: The holdout every lineage scores: shift3's replicate.
HOLDOUT = "synthetic-shift3-holdout"

#: Each pilot condition's interface, straight from the registry.
CONDITIONS_INTERFACE = {condition: spec.interface for condition, spec in phase1.CONDITIONS.items()}

#: Per-series ids the stubbed outcomes report, the arena's own shape.
PILOT_SERIES = ("synthetic-shift3/shift/0001", "synthetic-shift3/shift/0002")
HOLDOUT_SERIES = ("synthetic-shift3-holdout/shift/0001", "synthetic-shift3-holdout/shift/0002")

CHILD_A = child("0.5")
CHILD_B = child("0.7")
CHILD_C = child("0.9")
CHILD_D = child("0.3")


def scored(mase: float, series: Sequence[str] = PILOT_SERIES) -> EvalOutcome:
    """An ok outcome with the pooled MASE and the per-series detail the arena reports."""
    return EvalOutcome(
        status="ok",
        result_obj={
            "metrics": {"mase": mase},
            "series": [{"series_id": series_id, "mase": mase} for series_id in series],
        },
        series_wall_seconds=(0.01, 0.01),
        error=None,
        wall_seconds=0.05,
    )


# --- the module fixture: one pilot ledger, built once ---------------------------------------------


@dataclass
class Pilot:
    """The pilot ledger's paths and run ids, as the tests name them."""

    base: Path  # the fixture root; ``ledgers`` and the cache live under it
    ledger: Path
    lineages: dict[str, str]  # condition -> lineage run id
    holdouts: dict[str, str]  # condition -> holdout run id
    replay: str
    aborted: str


@contextlib.contextmanager
def _patched_lineage(
    responses: Sequence[LLMResponse],
    outcomes: Mapping[str, EvalOutcome],
    default: EvalOutcome,
) -> Any:
    """The engine's seams patched as ``test_phase1.install`` patches them."""
    with ExitStack() as stack:
        stack.enter_context(patch.object(phase1, "current_git_commit", lambda repo: "0" * 40))
        stack.enter_context(
            patch.object(phase1, "evaluate_source", StubEvaluator(dict(outcomes), default))
        )
        client = ScriptedClient(*responses)
        stack.enter_context(patch.object(phase1, "HttpsLLMClient", lambda config: client))
        stack.enter_context(patch.dict(os.environ, {"EVOLVETS_LLM_API_KEY": "test-key"}))
        yield


@contextlib.contextmanager
def _patched_holdout(
    default: EvalOutcome, outcomes: Mapping[str, EvalOutcome] | None = None
) -> Any:
    """The holdout's seams patched: git commit stubbed, scores stubbed per source."""
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(phase1_holdout, "current_git_commit", lambda repo: "0" * 40)
        )
        stack.enter_context(
            patch.object(phase1_holdout, "evaluate_source", StubEvaluator(outcomes or {}, default))
        )
        yield


def _lineage(
    pilot: Pilot,
    run_id: str,
    condition: str,
    *,
    spec: dict[str, Any],
    responses: Sequence[LLMResponse] = (),
    outcomes: Mapping[str, EvalOutcome] | None = None,
) -> int:
    """Run one stubbed lineage on synthetic-shift3 into the pilot ledger."""
    base_spec = lineage_spec(condition=condition, max_evaluations=4, **spec)
    with _patched_lineage(responses, outcomes or {}, scored(1.5)):
        return lineage_run(
            (pilot.base, pilot.base / "cache_store"),
            pilot.base,
            replace(base_spec, dataset="synthetic-shift3"),
            run_id,
        )


def _holdout(pilot: Pilot, condition: str) -> None:
    """Score one pilot lineage on the shift3 holdout, every artifact ok at 1.1."""
    run_id = f"holdout-{condition}"
    with _patched_holdout(scored(1.1, HOLDOUT_SERIES)):
        assert (
            holdout_run(
                (pilot.base, pilot.base / "cache_store"),
                pilot.base,
                from_run=pilot.lineages[condition],
                run_id=run_id,
            )
            == 0
        )
    pilot.holdouts[condition] = run_id


@pytest.fixture(scope="module")
def pilot(tmp_path_factory: pytest.TempPathFactory) -> Pilot:
    """Five conditions' lineages with holdouts, one replay and one aborted run.

    Every lineage runs at ``max_evaluations=4`` and the default spec limits,
    so the pilot's budgets are equal by construction; ``fixed`` spends its
    evaluation slots one per generation (more proposals) where
    ``evaluate_all`` spends them as they come (fewer proposals), both without
    an acceptance, while ``predictive`` accepts two climbs.
    """
    base = tmp_path_factory.mktemp("phase1-report")
    cache_root = base / "cache_store"
    for dataset in ("synthetic-shift3", HOLDOUT):
        source, series = public_inputs(dataset, base, base, per_family=1)
        build_cache(dataset, source, series, cache_root)
    made = Pilot(
        base=base,
        ledger=base / "ledgers",
        lineages={},
        holdouts={},
        replay="replay-predictive",
        aborted="lineage-aborted",
    )
    # the treatment: two accepted climbs (1.0 -> 0.8 -> 0.7), then a rejected repeat
    assert (
        _lineage(
            made,
            "lineage-predictive",
            "predictive",
            spec={"proposals_per_generation": 1},
            responses=[reply(CHILD_A), reply(CHILD_B), reply(CHILD_B)],
            outcomes={
                REFERENCE_ARTIFACT_SOURCE: scored(1.0),
                CHILD_A: scored(0.8),
                CHILD_B: scored(0.7),
            },
        )
        == 0
    )
    made.lineages["predictive"] = "lineage-predictive"
    _holdout(made, "predictive")
    # the replay of the treatment: recorded responses, zero tokens
    assert (
        _lineage(
            made,
            made.replay,
            "predictive",
            spec={"proposals_per_generation": 1, "replay_of": "lineage-predictive"},
            outcomes={
                REFERENCE_ARTIFACT_SOURCE: scored(1.0),
                CHILD_A: scored(0.8),
                CHILD_B: scored(0.7),
            },
        )
        == 0
    )
    # evaluate_all: every candidate evaluated until the budget closes, none accepted
    assert (
        _lineage(
            made,
            "lineage-evaluate-all",
            "evaluate_all",
            spec={"proposals_per_generation": 2, "eval_slots": 2},
            responses=[reply(CHILD_A), reply(CHILD_B), reply(CHILD_C), reply(CHILD_D)],
            outcomes={
                REFERENCE_ARTIFACT_SOURCE: scored(1.0),
                CHILD_A: scored(1.5),
                CHILD_B: scored(1.4),
                CHILD_C: scored(1.5),
                CHILD_D: scored(1.4),
            },
        )
        == 0
    )
    made.lineages["evaluate_all"] = "lineage-evaluate-all"
    _holdout(made, "evaluate_all")
    # fixed: one slot per generation, so more generations (and proposals) for
    # the same evaluation budget than evaluate_all, and no acceptance either
    assert (
        _lineage(
            made,
            "lineage-fixed",
            "fixed",
            spec={"proposals_per_generation": 2, "eval_slots": 1},
            responses=[reply(CHILD_A), reply(CHILD_B)] * 3,
            outcomes={
                REFERENCE_ARTIFACT_SOURCE: scored(1.0),
                CHILD_A: scored(1.5),
                CHILD_B: scored(1.4),
            },
        )
        == 0
    )
    made.lineages["fixed"] = "lineage-fixed"
    _holdout(made, "fixed")
    # the non-LLM track: random mutations, no tokens, no acceptance
    assert (
        _lineage(
            made,
            "lineage-random-param",
            "random_param",
            spec={"proposals_per_generation": 1},
            outcomes={REFERENCE_ARTIFACT_SOURCE: scored(1.0)},
        )
        == 0
    )
    made.lineages["random_param"] = "lineage-random-param"
    _holdout(made, "random_param")
    # the weights-space ablation: one accepted rows edit under weights-v1,
    # from the weights reference's own starting MASE (not the forecast
    # reference's 1.0 — its gain is measured against its own condition's start)
    assert (
        _lineage(
            made,
            "lineage-weights",
            "weights_only",
            spec={"proposals_per_generation": 1},
            responses=[reply(WEIGHTS_CHILD)] * 3,
            outcomes={
                WEIGHTS_REFERENCE_ARTIFACT_SOURCE: scored(1.2),
                WEIGHTS_CHILD: scored(0.8),
            },
        )
        == 0
    )
    made.lineages["weights_only"] = "lineage-weights"
    _holdout(made, "weights_only")
    # an aborted lineage: the baseline itself failed, so the run aborted. It
    # runs a condition no tabled run claims — two named lineages of one
    # condition are refused, tabled or not.
    assert (
        _lineage(
            made,
            made.aborted,
            "random",
            spec={"proposals_per_generation": 1},
            outcomes={REFERENCE_ARTIFACT_SOURCE: failed("worker: crash at import")},
        )
        == 2
    )
    return made


def _ids(pilot: Pilot, *, drop: Sequence[str] = (), add: Sequence[str] = ()) -> list[str]:
    """The pilot's named runs, minus dropped lineages' holdouts, plus additions.

    Dropping the replay's source lineage drops the replay too: a replay whose
    source is not named is itself refused, and no refusal test here wants that.
    """
    dropped = set(drop)
    if pilot.lineages.get("predictive") in dropped:
        dropped.add(pilot.replay)
    ids: list[str] = []
    for condition, run_id in pilot.lineages.items():
        if run_id in dropped:
            continue
        ids.append(run_id)
        if pilot.holdouts.get(condition, "") not in dropped:
            ids.append(pilot.holdouts[condition])
    if pilot.replay not in dropped:
        ids.append(pilot.replay)
    return ids + list(add)


def _report(pilot: Pilot, run_ids: Sequence[str]) -> Any:
    """Build the report data over the pilot's ledger."""
    return build_phase1_report_data(pilot.ledger, run_ids, bootstrap_seed=0)


def _records(pilot: Pilot, run_id: str) -> list[Record]:
    """One pilot run's records, chain-verified while reading."""
    return list(LedgerReader(pilot.ledger, run_id).records())


def _counts(pilot: Pilot, run_id: str) -> dict[str, int]:
    """One run's record-kind counts, plus its accepted count, from the records."""
    counts = {"proposal": 0, "prediction": 0, "allocation": 0, "evaluation": 0, "accepted": 0}
    for record in _records(pilot, run_id):
        if record.kind in counts and record.kind != "accepted":
            counts[record.kind] += 1
        elif record.kind == "acceptance" and record.payload.accepted:
            counts["accepted"] += 1
    return counts


def _spent(pilot: Pilot, run_id: str, field: str) -> float:
    """One run's left-to-right loop sum of a budget field."""
    total = 0.0
    for record in _records(pilot, run_id):
        total = total + getattr(record.budget, field)
    return total


# --- the hand-built copy helper -------------------------------------------------------------------


def copy_run(
    ledger_root: Path,
    source_id: str,
    new_id: str,
    *,
    config_replace: Mapping[str, object] | None = None,
    start_replace: Mapping[str, object] | None = None,
    summary_replace: Mapping[str, object] | None = None,
    summary_drop: Sequence[str] = (),
    keep_run_end: bool = True,
    drop_kinds: Sequence[str] = (),
    repeat_baseline: bool = False,
    budget_of: Callable[[Record], Budget] | None = None,
    mutate: Callable[[Record, Payload, list[str], BlobStore], tuple[Payload, list[str]]]
    | None = None,
) -> None:
    """Copy one run under a new id, preserving ids, references and blobs.

    Records are rebuilt with ``make_record`` over the same record ids and
    timestamps, so every ``refers_to`` and every blob linkage still holds and
    the copy verifies; exactly the named edit differs. ``mutate`` sees every
    record's payload and references with the copy's blob store, for edits
    that need new blobs; ``budget_of`` maps every record's budget, for edits
    of the charged compute.
    """
    run_dir = Path(ledger_root) / new_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir()
    shutil.copytree(Path(ledger_root) / source_id / "blobs", run_dir / "blobs")
    source_blobs = BlobStore(Path(ledger_root) / source_id / "blobs")
    blobs = BlobStore(run_dir / "blobs")
    records = list(LedgerReader(ledger_root, source_id).records())
    lines: dict[str, list[str]] = {"main": [], "sealed": []}
    state: dict[str, tuple[int, str]] = {stream: (0, ZERO_HASH) for stream in lines}
    for record in records:
        if record.kind in drop_kinds or (record.kind == "run_end" and not keep_run_end):
            continue
        payload: Payload = record.payload
        if record.kind == "run_start" and (start_replace or config_replace):
            assert isinstance(payload, RunStart)
            config = json.loads(source_blobs.get(payload.config_blob).decode("utf-8"))
            if config_replace:
                config.update(config_replace)
            payload = replace(
                payload,
                config_blob=blobs.put(canonical_bytes(config)),
                **(start_replace or {}),
            )
        if record.kind == "run_end" and (summary_replace or summary_drop):
            assert isinstance(payload, RunEnd)
            summary = {**payload.summary, **(summary_replace or {})}
            for key in summary_drop:
                summary.pop(key, None)
            payload = replace(payload, summary=summary)
        refers_to = list(record.refers_to)
        if mutate is not None:
            payload, refers_to = mutate(record, payload, refers_to, blobs)
        seq, prev_hash = state[record.stream]
        copied = make_record(
            stream=record.stream,
            seq=seq,
            prev_hash=prev_hash,
            run_id=new_id,
            kind=record.kind,
            payload=payload,
            refers_to=refers_to,
            budget=record.budget if budget_of is None else budget_of(record),
            created_at=record.created_at,
            record_id=record.record_id,
        )
        lines[record.stream].append(canonical_json(copied.to_dict()) + "\n")
        state[record.stream] = (seq + 1, copied.record_hash)
        if record.kind == "baseline" and repeat_baseline:
            seq, prev_hash = state[record.stream]
            duplicate = make_record(
                stream=record.stream,
                seq=seq,
                prev_hash=prev_hash,
                run_id=new_id,
                kind=record.kind,
                payload=payload,
                refers_to=refers_to,
                budget=record.budget if budget_of is None else budget_of(record),
                created_at=record.created_at,  # a fresh record_id: a second baseline
            )
            lines[record.stream].append(canonical_json(duplicate.to_dict()) + "\n")
            state[record.stream] = (seq + 1, duplicate.record_hash)
    for stream, stream_lines in lines.items():
        if stream_lines:
            (run_dir / f"{stream}.jsonl").write_text("".join(stream_lines), encoding="utf-8")


# --- Test 1: rows and order -----------------------------------------------------------------------


def test_rows_follow_registry_order_and_recount_from_records(pilot: Pilot) -> None:
    """Rows sit in CONDITIONS order, every number recounted from the records.

    Every condition ran under the same limits, so all rows end
    ``budget_exhausted`` with the same ``evaluations_charged`` (the budget's
    evaluations, one more than the evaluation count, baseline included),
    while ``fixed`` proposed more than ``evaluate_all`` for the same budget —
    the trade the pilot exists to measure. Row order does not follow the
    named ids' order.
    """
    data = _report(pilot, _ids(pilot))
    assert [row.condition for row in data.rows] == [
        "predictive",
        "evaluate_all",
        "fixed",
        "random_param",
        "weights_only",
    ]
    by_condition = {row.condition: row for row in data.rows}
    for condition, row in by_condition.items():
        counts = _counts(pilot, row.run_id)
        assert (row.proposals, row.predictions, row.allocations) == (
            counts["proposal"],
            counts["prediction"],
            counts["allocation"],
        )
        assert row.evaluations == counts["evaluation"]
        assert row.accepted == counts["accepted"]
        assert row.status == "budget_exhausted"
        assert row.evaluations_charged == counts["evaluation"] + 1
        assert row.interface == CONDITIONS_INTERFACE[condition]
        records = _records(pilot, row.run_id)
        assert row.starting_mase == _baseline_mase(records)
        assert row.best_mase == _accepted_best(records)
        assert row.gain == pytest.approx(row.starting_mase - row.best_mase)
        assert row.input_tokens == _token_sum(records, "llm_input_tokens")
        assert row.output_tokens == _token_sum(records, "llm_output_tokens")
        assert row.wall_seconds == _spent(pilot, row.run_id, "wall_seconds")
        assert row.gain_per_evaluation == pytest.approx(row.gain / row.evaluations_charged)
        assert row.gain_per_wall_hour == pytest.approx(row.gain * 3600.0 / row.wall_seconds)
    assert by_condition["predictive"].accepted == 2
    assert by_condition["predictive"].best_mase == pytest.approx(0.7)
    assert by_condition["weights_only"].accepted == 1
    assert by_condition["weights_only"].best_mase == pytest.approx(0.8)
    assert by_condition["weights_only"].interface == "weights-v1"
    fixed, evaluate_all = by_condition["fixed"], by_condition["evaluate_all"]
    assert (fixed.proposals, evaluate_all.proposals) == (6, 4)  # more rounds, same budget
    assert fixed.input_tokens > evaluate_all.input_tokens
    assert fixed.evaluations_charged == evaluate_all.evaluations_charged == 4
    random_param = by_condition["random_param"]
    assert (random_param.input_tokens, random_param.output_tokens) == (0, 0)
    assert random_param.gain == 0.0
    assert random_param.gain_per_million_tokens is None  # no tokens, no per-token gain
    assert by_condition["predictive"].gain_per_million_tokens == pytest.approx(
        by_condition["predictive"].gain
        * 1_000_000.0
        / (by_condition["predictive"].input_tokens + by_condition["predictive"].output_tokens)
    )
    # the one pilot around the table, and the flag order does not matter
    assert (data.dataset, data.pool_spec, data.foundation_models, data.seed) == (
        "synthetic-shift3",
        "simple",
        None,
        0,
    )
    assert data.git_commits == ("0" * 40,)
    assert data.untabled == ()
    assert data.missing_holdout == ()
    assert data.distinct_forecast_start_mase == (1.0,)  # every forecast-v2 row starts 1.0
    assert _report(pilot, list(reversed(_ids(pilot)))).rows == data.rows


def _baseline_mase(records: Sequence[Record]) -> float:
    """The single baseline record's pooled MASE."""
    payload = records[1].payload
    assert isinstance(payload, Baseline)
    return float(payload.metrics["mase"])


def _accepted_best(records: Sequence[Record]) -> float:
    """The last accepted edit's pooled MASE, the run's best."""
    by_id = {record.record_id: record for record in records}
    best = _baseline_mase(records)
    for record in records:
        if record.kind != "acceptance" or not record.payload.accepted:
            continue
        evaluation = by_id[record.refers_to[1]]
        best = float(evaluation.payload.metrics["mase"])
    return best


def _token_sum(records: Sequence[Record], field: str) -> int:
    """The records' left-to-right int sum of one token field."""
    total = 0
    for record in records:
        total += int(getattr(record.budget, field))
    return total


# --- Test 2: the summary cross-check --------------------------------------------------------------


def test_summary_mismatch_refuses_naming_run_and_key(pilot: Pilot) -> None:
    """A run_end summary the records do not reproduce is refused, run and key named."""
    end = _records(pilot, "lineage-predictive")[-1].payload
    assert isinstance(end, RunEnd)
    spent = end.summary["budget_spent"]
    assert isinstance(spent, dict)
    base_ids = _ids(pilot, drop=["lineage-predictive"])
    copy_run(
        pilot.ledger, "lineage-predictive", "cross-best", summary_replace={"best_mase": 0.1234}
    )
    with pytest.raises(Phase1ReportError, match=r"cross-best.*summary's best_mase"):
        _report(pilot, [*base_ids, "cross-best"])
    copy_run(
        pilot.ledger,
        "lineage-predictive",
        "cross-tokens",
        summary_replace={
            "budget_spent": {**spent, "llm_input_tokens": spent["llm_input_tokens"] + 1}
        },
    )
    with pytest.raises(Phase1ReportError, match=r"cross-tokens.*budget_spent\.llm_input_tokens"):
        _report(pilot, [*base_ids, "cross-tokens"])
    copy_run(pilot.ledger, "lineage-predictive", "cross-accepted", summary_replace={"accepted": 5})
    with pytest.raises(Phase1ReportError, match=r"cross-accepted.*summary's accepted"):
        _report(pilot, [*base_ids, "cross-accepted"])
    # an unedited copy of the same run cross-checks clean, so the check is not vacuous
    copy_run(pilot.ledger, "lineage-predictive", "cross-clean")
    data = _report(pilot, [*base_ids, "cross-clean"])
    assert data.rows[0].run_id == "cross-clean"
    assert data.missing_holdout == ("predictive",)  # its holdout was not copied along


# --- Test 9: the replay audit ---------------------------------------------------------------------


def test_replay_audit_row_and_zero_token_reproduction(pilot: Pilot) -> None:
    """The replay reports as an audit row, not a table row, at zero token cost."""
    data = _report(pilot, _ids(pilot))
    assert data.replay is not None
    audit = data.replay
    assert audit.run_id == pilot.replay
    assert audit.source_run_id == "lineage-predictive"
    assert audit.matches_source_run is True
    assert audit.timeout_status_mismatches == 0
    assert (audit.input_tokens, audit.output_tokens) == (0, 0)
    assert audit.wall_seconds == _spent(pilot, pilot.replay, "wall_seconds")
    counts = _counts(pilot, pilot.replay)
    assert audit.evaluations_charged == counts["evaluation"] + 1 == 4
    assert pilot.replay not in [row.run_id for row in data.rows]  # an audit, not a condition


# --- Test 10: untabled runs -----------------------------------------------------------------------


def test_aborted_and_unended_runs_listed_but_never_tabled(pilot: Pilot) -> None:
    """An aborted run and one without its run_end are listed with reasons, feed no row."""
    copy_run(pilot.ledger, "lineage-predictive", "no-end", keep_run_end=False)
    data = _report(pilot, [*_ids(pilot, drop=["lineage-predictive"]), pilot.aborted, "no-end"])
    assert [row.condition for row in data.rows] == [
        "evaluate_all",
        "fixed",
        "random_param",
        "weights_only",
    ]
    untabled = dict(data.untabled)
    assert set(untabled) == {"lineage-aborted", "no-end"}
    assert "baseline" in untabled["lineage-aborted"]  # the run's own error, surfaced
    assert untabled["no-end"] == "no run_end"
    with pytest.raises(Phase1ReportError, match="no tabled lineage"):
        _report(pilot, [pilot.aborted])
    with pytest.raises(Phase1ReportError, match="no tabled lineage"):
        _report(pilot, ["no-end"])


def test_a_condition_without_its_holdout_is_named(pilot: Pilot) -> None:
    """A tabled condition with no named holdout run lands in missing_holdout."""
    data = _report(pilot, _ids(pilot, drop=["holdout-fixed"]))
    assert data.missing_holdout == ("fixed",)
    assert [row.condition for row in data.rows] == [
        "predictive",
        "evaluate_all",
        "fixed",
        "random_param",
        "weights_only",
    ]


# --- Test 11: equal budgets -----------------------------------------------------------------------


def test_unequal_limits_refuse_the_replays_inflated_wall_does_not(pilot: Pilot) -> None:
    """Lineages under different limits refuse; the replay's own wall margin never does."""
    start = _records(pilot, "lineage-predictive")[0].payload
    assert isinstance(start, RunStart)
    copy_run(
        pilot.ledger,
        "lineage-predictive",
        "limits-predictive",
        start_replace={"budget_limits": replace(start.budget_limits, evaluations=5.0)},
    )
    with pytest.raises(Phase1ReportError, match=r"budget_limits\.evaluations"):
        _report(pilot, [*_ids(pilot, drop=["lineage-predictive"]), "limits-predictive"])
    # the pilot's own replay carries a larger wall limit than its source; not refused
    data = _report(pilot, _ids(pilot))
    assert data.replay is not None


# --- Test 12: one pilot, one table ----------------------------------------------------------------


def test_lineages_of_two_tables_refuse(pilot: Pilot) -> None:
    """A second dataset or pool spec among the tabled lineages refuses the report."""
    base_ids = _ids(pilot, drop=["lineage-predictive"])
    copy_run(
        pilot.ledger,
        "lineage-predictive",
        "one-dataset",
        config_replace={"dataset": "synthetic-shift2"},
    )
    with pytest.raises(Phase1ReportError, match="ran datasets"):
        _report(pilot, [*base_ids, "one-dataset"])
    copy_run(
        pilot.ledger,
        "lineage-predictive",
        "one-pool",
        config_replace={"pool_spec": "simple+foundation"},
    )
    with pytest.raises(Phase1ReportError, match="ran pool specs"):
        _report(pilot, [*base_ids, "one-pool"])


# --- Test 13: pairing refusals --------------------------------------------------------------------


def test_impossible_pairings_refuse(pilot: Pilot) -> None:
    """Duplicate ids or conditions, replay and holdout pairings, each refuse."""
    with pytest.raises(Phase1ReportError, match="duplicate run id"):
        _report(pilot, ["lineage-predictive", "lineage-predictive"])
    copy_run(pilot.ledger, "lineage-evaluate-all", "dupe-evaluate-all")
    with pytest.raises(Phase1ReportError, match="both ran condition"):
        _report(pilot, [*_ids(pilot), "dupe-evaluate-all"])
    copy_run(pilot.ledger, pilot.replay, "replay-copy")
    with pytest.raises(Phase1ReportError, match="two replay runs named"):
        _report(pilot, [*_ids(pilot), "replay-copy"])
    with pytest.raises(Phase1ReportError, match="names replay_of"):
        _report(pilot, [pilot.replay, "lineage-evaluate-all"])
    copy_run(
        pilot.ledger,
        pilot.replay,
        "replay-of-replay",
        config_replace={"replay_of": pilot.replay},
    )
    with pytest.raises(Phase1ReportError, match="two replay runs named"):
        _report(pilot, ["lineage-predictive", pilot.replay, "replay-of-replay"])
    copy_run(pilot.ledger, "holdout-predictive", "holdout-dupe")
    with pytest.raises(Phase1ReportError, match="two holdout runs named"):
        _report(pilot, ["lineage-predictive", "holdout-predictive", "holdout-dupe"])
    with pytest.raises(Phase1ReportError, match="names from_run"):
        _report(pilot, ["holdout-predictive", "lineage-evaluate-all"])
    copy_run(
        pilot.ledger,
        "holdout-predictive",
        "holdout-of-replay",
        config_replace={"from_run": pilot.replay},
    )
    with pytest.raises(Phase1ReportError, match="scores the replay"):
        _report(pilot, ["lineage-predictive", pilot.replay, "holdout-of-replay"])


def test_holdout_and_replay_mismatches_refuse(pilot: Pilot) -> None:
    """A holdout or replay of another pool or foundation set refuses."""
    copy_run(
        pilot.ledger,
        "holdout-predictive",
        "holdout-pool",
        config_replace={"pool_spec": "simple+foundation"},
    )
    with pytest.raises(Phase1ReportError, match="ran pool_spec"):
        _report(pilot, ["lineage-predictive", "holdout-pool"])
    copy_run(
        pilot.ledger,
        "holdout-predictive",
        "holdout-models",
        config_replace={"foundation_models": ["fake-model"]},
    )
    with pytest.raises(Phase1ReportError, match="disagree on foundation_models"):
        _report(pilot, ["lineage-predictive", "holdout-models"])
    copy_run(
        pilot.ledger,
        pilot.replay,
        "replay-models",
        config_replace={"foundation_models": ["fake-model"]},
    )
    with pytest.raises(Phase1ReportError, match="disagree on foundation_models"):
        _report(pilot, ["lineage-predictive", "replay-models"])
    copy_run(
        pilot.ledger,
        pilot.replay,
        "replay-condition",
        start_replace={"condition": "phase1-fixed"},
    )
    with pytest.raises(Phase1ReportError, match="runs condition"):
        _report(pilot, ["lineage-predictive", "replay-condition"])


# --- Test 14: run shape refusals ------------------------------------------------------------------


def test_a_run_id_with_no_ledger_refuses(pilot: Pilot) -> None:
    """A named run id that never ran is refused, never skipped."""
    with pytest.raises(Phase1ReportError, match="missing ledger stream file"):
        _report(pilot, [*_ids(pilot), "never-ran"])


def test_misshapen_runs_refuse(pilot: Pilot) -> None:
    """Non-Phase-1 or wrongly-declared runs, and broken record shapes, refuse."""
    copy_run(
        pilot.ledger,
        "lineage-evaluate-all",
        "not-phase1",
        start_replace={"condition": "gate1-dev-tune"},
    )
    with pytest.raises(Phase1ReportError, match="is not a Phase 1 run"):
        _report(pilot, ["not-phase1"])
    copy_run(
        pilot.ledger,
        "lineage-evaluate-all",
        "unknown-condition",
        start_replace={"condition": "phase1-nope"},
    )
    with pytest.raises(Phase1ReportError, match="unknown Phase 1 condition"):
        _report(pilot, ["unknown-condition"])
    copy_run(
        pilot.ledger,
        "lineage-evaluate-all",
        "no-predictions-declared",
        start_replace={"requires_predictions": False},
    )
    with pytest.raises(Phase1ReportError, match="must declare requires_predictions=true"):
        _report(pilot, ["no-predictions-declared"])
    copy_run(
        pilot.ledger,
        "holdout-predictive",
        "holdout-wants-predictions",
        start_replace={"requires_predictions": True},
    )
    with pytest.raises(Phase1ReportError, match="must declare requires_predictions=false"):
        _report(pilot, ["holdout-wants-predictions"])


def test_broken_lineage_shapes_refuse(pilot: Pilot) -> None:
    """A missing or doubled baseline, and a prediction pointing outside, refuse."""
    base_ids = _ids(pilot, drop=["lineage-evaluate-all"])
    copy_run(pilot.ledger, "lineage-evaluate-all", "no-baseline", drop_kinds=("baseline",))
    with pytest.raises(Phase1ReportError, match="found 0"):
        _report(pilot, [*base_ids, "no-baseline"])
    copy_run(pilot.ledger, "lineage-evaluate-all", "two-baselines", repeat_baseline=True)
    with pytest.raises(Phase1ReportError, match="found 2"):
        _report(pilot, [*base_ids, "two-baselines"])
    broken_first: list[bool] = []

    def break_prediction(
        record: Record, payload: Payload, refers_to: list[str], blobs: BlobStore
    ) -> tuple[Payload, list[str]]:
        if record.kind == "prediction" and not broken_first:
            broken_first.append(True)
            return payload, [str(uuid.uuid4())]
        return payload, refers_to

    copy_run(pilot.ledger, "lineage-evaluate-all", "orphan-prediction", mutate=break_prediction)
    with pytest.raises(Phase1ReportError, match="must refer to a proposal record"):
        _report(pilot, [*base_ids, "orphan-prediction"])


def test_broken_holdout_shapes_refuse(pilot: Pilot) -> None:
    """A holdout row that skips and a linkage blob at the wrong proposal refuse."""

    def skip_a_row(
        record: Record, payload: Payload, refers_to: list[str], blobs: BlobStore
    ) -> tuple[Payload, list[str]]:
        if (
            record.kind == "baseline"
            and isinstance(payload, Baseline)
            and payload.updater_id == "accept-01"
        ):
            return replace(payload, updater_id="accept-1"), refers_to
        return payload, refers_to

    copy_run(pilot.ledger, "holdout-predictive", "holdout-skip", mutate=skip_a_row)
    with pytest.raises(Phase1ReportError, match="the holdout rows are"):
        _report(pilot, ["lineage-predictive", "holdout-skip"])
    accepted = [
        record for record in _records(pilot, "lineage-predictive") if record.kind == "proposal"
    ]

    def relink(
        record: Record, payload: Payload, refers_to: list[str], blobs: BlobStore
    ) -> tuple[Payload, list[str]]:
        if (
            record.kind == "baseline"
            and isinstance(payload, Baseline)
            and payload.updater_id == "accept-02"
        ):
            assert payload.per_task_blob is not None
            blob = json.loads(blobs.get(payload.per_task_blob).decode("utf-8"))
            blob["proposal_record_id"] = accepted[0].record_id
            return replace(payload, per_task_blob=blobs.put(canonical_bytes(blob))), refers_to
        return payload, refers_to

    copy_run(pilot.ledger, "holdout-predictive", "holdout-relink", mutate=relink)
    with pytest.raises(Phase1ReportError, match="links proposal"):
        _report(pilot, ["lineage-predictive", "holdout-relink"])


# --- Test 15: protocol versions -------------------------------------------------------------------


def test_protocol_version_mixes_refuse(pilot: Pilot) -> None:
    """A run under different protocol versions never shares one table."""
    start = _records(pilot, "lineage-evaluate-all")[0].payload
    assert isinstance(start, RunStart)
    copy_run(
        pilot.ledger,
        "lineage-evaluate-all",
        "protocol-mix",
        start_replace={"protocol_versions": {**start.protocol_versions, "ledger": "9.9"}},
    )
    with pytest.raises(Phase1ReportError, match="protocol versions differ"):
        _report(pilot, [*_ids(pilot, drop=["lineage-evaluate-all"]), "protocol-mix"])


# --- Test 16: ledgers only, and no engine imports -------------------------------------------------


def test_the_builder_reads_ledger_files_only(pilot: Pilot, monkeypatch: MonkeyPatch) -> None:
    """Every file the builder opens is inside the pilot's ledger root."""
    opened: list[str] = []
    _record_opens(monkeypatch, opened)
    before = len(opened)
    data = _report(pilot, _ids(pilot))
    touched = opened[before:]
    assert touched  # it read the ledgers for real
    root = pilot.ledger.resolve()
    for path in touched:
        # the ledger root only: not the cache, the data or anything else nearby
        assert Path(path).resolve().is_relative_to(root), path
    assert data.rows


def test_the_module_names_no_engine_package() -> None:
    """The report module imports no arena, improver, selfmodel, allocator or updater."""
    source = Path(phase1_report_data.__file__).read_text(encoding="utf-8")
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    forbidden = (
        "evolvets.arena",
        "evolvets.improver",
        "evolvets.selfmodel",
        "evolvets.allocator",
        "evolvets.updater",
    )
    assert not [
        name
        for name in names
        if name in forbidden or any(name.startswith(f + ".") for f in forbidden)
    ]


# --- Tests 3-8 and 17: the numbers (feat/phase1-report-numbers) -----------------------------------


def scored_series(pooled: float, per_series: Sequence[float], series: Sequence[str]) -> EvalOutcome:
    """An ok outcome with a pooled MASE and distinct per-series MASE values."""
    return EvalOutcome(
        status="ok",
        result_obj={
            "metrics": {"mase": pooled},
            "series": [
                {"series_id": series_id, "mase": mase}
                for series_id, mase in zip(series, per_series, strict=True)
            ],
        },
        series_wall_seconds=(0.01,) * len(series),
        error=None,
        wall_seconds=0.05,
    )


def _bounds(differences: Sequence[float], seed: int) -> tuple[float, float]:
    """One condition's interval, recomputed independently of the builder.

    A fresh ``random.Random(seed)`` per condition is exactly what the
    builder's one-row ``_gap_rows`` table draws, so the bounds must agree
    draw for draw.
    """
    rng = random.Random(seed)
    means: list[float] = []
    for _resample in range(2000):
        total = 0.0
        for _draw in range(len(differences)):
            total = total + differences[rng.randrange(len(differences))]
        means.append(total / len(differences))
    means.sort()
    return means[50], means[1949]


def _loop_mean(values: Sequence[float]) -> float:
    """The left-to-right loop mean, recomputed independently."""
    total = 0.0
    for value in values:
        total = total + value
    return total / len(values)


# --- Test 3: the per-compute columns --------------------------------------------------------------


def test_per_compute_columns_divide_exactly_as_spelled(pilot: Pilot) -> None:
    """Every per-compute column equals its spelled division over the record sums."""
    data = _report(pilot, _ids(pilot))
    for row in data.rows:
        assert row.gain == row.starting_mase - row.best_mase
        assert row.gain_per_evaluation == row.gain / row.evaluations_charged
        tokens = row.input_tokens + row.output_tokens
        if tokens == 0:
            assert row.gain_per_million_tokens is None
        else:
            assert row.gain_per_million_tokens == row.gain * 1_000_000.0 / tokens
        assert row.gain_per_wall_hour == row.gain * 3600.0 / row.wall_seconds
    by_condition = {row.condition: row for row in data.rows}
    assert by_condition["random_param"].gain_per_million_tokens is None  # no LLM tokens
    for condition in ("predictive", "evaluate_all", "fixed", "weights_only"):
        assert by_condition[condition].gain_per_million_tokens is not None


def test_zero_wall_time_refuses_rather_than_dividing(pilot: Pilot) -> None:
    """A run whose records sum zero wall seconds is refused, not divided by.

    The hardening #178's review asked for: a degenerate hand-built ledger
    must meet a named refusal, never a ``ZeroDivisionError``.
    """
    end = _records(pilot, "lineage-fixed")[-1].payload
    assert isinstance(end, RunEnd)
    spent = end.summary["budget_spent"]
    assert isinstance(spent, dict)
    copy_run(
        pilot.ledger,
        "lineage-fixed",
        "zero-wall",
        budget_of=lambda record: replace(record.budget, wall_seconds=0.0),
        summary_replace={"budget_spent": {**spent, "wall_seconds": 0.0}},
    )
    with pytest.raises(Phase1ReportError, match=r"zero-wall.*zero wall seconds"):
        _report(pilot, [*_ids(pilot, drop=["lineage-fixed"]), "zero-wall"])


# --- Test 4: the series bootstrap -----------------------------------------------------------------


def test_series_bootstrap_matches_an_independent_recomputation(pilot: Pilot) -> None:
    """The interval equals a fresh-seeded independent draw over the per-series gains."""
    series = (
        "synthetic-shift3/shift/0001",
        "synthetic-shift3/shift/0002",
        "synthetic-shift3/shift/0003",
    )
    assert (
        _lineage(
            pilot,
            "lineage-series",
            "predictive",
            spec={"proposals_per_generation": 1},
            responses=[reply(CHILD_A), reply(CHILD_B), reply(CHILD_B)],
            outcomes={
                REFERENCE_ARTIFACT_SOURCE: scored_series(1.0, (0.8, 1.2, 1.1), series),
                CHILD_A: scored_series(0.9, (0.5, 1.1, 0.9), series),
            },
        )
        == 0
    )
    row = _report(pilot, ["lineage-series"]).rows[0]
    differences = [0.8 - 0.5, 1.2 - 1.1, 1.1 - 0.9]
    assert row.gain_interval is not None
    assert row.gain_interval.label == "predictive"
    assert row.gain_interval.mean == _loop_mean(differences)
    assert (row.gain_interval.lower, row.gain_interval.upper) == _bounds(differences, 0)


def test_no_acceptance_bootstraps_zero_and_one_series_gives_none(pilot: Pilot) -> None:
    """A lineage that accepted nothing has all-zero differences; one series has no interval."""
    data = _report(pilot, _ids(pilot))
    by_condition = {row.condition: row for row in data.rows}
    for condition in ("evaluate_all", "fixed", "random_param"):
        assert by_condition[condition].gain_interval == GapRow(
            label=condition, mean=0.0, lower=0.0, upper=0.0
        )
    # the accepting conditions: the baseline paired with the final accepted edit
    assert by_condition["predictive"].gain_interval == GapRow(
        label="predictive", mean=1.0 - 0.7, lower=1.0 - 0.7, upper=1.0 - 0.7
    )
    assert by_condition["weights_only"].gain_interval == GapRow(
        label="weights_only", mean=1.2 - 0.8, lower=1.2 - 0.8, upper=1.2 - 0.8
    )
    solo = (PILOT_SERIES[0],)
    assert (
        _lineage(
            pilot,
            "lineage-one-series",
            "predictive",
            spec={"proposals_per_generation": 1},
            responses=[reply(CHILD_B)] * 3,
            outcomes={
                REFERENCE_ARTIFACT_SOURCE: scored(1.0, solo),
                CHILD_B: scored(0.7, solo),
            },
        )
        == 0
    )
    assert _report(pilot, ["lineage-one-series"]).rows[0].gain_interval is None


def test_mismatched_series_ids_between_baseline_and_final_refuse(pilot: Pilot) -> None:
    """A final evaluation whose series ids are not the baseline's refuses the report."""
    base_ids = _ids(pilot, drop=["lineage-predictive"])
    records = _records(pilot, "lineage-predictive")
    by_id = {record.record_id: record for record in records}
    final = [
        by_id[record.refers_to[1]]
        for record in records
        if record.kind == "acceptance"
        and isinstance(record.payload, Acceptance)
        and record.payload.accepted
    ][-1]

    def rename(
        record: Record, payload: Payload, refers_to: list[str], blobs: BlobStore
    ) -> tuple[Payload, list[str]]:
        if record.record_id == final.record_id and isinstance(payload, Evaluation):
            assert payload.per_task_blob is not None
            blob = json.loads(blobs.get(payload.per_task_blob).decode("utf-8"))
            blob["result"]["series"][0]["series_id"] = "synthetic-shift3/renamed/0001"
            return (
                replace(payload, per_task_blob=blobs.put(canonical_bytes(blob))),
                refers_to,
            )
        return payload, refers_to

    copy_run(pilot.ledger, "lineage-predictive", "series-mismatch", mutate=rename)
    with pytest.raises(Phase1ReportError, match="series ids differ"):
        _report(pilot, [*base_ids, "series-mismatch"])


# --- Test 5: the starting values ------------------------------------------------------------------


def test_forecast_rows_share_one_start_while_weights_differs(pilot: Pilot) -> None:
    """The forecast-v2 rows start bit-equal; the weights row starts from its own reference."""
    data = _report(pilot, _ids(pilot))
    by_condition = {row.condition: row for row in data.rows}
    forecast_starts = {row.starting_mase for row in data.rows if row.interface == "forecast-v2"}
    assert len(forecast_starts) == 1
    assert data.distinct_forecast_start_mase == tuple(forecast_starts)
    weights = by_condition["weights_only"]
    assert weights.starting_mase not in forecast_starts
    assert weights.gain == weights.starting_mase - weights.best_mase  # against its own start


# --- Test 6: the calibration ----------------------------------------------------------------------


def _hand_calibration(
    records: Sequence[Record],
) -> tuple[list[tuple[float, float, dict[str, float]]], int, int]:
    """Hand-counted calibration: realized (parent, child mase, quantiles), not-ok, never.

    The parent is the running accepted value at the proposal record, exactly
    as the engine's fit history stores it.
    """
    by_id = {record.record_id: record for record in records}
    prediction_of: dict[str, Record] = {}
    evaluation_of: dict[str, Record] = {}
    parent_of: dict[str, float] = {}
    realized: list[tuple[float, float, dict[str, float]]] = []
    not_ok = 0
    never = 0
    running = 0.0
    for record in records:
        if record.kind == "baseline" and isinstance(record.payload, Baseline):
            running = float(record.payload.metrics["mase"])
        elif record.kind == "proposal":
            parent_of[record.record_id] = running
        elif record.kind == "prediction" and record.refers_to[0] not in prediction_of:
            prediction_of[record.refers_to[0]] = record
        elif record.kind == "evaluation" and record.refers_to[0] not in evaluation_of:
            evaluation_of[record.refers_to[0]] = record
        elif (
            record.kind == "acceptance"
            and isinstance(record.payload, Acceptance)
            and record.payload.accepted
        ):
            evaluation = by_id[record.refers_to[1]]
            assert isinstance(evaluation.payload, Evaluation)
            running = float(evaluation.payload.metrics["mase"])
    for proposal_id, parent in parent_of.items():
        if proposal_id not in prediction_of:
            continue  # nothing predicted, nothing measurable
        evaluation = evaluation_of.get(proposal_id)
        if evaluation is None:
            never += 1
        elif evaluation.payload.status != "ok":
            not_ok += 1
        else:
            assert isinstance(evaluation.payload, Evaluation)
            predicted = prediction_of[proposal_id].payload
            assert isinstance(predicted, Prediction)
            realized.append(
                (parent, float(evaluation.payload.metrics["mase"]), predicted.quantiles)
            )
    return realized, not_ok, never


def test_calibration_counts_coverages_and_means(pilot: Pilot) -> None:
    """Counts, coverages and means equal hand counts over the records' quantile pairs."""
    data = _report(pilot, _ids(pilot))
    assert [row.condition for row in data.calibration] == [row.condition for row in data.rows]
    run_of = {row.condition: row.run_id for row in data.rows}
    for row in data.calibration:
        realized, not_ok, never = _hand_calibration(_records(pilot, run_of[row.condition]))
        gains = [parent - child for parent, child, _quantiles in realized]
        inside_90 = sum(
            1 for parent, child, q in realized if q["q05"] <= parent - child <= q["q95"]
        )
        inside_50 = sum(
            1 for parent, child, q in realized if q["q25"] <= parent - child <= q["q75"]
        )
        counts = _counts(pilot, run_of[row.condition])
        assert (row.predictions, row.realized) == (counts["prediction"], len(realized))
        assert (row.not_ok_evaluated, row.never_evaluated) == (not_ok, never)
        assert row.coverage_90 == inside_90 / len(realized)
        assert row.coverage_50 == inside_50 / len(realized)
        assert row.mean_q50 == _loop_mean([q["q50"] for _p, _c, q in realized])
        assert row.mean_gain == _loop_mean(gains)
        assert row.mean_q50_error == _loop_mean(
            [q["q50"] - (parent - child) for parent, child, q in realized]
        )
    by_condition = {row.condition: row for row in data.calibration}
    assert by_condition["evaluate_all"].never_evaluated == 1  # the slot the budget closed on
    assert by_condition["fixed"].never_evaluated == 3  # one unevaluated proposal per generation


def test_a_no_proposal_lineage_calibrates_to_zeros(pilot: Pilot) -> None:
    """A lineage that proposed nothing has zero counts and no coverages, in its own build.

    Its ``llm_output_tokens`` limit is smaller than the pilot's, so the
    equal-budget refusal would reject it next to the standard fixtures; the
    task pins that it is reported alone.
    """
    assert (
        _lineage(
            pilot,
            "lineage-no-proposals",
            "predictive",
            spec={"proposals_per_generation": 1, "llm_output_tokens": 16_383},
            outcomes={REFERENCE_ARTIFACT_SOURCE: scored(1.0)},
        )
        == 0
    )
    data = _report(pilot, ["lineage-no-proposals"])
    assert data.calibration == (
        CalibrationRow(
            condition="predictive",
            predictions=0,
            realized=0,
            not_ok_evaluated=0,
            never_evaluated=0,
            coverage_90=None,
            coverage_50=None,
            mean_q50=None,
            mean_gain=None,
            mean_q50_error=None,
        ),
    )
    # nothing accepted, so the baseline is also the final artifact: zero differences
    assert data.rows[0].gain_interval == GapRow(label="predictive", mean=0.0, lower=0.0, upper=0.0)


# --- Test 7: the parent at proposal time ----------------------------------------------------------


def test_generation_two_uses_the_accepted_parent_not_the_baseline(pilot: Pilot) -> None:
    """A generation-2 proposal's realized gain subtracts from child A's MASE, not the baseline's."""
    realized, _not_ok, _never = _hand_calibration(_records(pilot, "lineage-predictive"))
    gains = [parent - child for parent, child, _quantiles in realized]
    assert gains == [1.0 - 0.8, 0.8 - 0.7, 0.7 - 0.7]
    by_condition = {row.condition: row for row in _report(pilot, _ids(pilot)).calibration}
    assert by_condition["predictive"].realized == 3
    assert by_condition["predictive"].mean_gain == _loop_mean(gains)


# --- Test 8: the safe inheritance -----------------------------------------------------------------


def test_inheritance_chain_rule_over_the_holdout_rows(pilot: Pilot) -> None:
    """Judged, hurt and rate follow the chain rule; no acceptance gives no rate."""
    data = _report(pilot, _ids(pilot))
    assert [row.condition for row in data.inheritance] == [row.condition for row in data.rows]
    by_condition = {row.condition: row for row in data.inheritance}
    predictive = by_condition["predictive"]
    assert (predictive.accepted, predictive.judged, predictive.hurt) == (2, 2, 0)
    assert predictive.safe_rate == 0.0
    assert (predictive.unjudgeable, predictive.not_ok_rows) == (0, 0)
    assert (predictive.start_holdout_mase, predictive.final_holdout_mase) == (1.1, 1.1)
    assert predictive.holdout_gain == 0.0
    weights = by_condition["weights_only"]
    assert (weights.accepted, weights.judged, weights.hurt, weights.safe_rate) == (1, 1, 0, 0.0)
    for condition in ("evaluate_all", "fixed", "random_param"):
        flat = by_condition[condition]
        assert (flat.accepted, flat.judged, flat.hurt) == (0, 0, 0)
        assert flat.safe_rate is None
        assert flat.holdout_gain == 0.0


def test_a_hurt_holdout_edit_is_judged_and_counted(pilot: Pilot) -> None:
    """A holdout edit that scored worse than its predecessor hurts; the rate is hurt/judged."""

    def bump(
        record: Record, payload: Payload, refers_to: list[str], blobs: BlobStore
    ) -> tuple[Payload, list[str]]:
        if (
            record.kind == "baseline"
            and isinstance(payload, Baseline)
            and payload.updater_id == "accept-01"
        ):
            assert payload.per_task_blob is not None
            blob = json.loads(blobs.get(payload.per_task_blob).decode("utf-8"))
            for entry in blob["result"]["series"]:
                entry["mase"] = 1.3
            return (
                replace(
                    payload, metrics={"mase": 1.3}, per_task_blob=blobs.put(canonical_bytes(blob))
                ),
                refers_to,
            )
        return payload, refers_to

    copy_run(pilot.ledger, "holdout-predictive", "holdout-hurt", mutate=bump)
    row = _report(pilot, ["lineage-predictive", "holdout-hurt"]).inheritance[0]
    assert (row.judged, row.hurt, row.safe_rate) == (2, 1, 0.5)
    assert (row.start_holdout_mase, row.final_holdout_mase, row.holdout_gain) == (1.1, 1.1, 0.0)


def test_a_not_ok_holdout_row_is_unjudgeable(pilot: Pilot) -> None:
    """The weights flat child, accepted in dev, scores invalid_weights on the holdout."""
    assert (
        _lineage(
            pilot,
            "lineage-weights-flat",
            "weights_only",
            spec={"proposals_per_generation": 1},
            responses=[reply(WEIGHTS_FLAT_CHILD)] * 3,
            outcomes={
                WEIGHTS_REFERENCE_ARTIFACT_SOURCE: scored(1.0),
                WEIGHTS_FLAT_CHILD: scored(0.8),  # the dev stub accepts the flat child
            },
        )
        == 0
    )
    invalid = EvalOutcome(
        status="invalid_weights",
        result_obj=None,
        series_wall_seconds=(),
        error="arena: weights rows must be tuples, not floats",
        wall_seconds=0.05,
    )
    with _patched_holdout(
        invalid, {WEIGHTS_REFERENCE_ARTIFACT_SOURCE: scored(1.1, HOLDOUT_SERIES)}
    ):
        assert (
            holdout_run(
                (pilot.base, pilot.base / "cache_store"),
                pilot.base,
                from_run="lineage-weights-flat",
                run_id="holdout-weights-flat",
            )
            == 0
        )
    row = _report(pilot, ["lineage-weights-flat", "holdout-weights-flat"]).inheritance[0]
    assert (row.accepted, row.judged, row.hurt, row.safe_rate) == (1, 0, 0, None)
    assert (row.unjudgeable, row.not_ok_rows) == (1, 1)
    assert row.start_holdout_mase == 1.1
    assert row.final_holdout_mase is None
    assert row.holdout_gain is None


def test_missing_and_untabled_holdouts_add_no_row(pilot: Pilot) -> None:
    """A condition without a named holdout is named as missing and adds no row.

    A holdout that is named but not tabled — the explicit decision for the
    reviewer's note on #178 — is listed untabled, feeds no inheritance row,
    and its condition is not "missing": the run exists, and the untabled
    list says why it fed nothing.
    """
    data = _report(pilot, _ids(pilot, drop=["holdout-fixed"]))
    assert "fixed" not in [row.condition for row in data.inheritance]
    assert data.missing_holdout == ("fixed",)
    copy_run(pilot.ledger, "holdout-predictive", "holdout-aborted", keep_run_end=False)
    data = _report(pilot, ["lineage-predictive", "holdout-aborted"])
    assert data.inheritance == ()
    assert data.missing_holdout == ()
    assert data.untabled == (("holdout-aborted", "no run_end"),)


# --- Test 17: regenerable -------------------------------------------------------------------------


def test_two_builds_are_equal_and_a_copied_ledger_regenerates(pilot: Pilot, tmp_path: Path) -> None:
    """Two builds over the same runs are equal; so is a build over a copied ledger root."""
    first = _report(pilot, _ids(pilot))
    assert _report(pilot, _ids(pilot)) == first
    copied = tmp_path / "ledgers"
    shutil.copytree(pilot.ledger, copied)
    assert build_phase1_report_data(copied, _ids(pilot), bootstrap_seed=0) == first
