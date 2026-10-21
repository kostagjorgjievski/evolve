"""Phase 1 report data: read the pilot's named runs, refuse, build the rows.

TASK_019 PR A, split by the size rule: this half reads the named lineage,
replay and holdout runs, classifies them, refuses everything that would make
the report unfaithful, and recomputes the per-condition counts, scores and
budget sums the summary must reproduce. The bootstrap, calibration and
inheritance numbers land in the follow-up PR against the same builder.

The report reads ledgers only: no cache root, no data root, no foundation
root, no process, no LLM. Every refusal is a :class:`Phase1ReportError`
raised before any row is built, naming the run (and record) it refuses — a
duplicate or unloadable run id, a non-Phase-1 run, an impossible replay or
holdout pairing, a protocol-version mix, lineages that did not run the same
pool under the same limits, a holdout whose rows disagree with its lineage's
acceptances, or a ``run_end`` summary the records do not reproduce. Runs
that aborted, or never wrote their ``run_end``, are listed untabled with
their reason and contribute nothing else (AGENTS.md 2.8: every run that
started is in the report).

Counts and token sums are ints; wall seconds accumulate left to right, as
everywhere in the arena.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evolvets.ledger.blobs import BlobError, BlobStore
from evolvets.ledger.payloads import Acceptance, Baseline, Evaluation, Proposal, RunEnd, RunStart
from evolvets.ledger.schema import Record
from evolvets.ledger.store import LedgerCorruptError, LedgerReader
from evolvets.runner.gate1_report_data import _obj_dict, _obj_float, _obj_str, _obj_str_list
from evolvets.runner.phase1 import CONDITIONS

#: The prefix every Phase 1 run's condition carries.
_LINEAGE_PREFIX = "phase1-"

#: The condition of a holdout run (``runner.phase1_holdout``'s own constant).
_HOLDOUT_CONDITION = "phase1-holdout"

#: The keys a holdout run's config blob holds, exactly.
_HOLDOUT_CONFIG_KEYS = frozenset(
    {"from_run", "dataset", "pool_spec", "foundation_models", "interface"}
)

#: The run_end statuses a tabled run ended with; ``aborted`` never tables.
_TABLED_STATUSES = frozenset({"completed", "budget_exhausted"})

#: Budget-limit fields that must agree across tabled lineages.
_LIMIT_FIELDS = (
    "llm_input_tokens",
    "llm_output_tokens",
    "gpu_seconds",
    "evaluations",
)


class Phase1ReportError(Exception):
    """Raised when the named runs cannot form one faithful Phase 1 report."""


@dataclass(frozen=True)
class ConditionRow:
    """One tabled lineage condition: its counts, scores and per-compute gains.

    ``gain_per_million_tokens`` is ``None`` when the token sum is 0
    (``random_param`` proposes without tokens); ``evaluations_charged`` is
    the ledger ``budget_spent`` evaluations, baseline included, so it is one
    more than the ``evaluations`` count.
    """

    condition: str
    interface: str
    run_id: str
    status: str  # run_end status: "completed" | "budget_exhausted"
    proposals: int
    predictions: int
    allocations: int
    evaluations: int  # evaluation records (the summary's count)
    accepted: int
    starting_mase: float
    best_mase: float
    gain: float  # starting_mase - best_mase
    input_tokens: int
    output_tokens: int
    wall_seconds: float  # budget_spent wall sum
    evaluations_charged: int  # budget_spent evaluations (baseline included)
    gain_per_evaluation: float
    gain_per_million_tokens: float | None  # None when the token sum is 0
    gain_per_wall_hour: float


@dataclass(frozen=True)
class ReplayAudit:
    """The replay run's audit row, copied from its ``run_end`` summary."""

    run_id: str
    source_run_id: str
    matches_source_run: bool
    timeout_status_mismatches: int
    input_tokens: int
    output_tokens: int
    wall_seconds: float
    evaluations_charged: int


@dataclass(frozen=True)
class Phase1ReportData:
    """Every number the Phase 1 pilot report shows, computed from the ledger alone."""

    dataset: str
    pool_spec: str
    foundation_models: tuple[str, ...] | None
    seed: int
    git_commits: tuple[str, ...]  # distinct run_start commits, sorted
    rows: tuple[ConditionRow, ...]  # CONDITIONS registry order
    missing_holdout: tuple[str, ...]  # conditions without one, registry order
    replay: ReplayAudit | None
    distinct_forecast_start_mase: tuple[float, ...]  # across forecast-v2 rows
    untabled: tuple[tuple[str, str], ...]  # (run_id, reason), sorted by run id


# --- blob reading; every malformed field is a refusal, never a guess ------------------------------


def _blob_json(blobs: BlobStore, ref: str, where: str) -> object:
    """Return one blob's parsed JSON, refusing anything unreadable."""
    try:
        data = blobs.get(ref)
    except BlobError as exc:
        raise Phase1ReportError(f"{where}: blob {ref} cannot be read: {exc}") from exc
    try:
        parsed: object = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise Phase1ReportError(f"{where}: blob {ref} is not JSON: {exc}") from exc
    return parsed


def _finite(value: object, where: str) -> float:
    """Return ``value`` as a float, refusing a non-number or a non-finite one."""
    number = _obj_float(value, where)
    if not math.isfinite(number):
        raise Phase1ReportError(f"{where} must be finite, got {number!r}")
    return number


def _whole(value: float, where: str) -> int:
    """Return a budget charge as an int, refusing a fractional one."""
    number = float(value)  # the schema admits ints; only fractions are refused
    if not number.is_integer():
        raise Phase1ReportError(f"{where} must be a whole count, got {value!r}")
    return int(number)


@dataclass(frozen=True)
class _SeriesScore:
    """One series' id and MASE, as a per-task blob reports them."""

    series_id: str
    mase: float


def _series_of(blobs: BlobStore, run_id: str, record: Record) -> tuple[_SeriesScore, ...] | None:
    """A scored record's per-series scores; ``None`` when the record is not ok.

    Every per-task blob must parse as an object; an ok record's series carry
    the finite MASE values the report reads, in blob order.
    """
    where = f"run {run_id} record {record.record_id}"
    payload = record.payload
    if not isinstance(payload, (Baseline, Evaluation)):
        raise Phase1ReportError(f"{where}: only baseline and evaluation records are scored")
    per_task = payload.per_task_blob
    if per_task is None:
        raise Phase1ReportError(f"{where}: a scored record without its per_task_blob")
    blob = _obj_dict(_blob_json(blobs, per_task, where), f"{where}: the per_task_blob")
    if payload.status != "ok":
        return None  # the engine records the error; there is no result to read
    result = _obj_dict(blob.get("result"), f"{where}: the per_task_blob result")
    raw_series = result.get("series")
    if not isinstance(raw_series, list):
        raise Phase1ReportError(f"{where}: the result's series must be a list, got {raw_series!r}")
    scores: list[_SeriesScore] = []
    for entry in raw_series:
        series = _obj_dict(entry, f"{where}: a series entry")
        scores.append(
            _SeriesScore(
                series_id=_obj_str(series.get("series_id"), f"{where}: a series_id"),
                mase=_finite(series.get("mase"), f"{where}: a series mase"),
            )
        )
    return tuple(scores)


def _generation_of(blobs: BlobStore, proposal: Record, where: str) -> int:
    """The generation a proposal was made at, read from its observation blob."""
    payload = proposal.payload
    if not isinstance(payload, Proposal):
        raise Phase1ReportError(f"{where}: record {proposal.record_id} is no proposal")
    observation = _obj_dict(
        _blob_json(blobs, payload.observation_blob, where),
        f"{where}: the observation blob of proposal {proposal.record_id}",
    )
    generation = observation.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise Phase1ReportError(
            f"{where}: proposal {proposal.record_id} holds no generation in its observation blob"
        )
    return generation


# --- reading and classifying the named runs -------------------------------------------------------


@dataclass
class _NamedRun:
    """One named run, read, verified and classified from its ``run_start``."""

    run_id: str
    records: list[Record]
    start: RunStart
    config: dict[str, Any]
    kind: str  # "lineage" | "holdout"
    condition: str | None  # the lineage suffix; None for a holdout
    replay_of: str | None  # the config's replay_of; a replay when not None
    end: RunEnd | None  # the run_end payload when it is the last record


def _tabled(run: _NamedRun) -> bool:
    """Whether the run ended ``completed`` or ``budget_exhausted``."""
    return run.end is not None and run.end.status in _TABLED_STATUSES


def _untabled_reason(run: _NamedRun) -> str:
    """Why a run is not tabled: its error, else its status, else no run_end."""
    if run.end is not None:
        error = run.end.summary.get("error")
        if error is not None:
            return error if isinstance(error, str) else str(error)
        return run.end.status
    return "no run_end"


def _read_named_run(root: Path, run_id: str) -> _NamedRun:
    """Verify one named run and classify it; Phase 1 runs are main-stream only."""
    reader = LedgerReader(root, run_id)
    verified = reader.verify(check_blobs=True)
    if not verified.ok:
        position = "?" if verified.seq is None else str(verified.seq)
        raise Phase1ReportError(
            f"run {run_id}: {verified.stream} stream, position {position}: {verified.reason}"
        )
    try:
        records = list(reader.records())
    except LedgerCorruptError as exc:
        raise Phase1ReportError(f"run {run_id}: {exc}") from exc
    for record in records:
        if record.stream != "main":
            raise Phase1ReportError(
                f"run {run_id}: a {record.stream}-stream record; Phase 1 runs are main-stream only"
            )
    first = records[0] if records else None
    if first is None or first.kind != "run_start" or not isinstance(first.payload, RunStart):
        raise Phase1ReportError(f"run {run_id}: the first record must be a run_start")
    start = first.payload
    config = _obj_dict(
        _blob_json(BlobStore(root / run_id / "blobs"), start.config_blob, f"run {run_id}"),
        f"run {run_id}: the config blob",
    )
    if not start.condition.startswith(_LINEAGE_PREFIX):
        raise Phase1ReportError(f"run {run_id}: condition {start.condition!r} is not a Phase 1 run")
    last = records[-1]
    end = last.payload if last.kind == "run_end" and isinstance(last.payload, RunEnd) else None
    if start.condition == _HOLDOUT_CONDITION:
        if set(config) != _HOLDOUT_CONFIG_KEYS:
            raise Phase1ReportError(
                f"run {run_id}: a holdout config must hold exactly the keys "
                f"{sorted(_HOLDOUT_CONFIG_KEYS)}, got {sorted(config)}"
            )
        if start.requires_predictions:
            raise Phase1ReportError(
                f"run {run_id}: a holdout run must declare requires_predictions=false"
            )
        return _NamedRun(run_id, records, start, config, "holdout", None, None, end)
    condition = start.condition[len(_LINEAGE_PREFIX) :]
    if condition not in CONDITIONS:
        raise Phase1ReportError(
            f"run {run_id}: unknown Phase 1 condition {start.condition!r}; "
            f"expected one of {list(CONDITIONS)}"
        )
    if not start.requires_predictions:
        raise Phase1ReportError(
            f"run {run_id}: the {condition!r} lineage must declare requires_predictions=true"
        )
    replay_of = config.get("replay_of")
    if replay_of is not None and not isinstance(replay_of, str):
        raise Phase1ReportError(
            f"run {run_id}: the config's replay_of must be a run id or null, got {replay_of!r}"
        )
    return _NamedRun(run_id, records, start, config, "lineage", condition, replay_of, end)


def _foundation_of(config: dict[str, Any], run_id: str) -> tuple[str, ...] | None:
    """The config's foundation_models as a tuple; ``None`` stays ``None``."""
    value = config.get("foundation_models")
    if value is None:
        return None
    return tuple(_obj_str_list(value, f"run {run_id}: config foundation_models"))


# --- the pairing of lineages, the replay and the holdouts -----------------------------------------


def _pair_runs(
    runs: Sequence[_NamedRun],
) -> tuple[list[_NamedRun], dict[str, _NamedRun], _NamedRun | None]:
    """Split lineages from holdouts and refuse impossible pairings.

    Returns the lineage runs, the holdout runs keyed by the lineage they
    score, and the replay run (``None`` when no replay was named). Two named
    lineages of one condition, a second replay, a replay whose source is not
    a named lineage run, an orphan holdout, a holdout of the replay and two
    holdouts for one lineage are each refused.
    """
    by_id = {run.run_id: run for run in runs}
    lineages = [run for run in runs if run.kind == "lineage"]
    holdouts_of: dict[str, _NamedRun] = {}
    replay: _NamedRun | None = None
    by_condition: dict[str, _NamedRun] = {}
    for run in lineages:
        if run.replay_of is not None:
            if replay is not None:
                raise Phase1ReportError(
                    f"two replay runs named: {replay.run_id!r} and {run.run_id!r}; "
                    "at most one may be reported"
                )
            replay = run
            continue  # a replay re-runs its source's condition; it claims no slot
        named = by_condition.get(run.condition or "")
        if named is not None:
            raise Phase1ReportError(
                f"runs {named.run_id!r} and {run.run_id!r} both ran condition "
                f"{run.condition!r}; name one"
            )
        by_condition[run.condition or ""] = run
    if replay is not None:
        if replay.replay_of == replay.run_id:
            raise Phase1ReportError(
                f"replay {replay.run_id!r} names itself as its replay_of source"
            )
        source = by_id.get(replay.replay_of or "")
        if source is None or source.kind != "lineage":
            raise Phase1ReportError(
                f"replay {replay.run_id!r} names replay_of {replay.replay_of!r}, "
                "which is not a named lineage run"
            )
    for run in runs:
        if run.kind != "holdout":
            continue
        from_run = _obj_str(run.config.get("from_run"), f"run {run.run_id}: config from_run")
        source = by_id.get(from_run)
        if source is None or source.kind != "lineage":
            raise Phase1ReportError(
                f"holdout {run.run_id!r} names from_run {from_run!r}, "
                "which is not a named lineage run"
            )
        if replay is not None and source is replay:
            raise Phase1ReportError(
                f"holdout {run.run_id!r} scores the replay {from_run!r}; "
                "it belongs to the source lineage"
            )
        named = holdouts_of.get(from_run)
        if named is not None:
            raise Phase1ReportError(
                f"two holdout runs named for lineage {from_run!r}: "
                f"{named.run_id!r} and {run.run_id!r}"
            )
        holdouts_of[from_run] = run
    return lineages, holdouts_of, replay


# --- the per-run checks of tabled runs ------------------------------------------------------------


def _check_lineage_run(root: Path, run: _NamedRun) -> None:
    """Refuse a tabled lineage whose shape the report cannot trust."""
    records = run.records
    where = f"run {run.run_id}"
    baselines = [record for record in records if record.kind == "baseline"]
    if len(baselines) != 1 or len(records) < 2 or records[1] is not baselines[0]:
        raise Phase1ReportError(
            f"{where}: exactly one baseline record, immediately after run_start, "
            f"is required; found {len(baselines)}"
        )
    baseline = baselines[0].payload
    if not isinstance(baseline, Baseline):
        raise Phase1ReportError(f"{where}: the baseline record carries no baseline payload")
    reference = CONDITIONS[run.condition or ""].reference_updater_id
    if baseline.updater_id != reference:
        raise Phase1ReportError(
            f"{where}: the baseline names updater {baseline.updater_id!r}, "
            f"not the condition's reference {reference!r}"
        )
    if baseline.status != "ok":
        raise Phase1ReportError(
            f"{where}: a tabled run cannot carry a {baseline.status!r} baseline"
        )
    _finite(baseline.metrics.get("mase"), f"{where}: the baseline metrics mase")
    by_id = {record.record_id: record for record in records}
    for record in records:
        if record.kind == "prediction":
            target = by_id.get(record.refers_to[0])
            if target is None or target.kind != "proposal":
                raise Phase1ReportError(
                    f"{where} record {record.record_id}: a prediction must refer "
                    "to a proposal record of this run"
                )
        elif record.kind == "acceptance":
            proposal = by_id.get(record.refers_to[0])
            evaluation = by_id.get(record.refers_to[1])
            if proposal is None or proposal.kind != "proposal":
                raise Phase1ReportError(
                    f"{where} record {record.record_id}: an acceptance must refer "
                    "to its proposal record first"
                )
            if evaluation is None or evaluation.kind != "evaluation":
                raise Phase1ReportError(
                    f"{where} record {record.record_id}: an acceptance must refer "
                    "to its evaluation record second"
                )
    blobs = BlobStore(root / run.run_id / "blobs")
    for record in records:
        if record.kind in ("baseline", "evaluation"):
            _series_of(blobs, run.run_id, record)


def _accepted_proposals(lineage: _NamedRun) -> list[Record]:
    """The lineage's accepted proposals' records, in ledger order."""
    by_id = {record.record_id: record for record in lineage.records}
    out: list[Record] = []
    for record in lineage.records:
        if record.kind != "acceptance":
            continue
        payload = record.payload
        if not (isinstance(payload, Acceptance) and payload.accepted):
            continue
        proposal = by_id.get(record.refers_to[0])
        if proposal is None or proposal.kind != "proposal":
            raise Phase1ReportError(
                f"run {lineage.run_id} record {record.record_id}: an acceptance "
                "must refer to its proposal record first"
            )
        out.append(proposal)
    return out


def _check_holdout_run(root: Path, holdout: _NamedRun, lineage: _NamedRun) -> None:
    """Refuse a tabled holdout whose rows disagree with its lineage's acceptances."""
    pair = f"holdout {holdout.run_id!r} of lineage {lineage.run_id!r}"
    baselines = [record for record in holdout.records if record.kind == "baseline"]
    ids: list[str] = []
    for record in baselines:
        payload = record.payload
        if not isinstance(payload, Baseline):
            raise Phase1ReportError(f"{pair}: record {record.record_id} is no baseline")
        ids.append(payload.updater_id)
    accepted = _accepted_proposals(lineage)
    expected = ["start"] + [f"accept-{index:02d}" for index in range(1, len(accepted) + 1)]
    if ids != expected:
        raise Phase1ReportError(
            f"{pair}: the holdout rows are {ids}, but the lineage accepted "
            f"{len(accepted)} edit(s) and the rows must be {expected}"
        )
    holdout_blobs = BlobStore(root / holdout.run_id / "blobs")
    lineage_blobs = BlobStore(root / lineage.run_id / "blobs")
    proposals = [record for record in lineage.records if record.kind == "proposal"]
    for index, record in enumerate(baselines):
        payload = record.payload
        assert isinstance(payload, Baseline)  # the loop above established it
        where = f"{pair} record {record.record_id}"
        if payload.per_task_blob is None:
            raise Phase1ReportError(f"{where}: a scored record without its per_task_blob")
        blob = _obj_dict(
            _blob_json(holdout_blobs, payload.per_task_blob, where), f"{where}: the per_task_blob"
        )
        linked = _obj_str(blob.get("lineage_run_id"), f"{where}: lineage_run_id")
        if linked != lineage.run_id:
            raise Phase1ReportError(f"{where}: links lineage {linked!r}, not {lineage.run_id!r}")
        if index == 0:
            expected_proposal = proposals[0].record_id if proposals else None
        else:
            expected_proposal = accepted[index - 1].record_id
        proposal_id = blob.get("proposal_record_id")
        if proposal_id != expected_proposal:
            raise Phase1ReportError(
                f"{where}: links proposal {proposal_id!r}, expected {expected_proposal!r}"
            )
        if index > 0:
            expected_generation = _generation_of(lineage_blobs, accepted[index - 1], pair)
            generation = blob.get("generation")
            if generation != expected_generation:
                raise Phase1ReportError(
                    f"{where}: names generation {generation!r}, expected {expected_generation!r}"
                )
        _series_of(holdout_blobs, holdout.run_id, record)


def _check_holdout_pool(lineage: _NamedRun, holdout: _NamedRun) -> None:
    """Refuse a holdout scored on another pool or with other foundation models."""
    holdout_pool = _obj_str(
        holdout.config.get("pool_spec"), f"run {holdout.run_id}: config pool_spec"
    )
    lineage_pool = _obj_str(
        lineage.config.get("pool_spec"), f"run {lineage.run_id}: config pool_spec"
    )
    if holdout_pool != lineage_pool:
        raise Phase1ReportError(
            f"holdout {holdout.run_id!r} ran pool_spec {holdout_pool!r}, "
            f"its lineage {lineage.run_id!r} ran {lineage_pool!r}"
        )
    holdout_models = _foundation_of(holdout.config, holdout.run_id)
    lineage_models = _foundation_of(lineage.config, lineage.run_id)
    if holdout_models != lineage_models:
        raise Phase1ReportError(
            f"holdout {holdout.run_id!r} and lineage {lineage.run_id!r} disagree on "
            f"foundation_models: {holdout_models!r} vs {lineage_models!r}"
        )


# --- the recomputed numbers and the summary cross-check -------------------------------------------


@dataclass(frozen=True)
class _LineageNumbers:
    """One tabled lineage's recomputed counts, scores and budget sums."""

    proposals: int
    predictions: int
    allocations: int
    evaluations: int
    accepted: int
    starting_mase: float
    best_mase: float
    input_tokens: int
    output_tokens: int
    wall_seconds: float
    evaluations_charged: float


def _numbers(run: _NamedRun) -> _LineageNumbers:
    """Recompute one lineage's summary numbers from its records, left to right."""
    run_id = run.run_id
    proposals = predictions = allocations = evaluations = accepted = 0
    input_tokens = 0
    output_tokens = 0
    wall_seconds = 0.0
    evaluations_charged = 0.0
    starting_mase = 0.0
    best_mase = 0.0
    by_id = {record.record_id: record for record in run.records}
    for record in run.records:
        where = f"run {run_id} record {record.record_id}: budget"
        wall_seconds = wall_seconds + record.budget.wall_seconds
        input_tokens += _whole(record.budget.llm_input_tokens, f"{where} llm_input_tokens")
        output_tokens += _whole(record.budget.llm_output_tokens, f"{where} llm_output_tokens")
        if record.kind == "proposal":
            proposals += 1
        elif record.kind == "prediction":
            predictions += 1
        elif record.kind == "allocation":
            allocations += 1
        elif record.kind in ("evaluation", "baseline"):
            payload = record.payload
            if not isinstance(payload, (Baseline, Evaluation)):
                raise Phase1ReportError(
                    f"run {run_id} record {record.record_id}: no scored payload"
                )
            if record.kind == "evaluation":
                evaluations += 1
            evaluations_charged = evaluations_charged + record.budget.evaluations
            if record.kind == "baseline":
                starting_mase = _finite(
                    payload.metrics.get("mase"), f"run {run_id} record {record.record_id}: mase"
                )
                best_mase = starting_mase
        elif record.kind == "acceptance":
            payload = record.payload
            if not isinstance(payload, Acceptance):
                raise Phase1ReportError(
                    f"run {run_id} record {record.record_id}: no acceptance payload"
                )
            if payload.accepted:
                accepted += 1
                evaluation = by_id.get(record.refers_to[1])
                scored = evaluation.payload if evaluation is not None else None
                if (
                    evaluation is None
                    or evaluation.kind != "evaluation"
                    or not isinstance(scored, Evaluation)
                ):
                    raise Phase1ReportError(
                        f"run {run_id} record {record.record_id}: an acceptance must "
                        "refer to its evaluation record second"
                    )
                best_mase = _finite(
                    scored.metrics.get("mase"),
                    f"run {run_id} record {evaluation.record_id}: the accepted mase",
                )
    return _LineageNumbers(
        proposals=proposals,
        predictions=predictions,
        allocations=allocations,
        evaluations=evaluations,
        accepted=accepted,
        starting_mase=starting_mase,
        best_mase=best_mase,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        wall_seconds=wall_seconds,
        evaluations_charged=evaluations_charged,
    )


def _cross_check_summary(run: _NamedRun, numbers: _LineageNumbers) -> None:
    """Refuse every run_end summary key the records do not reproduce."""
    end = run.end
    assert end is not None  # the caller tabled the run
    summary = end.summary
    spent = _obj_dict(summary.get("budget_spent"), f"run {run.run_id}: summary budget_spent")
    expected: dict[str, object] = {
        "condition": run.condition,
        "proposals": numbers.proposals,
        "predictions": numbers.predictions,
        "allocations": numbers.allocations,
        "evaluations": numbers.evaluations,
        "accepted": numbers.accepted,
        "starting_mase": numbers.starting_mase,
        "best_mase": numbers.best_mase,
        "replay_of": run.replay_of,
    }
    for key, recomputed in expected.items():
        actual = summary.get(key)
        if actual != recomputed:
            raise Phase1ReportError(
                f"run {run.run_id}: the run_end summary's {key} says {actual!r}, "
                f"the records say {recomputed!r}"
            )
    charged = {
        "llm_input_tokens": numbers.input_tokens,
        "llm_output_tokens": numbers.output_tokens,
        "wall_seconds": numbers.wall_seconds,
        "evaluations": numbers.evaluations_charged,
    }
    for key, recomputed in charged.items():
        actual = spent.get(key)
        if actual != recomputed:
            raise Phase1ReportError(
                f"run {run.run_id}: the run_end summary's budget_spent.{key} says "
                f"{actual!r}, the records say {recomputed!r}"
            )


# --- the entry point ------------------------------------------------------------------------------


def build_phase1_report_data(
    ledger_root: Path, run_ids: Sequence[str], *, bootstrap_seed: int
) -> Phase1ReportData:
    """Read the named runs and compute the report's data from the ledger alone.

    Raises :class:`Phase1ReportError` on every refusal the report's rules
    define; ``bootstrap_seed`` seeds the series bootstrap, which the
    follow-up PR of this task adds.
    """
    del bootstrap_seed  # the bootstrap lands with the numbers PR of TASK_019 PR A
    root = Path(ledger_root)
    if not run_ids:
        raise Phase1ReportError("at least one run id is required")
    seen: set[str] = set()
    for run_id in run_ids:
        if run_id in seen:
            raise Phase1ReportError(f"duplicate run id {run_id!r} in {list(run_ids)}")
        seen.add(run_id)
    runs = [_read_named_run(root, run_id) for run_id in run_ids]
    lineages, holdouts_of, replay = _pair_runs(runs)

    # every run that started is in the report; only tabled runs feed the tables
    row_sources = [run for run in lineages if run is not replay and _tabled(run)]
    untabled = tuple(
        sorted(
            ((run.run_id, _untabled_reason(run)) for run in runs if not _tabled(run)),
            key=lambda item: item[0],
        )
    )
    if not row_sources:
        raise Phase1ReportError("no tabled lineage run remains: there is nothing to report")

    # protocol versions agree across every named run
    versions = runs[0].start.protocol_versions
    for run in runs[1:]:
        if run.start.protocol_versions != versions:
            raise Phase1ReportError(
                f"protocol versions differ: run {runs[0].run_id} has {versions}, "
                f"run {run.run_id} has {run.start.protocol_versions}"
            )

    # one pilot, one table: the tabled lineages ran one pool under one seed
    # and one set of limits (the replay's inflated wall limit is its own)
    head = row_sources[0]
    dataset = _obj_str(head.config.get("dataset"), f"run {head.run_id}: config dataset")
    pool_spec = _obj_str(head.config.get("pool_spec"), f"run {head.run_id}: config pool_spec")
    foundation = _foundation_of(head.config, head.run_id)
    for run in row_sources[1:]:
        other_dataset = _obj_str(run.config.get("dataset"), f"run {run.run_id}: config dataset")
        other_pool = _obj_str(run.config.get("pool_spec"), f"run {run.run_id}: config pool_spec")
        if other_dataset != dataset:
            raise Phase1ReportError(
                f"lineage runs {head.run_id!r} and {run.run_id!r} ran datasets "
                f"{dataset!r} and {other_dataset!r}; one report is one table"
            )
        if other_pool != pool_spec:
            raise Phase1ReportError(
                f"lineage runs {head.run_id!r} and {run.run_id!r} ran pool specs "
                f"{pool_spec!r} and {other_pool!r}; one report is one table"
            )
        if _foundation_of(run.config, run.run_id) != foundation:
            raise Phase1ReportError(
                f"lineage runs {head.run_id!r} and {run.run_id!r} ran foundation models "
                f"{foundation!r} and {_foundation_of(run.config, run.run_id)!r}"
            )
        if run.start.seed != head.start.seed:
            raise Phase1ReportError(
                f"lineage runs {head.run_id!r} and {run.run_id!r} ran seeds "
                f"{head.start.seed} and {run.start.seed}; one report is one pilot"
            )
        for field in (*_LIMIT_FIELDS, "wall_seconds"):
            if getattr(run.start.budget_limits, field) != getattr(head.start.budget_limits, field):
                raise Phase1ReportError(
                    f"lineage runs {head.run_id!r} and {run.run_id!r} declare different "
                    f"budget_limits.{field}: the equal-budget rule"
                )

    # the per-run shape checks, then the summary cross-check, per tabled run
    tabled_lineages = [run for run in lineages if _tabled(run)]
    for run in tabled_lineages:
        _check_lineage_run(root, run)
    for lineage_run_id, holdout in sorted(holdouts_of.items()):
        if not _tabled(holdout):
            continue
        lineage = next(run for run in runs if run.run_id == lineage_run_id)
        _check_holdout_run(root, holdout, lineage)
        _check_holdout_pool(lineage, holdout)
    if replay is not None and _tabled(replay):
        source = next(run for run in runs if run.run_id == replay.replay_of)
        if replay.condition != source.condition:
            raise Phase1ReportError(
                f"replay {replay.run_id!r} runs condition {replay.condition!r}, "
                f"its source {source.run_id!r} ran {source.condition!r}"
            )
        for key in ("dataset", "pool_spec"):
            if replay.config.get(key) != source.config.get(key):
                raise Phase1ReportError(
                    f"replay {replay.run_id!r} ran {key} {replay.config.get(key)!r}, "
                    f"its source ran {source.config.get(key)!r}"
                )
        if _foundation_of(replay.config, replay.run_id) != _foundation_of(
            source.config, source.run_id
        ):
            raise Phase1ReportError(
                f"replay {replay.run_id!r} and its source {source.run_id!r} disagree "
                "on foundation_models"
            )
        for field in _LIMIT_FIELDS:
            if getattr(replay.start.budget_limits, field) != getattr(
                source.start.budget_limits, field
            ):
                raise Phase1ReportError(
                    f"replay {replay.run_id!r} and its source {source.run_id!r} declare "
                    f"different budget_limits.{field}"
                )
    numbers = {run.run_id: _numbers(run) for run in tabled_lineages}
    for run in tabled_lineages:
        _cross_check_summary(run, numbers[run.run_id])

    # the rows, in CONDITIONS registry order over the tabled conditions
    row_of = {run.condition: run for run in row_sources}
    rows: list[ConditionRow] = []
    for condition, spec in CONDITIONS.items():
        row_source = row_of.get(condition)
        if row_source is None:
            continue
        counted = numbers[row_source.run_id]
        gain = counted.starting_mase - counted.best_mase
        tokens = counted.input_tokens + counted.output_tokens
        charged = _whole(
            counted.evaluations_charged,
            f"run {row_source.run_id}: the charged evaluations",
        )
        rows.append(
            ConditionRow(
                condition=condition,
                interface=spec.interface,
                run_id=row_source.run_id,
                status=row_source.end.status if row_source.end is not None else "",
                proposals=counted.proposals,
                predictions=counted.predictions,
                allocations=counted.allocations,
                evaluations=counted.evaluations,
                accepted=counted.accepted,
                starting_mase=counted.starting_mase,
                best_mase=counted.best_mase,
                gain=gain,
                input_tokens=counted.input_tokens,
                output_tokens=counted.output_tokens,
                wall_seconds=counted.wall_seconds,
                evaluations_charged=charged,
                gain_per_evaluation=gain / counted.evaluations_charged,
                gain_per_million_tokens=None if tokens == 0 else gain * 1_000_000.0 / tokens,
                gain_per_wall_hour=gain * 3600.0 / counted.wall_seconds,
            )
        )

    # the replay's audit row, copied from its run_end
    audit: ReplayAudit | None = None
    if replay is not None and _tabled(replay):
        end = replay.end
        assert end is not None
        matches = end.summary.get("matches_source_run")
        if not isinstance(matches, bool):
            raise Phase1ReportError(
                f"run {replay.run_id}: the replay's run_end must carry a boolean "
                f"matches_source_run, got {matches!r}"
            )
        mismatches = end.summary.get("timeout_status_mismatches")
        if isinstance(mismatches, bool) or not isinstance(mismatches, int):
            raise Phase1ReportError(
                f"run {replay.run_id}: the replay's run_end must carry an int "
                f"timeout_status_mismatches, got {mismatches!r}"
            )
        counted = numbers[replay.run_id]
        audit = ReplayAudit(
            run_id=replay.run_id,
            source_run_id=replay.replay_of or "",
            matches_source_run=matches,
            timeout_status_mismatches=mismatches,
            input_tokens=counted.input_tokens,
            output_tokens=counted.output_tokens,
            wall_seconds=counted.wall_seconds,
            evaluations_charged=_whole(
                counted.evaluations_charged,
                f"run {replay.run_id}: the charged evaluations",
            ),
        )

    # the forecast-v2 starting values, first-seen in registry order
    distinct: list[float] = []
    for row in rows:
        if CONDITIONS[row.condition].interface != "forecast-v2":
            continue  # the weights-v1 row keeps its own reference
        if row.starting_mase not in distinct:
            distinct.append(row.starting_mase)
    missing_holdout = tuple(row.condition for row in rows if row.run_id not in holdouts_of)
    return Phase1ReportData(
        dataset=dataset,
        pool_spec=pool_spec,
        foundation_models=foundation,
        seed=head.start.seed,
        git_commits=tuple(sorted({run.start.git_commit for run in runs})),
        rows=tuple(rows),
        missing_holdout=missing_holdout,
        replay=audit,
        distinct_forecast_start_mase=tuple(distinct),
        untabled=untabled,
    )
