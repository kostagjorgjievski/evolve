"""Gate 1 report data: read named runs, refuse inconsistencies, compute every number.

PR C2 renders what this module computes; nothing here formats Markdown or
parses a command line. ``build_report_data`` re-derives every number from
the ledger alone: it verifies each named run's chain and blobs, refuses
anything that disagrees — a protocol-version mix across runs, a result that
does not match its run's config blob, an oracle whose series differ from a
replay's, a ``run_end`` selection the records do not reproduce, two
complete runs over one pool, a sealed-stream record — and then turns the
completed runs' records into plain dataclasses: run rows, protocol
versions, and per ``(dataset, pool spec)`` the tuned-baseline rows, the
hindsight oracles, the gaps of the best tuned baseline with their paired
bootstrap intervals, the per-family gaps of the synthetic pool, the
leakage flag and the largest per-series wall-clock. Incomplete runs
(aborted, budget-exhausted, or missing their ``run_end``) are listed with
their ``validate_run`` violations and never tabled.

Float accumulation is explicit and left-to-right, as everywhere in the
arena; the only randomness is the bootstrap, seeded afresh for each Gaps
table from the one ``bootstrap_seed``, so the same ledger and seed give
bit-identical data.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evolvets.arena.oracles import ORACLE_SWITCHES
from evolvets.arena.sealed.client import SERIES_TIMEOUT_SECONDS
from evolvets.arena.synthetic.grammar import FAMILIES
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import Baseline, RunEnd, RunStart
from evolvets.ledger.schema import Record
from evolvets.ledger.store import LedgerCorruptError, LedgerReader
from evolvets.ledger.validate import validate_run
from evolvets.runner.gate1_grids import Selection, select_configs
from evolvets.runner.gate1_probe import PROBE_CONDITION
from evolvets.updater.baselines import BASELINE_IDS

#: Bootstrap resamples behind every Gaps interval.
BOOTSTRAP_RESAMPLES = 2000
#: Indexes into the sorted resample means that bound the 95% interval.
_INTERVAL_LOW_INDEX = 50
_INTERVAL_HIGH_INDEX = 1949
#: The dataset whose series ids are ``<family>/<index>`` and whose results
#: carry the true-best forecast.
_SYNTHETIC_DATASET = "synthetic-dev"


class ReportError(Exception):
    """Raised when the ledger behind a report cannot be trusted or disagrees with itself."""


# --- the dataclasses C2 renders -------------------------------------------------------------------


@dataclass(frozen=True)
class RunRow:
    """One named run as the Runs section reports it."""

    run_id: str
    condition: str
    status: str  # run_end status, or "incomplete" when the run has no run_end
    git_commit: str
    records: int
    violations: tuple[str, ...]  # validate_run output; empty means none


@dataclass(frozen=True)
class TunedRow:
    """One baseline's tuned row: its selection and what tuning it cost.

    ``config`` and ``mase`` are ``None`` when no grid point produced an ok
    result; the cost columns are filled from the records either way.
    """

    baseline: str
    config: dict[str, object] | None
    mase: float | None
    tried: int
    failed: int  # records whose status is not "ok"
    evaluations: float  # loop-sum of the records' budget.evaluations
    wall_seconds: float  # loop-sum of the records' budget.wall_seconds


@dataclass(frozen=True)
class OracleNumbers:
    """One pool's hindsight oracles, as the Oracles table reports them.

    ``true_best_mase`` is the synthetic true-best forecast, read from the
    baseline results' metrics; it is ``None`` on real pools.
    """

    true_best_mase: float | None
    global_expert: str
    global_expert_mase: float
    best_expert_mase: float
    switch_mase: tuple[float, ...]  # one per ORACLE_SWITCHES entry
    select_mase: float


@dataclass(frozen=True)
class GapRow:
    """One oracle's gap from the best tuned baseline, over one pool's series.

    ``lower`` and ``upper`` are ``None`` with a single series: the mean is
    printed, the interval is not defined.
    """

    label: str
    mean: float  # loop-mean of d_s = baseline_mase_s - oracle_s
    lower: float | None
    upper: float | None


@dataclass(frozen=True)
class FamilyGaps:
    """The same Gaps rows over one synthetic family's series only."""

    family: str
    rows: tuple[GapRow, ...]


@dataclass(frozen=True)
class PoolSection:
    """Everything the report says about one ``(dataset, pool spec)`` of complete runs.

    ``gaps`` is ``None`` when no baseline has an ok configuration on this
    pool: there is no best tuned baseline and no gap to compute.
    ``family_gaps`` is non-empty only on ``synthetic-dev``.
    """

    dataset: str
    pool_spec: str
    leakage_risk: bool
    tuned: tuple[TunedRow, ...]
    oracles: OracleNumbers
    gaps: tuple[GapRow, ...] | None
    family_gaps: tuple[FamilyGaps, ...]
    max_series_wall_seconds: float | None  # largest series_wall_seconds over all replays


@dataclass(frozen=True)
class ReportData:
    """Every number the Gate 1 dev report shows, computed from the ledger alone."""

    run_rows: tuple[RunRow, ...]
    protocol_versions: dict[str, str]
    sections: tuple[PoolSection, ...]  # leakage_risk == False, in config order
    leakage_sections: tuple[PoolSection, ...]  # leakage_risk == True, in config order
    series_timeout_seconds: float


# --- JSON accessors; every malformed field is a refusal, never a guess----------------------------


def _obj_dict(value: object, where: str) -> dict[str, Any]:
    """Return ``value`` as a JSON object, else refuse."""
    if not isinstance(value, dict):
        raise ReportError(f"{where} must be a JSON object, got {value!r}")
    return value


def _obj_str(value: object, where: str) -> str:
    """Return ``value`` as a string, else refuse."""
    if not isinstance(value, str):
        raise ReportError(f"{where} must be a string, got {value!r}")
    return value


def _obj_float(value: object, where: str) -> float:
    """Return ``value`` as a number (bools rejected), else refuse."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{where} must be a number, got {value!r}")
    return float(value)


def _obj_bool(value: object, where: str) -> bool:
    """Return ``value`` as a bool, else refuse."""
    if not isinstance(value, bool):
        raise ReportError(f"{where} must be a bool, got {value!r}")
    return value


def _obj_str_list(value: object, where: str) -> list[str]:
    """Return ``value`` as a list of strings, else refuse."""
    if not isinstance(value, list):
        raise ReportError(f"{where} must be a list, got {value!r}")
    return [_obj_str(item, f"{where} entry") for item in value]


def _obj_float_list(value: object, where: str) -> list[float]:
    """Return ``value`` as a list of numbers, else refuse."""
    if not isinstance(value, list):
        raise ReportError(f"{where} must be a list, got {value!r}")
    return [_obj_float(item, f"{where} entry") for item in value]


def _blob_reader(root: Path, run_id: str) -> Callable[[str], object]:
    """Return a ``read_blob`` over one run's store, as ``select_configs`` takes it."""

    def read(ref: str) -> object:
        try:
            data = BlobStore(root / run_id / "blobs").get(ref)
            parsed: object = json.loads(data.decode("utf-8"))
        except (ValueError, OSError) as exc:
            raise ReportError(f"run {run_id}: blob {ref} cannot be read: {exc}") from exc
        return parsed

    return read


def _result_of(run_id: str, blob: object) -> dict[str, Any]:
    """Return the ``result`` object of one baseline record's blob."""
    return _obj_dict(_obj_dict(blob, f"run {run_id}: a baseline blob").get("result"), "result")


# --- left-to-right float arithmetic, as everywhere in the arena -----------------------------------


def _loop_sum(values: Sequence[float]) -> float:
    """Return the left-to-right loop sum of ``values``."""
    total = 0.0
    for value in values:
        total = total + value
    return total


def _loop_mean(values: Sequence[float]) -> float:
    """Return the left-to-right loop mean of ``values``."""
    return _loop_sum(values) / len(values)


def _lowest(values: Sequence[float]) -> float:
    """The minimum of ``values`` by a left-to-right scan (first minimum wins ties)."""
    low = values[0]
    for value in values[1:]:
        low = value if value < low else low
    return low


def _gap_rows(labeled: Sequence[tuple[str, Sequence[float]]], seed: int) -> tuple[GapRow, ...]:
    """Compute one Gaps table: each row's mean and paired bootstrap interval.

    One ``random.Random(seed)`` serves the whole table, in row order, so a
    table's numbers depend on nothing outside it; a single-series table has
    no interval to draw.
    """
    rng = random.Random(seed)
    rows: list[GapRow] = []
    for label, differences in labeled:
        n = len(differences)
        mean = _loop_mean(differences)
        if n == 1:
            rows.append(GapRow(label=label, mean=mean, lower=None, upper=None))
            continue
        means: list[float] = []
        for _resample in range(BOOTSTRAP_RESAMPLES):
            total = 0.0
            for _draw in range(n):
                total = total + differences[rng.randrange(n)]
            means.append(total / n)
        means.sort()
        rows.append(
            GapRow(
                label=label,
                mean=mean,
                lower=means[_INTERVAL_LOW_INDEX],
                upper=means[_INTERVAL_HIGH_INDEX],
            )
        )
    return tuple(rows)


# --- reading one named run ------------------------------------------------------------------------


class _Pool:
    """One complete run's records for one ``(dataset, pool spec)``."""

    def __init__(self) -> None:
        self.oracle: tuple[Record, dict[str, Any]] | None = None
        self.by_baseline: dict[str, list[tuple[Record, dict[str, Any]]]] = {}


@dataclass
class _RunPools:
    """A complete run's pools, keyed for the sections, with its own selection."""

    pools: dict[tuple[str, str], _Pool]
    selection: Selection


def _read_run(root: Path, run_id: str) -> list[Record]:
    """Verify one run's chain and blobs and return its records; sealed records refuse."""
    reader = LedgerReader(root, run_id)
    verified = reader.verify(check_blobs=True)
    if not verified.ok:
        position = "?" if verified.seq is None else str(verified.seq)
        raise ReportError(
            f"run {run_id}: {verified.stream} stream, position {position}: {verified.reason}"
        )
    try:
        records = list(reader.records())
    except LedgerCorruptError as exc:
        raise ReportError(f"run {run_id}: {exc}") from exc
    for record in records:
        if record.stream == "sealed":
            raise ReportError(
                f"run {run_id}: sealed results are reported by the sealed Gate 1 task"
            )
    return records


def _run_start(
    root: Path, run_id: str, records: Sequence[Record]
) -> tuple[RunStart, dict[str, Any]]:
    """Return one run's ``run_start`` payload and its parsed config blob."""
    first = next((record for record in records if record.stream == "main"), None)
    if first is None or first.kind != "run_start" or not isinstance(first.payload, RunStart):
        raise ReportError(f"run {run_id}: the first main record must be a run_start")
    read = _blob_reader(root, run_id)
    config = _obj_dict(read(first.payload.config_blob), f"run {run_id}: the config blob")
    if config.get("experiment") != "gate1":
        raise ReportError(
            f"run {run_id}: the config names experiment {config.get('experiment')!r}, not 'gate1'"
        )
    return first.payload, config


def _run_end(records: Sequence[Record]) -> Record | None:
    """Return the run's ``run_end`` record, or ``None`` when it never ended."""
    for record in records:
        if record.stream == "main" and record.kind == "run_end":
            return record
    return None


def _check_result(
    run_id: str, record: Record, result: dict[str, Any], config: dict[str, Any], arena: str
) -> None:
    """Refuse a result whose pool pins disagree with its run's config blob."""
    where = f"run {run_id} record {record.record_id}"
    version = _obj_str(result.get("arena_version"), f"{where}: result.arena_version")
    if version != arena:
        raise ReportError(f"{where}: result arena_version {version!r} != run's {arena!r}")
    dataset = _obj_str(result.get("dataset"), f"{where}: result.dataset")
    pool_spec = _obj_str(result.get("pool_spec"), f"{where}: result.pool_spec")
    datasets = _obj_str_list(config.get("datasets"), f"run {run_id}: config datasets")
    pool_specs = _obj_str_list(config.get("pool_specs"), f"run {run_id}: config pool_specs")
    if dataset not in datasets:
        raise ReportError(f"{where}: dataset {dataset!r} is not in the run's config datasets")
    if pool_spec not in pool_specs:
        raise ReportError(f"{where}: pool spec {pool_spec!r} is not in the run's config pool_specs")
    cache_id = _obj_str(result.get("cache_id"), f"{where}: result.cache_id")
    simple = _obj_dict(config.get("simple_cache_ids"), f"run {run_id}: config simple_cache_ids")
    expected_cache = simple.get(dataset)
    if expected_cache != cache_id:
        raise ReportError(
            f"{where}: cache_id {cache_id!r} != the config's simple cache "
            f"{expected_cache!r} for {dataset!r}"
        )
    foundation = _obj_str_list(
        result.get("foundation_cache_ids"), f"{where}: result.foundation_cache_ids"
    )
    nested = _obj_dict(
        config.get("foundation_cache_ids"), f"run {run_id}: config foundation_cache_ids"
    )
    per_dataset = _obj_dict(
        nested.get(dataset), f"run {run_id}: config foundation_cache_ids[{dataset!r}]"
    )
    expected_foundation = per_dataset.get(pool_spec)
    if expected_foundation != foundation:
        raise ReportError(
            f"{where}: foundation_cache_ids {foundation} != the config's "
            f"{expected_foundation} for {dataset!r}/{pool_spec!r}"
        )


def _series_ids(run_id: str, result: dict[str, Any]) -> list[str]:
    """Return a result's per-series ids, in order."""
    series = result.get("series")
    if not isinstance(series, list):
        raise ReportError(f"run {run_id}: a result's series must be a list, got {series!r}")
    ids: list[str] = []
    for entry in series:
        ids.append(_obj_str(_obj_dict(entry, "a series entry").get("series_id"), "series_id"))
    return ids


def _pools_of(
    root: Path, run_id: str, records: Sequence[Record], config: dict[str, Any], start: RunStart
) -> _RunPools:
    """Group a complete run's dev baseline records by ``(dataset, pool spec)``."""
    read = _blob_reader(root, run_id)
    arena = start.protocol_versions["arena"]
    pools: dict[tuple[str, str], _Pool] = {}
    for record in records:
        if record.stream != "main" or record.kind != "baseline":
            continue
        payload = record.payload
        if not isinstance(payload, Baseline) or payload.per_task_blob is None:
            continue
        blob = _obj_dict(read(payload.per_task_blob), f"run {run_id}: a baseline blob")
        result = _result_of(run_id, blob)
        _check_result(run_id, record, result, config, arena)
        dataset = _obj_str(result.get("dataset"), f"run {run_id}: result.dataset")
        pool_spec = _obj_str(result.get("pool_spec"), f"run {run_id}: result.pool_spec")
        pool = pools.setdefault((dataset, pool_spec), _Pool())
        if payload.updater_id == "oracle":
            if pool.oracle is not None:
                raise ReportError(f"run {run_id}: {dataset}/{pool_spec} has two oracle records")
            pool.oracle = (record, blob)
        elif payload.updater_id in BASELINE_IDS:
            pool.by_baseline.setdefault(payload.updater_id, []).append((record, blob))
    for (dataset, pool_spec), pool in pools.items():
        if pool.oracle is None:
            raise ReportError(f"run {run_id}: {dataset}/{pool_spec} has replays but no oracle")
        oracle_ids = _series_ids(run_id, _result_of(run_id, pool.oracle[1]))
        for entries in pool.by_baseline.values():
            for record, blob in entries:
                payload = record.payload
                if not isinstance(payload, Baseline) or payload.status != "ok":
                    continue  # a failed result has no series to compare
                ids = _series_ids(run_id, _result_of(run_id, blob))
                if ids != oracle_ids:
                    raise ReportError(
                        f"run {run_id}: {dataset}/{pool_spec} series ids differ between the "
                        f"oracle and record {record.record_id}"
                    )
    selection = select_configs(list(records), read)
    end = _run_end(records)
    summary = end.payload.summary if end is not None and isinstance(end.payload, RunEnd) else {}
    if summary.get("selected") != selection:
        raise ReportError(
            f"run {run_id}: the run_end summary's 'selected' does not equal select_configs "
            "over the run's records"
        )
    return _RunPools(pools=pools, selection=selection)


# --- one pool section-----------------------------------------------------------------------------


def _tuned_rows(
    pool: _Pool,
    picks: dict[str, dict[str, Any] | None],
    *,
    updater_ids: Sequence[str] = BASELINE_IDS,
) -> tuple[TunedRow, ...]:
    """Build the tuned rows, in ``updater_ids`` order, from one pool's records.

    The baselines' ids for a tune run's pools; the learned ids
    (``LEARNED_IDS``) for a probe run's (TASK_014 PR D).
    """
    rows: list[TunedRow] = []
    for baseline_id in updater_ids:
        entries = pool.by_baseline.get(baseline_id)
        if not entries:
            continue
        failed = 0
        evaluations = 0.0
        wall_seconds = 0.0
        for record, _blob in entries:
            payload = record.payload
            if isinstance(payload, Baseline) and payload.status != "ok":
                failed += 1
            evaluations = evaluations + record.budget.evaluations
            wall_seconds = wall_seconds + record.budget.wall_seconds
        pick = picks.get(baseline_id)
        rows.append(
            TunedRow(
                baseline=baseline_id,
                config=None
                if pick is None
                else _obj_dict(pick.get("config"), f"{baseline_id}'s selected config"),
                mase=None if pick is None else _obj_float(pick.get("mase"), "the selected mase"),
                tried=len(entries),
                failed=failed,
                evaluations=evaluations,
                wall_seconds=wall_seconds,
            )
        )
    return tuple(rows)


def _metric(metrics: Mapping[str, float], key: str, where: str) -> float:
    """Return one oracle metric, refusing when the record does not carry it."""
    value = metrics.get(key)
    if value is None:
        raise ReportError(f"{where} has no {key!r} metric")
    return value


def _oracle_of(run_id: str, pool: _Pool) -> tuple[Baseline, dict[str, Any]]:
    """Return one pool's oracle payload and blob, refusing when it is missing."""
    if pool.oracle is None:
        raise ReportError(f"run {run_id}: a replayed pool has no oracle record")
    payload = pool.oracle[0].payload
    if not isinstance(payload, Baseline):
        raise ReportError(f"run {run_id}: the oracle record must carry a baseline payload")
    return payload, _result_of(run_id, pool.oracle[1])


def _oracle_numbers(run_id: str, pool: _Pool) -> OracleNumbers:
    """Build the Oracles rows from the oracle record and the baseline metrics."""
    payload, result = _oracle_of(run_id, pool)
    dataset = _obj_str(result.get("dataset"), "the oracle's dataset")
    pool_spec = _obj_str(result.get("pool_spec"), "the oracle's pool_spec")
    where = f"{dataset}/{pool_spec}"
    true_best: float | None = None
    for baseline_id in BASELINE_IDS:
        for record, _blob in pool.by_baseline.get(baseline_id, []):
            candidate = record.payload
            if isinstance(candidate, Baseline) and candidate.status == "ok":
                value = candidate.metrics.get("oracle_mase")
                if value is not None:
                    true_best = value
                    break
        if true_best is not None:
            break
    switches = tuple(
        _metric(payload.metrics, f"switch_mase_m{m}", f"the oracle record of {where}")
        for m in ORACLE_SWITCHES
    )
    return OracleNumbers(
        true_best_mase=true_best,
        global_expert=_obj_str(result.get("global_expert"), f"{where}: global_expert"),
        global_expert_mase=_metric(payload.metrics, "global_expert_mase", f"the oracle of {where}"),
        best_expert_mase=_metric(payload.metrics, "best_expert_mase", f"the oracle of {where}"),
        switch_mase=switches,
        select_mase=_metric(payload.metrics, "select_mase", f"the oracle of {where}"),
    )


def _max_series_wall_seconds(run_id: str, pool: _Pool) -> float | None:
    """The largest ``series_wall_seconds`` entry over all the pool's replays."""
    largest: float | None = None
    for entries in pool.by_baseline.values():
        for record, blob in entries:
            where = f"run {run_id} record {record.record_id}"
            walls = _obj_float_list(
                blob.get("series_wall_seconds"), f"{where}: series_wall_seconds"
            )
            for wall in walls:
                if largest is None or wall > largest:
                    largest = wall
    return largest


def _labeled_differences(
    baseline_mase: Sequence[float],
    oracle_true_best: Sequence[float | None],
    expert_rows: Sequence[Sequence[float]],
    switch_rows: Sequence[Sequence[float]],
    select_values: Sequence[float],
    global_index: int,
    global_expert: str,
) -> list[tuple[str, list[float]]]:
    """The per-series differences ``d_s = baseline_mase_s - oracle_s``, one row per oracle."""
    labeled: list[tuple[str, list[float]]] = []
    if len(oracle_true_best) > 0 and all(value is not None for value in oracle_true_best):
        true_best_row: list[float] = []
        for i in range(len(baseline_mase)):
            value = oracle_true_best[i]
            if value is not None:  # always true here; keeps the subtraction typed
                true_best_row.append(baseline_mase[i] - value)
        labeled.append(("true-best forecast", true_best_row))
    labeled.append(
        (
            f"best fixed expert ({global_expert})",
            [baseline_mase[i] - expert_rows[i][global_index] for i in range(len(baseline_mase))],
        )
    )
    labeled.append(
        (
            "best expert per series",
            [baseline_mase[i] - _lowest(expert_rows[i]) for i in range(len(baseline_mase))],
        )
    )
    for index, m in enumerate(ORACLE_SWITCHES):
        labeled.append(
            (
                f"switch_mase_m{m}",
                [baseline_mase[i] - switch_rows[i][index] for i in range(len(baseline_mase))],
            )
        )
    labeled.append(
        (
            "select_mase",
            [baseline_mase[i] - select_values[i] for i in range(len(baseline_mase))],
        )
    )
    return labeled


def _best_baseline(
    dataset: str, pool_spec: str, selection: Selection
) -> tuple[str, dict[str, Any]] | None:
    """The baseline with the lowest selected dev MASE; an exact tie keeps the earlier id."""
    picks = selection.get(dataset, {}).get(pool_spec, {})
    best: tuple[str, dict[str, Any]] | None = None
    best_mase = 0.0
    for baseline_id in BASELINE_IDS:
        pick = picks.get(baseline_id)
        if pick is None:
            continue
        mase = _obj_float(pick.get("mase"), f"{baseline_id}'s selected mase")
        if best is None or mase < best_mase:
            best = (baseline_id, pick)
            best_mase = mase
    return best


def _build_section(
    run_id: str,
    dataset: str,
    pool_spec: str,
    pool: _Pool,
    selection: Selection,
    bootstrap_seed: int,
) -> PoolSection:
    """Compute everything one ``(dataset, pool spec)`` section shows."""
    _oracle_payload, oracle_result = _oracle_of(run_id, pool)
    oracles = _oracle_numbers(run_id, pool)
    picks = selection.get(dataset, {}).get(pool_spec, {})
    tuned = _tuned_rows(pool, picks)
    gaps: tuple[GapRow, ...] | None = None
    family_gaps: tuple[FamilyGaps, ...] = ()
    best = _best_baseline(dataset, pool_spec, selection)
    if best is not None:
        best_id, pick = best
        record_id = _obj_str(pick.get("record_id"), f"{best_id}'s selected record_id")
        found = next(
            (
                (record, blob)
                for record, blob in pool.by_baseline.get(best_id, [])
                if record.record_id == record_id
            ),
            None,
        )
        if found is None:
            raise ReportError(
                f"run {run_id}: {dataset}/{pool_spec} selection names record {record_id}, "
                f"which is not a {best_id} record of this run"
            )
        best_result = _result_of(run_id, found[1])
        where = f"run {run_id} record {record_id}"
        best_series = best_result.get("series")
        if not isinstance(best_series, list):
            raise ReportError(f"{where}: the result's series must be a list")
        baseline_mase: list[float] = []
        oracle_true_best: list[float | None] = []
        for entry in best_series:
            scored = _obj_dict(entry, f"{where}: a series entry")
            baseline_mase.append(_obj_float(scored.get("mase"), f"{where}: series mase"))
            true_best = scored.get("oracle_mase")
            if true_best is not None:
                true_best = _obj_float(true_best, f"{where}: series oracle_mase")
            oracle_true_best.append(true_best)
        oracle_series = oracle_result.get("series")
        if not isinstance(oracle_series, list):
            raise ReportError(f"run {run_id}: the oracle's series must be a list")
        expert_rows: list[list[float]] = []
        switch_rows: list[list[float]] = []
        select_values: list[float] = []
        for entry in oracle_series:
            per_series = _obj_dict(entry, f"run {run_id}: an oracle series entry")
            expert_rows.append(_obj_float_list(per_series.get("expert_mase"), "expert_mase"))
            switch_rows.append(_obj_float_list(per_series.get("switch_mase"), "switch_mase"))
            select_values.append(_obj_float(per_series.get("select_mase"), "select_mase"))
        expert_names = _obj_str_list(oracle_result.get("expert_names"), "expert_names")
        global_expert = _obj_str(oracle_result.get("global_expert"), "global_expert")
        if global_expert not in expert_names:
            raise ReportError(
                f"run {run_id}: global expert {global_expert!r} is not one of {expert_names}"
            )
        labeled = _labeled_differences(
            baseline_mase,
            oracle_true_best,
            expert_rows,
            switch_rows,
            select_values,
            expert_names.index(global_expert),
            global_expert,
        )
        gaps = _gap_rows(labeled, bootstrap_seed)
        if dataset == _SYNTHETIC_DATASET:
            series_ids = _series_ids(run_id, best_result)
            by_family: dict[str, list[int]] = {}
            for index, series_id in enumerate(series_ids):
                by_family.setdefault(series_id.split("/", 1)[0], []).append(index)
            tables: list[FamilyGaps] = []
            for family in FAMILIES:
                indexes = by_family.get(family)
                if not indexes:
                    continue
                family_labeled = [
                    (label, [differences[i] for i in indexes]) for label, differences in labeled
                ]
                tables.append(
                    FamilyGaps(family=family, rows=_gap_rows(family_labeled, bootstrap_seed))
                )
            family_gaps = tuple(tables)
    return PoolSection(
        dataset=dataset,
        pool_spec=pool_spec,
        leakage_risk=_obj_bool(oracle_result.get("leakage_risk"), "leakage_risk"),
        tuned=tuned,
        oracles=oracles,
        gaps=gaps,
        family_gaps=family_gaps,
        max_series_wall_seconds=_max_series_wall_seconds(run_id, pool),
    )


# --- the entry point------------------------------------------------------------------------------


def build_report_data(
    ledger_root: Path, run_ids: Sequence[str], *, bootstrap_seed: int
) -> ReportData:
    """Read the named runs and compute the report's numbers from the ledger alone.

    Raises :class:`ReportError` on any refusal the report's rules define:
    a duplicate or unloadable run id, a sealed-stream record, a config that
    is not a Gate 1 experiment, a protocol-version mix across runs, a
    result that disagrees with its run's config, mismatched series ids, a
    ``run_end`` selection the records do not reproduce, or two complete
    runs over one ``(dataset, pool spec)``.
    """
    if not run_ids:
        raise ReportError("at least one run id is required")
    seen: set[str] = set()
    for run_id in run_ids:
        if run_id in seen:
            raise ReportError(f"duplicate run id {run_id!r} in {list(run_ids)}")
        seen.add(run_id)
    root = Path(ledger_root)
    rows: list[RunRow] = []
    sections: list[PoolSection] = []
    leakage_sections: list[PoolSection] = []
    claimed: dict[tuple[str, str], str] = {}
    versions: dict[str, str] | None = None
    versions_run = ""
    for run_id in run_ids:
        records = _read_run(root, run_id)
        start, config = _run_start(root, run_id, records)
        if start.condition == PROBE_CONDITION:
            raise ReportError(
                f"run {run_id}: a {PROBE_CONDITION} run is named with --probe-run, not --run"
            )
        if versions is None:
            versions = dict(start.protocol_versions)
            versions_run = run_id
        elif dict(start.protocol_versions) != versions:
            raise ReportError(
                f"protocol versions differ: run {versions_run} has {versions}, "
                f"run {run_id} has {dict(start.protocol_versions)}"
            )
        end = _run_end(records)
        status = (
            end.payload.status
            if end is not None and isinstance(end.payload, RunEnd)
            else "incomplete"
        )
        rows.append(
            RunRow(
                run_id=run_id,
                condition=start.condition,
                status=status,
                git_commit=start.git_commit,
                records=len(records),
                violations=tuple(validate_run(records)),
            )
        )
        if status != "completed":
            continue  # listed in the Runs rows, never tabled
        run_pools = _pools_of(root, run_id, records, config, start)
        for dataset in _obj_str_list(config.get("datasets"), f"run {run_id}: config datasets"):
            for pool_spec in _obj_str_list(
                config.get("pool_specs"), f"run {run_id}: config pool_specs"
            ):
                pool = run_pools.pools.get((dataset, pool_spec))
                if pool is None:
                    continue
                owner = claimed.get((dataset, pool_spec))
                if owner is not None:
                    raise ReportError(
                        f"two complete runs cover {dataset}/{pool_spec}: {owner} and {run_id}"
                    )
                claimed[(dataset, pool_spec)] = run_id
                section = _build_section(
                    run_id, dataset, pool_spec, pool, run_pools.selection, bootstrap_seed
                )
                if section.leakage_risk:
                    leakage_sections.append(section)
                else:
                    sections.append(section)
    assert versions is not None  # run_ids is non-empty, so one run set them
    return ReportData(
        run_rows=tuple(rows),
        protocol_versions=versions,
        sections=tuple(sections),
        leakage_sections=tuple(leakage_sections),
        series_timeout_seconds=SERIES_TIMEOUT_SECONDS,
    )
