"""The learned-updater probe's data: gains and verdicts from the ledger alone.

:func:`build_probe_data` reads the named tune runs and the probe runs pinned
to them (TASK_014 PR C's ``gate1_probe``) and turns the completed runs'
records into plain dataclasses: one row per probe run — incomplete runs
listed, never tabled — and, for each ``(dataset, pool spec)`` a complete
probe run covers, the best tuned baseline of that pool's tune run, the
learned updaters' tuned rows with their cost columns, and each learned
updater's per-series gain over that baseline with its paired bootstrap
interval and mechanical verdict. Every refusal the report's rules defines
is a :class:`~evolvets.runner.gate1_report_data.ReportError`: an empty or
duplicated probe id, an id named on both sides, a probe run that does not
verify, a sealed-stream record, a run of another condition, a
protocol-version mix, a config that pins a tune run other than the named
ones or a different ``run_end`` record, a named tune run that is not
completed, a result whose pool pins disagree with either config, series ids
that differ from the tune run's oracle, a ``run_end`` selection the records
do not reproduce, two complete probe runs over one pool, or a record that
is not a learned updater's.

The tune runs are read through ``gate1_report_data``'s own helpers, so both
halves of the report refuse the same things about them; the gains reuse
``_gap_rows`` — one freshly seeded table per pool, in ``LEARNED_IDS``
order — so the same ledger and seed give bit-identical data. Nothing here
formats Markdown or parses a command line; that is ``gate1_report``'s half
of the PR.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evolvets.ledger.payloads import Baseline, RunEnd
from evolvets.ledger.schema import Record
from evolvets.ledger.validate import validate_run
from evolvets.runner.gate1_grids import select_configs
from evolvets.runner.gate1_probe import PROBE_CONDITION
from evolvets.runner.gate1_report_data import (
    GapRow,
    ReportError,
    TunedRow,
    _best_baseline,
    _blob_reader,
    _check_result,
    _gap_rows,
    _obj_bool,
    _obj_dict,
    _obj_float,
    _obj_str,
    _obj_str_list,
    _Pool,
    _pools_of,
    _read_run,
    _result_of,
    _run_end,
    _run_start,
    _RunPools,
    _series_ids,
    _tuned_rows,
)
from evolvets.updater.learned import LEARNED_IDS

# --- the dataclasses the report's probe half renders ----------------------------------------------


@dataclass(frozen=True)
class ProbeRunRow:
    """One probe run as the Probe runs table reports it."""

    run_id: str
    status: str  # run_end status, or "incomplete" when the run has no run_end
    git_commit: str
    tune_run_id: str  # the tune run the probe's config pins
    records: int
    violations: tuple[str, ...]  # validate_run output; empty means none


@dataclass(frozen=True)
class GainRow:
    """One learned updater's gain over the best tuned baseline, over one pool's series.

    ``mean`` is the loop-mean of ``d_s = best_baseline_mase_s -
    learned_mase_s``, positive favouring the learned updater; ``lower`` and
    ``upper`` are ``None`` with a single series, and so is the verdict
    ``n/a``.
    """

    updater: str
    mean: float
    lower: float | None
    upper: float | None
    verdict: str  # "beats" | "worse" | "inside" | "n/a"


@dataclass(frozen=True)
class ProbeSection:
    """Everything the probe says about one ``(dataset, pool spec)`` of complete runs.

    ``best_baseline`` and ``best_baseline_mase`` are ``None`` when no
    baseline has an ok configuration on this pool: there is nothing to gain
    against, so ``gains`` is empty. ``gains`` holds one row per learned
    updater with an ok selection, in ``LEARNED_IDS`` order.
    """

    dataset: str
    pool_spec: str
    leakage_risk: bool
    best_baseline: str | None
    best_baseline_mase: float | None
    learned: tuple[TunedRow, ...]  # LEARNED_IDS order, as _tuned_rows builds them
    gains: tuple[GainRow, ...]


@dataclass(frozen=True)
class ProbeData:
    """Every number the report's probe half shows, computed from the ledger alone."""

    run_rows: tuple[ProbeRunRow, ...]
    sections: tuple[ProbeSection, ...]  # leakage_risk == False, in tune config order
    leakage_sections: tuple[ProbeSection, ...]  # leakage_risk == True, in tune config order


# --- internal holders -----------------------------------------------------------------------------


@dataclass
class _TuneRun:
    """One completed tune run as the probe's reference."""

    run_id: str
    config: dict[str, Any]
    run_end_record_id: str
    pools: _RunPools
    oracle_ids: dict[tuple[str, str], list[str]]  # per covered pool, the oracle's series ids


class _ProbePool(_Pool):
    """One complete probe run's records for one ``(dataset, pool spec)``.

    A probe run records no oracle — the oracles are its tune run's — so
    ``oracle`` stays ``None``; ``_tuned_rows`` reads only ``by_baseline``.
    """

    def __init__(self, run_id: str) -> None:
        super().__init__()
        self.run_id = run_id


@dataclass
class _Covered:
    """One probe pool together with its own run's selection picks for it."""

    pool: _ProbePool
    picks: dict[str, dict[str, Any] | None]


def _verdict(row: GapRow) -> str:
    """The mechanical verdict of one gain row: ``beats``, ``worse``, ``inside`` or ``n/a``."""
    if row.lower is None or row.upper is None:
        return "n/a"  # a single series: no interval to draw
    if row.lower > 0.0:
        return "beats"
    if row.upper < 0.0:
        return "worse"
    return "inside"


# --- reading the named runs -----------------------------------------------------------------------


def _read_tune_runs(
    root: Path, tune_run_ids: Sequence[str]
) -> tuple[dict[str, _TuneRun], dict[str, str]]:
    """Read every named tune run as the probe's reference; refuse anything unusable.

    Each must verify (blobs included), name the Gate 1 experiment, end
    ``completed`` and reproduce its own ``run_end`` selection — all through
    ``gate1_report_data``'s readers, so the report's two halves refuse the
    same things — and all must pin one set of protocol versions.
    """
    tunes: dict[str, _TuneRun] = {}
    versions: dict[str, str] | None = None
    versions_run = ""
    for tune_run_id in tune_run_ids:
        records = _read_run(root, tune_run_id)
        start, config = _run_start(root, tune_run_id, records)
        if versions is None:
            versions = dict(start.protocol_versions)
            versions_run = tune_run_id
        elif dict(start.protocol_versions) != versions:
            raise ReportError(
                f"protocol versions differ: run {versions_run} has {versions}, "
                f"run {tune_run_id} has {dict(start.protocol_versions)}"
            )
        end = _run_end(records)
        if end is None or not isinstance(end.payload, RunEnd) or end.payload.status != "completed":
            raise ReportError(
                f"tune run {tune_run_id} is not completed; a probe's baselines are read "
                "from a completed tune run"
            )
        run_pools = _pools_of(root, tune_run_id, records, config, start)
        oracle_ids: dict[tuple[str, str], list[str]] = {}
        for key, pool in run_pools.pools.items():
            if pool.oracle is None:
                continue  # unreachable: _pools_of refuses a replayed pool without an oracle
            oracle_ids[key] = _series_ids(tune_run_id, _result_of(tune_run_id, pool.oracle[1]))
        tunes[tune_run_id] = _TuneRun(
            run_id=tune_run_id,
            config=config,
            run_end_record_id=end.record_id,
            pools=run_pools,
            oracle_ids=oracle_ids,
        )
    assert versions is not None  # tune_run_ids is non-empty, so one run set them
    return tunes, versions


def _pin_against_tune(
    run_id: str,
    config: dict[str, Any],
    dataset: str,
    pool_spec: str,
    tune: _TuneRun,
) -> None:
    """Refuse a probe pool the named tune run did not replay, or replayed from other caches."""
    where = f"run {run_id}: {dataset}/{pool_spec}"
    datasets = _obj_str_list(
        tune.config.get("datasets"), f"tune run {tune.run_id}: config datasets"
    )
    pool_specs = _obj_str_list(
        tune.config.get("pool_specs"), f"tune run {tune.run_id}: config pool_specs"
    )
    if dataset not in datasets:
        raise ReportError(f"{where}: tune run {tune.run_id} does not cover dataset {dataset!r}")
    if pool_spec not in pool_specs:
        raise ReportError(f"{where}: tune run {tune.run_id} does not cover pool spec {pool_spec!r}")
    simple = _obj_dict(config.get("simple_cache_ids"), f"run {run_id}: config simple_cache_ids")
    tune_simple = _obj_dict(
        tune.config.get("simple_cache_ids"), f"tune run {tune.run_id}: config simple_cache_ids"
    )
    if simple.get(dataset) != tune_simple.get(dataset):
        raise ReportError(
            f"{where}: probe config simple cache {simple.get(dataset)!r} != the tune run's "
            f"{tune_simple.get(dataset)!r}"
        )
    nested = _obj_dict(
        config.get("foundation_cache_ids"), f"run {run_id}: config foundation_cache_ids"
    )
    tune_nested = _obj_dict(
        tune.config.get("foundation_cache_ids"),
        f"tune run {tune.run_id}: config foundation_cache_ids",
    )
    per_dataset = _obj_dict(
        nested.get(dataset), f"run {run_id}: config foundation_cache_ids[{dataset!r}]"
    )
    tune_per_dataset = _obj_dict(
        tune_nested.get(dataset),
        f"tune run {tune.run_id}: config foundation_cache_ids[{dataset!r}]",
    )
    if per_dataset.get(pool_spec) != tune_per_dataset.get(pool_spec):
        raise ReportError(
            f"{where}: probe config foundation caches {per_dataset.get(pool_spec)!r} != the "
            f"tune run's {tune_per_dataset.get(pool_spec)!r}"
        )


def _read_probe_run(
    root: Path,
    run_id: str,
    tunes: Mapping[str, _TuneRun],
    versions: dict[str, str],
) -> tuple[ProbeRunRow, dict[tuple[str, str, str], _Covered]]:
    """Read one probe run into its row and its covered pools; refuse anything unusable.

    The run must verify (blobs included), open with a ``run_start`` of the
    probe condition whose config pins one of the named tune runs — by id and
    by ``run_end`` record — and pin the tune runs' protocol versions. A
    completed run's records are then grouped by ``(dataset, pool spec)``:
    every record must be a learned updater's, every result must agree with
    the probe config's pool pins and the tune run's, every ok result's
    series ids must equal the tune run's oracle ids, and the ``run_end``
    selection must reproduce. An incomplete run is only listed.
    """
    records = _read_run(root, run_id)
    start, config = _run_start(root, run_id, records)
    if start.condition != PROBE_CONDITION:
        raise ReportError(
            f"run {run_id}: the first main record's condition is {start.condition!r}, "
            f"not {PROBE_CONDITION!r}"
        )
    if dict(start.protocol_versions) != versions:
        raise ReportError(
            f"protocol versions differ: the tune runs have {versions}, "
            f"probe run {run_id} has {dict(start.protocol_versions)}"
        )
    named = _obj_str(config.get("tune_run_id"), f"run {run_id}: config tune_run_id")
    tune = tunes.get(named)
    if tune is None:
        raise ReportError(
            f"run {run_id}: config names tune run {named!r}, not one of the reported "
            f"tune runs {list(tunes)}"
        )
    pinned_end = _obj_str(
        config.get("tune_run_end_record_id"), f"run {run_id}: config tune_run_end_record_id"
    )
    if pinned_end != tune.run_end_record_id:
        raise ReportError(
            f"run {run_id}: config tune_run_end_record_id {pinned_end!r} != tune run "
            f"{named!r}'s run_end record {tune.run_end_record_id!r}"
        )
    end = _run_end(records)
    end_payload = end.payload if end is not None else None
    status = end_payload.status if isinstance(end_payload, RunEnd) else "incomplete"
    row = ProbeRunRow(
        run_id=run_id,
        status=status,
        git_commit=start.git_commit,
        tune_run_id=named,
        records=len(records),
        violations=tuple(validate_run(records)),
    )
    if status != "completed":
        return row, {}
    read = _blob_reader(root, run_id)
    arena = start.protocol_versions["arena"]
    pools: dict[tuple[str, str], _ProbePool] = {}
    for record in records:
        if record.stream != "main" or record.kind != "baseline":
            continue
        payload = record.payload
        if not isinstance(payload, Baseline) or payload.per_task_blob is None:
            continue
        if payload.updater_id not in LEARNED_IDS:
            raise ReportError(
                f"run {run_id} record {record.record_id}: updater id "
                f"{payload.updater_id!r} is not one of the learned ids {list(LEARNED_IDS)}"
            )
        blob = _obj_dict(read(payload.per_task_blob), f"run {run_id}: a baseline blob")
        result = _result_of(run_id, blob)
        _check_result(run_id, record, result, config, arena)
        key = (
            _obj_str(result.get("dataset"), f"run {run_id}: result.dataset"),
            _obj_str(result.get("pool_spec"), f"run {run_id}: result.pool_spec"),
        )
        pools.setdefault(key, _ProbePool(run_id)).by_baseline.setdefault(
            payload.updater_id, []
        ).append((record, blob))
    selection = select_configs(list(records), read, updater_ids=LEARNED_IDS)
    assert isinstance(end_payload, RunEnd)  # a completed status came from a run_end payload
    if end_payload.summary.get("selected") != selection:
        raise ReportError(
            f"run {run_id}: the run_end summary's 'selected' does not equal select_configs "
            "over the run's records"
        )
    covered: dict[tuple[str, str, str], _Covered] = {}
    for (dataset, pool_spec), pool in pools.items():
        _pin_against_tune(run_id, config, dataset, pool_spec, tune)
        oracle_ids = tune.oracle_ids.get((dataset, pool_spec))
        if oracle_ids is None:
            raise ReportError(
                f"run {run_id}: {dataset}/{pool_spec} has no oracle in its tune run "
                f"{tune.run_id}, so its series cannot be compared"
            )
        for entries in pool.by_baseline.values():
            for record, blob in entries:
                if not isinstance(record.payload, Baseline) or record.payload.status != "ok":
                    continue  # a failed result has no series to compare
                if _series_ids(run_id, _result_of(run_id, blob)) != oracle_ids:
                    raise ReportError(
                        f"run {run_id}: {dataset}/{pool_spec} series ids differ between the "
                        f"tune run's oracle and record {record.record_id}"
                    )
        covered[(named, dataset, pool_spec)] = _Covered(
            pool=pool,
            picks=selection.get(dataset, {}).get(pool_spec, {}),
        )
    return row, covered


# --- one probe section ----------------------------------------------------------------------------


def _selected_series(
    run_id: str,
    updater_id: str,
    record_id: str,
    entries: Sequence[tuple[Record, dict[str, Any]]],
) -> list[float]:
    """One selected record's per-series MASE, in series order; refuse a record that is absent."""
    found = next(
        ((record, blob) for record, blob in entries if record.record_id == record_id),
        None,
    )
    if found is None:
        raise ReportError(
            f"run {run_id}: the selection names record {record_id}, which is not a "
            f"{updater_id} record of this run"
        )
    where = f"run {run_id} record {record_id}"
    series = _result_of(run_id, found[1]).get("series")
    if not isinstance(series, list):
        raise ReportError(f"{where}: the result's series must be a list")
    mase: list[float] = []
    for entry in series:
        scored = _obj_dict(entry, f"{where}: a series entry")
        mase.append(_obj_float(scored.get("mase"), f"{where}: series mase"))
    return mase


def _build_probe_section(
    dataset: str,
    pool_spec: str,
    found: _Covered,
    tune: _TuneRun,
    bootstrap_seed: int,
) -> ProbeSection:
    """Compute everything one probe section shows: the best baseline's gains per learned updater."""
    tune_pool = tune.pools.pools.get((dataset, pool_spec))
    if tune_pool is None or tune_pool.oracle is None:
        raise ReportError(
            f"tune run {tune.run_id}: {dataset}/{pool_spec} has no oracle record, so the "
            "probe over it cannot be compared"
        )
    leakage_risk = _obj_bool(
        _result_of(tune.run_id, tune_pool.oracle[1]).get("leakage_risk"), "leakage_risk"
    )
    best = _best_baseline(dataset, pool_spec, tune.pools.selection)
    best_baseline: str | None = None
    best_mase: float | None = None
    baseline_mase: list[float] | None = None
    if best is not None:
        best_id, pick = best
        best_baseline = best_id
        best_mase = _obj_float(pick.get("mase"), f"{best_id}'s selected mase")
        record_id = _obj_str(pick.get("record_id"), f"{best_id}'s selected record_id")
        baseline_mase = _selected_series(
            tune.run_id, best_id, record_id, tune_pool.by_baseline.get(best_id, [])
        )
    learned = _tuned_rows(found.pool, found.picks, updater_ids=LEARNED_IDS)
    gains: tuple[GainRow, ...] = ()
    if best is not None and baseline_mase is not None:
        labeled: list[tuple[str, list[float]]] = []
        for learned_id in LEARNED_IDS:
            learned_pick = found.picks.get(learned_id)
            if learned_pick is None:
                continue  # no ok configuration: no gain row, the verdict cell reads "none"
            record_id = _obj_str(
                learned_pick.get("record_id"), f"{learned_id}'s selected record_id"
            )
            learned_mase = _selected_series(
                found.pool.run_id,
                learned_id,
                record_id,
                found.pool.by_baseline.get(learned_id, []),
            )
            if len(learned_mase) != len(baseline_mase):
                raise ReportError(
                    f"run {found.pool.run_id} record {record_id}: {len(learned_mase)} series "
                    f"against the best baseline's {len(baseline_mase)}, after both matched "
                    "the oracle's series ids"
                )
            differences = [baseline_mase[i] - learned_mase[i] for i in range(len(baseline_mase))]
            labeled.append((learned_id, differences))
        gains = tuple(
            GainRow(
                updater=label,
                mean=row.mean,
                lower=row.lower,
                upper=row.upper,
                verdict=_verdict(row),
            )
            for (label, _differences), row in zip(
                labeled, _gap_rows(labeled, bootstrap_seed), strict=True
            )
        )
    return ProbeSection(
        dataset=dataset,
        pool_spec=pool_spec,
        leakage_risk=leakage_risk,
        best_baseline=best_baseline,
        best_baseline_mase=best_mase,
        learned=learned,
        gains=gains,
    )


# --- the entry point ------------------------------------------------------------------------------


def build_probe_data(
    ledger_root: Path,
    tune_run_ids: Sequence[str],
    probe_run_ids: Sequence[str],
    *,
    bootstrap_seed: int,
) -> ProbeData:
    """Read the named tune and probe runs and compute the probe's numbers from the ledger alone.

    Raises :class:`ReportError` on any refusal the module docstring lists.
    Incomplete probe runs (aborted, budget-exhausted, or missing their
    ``run_end``) appear in ``run_rows`` and never contribute a section;
    sections cover each ``(dataset, pool spec)`` one complete probe run
    replays, ordered by the tune runs' config order and split by
    ``leakage_risk``.
    """
    if not tune_run_ids:
        raise ReportError("at least one tune run id is required")
    named_ids: set[str] = set()
    for run_id in tune_run_ids:
        if run_id in named_ids:
            raise ReportError(f"duplicate tune run id {run_id!r} in {list(tune_run_ids)}")
        named_ids.add(run_id)
    if not probe_run_ids:
        raise ReportError("at least one probe run id is required")
    probe_seen: set[str] = set()
    for run_id in probe_run_ids:
        if run_id in probe_seen:
            raise ReportError(f"duplicate probe run id {run_id!r} in {list(probe_run_ids)}")
        if run_id in named_ids:
            raise ReportError(f"run id {run_id!r} is named as both a tune run and a probe run")
        probe_seen.add(run_id)
    root = Path(ledger_root)
    tunes, versions = _read_tune_runs(root, tune_run_ids)
    rows: list[ProbeRunRow] = []
    covered: dict[tuple[str, str, str], _Covered] = {}
    claimed: dict[tuple[str, str], str] = {}
    for run_id in probe_run_ids:
        row, pools = _read_probe_run(root, run_id, tunes, versions)
        rows.append(row)
        for (named, dataset, pool_spec), pool in pools.items():
            owner = claimed.get((dataset, pool_spec))
            if owner is not None:
                raise ReportError(
                    f"two complete probe runs cover {dataset}/{pool_spec}: {owner} and {run_id}"
                )
            claimed[(dataset, pool_spec)] = run_id
            covered[(named, dataset, pool_spec)] = pool
    sections: list[ProbeSection] = []
    leakage_sections: list[ProbeSection] = []
    for tune_run_id in tune_run_ids:
        tune = tunes[tune_run_id]
        datasets = _obj_str_list(
            tune.config.get("datasets"), f"tune run {tune_run_id}: config datasets"
        )
        pool_specs = _obj_str_list(
            tune.config.get("pool_specs"), f"tune run {tune_run_id}: config pool_specs"
        )
        for dataset in datasets:
            for pool_spec in pool_specs:
                found = covered.get((tune_run_id, dataset, pool_spec))
                if found is None:
                    continue
                section = _build_probe_section(dataset, pool_spec, found, tune, bootstrap_seed)
                if section.leakage_risk:
                    leakage_sections.append(section)
                else:
                    sections.append(section)
    return ProbeData(
        run_rows=tuple(rows),
        sections=tuple(sections),
        leakage_sections=tuple(leakage_sections),
    )
