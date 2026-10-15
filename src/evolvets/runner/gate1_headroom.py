"""Headroom scan data: how far each pool sits above its best tuned baseline.

:func:`build_headroom_data` (TASK_015 PR B) reads the named tune runs
exactly as the Gate 1 report does — it calls ``build_report_data`` first, so
every one of its refusals applies unchanged with the same message, and the
run rows and protocol versions are copied from that result — and then, for
each pool of each completed run, in the report's order, measures each
hindsight target's headroom over the pool's best tuned baseline:

    headroom = (best tuned baseline MASE - target MASE) / best tuned baseline MASE

summed over the pool's series, with a paired bootstrap interval over series
computed as the report's Gaps intervals are (2000 resamples, indexes 50 and
1949) but of the ratio itself. The targets are the switching oracles with 8
and 32 switches and the best expert per series, plus — on synthetic pools
only, where every series carries ``oracle_mase`` — the true-best forecast.
Sections are ranked by the lower end of the switch-8 headroom interval,
separately for leakage-risk and other pools, ties broken by the headroom
itself; a pool whose every baseline failed has no best baseline, so it is
listed among the unranked and never ranked. :func:`render_headroom` (PR B's
rendering half) turns that data into the fixed Markdown layout — the ranked
tables with the per-pool detail behind them — borrowing the Gate 1 report's
float and table formatting so the two reports cannot drift. The command line
around both halves lives in ``runner.cli``, as the report's does.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evolvets.runner.gate1_report import _VERSION_KEYS, _fmt, _table
from evolvets.runner.gate1_report_data import (
    _INTERVAL_HIGH_INDEX,
    _INTERVAL_LOW_INDEX,
    BOOTSTRAP_RESAMPLES,
    ReportError,
    RunRow,
    _best_baseline,
    _labeled_targets,
    _loop_mean,
    _loop_sum,
    _obj_bool,
    _obj_dict,
    _obj_float,
    _obj_float_list,
    _obj_str,
    _obj_str_list,
    _oracle_of,
    _PoolRef,
    _read_named_runs,
    _result_of,
    _selected_entry,
    _series_ids,
    build_report_data,
)

#: The targets every pool is measured against, in table order: the switching
#: oracles with 8 and 32 switches, the best expert per series, and — synthetic
#: pools only — the true-best forecast. The labels are the Gaps table's.
HEADROOM_TARGETS: tuple[str, ...] = (
    "switch_mase_m8",
    "switch_mase_m32",
    "best expert per series",
    "true-best forecast",
)
#: The row the ranking sorts by: the switch-8 oracle's headroom.
RANK_TARGET = "switch_mase_m8"


# --- the dataclasses the rendering half shows -----------------------------------------------------


@dataclass(frozen=True)
class HeadroomRow:
    """One target's headroom from the best tuned baseline, over one pool's series.

    ``target_mase`` and ``gap`` are loop-means of ``t_s`` and ``d_s = b_s -
    t_s``; ``gap`` equals the Gate 1 report's ``GapRow.mean`` for the same
    label. ``lower`` and ``upper`` are ``None`` with a single series: there
    is nothing to resample.
    """

    target: str  # a HEADROOM_TARGETS label, as the Gaps table names it
    target_mase: float  # loop-mean of t_s
    gap: float  # loop-mean of b_s - t_s, equal to the Gate 1 report's GapRow.mean
    headroom: float  # (B - T) / B, where B and T are the loop-sums of b_s and t_s
    lower: float | None
    upper: float | None  # None with a single series


@dataclass(frozen=True)
class HeadroomSection:
    """Everything the headroom scan says about one ``(dataset, pool spec)``.

    ``rows`` is empty when no baseline has an ok configuration on this pool:
    there is no best tuned baseline to measure from, and the section is
    listed among the unranked rather than ranked.
    """

    run_id: str
    dataset: str
    pool_spec: str
    leakage_risk: bool
    series: int
    best_baseline: str | None
    best_baseline_mase: float | None
    replay_wall_seconds: float | None  # budget.wall_seconds of the best baseline's selected record
    tune_wall_seconds: float  # loop-sum of budget.wall_seconds over the pool's baseline records
    rows: tuple[HeadroomRow, ...]  # HEADROOM_TARGETS order; () without a best baseline


@dataclass(frozen=True)
class HeadroomData:
    """Every number the headroom report shows, computed from the ledger alone."""

    run_rows: tuple[RunRow, ...]
    protocol_versions: dict[str, str]
    ranked: tuple[HeadroomSection, ...]  # leakage_risk False, with rows, ranked
    leakage_ranked: tuple[HeadroomSection, ...]  # leakage_risk True, with rows, ranked
    unranked: tuple[HeadroomSection, ...]  # no best baseline, in config order


# --- the headroom arithmetic ----------------------------------------------------------------------


def _headroom_ratio(baseline_total: float, target_total: float) -> float:
    """Return ``(B - T) / B``, refusing a zero baseline: headroom is a share of it."""
    if baseline_total == 0.0:
        raise ReportError("headroom undefined: zero baseline MASE")
    return (baseline_total - target_total) / baseline_total


def _headroom_rows(
    baseline_mase: Sequence[float],
    targets: Sequence[tuple[str, Sequence[float]]],
    *,
    bootstrap_seed: int,
) -> tuple[HeadroomRow, ...]:
    """One section's rows, in ``HEADROOM_TARGETS`` order, one bootstrap per section.

    One ``random.Random(bootstrap_seed)`` serves the whole section, in row
    order; each of its 2000 resamples draws ``n`` series indexes and takes
    the ratio of the resampled sums, so a resample whose baseline sum is
    zero refuses exactly as a zero baseline does.
    """
    by_label = {label: values for label, values in targets}
    n = len(baseline_mase)
    rng = random.Random(bootstrap_seed)
    rows: list[HeadroomRow] = []
    for target in HEADROOM_TARGETS:
        values = by_label.get(target)
        if values is None:
            continue  # "true-best forecast" on a real pool: no series carries it
        differences = [baseline_mase[i] - values[i] for i in range(n)]
        headroom = _headroom_ratio(_loop_sum(baseline_mase), _loop_sum(values))
        lower: float | None = None
        upper: float | None = None
        if n > 1:
            resamples: list[float] = []
            for _resample in range(BOOTSTRAP_RESAMPLES):
                baseline_total = 0.0
                target_total = 0.0
                for _draw in range(n):
                    index = rng.randrange(n)
                    baseline_total = baseline_total + baseline_mase[index]
                    target_total = target_total + values[index]
                resamples.append(_headroom_ratio(baseline_total, target_total))
            resamples.sort()
            lower = resamples[_INTERVAL_LOW_INDEX]
            upper = resamples[_INTERVAL_HIGH_INDEX]
        rows.append(
            HeadroomRow(
                target=target,
                target_mase=_loop_mean(values),
                gap=_loop_mean(differences),
                headroom=headroom,
                lower=lower,
                upper=upper,
            )
        )
    return tuple(rows)


def _rank(sections: Sequence[HeadroomSection]) -> tuple[HeadroomSection, ...]:
    """Rank sections by descending switch-8 headroom, lower bound first; ties keep order.

    The lower bound is the ranking key because a pool earns its place by
    what its interval promises, not by its point estimate; a single-series
    section has no interval, so it ranks by its headroom. The second key
    breaks a tie on the bound, and the stable sort keeps the report's order
    on a full tie.
    """

    def key(section: HeadroomSection) -> tuple[float, float]:
        row = next(row for row in section.rows if row.target == RANK_TARGET)
        return (row.headroom if row.lower is None else row.lower, row.headroom)

    return tuple(sorted(sections, key=key, reverse=True))


# --- one pool section -----------------------------------------------------------------------------


def _build_headroom_section(ref: _PoolRef, *, bootstrap_seed: int) -> HeadroomSection:
    """Compute one pool's headroom section: its best baseline's rows, or none."""
    run_id = ref.run_id
    _oracle_payload, oracle_result = _oracle_of(run_id, ref.pool)
    tune_wall_seconds = 0.0
    for entries in ref.pool.by_baseline.values():
        for record, _blob in entries:
            tune_wall_seconds = tune_wall_seconds + record.budget.wall_seconds
    series = len(_series_ids(run_id, oracle_result))
    rows: tuple[HeadroomRow, ...] = ()
    best_baseline: str | None = None
    best_mase: float | None = None
    replay_wall_seconds: float | None = None
    best = _best_baseline(ref.dataset, ref.pool_spec, ref.selection)
    if best is not None:
        best_id, pick = best
        record, blob = _selected_entry(run_id, ref.dataset, ref.pool_spec, ref.pool, best)
        best_baseline = best_id
        best_mase = _obj_float(pick.get("mase"), f"{best_id}'s selected mase")
        replay_wall_seconds = record.budget.wall_seconds
        best_result = _result_of(run_id, blob)
        where = f"run {run_id} record {record.record_id}"
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
        rows = _headroom_rows(
            baseline_mase,
            _labeled_targets(
                oracle_true_best,
                expert_rows,
                switch_rows,
                select_values,
                expert_names.index(global_expert),
                global_expert,
            ),
            bootstrap_seed=bootstrap_seed,
        )
        series = len(baseline_mase)
    return HeadroomSection(
        run_id=run_id,
        dataset=ref.dataset,
        pool_spec=ref.pool_spec,
        leakage_risk=_obj_bool(oracle_result.get("leakage_risk"), "leakage_risk"),
        series=series,
        best_baseline=best_baseline,
        best_baseline_mase=best_mase,
        replay_wall_seconds=replay_wall_seconds,
        tune_wall_seconds=tune_wall_seconds,
        rows=rows,
    )


# --- the entry point ------------------------------------------------------------------------------


def build_headroom_data(
    ledger_root: Path, run_ids: Sequence[str], *, bootstrap_seed: int
) -> HeadroomData:
    """Read the named runs and compute the headroom scan's numbers from the ledger alone.

    The Gate 1 report is built first, so every refusal of
    :func:`~evolvets.runner.gate1_report_data.build_report_data` applies
    unchanged, with the same :class:`ReportError` message; the run rows and
    protocol versions are that result's. Each pool of each completed run
    then becomes one section, in the report's order, ranked as the module
    docstring describes.
    """
    report = build_report_data(ledger_root, run_ids, bootstrap_seed=bootstrap_seed)
    _rows, _versions, pools = _read_named_runs(Path(ledger_root), run_ids)
    sections = [_build_headroom_section(ref, bootstrap_seed=bootstrap_seed) for ref in pools]
    with_rows = [section for section in sections if section.rows]
    return HeadroomData(
        run_rows=report.run_rows,
        protocol_versions=report.protocol_versions,
        ranked=_rank([section for section in with_rows if not section.leakage_risk]),
        leakage_ranked=_rank([section for section in with_rows if section.leakage_risk]),
        unranked=tuple(section for section in sections if not section.rows),
    )


# --- the rendering (PR B's second half) -----------------------------------------------------------


#: The Notes sentences, fixed: what headroom is and how its intervals are
#: drawn, where the numbers were computed, what the oracles do and do not
#: bound, what the cost columns count, and how the ranking sorts. The first
#: line's minus sign is U+2212, as TASK_015 writes it (escaped for RUF001).
_NOTES = (
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

#: The ranked tables' columns: the pool's identity, the best tuned baseline
#: and its MASE, one headroom column per target with the switch-8 row's
#: interval beside it (the ranking's key), and the two cost columns.
_RANKED_HEADER = (
    "rank",
    "dataset",
    "pool",
    "series",
    "best tuned baseline",
    "baseline MASE",
    "switch-8 headroom",
    "95% interval",
    "switch-32 headroom",
    "best-per-series headroom",
    "true-best headroom",
    "replay seconds",
    "tune seconds",
)


def _interval(row: HeadroomRow) -> str:
    """The interval cell: ``n/a`` with a single series, there being nothing to draw."""
    if row.lower is None or row.upper is None:
        return "n/a"
    return f"[{_fmt(row.lower)}, {_fmt(row.upper)}]"


def _row_of(section: HeadroomSection, target: str) -> HeadroomRow | None:
    """One section's row for a target, or ``None`` where the pool carries no such row."""
    for row in section.rows:
        if row.target == target:
            return row
    return None


def _runs_table(data: HeadroomData) -> list[str]:
    """The Runs table: every named run, complete or not, with its violations."""
    return _table(
        ("run id", "condition", "status", "git commit", "records", "validate_run"),
        [
            (
                row.run_id,
                row.condition,
                row.status,
                row.git_commit,
                str(row.records),
                "; ".join(row.violations) if row.violations else "none",
            )
            for row in data.run_rows
        ],
    )


def _ranked_table(sections: Sequence[HeadroomSection]) -> list[str]:
    """One ranked table: each section's row, one headroom column per target.

    The interval beside the switch-8 headroom is that row's — the bound the
    ranking sorts by. A section is in a ranked list only by carrying rows, so
    its best baseline and its always-present targets exist; the true-best row
    is the one a real pool lacks, and renders as ``n/a``.
    """
    rows: list[tuple[str, ...]] = []
    for rank, section in enumerate(sections, start=1):
        assert section.best_baseline is not None  # a ranked section carried rows
        assert section.best_baseline_mase is not None
        assert section.replay_wall_seconds is not None  # its selected record exists
        switch8 = _row_of(section, "switch_mase_m8")
        switch32 = _row_of(section, "switch_mase_m32")
        best_per_series = _row_of(section, "best expert per series")
        true_best = _row_of(section, "true-best forecast")
        assert switch8 is not None and switch32 is not None and best_per_series is not None
        rows.append(
            (
                str(rank),
                section.dataset,
                section.pool_spec,
                str(section.series),
                section.best_baseline,
                _fmt(section.best_baseline_mase),
                _fmt(switch8.headroom),
                _interval(switch8),
                _fmt(switch32.headroom),
                _fmt(best_per_series.headroom),
                "n/a" if true_best is None else _fmt(true_best.headroom),
                _fmt(section.replay_wall_seconds),
                _fmt(section.tune_wall_seconds),
            )
        )
    return _table(_RANKED_HEADER, rows)


def _pool_detail(section: HeadroomSection) -> list[str]:
    """One pool's subsection of the per-pool detail: its best baseline and every row."""
    assert section.best_baseline is not None  # only sections with rows are detailed
    assert section.best_baseline_mase is not None
    heading = f"### {section.dataset}, pool {section.pool_spec}"
    if section.leakage_risk:
        heading += " (leakage risk)"
    return [
        heading,
        "",
        f"Best tuned baseline: {section.best_baseline} "
        f"(dev MASE {_fmt(section.best_baseline_mase)}), run {section.run_id}",
        "",
        *_table(
            ("target", "target MASE", "gap", "headroom", "95% interval"),
            [
                (
                    row.target,
                    _fmt(row.target_mase),
                    _fmt(row.gap),
                    _fmt(row.headroom),
                    _interval(row),
                )
                for row in section.rows
            ],
        ),
        "",
    ]


def render_headroom(data: HeadroomData, run_ids: Sequence[str], *, bootstrap_seed: int) -> str:
    """Render the headroom scan of already-built data as fixed Markdown.

    The layout is fixed: the generating command (run ids and bootstrap seed
    only, so the output depends on nothing else), the Runs table, the protocol
    versions, the ranked table (and the leakage-risk one, the heading and
    table only when such a section exists), the pools without a tuned
    baseline (only when there are any), the per-pool detail — ranked sections
    first, then leakage-risk ones, both in rank order — and the fixed Notes.
    """
    runs = " ".join(f"--run {run_id}" for run_id in run_ids)
    command = f"evolvets gate1 headroom {runs} --bootstrap-seed {bootstrap_seed}"
    lines = [
        "# Headroom scan (dev)",
        "",
        f"generated by `{command}`",
        "",
        "## Runs",
        "",
    ]
    lines.extend(_runs_table(data))
    lines.extend(["", "## Protocol versions", ""])
    lines.extend(
        _table(
            ("protocol", "version"), [(key, data.protocol_versions[key]) for key in _VERSION_KEYS]
        )
    )
    lines.extend(["", "## Ranked headroom", ""])
    lines.extend(_ranked_table(data.ranked))
    if data.leakage_ranked:
        lines.extend(["", "## Ranked headroom, leakage-risk pools", ""])
        lines.extend(_ranked_table(data.leakage_ranked))
    if data.unranked:
        lines.extend(["", "## Pools without a tuned baseline", ""])
        lines.extend(
            f"- {section.dataset}, pool {section.pool_spec} (run {section.run_id})"
            for section in data.unranked
        )
    lines.extend(["", "## Per-pool detail", ""])
    for section in (*data.ranked, *data.leakage_ranked):
        lines.extend(_pool_detail(section))
    lines.extend(["## Notes", "", *_NOTES, ""])
    return "\n".join(lines)
