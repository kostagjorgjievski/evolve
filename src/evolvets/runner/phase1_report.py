"""Phase 1 report: the deterministic Markdown rendering of the pilot's data.

``build_phase1_report`` renders a :class:`Phase1ReportData` — every refusal
and every number lives in ``build_phase1_report_data`` — into the document
TASK_019 fixes: what the pilot is and where every number regenerates from,
one line per named run (every run that started is in the report, AGENTS.md
2.8), one Conditions table in registry order with the per-compute columns,
the series-bootstrap gain intervals, the self-model calibration, the
safe-inheritance chain, the replay audit, the runs not tabled with their
reasons, and the fixed Notes. Every float is ``f"{x:.4f}"``, counts and
tokens are ints, and ``None`` renders ``n/a``, so the same ledger and seed
give byte-identical Markdown wherever the report is regenerated.

Statuses render where the data carries them: the condition rows' own
``run_end`` status, and the untabled runs' reasons. A tabled holdout or
replay shows no status cell — the data holds none, and the report states
nothing it did not read.
"""

from __future__ import annotations

from collections.abc import Sequence

from evolvets.runner.phase1_report_data import Phase1ReportData

__all__ = ["build_phase1_report"]

#: The Notes sentences, fixed; the start-values one is conditional and
#: rendered separately, for it appears only when the values disagree.
_NOTES = (
    "- One seed, one run per condition, so cross-condition differences carry no "
    "seed-level uncertainty.",
    "- `budget_exhausted` is the designed end of a lineage under its evaluation "
    "limit, tabled like `completed`.",
    "- `weights_only` starts from the weights reference, so its starting MASE is "
    "not the other conditions' and each gain is measured against its own "
    "condition's start.",
    "- `random_param` proposes without LLM tokens, so its gain per 1M tokens is `n/a`.",
    "- The safe-inheritance rate is a share of accepted edits, never of evaluations.",
)


def _fmt(value: float) -> str:
    """Format one number the way every number in the report is formatted."""
    return f"{value:.4f}"


def _opt(value: float | None) -> str:
    """Format one optional float; ``None`` renders ``n/a``."""
    return "n/a" if value is None else _fmt(value)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render one Markdown table: a header row, a rule, then the rows."""
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _interval_cell(mean: float | None, lower: float | None, upper: float | None) -> list[str]:
    """The gain-interval cells: mean and bounds, each ``n/a`` when there are none."""
    if lower is None or upper is None:
        return [_opt(mean), "n/a"]
    return [_opt(mean), f"[{_fmt(lower)}, {_fmt(upper)}]"]


def _conditions_table(data: Phase1ReportData) -> list[str]:
    """The Conditions table: one row per condition, registry order, all columns."""
    return _table(
        (
            "condition",
            "interface",
            "status",
            "proposals",
            "predictions",
            "evaluations (charged)",
            "accepted",
            "start MASE",
            "best MASE",
            "gain",
            "gain/evaluation",
            "gain per 1M tokens",
            "gain per wall hour",
            "input tokens",
            "output tokens",
            "wall seconds",
        ),
        [
            (
                row.condition,
                row.interface,
                row.status,
                str(row.proposals),
                str(row.predictions),
                str(row.evaluations_charged),
                str(row.accepted),
                _fmt(row.starting_mase),
                _fmt(row.best_mase),
                _fmt(row.gain),
                _fmt(row.gain_per_evaluation),
                _opt(row.gain_per_million_tokens),
                _fmt(row.gain_per_wall_hour),
                str(row.input_tokens),
                str(row.output_tokens),
                _fmt(row.wall_seconds),
            )
            for row in data.rows
        ],
    )


def _gain_table(data: Phase1ReportData) -> list[str]:
    """The gain-interval table: one row per condition's series bootstrap."""
    rows: list[tuple[str, ...]] = []
    for row in data.rows:
        interval = row.gain_interval
        rows.append(
            (
                row.condition,
                *_interval_cell(
                    None if interval is None else interval.mean,
                    None if interval is None else interval.lower,
                    None if interval is None else interval.upper,
                ),
            )
        )
    return _table(("condition", "mean", "95% interval"), rows)


def _calibration_table(data: Phase1ReportData) -> list[str]:
    """The calibration table: counts, coverages and means per condition."""
    return _table(
        (
            "condition",
            "predictions",
            "realized",
            "not-ok evaluated",
            "never evaluated",
            "coverage 90",
            "coverage 50",
            "mean q50",
            "mean realized gain",
            "mean q50 error",
        ),
        [
            (
                row.condition,
                str(row.predictions),
                str(row.realized),
                str(row.not_ok_evaluated),
                str(row.never_evaluated),
                _opt(row.coverage_90),
                _opt(row.coverage_50),
                _opt(row.mean_q50),
                _opt(row.mean_gain),
                _opt(row.mean_q50_error),
            )
            for row in data.calibration
        ],
    )


def _inheritance_table(data: Phase1ReportData) -> list[str]:
    """The safe-inheritance table: one row per condition with a holdout run."""
    return _table(
        (
            "condition",
            "accepted",
            "judged",
            "hurt",
            "safe-inheritance rate",
            "unjudgeable",
            "not-ok rows",
            "start holdout MASE",
            "final holdout MASE",
            "holdout gain",
        ),
        [
            (
                row.condition,
                str(row.accepted),
                str(row.judged),
                str(row.hurt),
                _opt(row.safe_rate),
                str(row.unjudgeable),
                str(row.not_ok_rows),
                _opt(row.start_holdout_mase),
                _opt(row.final_holdout_mase),
                _opt(row.holdout_gain),
            )
            for row in data.inheritance
        ],
    )


def _runs_lines(data: Phase1ReportData) -> list[str]:
    """One line per named tabled run: the conditions with their holdouts, the replay last.

    The order is the registry's, each condition followed by its holdout's line,
    so the section never follows the ``--run`` flags' order; untabled runs are
    the next section's, with their reasons.
    """
    holdout_of = {row.condition: row.holdout_run_id for row in data.inheritance}
    lines: list[str] = []
    for row in data.rows:
        lines.append(f"- `{row.run_id}`: condition {row.condition} ({row.status})")
        if row.condition in holdout_of:
            lines.append(f"- `{holdout_of[row.condition]}`: holdout of {row.condition}")
    if data.replay is not None:
        lines.append(f"- `{data.replay.run_id}`: replay of `{data.replay.source_run_id}`")
    return lines


def _replay_lines(data: Phase1ReportData) -> list[str]:
    """The replay audit: its row, then the fixed sentences its findings call for."""
    replay = data.replay
    if replay is None:
        return ["No replay run was named."]
    lines = _table(
        (
            "run",
            "source",
            "matches_source_run",
            "timeout status mismatches",
            "input tokens",
            "output tokens",
            "wall seconds",
            "evaluations charged",
        ),
        [
            (
                f"`{replay.run_id}`",
                f"`{replay.source_run_id}`",
                str(replay.matches_source_run).lower(),
                str(replay.timeout_status_mismatches),
                str(replay.input_tokens),
                str(replay.output_tokens),
                _fmt(replay.wall_seconds),
                str(replay.evaluations_charged),
            )
        ],
    )
    lines.append("")
    lines.append("Tokens are zero by construction; the evaluations are honestly re-charged.")
    if replay.matches_source_run:
        lines.append("`matches_source_run` true means the aligned records agree bit-for-bit.")
    else:
        lines.append(
            "`matches_source_run` is false: an audit finding (temperature noise around "
            "one prompt), stated, not repaired."
        )
    if replay.timeout_status_mismatches > 0:
        lines.append(
            f"`timeout_status_mismatches` is {replay.timeout_status_mismatches}: an audit "
            "finding (a timeout that did not reproduce), stated, not repaired."
        )
    return lines


def build_phase1_report(data: Phase1ReportData) -> str:
    """Render the Phase 1 pilot report of ``data``, from the ledger alone.

    The document depends only on the data (hence on the ledger contents and
    the bootstrap seed); nothing here reads a file, a clock or the
    environment.
    """
    commits = ", ".join(f"`{commit}`" for commit in data.git_commits)
    lines = [
        "# Phase 1 report",
        "",
        "## What this is",
        "",
        f"One lineage per named condition on `{data.dataset}`, pool `{data.pool_spec}`, "
        f"seed {data.seed}, under identical budget limits.",
        "The gain columns divide by the ledger `budget_spent` sums.",
        f"Every number regenerates from these ledger records at git commits {commits}.",
        "",
        "## Runs",
        "",
        *_runs_lines(data),
        "",
        "## Conditions",
        "",
    ]
    lines.extend(_conditions_table(data))
    lines.extend(
        [
            "",
            "## Gain uncertainty (series bootstrap)",
            "",
            "These are series-resampling intervals over the dev pool, not seed replicates.",
            "",
        ]
    )
    lines.extend(_gain_table(data))
    lines.extend(
        [
            "",
            "## Self-model calibration",
            "",
            "Coverage is only measurable where an evaluation happened.",
            "",
        ]
    )
    lines.extend(_calibration_table(data))
    lines.extend(["", "## Safe inheritance (holdout)", ""])
    if data.missing_holdout:
        lines.append(f"No holdout run named for: {', '.join(data.missing_holdout)}.")
        lines.append("")
    lines.extend(_inheritance_table(data))
    lines.extend(["", "## Replay audit", "", *_replay_lines(data), "", "## Runs not tabled", ""])
    if data.untabled:
        lines.extend(f"- `{run_id}`: {reason}" for run_id, reason in data.untabled)
    else:
        lines.append("None.")
    notes = [*_NOTES]
    if len(data.distinct_forecast_start_mase) > 1:
        values = ", ".join(_fmt(value) for value in data.distinct_forecast_start_mase)
        notes.insert(
            2,
            f"- The forecast-v2 conditions' starting MASE values are not bit-equal: {values}.",
        )
    lines.extend(["", "## Notes", "", *notes, ""])
    return "\n".join(lines)
