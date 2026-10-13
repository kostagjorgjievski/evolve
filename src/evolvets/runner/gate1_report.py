"""Gate 1 dev report: the deterministic Markdown rendering of PR C1's data.

``build_report`` reads the named runs through ``build_report_data`` — every
refusal and every number lives there — and renders the sections in a fixed
order: the generating command (run ids and bootstrap seed only, so the output
depends on nothing else), the Runs table with its ``validate_run``
violations, the protocol versions, one subsection per ``(dataset, pool
spec)`` with the leakage-risk pools split into their own section, the
per-series wall-clock evidence for the ``SERIES_TIMEOUT_SECONDS`` review,
and the fixed Notes. Every float is ``f"{x:.4f}"`` and every config
``canonical_json``, so the same ledger and seed give byte-identical Markdown
wherever it is regenerated.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from evolvets.arena.oracles import ORACLE_SWITCHES
from evolvets.ledger.canonical import canonical_json
from evolvets.runner.gate1_report_data import (
    GapRow,
    OracleNumbers,
    PoolSection,
    ReportData,
    ReportError,
    TunedRow,
    build_report_data,
)

__all__ = ["ReportError", "build_report"]

#: The protocol versions table's rows, in the order ``protocol_versions()`` lists them.
_VERSION_KEYS = ("ledger", "arena", "generator", "cache_manifest")
#: The sentence that introduces the leakage-risk pools.
_LEAKAGE_SENTENCE = "A pooled foundation model may have seen this dataset in pretraining."
#: The Notes sentences, fixed: dev numbers tuned on the same series, oracle
#: bounds that hold for single-expert selection only, and the owner's gate.
_NOTES = (
    "- These are dev numbers: each selected configuration was chosen on these same series, "
    "so the dev gaps favour that baseline's fit.",
    "- The expert-pool oracles bound single-expert selection only and are not a floor "
    "for weighted combinations.",
    "- The Gate 1 decision is the owner's.",
)


def _fmt(value: float) -> str:
    """Format one number the way every number in the report is formatted."""
    return f"{value:.4f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Render one Markdown table: a header row, a rule, then the rows."""
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def _best_tuned(tuned: Sequence[TunedRow]) -> TunedRow | None:
    """The tuned row with the lowest dev MASE; an exact tie keeps the earlier id.

    The rows arrive in ``BASELINE_IDS`` order and ties are resolved by exact
    ``<``, the same rule ``build_report_data`` used when it computed the gaps.
    """
    best: TunedRow | None = None
    best_mase = 0.0
    for row in tuned:
        if row.mase is None:
            continue
        if best is None or row.mase < best_mase:
            best = row
            best_mase = row.mase
    return best


def _tuned_table(section: PoolSection) -> list[str]:
    """The Tuned baselines table: the selection and what tuning it cost."""
    rows = [
        (
            row.baseline,
            "none" if row.config is None else canonical_json(row.config),
            "none" if row.mase is None else _fmt(row.mase),
            str(row.tried),
            str(row.failed),
            _fmt(row.evaluations),
            _fmt(row.wall_seconds),
        )
        for row in section.tuned
    ]
    return _table(
        (
            "baseline",
            "selected config",
            "dev MASE",
            "configurations",
            "failed",
            "evaluations",
            "wall seconds",
        ),
        rows,
    )


def _oracle_table(oracles: OracleNumbers) -> list[str]:
    """The Oracles table: every hindsight bound the arena computes for a pool.

    The true-best forecast is a synthetic-only oracle (only the generator
    knows it), so real pools carry no such row — as they carry no true-best
    gap row.
    """
    rows: list[tuple[str, str]] = []
    if oracles.true_best_mase is not None:
        rows.append(("true-best forecast", _fmt(oracles.true_best_mase)))
    rows.extend(
        [
            (f"best fixed expert ({oracles.global_expert})", _fmt(oracles.global_expert_mase)),
            ("best expert per series", _fmt(oracles.best_expert_mase)),
        ]
    )
    for m, value in zip(ORACLE_SWITCHES, oracles.switch_mase, strict=True):
        rows.append((f"switch_mase_m{m}", _fmt(value)))
    rows.append(("select_mase", _fmt(oracles.select_mase)))
    return _table(("oracle", "dev MASE"), rows)


def _gap_table(rows: Sequence[GapRow]) -> list[str]:
    """One Gaps table: each oracle's mean gap and its 95% paired bootstrap interval."""
    return _table(
        ("oracle", "mean gap", "95% interval"),
        [(row.label, _fmt(row.mean), _interval(row)) for row in rows],
    )


def _interval(row: GapRow) -> str:
    """The interval cell: ``n/a`` with a single series, there being nothing to draw."""
    if row.lower is None or row.upper is None:
        return "n/a"
    return f"[{_fmt(row.lower)}, {_fmt(row.upper)}]"


def _pool_section(section: PoolSection) -> list[str]:
    """Everything one ``(dataset, pool spec)`` subsection shows, tables and gaps."""
    lines = [f"## {section.dataset}, pool {section.pool_spec}", ""]
    lines.append("### Tuned baselines")
    lines.append("")
    lines.extend(_tuned_table(section))
    lines.append("")
    lines.append("### Oracles")
    lines.append("")
    lines.extend(_oracle_table(section.oracles))
    lines.append("")
    best = _best_tuned(section.tuned)
    if section.gaps is None or best is None:
        lines.append("No tuned baseline has an ok configuration on this pool.")
        lines.append("")
    else:
        lines.append(f"### Gaps from the best tuned baseline ({best.baseline})")
        lines.append("")
        lines.extend(_gap_table(section.gaps))
        lines.append("")
    for family in section.family_gaps:
        lines.append(f"### Family gaps: {family.family}")
        lines.append("")
        lines.extend(_gap_table(family.rows))
        lines.append("")
    return lines


def _runs_table(data: ReportData) -> list[str]:
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


def _wall_clock_table(data: ReportData) -> list[str]:
    """The per-series wall-clock table, every pool with its largest series time."""
    timeout = _fmt(data.series_timeout_seconds)
    return _table(
        ("dataset", "pool", "max series wall seconds", "SERIES_TIMEOUT_SECONDS"),
        [
            (
                section.dataset,
                section.pool_spec,
                "n/a"
                if section.max_series_wall_seconds is None
                else _fmt(section.max_series_wall_seconds),
                timeout,
            )
            for section in (*data.sections, *data.leakage_sections)
        ],
    )


def build_report(ledger_root: Path, run_ids: Sequence[str], *, bootstrap_seed: int) -> str:
    """Render the Gate 1 dev report of the named runs, from the ledger alone.

    Every refusal is :class:`ReportError` from ``build_report_data``; the
    returned Markdown depends only on the ledger contents and
    ``bootstrap_seed``.
    """
    data = build_report_data(ledger_root, run_ids, bootstrap_seed=bootstrap_seed)
    runs = " ".join(f"--run {run_id}" for run_id in run_ids)
    lines = [
        "# Gate 1 report (dev)",
        "",
        f"generated by `evolvets gate1 report {runs} --bootstrap-seed {bootstrap_seed}`",
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
    lines.append("")
    for section in data.sections:
        lines.extend(_pool_section(section))
    lines.extend(["## Leakage-risk results", ""])
    if data.leakage_sections:
        lines.extend([_LEAKAGE_SENTENCE, ""])
        for section in data.leakage_sections:
            lines.extend(_pool_section(section))
    else:
        lines.extend(["No leakage-risk results.", ""])
    lines.extend(["## Per-series wall-clock", ""])
    lines.extend(_wall_clock_table(data))
    lines.extend(["", "## Notes", "", *_NOTES, ""])
    return "\n".join(lines)
