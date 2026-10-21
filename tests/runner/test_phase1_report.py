"""The Phase 1 report's rendering and CLI (TASK_019 PR B).

Over the PR A fixtures: the pilot ledger of ``test_phase1_report_data``
(rebuilt once for this module, as each module builds its own) feeds the nine
required tests — byte-identical regeneration and the ``--out``/``--check``
CLI (1), the refusal that writes nothing (2), the sections in their fixed
order with every named run listed (3), the Conditions table's registry
order, formats and ``budget_exhausted`` status (4), the calibration's
shares and ``n/a`` cells (5), the inheritance rates, unjudgeable counts and
the missing-holdout sentence (6), the replay audit in both match cases
(7), the Notes' conditional sentences (8) and the CLI defaults (9). The
numbers themselves are PR A's, tested there against hand counts.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from evolvets.improver.artifacts import REFERENCE_ARTIFACT_SOURCE
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.canonical import canonical_bytes
from evolvets.ledger.payloads import Baseline, Payload
from evolvets.ledger.schema import Record
from evolvets.runner import cli
from evolvets.runner.phase1_report import build_phase1_report
from evolvets.runner.phase1_report_data import build_phase1_report_data
from tests.runner.test_phase1_report_data import (
    Pilot,
    _counts,
    _hand_calibration,
    _ids,
    _lineage,
    _records,
    copy_run,
    scored,
)
from tests.runner.test_phase1_report_data import pilot as _pilot

#: The PR A pilot fixture, under the name pytest resolves it by; the tests'
#: ``pilot`` parameters shadow this binding, so it rides in under an alias.
pilot = _pilot


def _report(pilot: Pilot, run_ids: list[str]) -> str:
    """Render the pilot's report over the named runs at the default seed."""
    return build_phase1_report(build_phase1_report_data(pilot.ledger, run_ids, bootstrap_seed=0))


def _flags(run_ids: list[str]) -> list[str]:
    """The ``--run`` flags of one CLI invocation."""
    return [flag for run_id in run_ids for flag in ("--run", run_id)]


def _table_rows(report: str, heading: str) -> list[list[str]]:
    """One section's Markdown table as rows of stripped cells, header dropped."""
    lines = report.splitlines()
    start = lines.index(f"## {heading}")
    rows: list[list[str]] = []
    for line in lines[start + 1 :]:
        if not line.startswith("|"):
            if rows:
                break  # the table ended
            continue  # the section's opening sentence
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if all(cell == "---" for cell in cells):
            continue
        rows.append(cells)
    return rows[1:]


def _cells(value: float | None) -> str:
    """Format one optional float the way the report renders it."""
    return "n/a" if value is None else f"{value:.4f}"


# --- Test 1: regenerable --------------------------------------------------------------------------


def test_two_builds_are_byte_identical_and_a_copied_ledger_regenerates(
    pilot: Pilot, tmp_path: Path
) -> None:
    """Two builds give identical strings, from this ledger root or a copy of it."""
    first = _report(pilot, _ids(pilot))
    assert _report(pilot, _ids(pilot)) == first
    copied = tmp_path / "ledgers"
    shutil.copytree(pilot.ledger, copied)
    assert (
        build_phase1_report(build_phase1_report_data(copied, _ids(pilot), bootstrap_seed=0))
        == first
    )
    assert first.endswith("\n")


def test_cli_out_writes_and_check_accepts(
    pilot: Pilot, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--out writes the built report; --check against the same file exits 0."""
    out = tmp_path / "report.md"
    args = [
        "phase1",
        "report",
        "--ledger-root",
        str(pilot.ledger),
        *_flags(_ids(pilot)),
    ]
    assert cli.main([*args, "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == _report(pilot, _ids(pilot))
    assert cli.main([*args, "--check", str(out)]) == 0
    assert capsys.readouterr().out.startswith("OK ")


def test_cli_check_names_the_first_differing_line(
    pilot: Pilot, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A one-character change fails the check at that line's number."""
    lines = _report(pilot, _ids(pilot)).splitlines()
    target = next(i for i, line in enumerate(lines) if line.startswith("| predictive |"))
    lines[target] = lines[target][:-1] + ("0" if lines[target].endswith("1") else "1")
    tampered = tmp_path / "tampered.md"
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert (
        cli.main(
            [
                "phase1",
                "report",
                "--ledger-root",
                str(pilot.ledger),
                *_flags(_ids(pilot)),
                "--check",
                str(tampered),
            ]
        )
        == 1
    )
    assert f"differs at line {target + 1}" in capsys.readouterr().out


# --- Test 2: the refusal --------------------------------------------------------------------------


def test_cli_refusal_is_one_fail_line_and_writes_nothing(
    pilot: Pilot, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A duplicate condition among --run flags prints FAIL, exits 1, writes nothing."""
    copy_run(pilot.ledger, "lineage-evaluate-all", "dupe-evaluate-all")
    out = tmp_path / "report.md"
    assert (
        cli.main(
            [
                "phase1",
                "report",
                "--ledger-root",
                str(pilot.ledger),
                *_flags([*_ids(pilot), "dupe-evaluate-all"]),
                "--out",
                str(out),
            ]
        )
        == 1
    )
    assert capsys.readouterr().out.startswith("FAIL ")
    assert not out.exists()


# --- Test 3: sections and order -------------------------------------------------------------------


def test_the_sections_appear_in_the_fixed_order_and_runs_lists_every_run(pilot: Pilot) -> None:
    """The headings sit in the listed order; the Runs block names every named run."""
    report = _report(pilot, _ids(pilot))
    headings = [line for line in report.splitlines() if line.startswith("## ")]
    assert headings == [
        "## What this is",
        "## Runs",
        "## Conditions",
        "## Gain uncertainty (series bootstrap)",
        "## Self-model calibration",
        "## Safe inheritance (holdout)",
        "## Replay audit",
        "## Runs not tabled",
        "## Notes",
    ]
    runs = report[report.index("## Runs") : report.index("## Conditions")]
    for run_id in _ids(pilot):
        assert f"`{run_id}`" in runs


# --- Test 4: the conditions table -----------------------------------------------------------------


def test_the_conditions_table_orders_formats_and_tables_exhaustion(pilot: Pilot) -> None:
    """Registry row order, .4f floats, int counts, n/a tokens, status shown."""
    data = build_phase1_report_data(pilot.ledger, _ids(pilot), bootstrap_seed=0)
    rows = _table_rows(_report(pilot, _ids(pilot)), "Conditions")
    assert [cells[0] for cells in rows] == [row.condition for row in data.rows]
    for cells, row in zip(rows, data.rows, strict=True):
        assert cells[:7] == [
            row.condition,
            row.interface,
            row.status,
            str(row.proposals),
            str(row.predictions),
            str(row.evaluations_charged),  # the charged count, not the record count
            str(row.accepted),
        ]
        assert cells[7:11] == [
            f"{value:.4f}"
            for value in (row.starting_mase, row.best_mase, row.gain, row.gain_per_evaluation)
        ]
        assert cells[11] == _cells(row.gain_per_million_tokens)
        assert cells[12] == f"{row.gain_per_wall_hour:.4f}"
        assert cells[13:15] == [str(row.input_tokens), str(row.output_tokens)]  # ints, not .4f
        assert cells[15] == f"{row.wall_seconds:.4f}"
    by_condition = {cells[0]: cells for cells in rows}
    assert by_condition["random_param"][11] == "n/a"  # proposes without LLM tokens
    assert all(cells[2] == "budget_exhausted" for cells in rows)  # tabled with its status


# --- Test 5: the calibration ----------------------------------------------------------------------


def test_calibration_renders_the_fixtures_hand_counts(pilot: Pilot) -> None:
    """The count columns hold the hand counts; shares and means render .4f or n/a."""
    data = build_phase1_report_data(pilot.ledger, _ids(pilot), bootstrap_seed=0)
    rows = _table_rows(_report(pilot, _ids(pilot)), "Self-model calibration")
    by_condition = {row.condition: row for row in data.calibration}
    run_of = {row.condition: row.run_id for row in data.rows}
    for cells in rows:
        row = by_condition[cells[0]]
        realized, not_ok, never = _hand_calibration(_records(pilot, run_of[cells[0]]))
        assert cells[1:5] == [
            str(_counts(pilot, run_of[cells[0]])["prediction"]),
            str(len(realized)),
            str(not_ok),
            str(never),
        ]
        assert cells[5:] == [
            _cells(value)
            for value in (
                row.coverage_90,
                row.coverage_50,
                row.mean_q50,
                row.mean_gain,
                row.mean_q50_error,
            )
        ]
    shares = {cells[0]: cells[5] for cells in rows}
    assert shares["predictive"] == f"{by_condition['predictive'].coverage_90:.4f}"


def test_a_no_proposal_lineage_renders_na_coverages_in_its_own_build(pilot: Pilot) -> None:
    """The no-proposal condition renders zeros and n/a, from its own report build.

    Its output-token limit is smaller than the pilot's, so the equal-budget
    refusal would reject it next to the standard fixtures; as PR A test 6
    pinned, it is reported alone.
    """
    assert (
        _lineage(
            pilot,
            "render-no-proposals",
            "predictive",
            spec={"proposals_per_generation": 1, "llm_output_tokens": 16_383},
            outcomes={REFERENCE_ARTIFACT_SOURCE: scored(1.0)},
        )
        == 0
    )
    report = _report(pilot, ["render-no-proposals"])
    (cells,) = _table_rows(report, "Self-model calibration")
    assert cells[0] == "predictive"
    assert cells[1:5] == ["0", "0", "0", "0"]
    assert cells[5:] == ["n/a"] * 5


# --- Test 6: the safe inheritance -----------------------------------------------------------------


def test_inheritance_renders_rates_unjudgeable_and_the_missing_sentence(pilot: Pilot) -> None:
    """The rate is hurt/judged .4f or n/a; unjudgeable and not-ok shown; the
    missing-holdout sentence appears exactly when a condition lacks its holdout."""
    data = build_phase1_report_data(pilot.ledger, _ids(pilot), bootstrap_seed=0)
    report = _report(pilot, _ids(pilot))
    rows = _table_rows(report, "Safe inheritance (holdout)")
    by_condition = {row.condition: row for row in data.inheritance}
    for cells in rows:
        row = by_condition[cells[0]]
        assert cells[1:7] == [
            str(row.accepted),
            str(row.judged),
            str(row.hurt),
            _cells(row.safe_rate),
            str(row.unjudgeable),
            str(row.not_ok_rows),
        ]
        assert cells[7:] == [
            _cells(value)
            for value in (row.start_holdout_mase, row.final_holdout_mase, row.holdout_gain)
        ]
    by_name = {cells[0]: cells for cells in rows}
    assert by_name["predictive"][4] == "0.0000"  # hurt 0 of judged 2
    assert by_name["evaluate_all"][4] == "n/a"  # judged 0: nothing accepted
    assert "No holdout run named for:" not in report
    dropped = _report(pilot, _ids(pilot, drop=["holdout-fixed"]))
    assert "No holdout run named for: fixed." in dropped


def _bump_holdout_mase(
    record: Record, payload: Payload, refers_to: list[str], blobs: BlobStore
) -> tuple[Payload, list[str]]:
    """A hand-built holdout variant's accept-01 row, scored worse than start."""
    if (
        record.kind == "baseline"
        and isinstance(payload, Baseline)
        and payload.updater_id == "accept-01"
    ):
        assert payload.per_task_blob is not None
        blob: dict[str, Any] = json.loads(blobs.get(payload.per_task_blob).decode("utf-8"))
        for entry in blob["result"]["series"]:
            entry["mase"] = 1.3
        return (
            replace(payload, metrics={"mase": 1.3}, per_task_blob=blobs.put(canonical_bytes(blob))),
            refers_to,
        )
    return payload, refers_to


def _break_holdout_row(
    record: Record, payload: Payload, refers_to: list[str], blobs: BlobStore
) -> tuple[Payload, list[str]]:
    """A hand-built holdout variant's accept-01 row, scored not-ok."""
    if (
        record.kind == "baseline"
        and isinstance(payload, Baseline)
        and payload.updater_id == "accept-01"
    ):
        broken = replace(payload, status="invalid_weights", metrics={}, error="arena: bad rows")
        return broken, refers_to
    return payload, refers_to


def test_a_hurt_or_not_ok_holdout_edit_renders_its_cells(pilot: Pilot) -> None:
    """A hurt edit rates 0.5000; a not-ok row is unjudgeable, counted, rate n/a."""
    copy_run(pilot.ledger, "holdout-predictive", "render-holdout-hurt", mutate=_bump_holdout_mase)
    (cells,) = _table_rows(
        _report(pilot, ["lineage-predictive", "render-holdout-hurt"]), "Safe inheritance (holdout)"
    )
    assert cells[1:7] == ["2", "2", "1", "0.5000", "0", "0"]
    copy_run(pilot.ledger, "holdout-predictive", "render-holdout-not-ok", mutate=_break_holdout_row)
    (cells,) = _table_rows(
        _report(pilot, ["lineage-predictive", "render-holdout-not-ok"]),
        "Safe inheritance (holdout)",
    )
    assert cells[1:7] == ["2", "0", "0", "n/a", "2", "1"]  # both pairs span the broken row
    assert cells[7:] == ["1.1000", "1.1000", "0.0000"]  # start and final ok: the gain is 0.0


# --- Test 7: the replay audit ---------------------------------------------------------------------


def _replay_section(report: str) -> str:
    """The Replay audit section's text, up to the runs not tabled."""
    return report[report.index("## Replay audit") : report.index("## Runs not tabled")]


def test_the_replay_audit_renders_both_match_cases(pilot: Pilot) -> None:
    """The audit row and its fixed sentences, in the true and the false case."""
    section = _replay_section(_report(pilot, _ids(pilot)))
    assert "`replay-predictive`" in section and "`lineage-predictive`" in section
    assert "| true | 0 |" in section  # matches_source_run true, no timeout mismatches
    assert "bit-for-bit" in section
    assert "honestly re-charged" in section
    assert "audit finding" not in section
    copy_run(
        pilot.ledger,
        pilot.replay,
        "render-replay-false",
        summary_replace={"matches_source_run": False},
    )
    false_case = _replay_section(
        _report(pilot, [*_ids(pilot, drop=[pilot.replay]), "render-replay-false"])
    )
    assert "| false | 0 |" in false_case
    assert "temperature noise around one prompt" in false_case
    assert "stated, not repaired" in false_case
    assert "bit-for-bit" not in false_case
    copy_run(
        pilot.ledger,
        pilot.replay,
        "render-replay-timeout",
        summary_replace={"timeout_status_mismatches": 1},
    )
    timeout_case = _replay_section(
        _report(pilot, [*_ids(pilot, drop=[pilot.replay]), "render-replay-timeout"])
    )
    assert "| true | 1 |" in timeout_case
    assert "a timeout that did not reproduce" in timeout_case
    no_replay = _report(pilot, _ids(pilot, drop=["lineage-predictive", "holdout-predictive"]))
    assert "No replay run was named." in _replay_section(no_replay)


# --- Test 8: the notes ----------------------------------------------------------------------------


def test_the_notes_sentences_appear_exactly_when_they_hold(pilot: Pilot) -> None:
    """The five fixed Notes always; the bit-equal sentence only on disagreement."""
    report = _report(pilot, _ids(pilot))
    notes = report[report.index("## Notes") :]
    for fragment in (
        "no seed-level uncertainty",
        "designed end of a lineage",
        "weights reference",
        "proposes without LLM tokens",
        "share of accepted edits",
    ):
        assert fragment in notes
    assert "not bit-equal" not in notes  # the fixture's forecast-v2 starts agree

    def shift_start(
        record: Record, payload: Payload, refers_to: list[str], blobs: BlobStore
    ) -> tuple[Payload, list[str]]:
        """A shifted baseline: the copy starts at 1.05, records and summary agree."""
        if record.kind == "baseline" and isinstance(payload, Baseline):
            return replace(payload, metrics={**payload.metrics, "mase": 1.05}), refers_to
        return payload, refers_to

    copy_run(
        pilot.ledger,
        "lineage-evaluate-all",
        "render-start-shifted",
        mutate=shift_start,
        summary_replace={"starting_mase": 1.05, "best_mase": 1.05},
    )
    shifted_ids = [
        *_ids(pilot, drop=["lineage-evaluate-all", "holdout-evaluate-all"]),
        "render-start-shifted",
    ]
    shifted = _report(pilot, shifted_ids)
    assert (
        "The forecast-v2 conditions' starting MASE values are not bit-equal: "
        "1.0000, 1.0500." in shifted[shifted.index("## Notes") :]
    )


# --- Test 9: the CLI defaults ---------------------------------------------------------------------


def test_the_cli_defaults_and_output_flags() -> None:
    """--ledger-root defaults to ledgers, --bootstrap-seed to 0; --out and
    --check are mutually exclusive and one is required."""
    parser = cli.build_parser()
    args = parser.parse_args(["phase1", "report", "--run", "x", "--out", "p"])
    assert args.ledger_root == "ledgers"
    assert args.bootstrap_seed == 0
    assert args.out == "p" and args.check is None
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["phase1", "report", "--run", "x"])
    assert raised.value.code == 2
    with pytest.raises(SystemExit) as raised:
        parser.parse_args(["phase1", "report", "--run", "x", "--out", "p", "--check", "c"])
    assert raised.value.code == 2
