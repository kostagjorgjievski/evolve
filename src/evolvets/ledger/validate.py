"""Cross-record ordering rules that a single record cannot check alone.

The hash chain proves integrity; these rules prove the run happened in an
order the protocol allows — predictions logged before the evaluations they
forecast, references pointing at records that already existed, budgets
inside the limits the run declared.
"""

from __future__ import annotations

from collections.abc import Sequence

from evolvets.ledger.payloads import RunStart
from evolvets.ledger.schema import Record

_BUDGET_FIELDS = (
    "llm_input_tokens",
    "llm_output_tokens",
    "wall_seconds",
    "gpu_seconds",
    "evaluations",
)


def validate_run(records: Sequence[Record], uses_predictions: bool | None = None) -> list[str]:
    """Return every ordering violation found in one run's records.

    ``records`` should cover both streams (``LedgerReader.records()`` order).
    ``uses_predictions`` defaults to auto-detection: the rule is enforced
    when the run contains at least one prediction record (condition strings
    are free-form in schema v1, so presence is the detectable signal).
    """
    violations: list[str] = []
    by_id: dict[str, Record] = {}
    for record in records:
        if record.record_id in by_id:
            violations.append(f"duplicate record_id {record.record_id}")
        else:
            by_id[record.record_id] = record

    main = [record for record in records if record.stream == "main"]
    run_start = next(
        (r for r in main if r.kind == "run_start" and isinstance(r.payload, RunStart)), None
    )
    if not main:
        violations.append("main stream has no records")
    elif main[0].kind != "run_start":
        violations.append("the first record of main must be run_start")
    run_ends = [r for r in main if r.kind == "run_end"]
    if len(run_ends) > 1:
        violations.append(f"{len(run_ends)} run_end records; at most one is allowed")
    if run_ends and main and run_ends[0] is not main[-1]:
        violations.append("run_end must be the last record of main")

    for record in records:
        for ref in record.refers_to:
            target = by_id.get(ref)
            if target is None:
                violations.append(
                    f"{record.kind} at seq {record.seq} refers to unknown record {ref}"
                )
            elif target.created_at > record.created_at:
                violations.append(
                    f"{record.kind} at seq {record.seq} refers to {ref} created later"
                )
            elif target.stream == record.stream and target.seq >= record.seq:
                violations.append(
                    f"{record.kind} at seq {record.seq} refers to {ref} at seq {target.seq}"
                )
        if record.kind == "acceptance":
            proposal = by_id.get(record.refers_to[0])
            evaluation = by_id.get(record.refers_to[1])
            if proposal is None or proposal.kind != "proposal":
                violations.append(
                    f"acceptance at seq {record.seq} does not refer to a proposal first"
                )
            if evaluation is None or evaluation.kind != "evaluation":
                violations.append(
                    f"acceptance at seq {record.seq} does not refer to an evaluation second"
                )

    if uses_predictions is None:
        uses_predictions = any(record.kind == "prediction" for record in records)
    if uses_predictions:
        predictions_by_proposal: dict[str, list[Record]] = {}
        for record in records:
            if record.kind == "prediction":
                predictions_by_proposal.setdefault(record.refers_to[0], []).append(record)
        for record in records:
            if record.kind != "evaluation":
                continue
            logged_before = any(
                (prediction.stream == record.stream and prediction.seq < record.seq)
                or (
                    prediction.stream != record.stream
                    and prediction.created_at <= record.created_at
                )
                for prediction in predictions_by_proposal.get(record.refers_to[0], [])
            )
            if not logged_before:
                violations.append(
                    f"evaluation at seq {record.seq} ({record.stream}) has no prediction "
                    f"logged before it for proposal {record.refers_to[0]}"
                )

    if run_start is not None and isinstance(run_start.payload, RunStart):
        limits = run_start.payload.budget_limits
        for field in _BUDGET_FIELDS:
            spent = sum(getattr(record.budget, field) for record in records)
            if spent > getattr(limits, field):
                violations.append(
                    f"budget {field} exceeded: spent {spent} > limit {getattr(limits, field)}"
                )
    return violations
