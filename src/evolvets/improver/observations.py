"""The observation object: what the improver is shown before it proposes (TASK_017 PR A2).

The observation is built from the lineage's own ledger records alone — never
from a pool, a dataset or the holdout — and every evaluation that reaches it
must be a dev evaluation: a sealed or live score can never enter a prompt
(:class:`ImproverError` otherwise). Each history entry is one evaluated
proposal, its ``family`` recomputed from the artifact blobs (not trusted from
a field), its ``gain`` the parent artifact's dev MASE minus the child's
(positive is better), and its ``accepted`` read from the acceptance record
that follows it. ``parent_metrics`` is the metrics dict of the artifact the
next proposal will edit — the evaluation of the most recent accepted
proposal — and ``{}`` at generation 0, where the starting artifact has no
evaluation of its own in the lineage.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import cast

from evolvets.improver.artifacts import artifact_family
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import Acceptance, Evaluation, Proposal
from evolvets.ledger.schema import Record


class ImproverError(RuntimeError):
    """Raised when records would leak a non-dev score into an observation."""


def _artifact_text(blobs: BlobStore, ref: str) -> str:
    """The artifact source stored under ``ref`` (blobs hold canonical JSON of the text)."""
    return cast("str", json.loads(blobs.get(ref).decode("utf-8")))


def _generation_of(proposal: Record, by_id: dict[str, Record]) -> int:
    """A proposal's generation: 0 when it edits the starting artifact, else parent's + 1."""
    generation = 0
    current = proposal
    while current.refers_to:
        parent = by_id.get(current.refers_to[0])
        if parent is None or parent.kind != "proposal":
            break
        generation += 1
        current = parent
    return generation


def observation_object(
    blobs: BlobStore, records: Sequence[Record], generation: int, max_history: int = 20
) -> dict[str, object]:
    """Build the observation for the proposal about to be made at ``generation``.

    ``records`` are the lineage's own records in ledger order (both streams).
    Proposals whose evaluation failed or has not happened are skipped, as is a
    proposal whose parent artifact has no dev score in the lineage yet — the
    generation-0 proposal, whose parent is the starting artifact. The history
    is capped at the last ``max_history`` entries.
    """
    proposals: list[Record] = []
    by_id: dict[str, Record] = {}
    evaluation_of: dict[str, Record] = {}
    accepted_of: dict[str, bool] = {}
    metrics_of_artifact: dict[str, dict[str, float]] = {}
    parent_metrics: dict[str, float] | None = None
    for record in records:
        by_id[record.record_id] = record
        if record.kind == "proposal" and isinstance(record.payload, Proposal):
            proposals.append(record)
        elif record.kind == "evaluation" and isinstance(record.payload, Evaluation):
            if record.payload.pool != "dev":
                raise ImproverError(
                    f"an evaluation with pool {record.payload.pool!r} reached the improver; "
                    "only dev-pool scores may enter an observation"
                )
            evaluation_of[record.refers_to[0]] = record
            if record.payload.status == "ok":
                evaluated = by_id.get(record.refers_to[0])
                if evaluated is not None and isinstance(evaluated.payload, Proposal):
                    metrics_of_artifact[evaluated.payload.child_artifact] = dict(
                        record.payload.metrics
                    )
        elif record.kind == "acceptance" and isinstance(record.payload, Acceptance):
            accepted_of[record.refers_to[0]] = record.payload.accepted
            if record.payload.accepted:
                evaluation = by_id.get(record.refers_to[1])
                if evaluation is not None and isinstance(evaluation.payload, Evaluation):
                    parent_metrics = dict(evaluation.payload.metrics)
    history: list[dict[str, object]] = []
    for proposal in proposals:
        evaluation = evaluation_of.get(proposal.record_id)
        if evaluation is None or not isinstance(evaluation.payload, Evaluation):
            continue
        if evaluation.payload.status != "ok" or not isinstance(proposal.payload, Proposal):
            continue
        artifact_metrics = metrics_of_artifact.get(proposal.payload.parent_artifact)
        if artifact_metrics is None:
            continue
        history.append(
            {
                "generation": _generation_of(proposal, by_id),
                "family": artifact_family(
                    _artifact_text(blobs, proposal.payload.parent_artifact),
                    _artifact_text(blobs, proposal.payload.child_artifact),
                ),
                "gain": artifact_metrics["mase"] - evaluation.payload.metrics["mase"],
                "accepted": accepted_of.get(proposal.record_id, False),
            }
        )
    capped = history[len(history) - max_history :] if max_history > 0 else []
    return {
        "generation": generation,
        "parent_metrics": dict(parent_metrics) if parent_metrics is not None else {},
        "history": capped,
    }
