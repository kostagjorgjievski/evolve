"""Holdout scoring of a finished Phase 1 lineage's accepted edits (TASK_018 PR C).

A holdout run is a reader of one lineage's ledger and nothing more. It
resolves the lineage's starting artifact and every accepted child from the
records, replays each over a fresh-seed holdout replicate
(:mod:`evolvets.arena.synthetic.holdout`) through the same subprocess
evaluator the lineage used, and writes one ``baseline`` record per artifact:
``start`` first, then ``accept-01``, ``accept-02`` ... in acceptance order.
The run holds no proposals, predictions, allocations, evaluations or
acceptances, and this module imports nothing from ``evolvets.improver`` —
it reads the lineage's records and blobs and delegates every evaluation —
so a holdout score can never feed an observation.

The holdout never selects anything, either way round: only a lineage whose
``run_end`` is ``completed`` or ``budget_exhausted`` may be scored, and only
on the holdout of its own dataset family (a shift3 lineage scores on
``synthetic-shift3-holdout``), so a holdout pool is never reached by a code
path that selects. Non-ok holdout scores are recorded as they are: the
safe-inheritance rate they feed is entry 20's to compute.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolvets.arena.dev import load_dev_pool
from evolvets.arena.synthetic.holdout import HOLDOUT_PRESETS
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import Acceptance, Baseline, Proposal, RunEnd, RunStart
from evolvets.ledger.schema import Budget, Record
from evolvets.ledger.store import LedgerReader
from evolvets.runner.phase1 import (
    CONDITIONS,
    EVAL_WALL_MARGIN_SECONDS,
    Condition,
    # the pinned reference of each object space and the blob reader for
    # artifact sources are the lineage engine's own; borrowing them keeps the
    # pin single-sourced — and this module free of evolvets.improver
    _artifact_text,
    _reference_artifact,
)
from evolvets.runner.phase1_eval import EvalOutcome, evaluate_source
from evolvets.runner.runs import Run, current_git_commit
from evolvets.updater.interfaces import ConfigValue

#: The holdout run's condition; its run_start also declares seed 0 and
#: ``requires_predictions=False`` — a prediction record would be a violation.
_HOLDOUT_CONDITION = "phase1-holdout"

#: The seed every holdout score runs under: fixed, not the lineage's.
_HOLDOUT_SEED = 0

#: The updater_id of the starting artifact's record; accepted edits follow
#: as ``accept-01``, ``accept-02`` ... in acceptance order.
_START_ID = "start"

#: The run_start conditions of Phase 1 lineages: ``phase1-<condition id>``.
_LINEAGE_PREFIX = "phase1-"

#: The budget sums the run_end summary reports.
_BUDGET_SUMS = ("llm_input_tokens", "llm_output_tokens", "wall_seconds", "evaluations")


class HoldoutError(RuntimeError):
    """Raised when a holdout cannot run: a bad request or an unusable lineage."""


@dataclass(frozen=True)
class _Scored:
    """One artifact a holdout scores, and the lineage record it links back to."""

    updater_id: str  # "start" | "accept-01" | ...
    source: str  # the artifact source the evaluator replays
    proposal_record_id: str | None  # None only when no proposal parents the start
    generation: int  # the generation the scored artifact was proposed at


def _read_lineage(
    ledger_root: Path, from_run: str, dataset: str
) -> tuple[list[Record], Condition, float, float]:
    """Read and check the lineage to score; refuse anything but a finished Phase 1 run.

    Returns its records (both streams, chain and blobs verified), its
    condition — the interface and the reference config come from it — and
    the evaluation limits its own evaluations ran under, which the holdout
    reuses. An aborted or missing run, a run of another condition family and
    a lineage of another dataset family than the holdout's are refused
    before anything is written.
    """
    reader = LedgerReader(Path(ledger_root), from_run)
    chain = reader.verify(check_blobs=True)
    if not chain.ok:
        raise HoldoutError(f"lineage {from_run!r} does not verify: {chain.reason}")
    records = list(reader.records())
    starts = [record for record in records if record.kind == "run_start"]
    if len(starts) != 1 or not isinstance(starts[0].payload, RunStart):
        raise HoldoutError(f"lineage {from_run!r} has no single run_start")
    ends = [record for record in records if record.kind == "run_end"]
    if not ends or not isinstance(ends[-1].payload, RunEnd):
        raise HoldoutError(f"lineage {from_run!r} has no run_end")
    if ends[-1].payload.status not in ("completed", "budget_exhausted"):
        raise HoldoutError(
            f"lineage {from_run!r} ended {ends[-1].payload.status!r}; "
            "only finished lineages score on a holdout"
        )
    start = starts[0].payload
    if not start.condition.startswith(_LINEAGE_PREFIX):
        raise HoldoutError(
            f"run {from_run!r} holds condition {start.condition!r}, not a phase1 lineage"
        )
    condition = CONDITIONS.get(start.condition[len(_LINEAGE_PREFIX) :])
    if condition is None:
        raise HoldoutError(f"run {from_run!r} holds unknown phase1 condition {start.condition!r}")
    blobs = BlobStore(Path(ledger_root) / from_run / "blobs")
    config: object = json.loads(blobs.get(start.config_blob).decode("utf-8"))
    if not isinstance(config, dict):
        raise HoldoutError(f"lineage {from_run!r} holds a config blob that is not an object")
    base = HOLDOUT_PRESETS[dataset].base
    if config.get("dataset") != base:
        raise HoldoutError(
            f"lineage {from_run!r} ran dataset {config.get('dataset')!r}; {dataset!r} scores "
            f"the holdout family of {base!r}"
        )
    limits: list[float] = []
    for key in ("eval_timeout_seconds", "eval_memory_gib"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise HoldoutError(
                f"lineage {from_run!r} records no usable {key}; cannot size its holdout"
            )
        limits.append(float(value))
    return records, condition, limits[0], limits[1]


def _generation_of(blobs: BlobStore, proposal: Record) -> int:
    """The generation a proposal was made at, read from its own observation blob."""
    payload = proposal.payload
    assert isinstance(payload, Proposal)
    observation: object = json.loads(blobs.get(payload.observation_blob).decode("utf-8"))
    generation = observation.get("generation") if isinstance(observation, dict) else None
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise HoldoutError(
            f"proposal {proposal.record_id} holds no generation in its observation blob"
        )
    return generation


def _artifacts_to_score(
    blobs: BlobStore,
    records: Sequence[Record],
    from_run: str,
    reference_source: str,
) -> list[_Scored]:
    """The lineage's artifacts to score: the starting artifact, then every accepted child.

    The starting artifact is the first proposal's ``parent_artifact`` blob; a
    lineage that proposed nothing is scored from the pinned reference source
    of its condition's object space. Each accepted edit contributes its
    acceptance's ``new_artifact``, checked equal to the child artifact of the
    proposal that acceptance judged.
    """
    by_id = {record.record_id: record for record in records}
    proposals = [record for record in records if record.kind == "proposal"]
    out: list[_Scored] = []
    if proposals:
        first = proposals[0]
        payload = first.payload
        assert isinstance(payload, Proposal)
        out.append(
            _Scored(
                updater_id=_START_ID,
                source=_artifact_text(blobs, payload.parent_artifact),
                proposal_record_id=first.record_id,
                generation=_generation_of(blobs, first),
            )
        )
    else:
        # no proposal parents it: it would have been generation 0's parent
        out.append(_Scored(_START_ID, reference_source, None, 0))
    accepted = 0
    for record in records:
        if record.kind != "acceptance":
            continue
        payload = record.payload
        if not isinstance(payload, Acceptance) or not payload.accepted:
            continue
        accepted += 1
        proposal = by_id.get(record.refers_to[0])
        if proposal is None or proposal.kind != "proposal":
            raise HoldoutError(
                f"acceptance {record.record_id} of {from_run!r} does not refer to a proposal"
            )
        proposal_payload = proposal.payload
        assert isinstance(proposal_payload, Proposal)
        if payload.new_artifact is None or payload.new_artifact != proposal_payload.child_artifact:
            raise HoldoutError(
                f"acceptance {record.record_id} of {from_run!r} does not score the child "
                "artifact of the proposal it judged"
            )
        out.append(
            _Scored(
                updater_id=f"accept-{accepted:02d}",
                source=_artifact_text(blobs, payload.new_artifact),
                proposal_record_id=proposal.record_id,
                generation=_generation_of(blobs, proposal),
            )
        )
    return out


def _request(
    source: str,
    config: Mapping[str, ConfigValue],
    *,
    dataset: str,
    pool_spec: str,
    foundation_models: tuple[str, ...] | None,
    interface: str,
    cache_root: Path,
    data_root: Path,
    manifest_dir: Path,
    foundation_root: Path,
) -> dict[str, object]:
    """One evaluator request: this source at the reference config, seed 0, holdout pool."""
    return {
        "source": source,
        "config": dict(config),
        "seed": _HOLDOUT_SEED,
        "interface": interface,
        "dataset": dataset,
        "pool_spec": pool_spec,
        "foundation_models": list(foundation_models) if foundation_models is not None else [],
        "cache_root": str(cache_root),
        "data_root": str(data_root),
        "manifest_dir": str(manifest_dir),
        "foundation_root": str(foundation_root),
    }


def _metrics(outcome: EvalOutcome) -> dict[str, float]:
    """The pooled metrics of an ``ok`` outcome, ``{}`` otherwise."""
    if outcome.status != "ok" or outcome.result_obj is None:
        return {}
    metrics = outcome.result_obj.get("metrics", {})
    return dict(metrics) if isinstance(metrics, dict) else {}


def run_holdout(
    from_run: str,
    *,
    dataset: str,
    pool_spec: str,
    foundation_models: tuple[str, ...] | None,
    ledger_root: Path,
    cache_root: Path,
    data_root: Path,
    manifest_dir: Path,
    foundation_root: Path,
    run_id: str,
) -> int:
    """Score one finished lineage's accepted edits on a holdout replicate.

    0 when the holdout run completed, 2 when it aborted (a crash mid-scoring
    ends the run ``aborted`` through the ``Run`` context). Every refusal
    before ``run_start`` — a non-holdout dataset, an unusable lineage, an
    unloadable pool, a dirty tree, a reused run id — raises
    :class:`HoldoutError`, :class:`~evolvets.arena.errors.ArenaError` or
    :class:`~evolvets.runner.runs.RunError` with nothing written. Each score
    charges ``evaluations=1`` and its parent-measured wall, and the budget is
    sized exactly for the artifacts there are to score — the lineage's own
    per-evaluation deadline plus margin, once per artifact.
    """
    started = time.monotonic()
    if dataset not in HOLDOUT_PRESETS:
        raise HoldoutError(
            f"dataset must be one of {list(HOLDOUT_PRESETS)}, got {dataset!r}: "
            "holdouts score accepted edits only and are never selected on"
        )
    records, condition, timeout_seconds, memory_gib = _read_lineage(
        Path(ledger_root), from_run, dataset
    )
    blobs = BlobStore(Path(ledger_root) / from_run / "blobs")
    reference_source, reference_config = _reference_artifact(condition)
    artifacts = _artifacts_to_score(blobs, records, from_run, reference_source)
    # the pool loads before the run starts: no run starts that cannot run
    load_dev_pool(
        dataset,
        Path(cache_root),
        Path(data_root),
        Path(manifest_dir),
        pool_spec=pool_spec,
        foundation_root=Path(foundation_root) if pool_spec == "simple+foundation" else None,
        foundation_models=(foundation_models if pool_spec == "simple+foundation" else None),
    )
    run = Run.start(
        Path(ledger_root),
        run_id,
        condition=_HOLDOUT_CONDITION,
        seed=_HOLDOUT_SEED,
        config={
            "from_run": from_run,
            "dataset": dataset,
            "pool_spec": pool_spec,
            "foundation_models": (
                list(foundation_models) if foundation_models is not None else None
            ),
            "interface": condition.interface,
        },
        budget_limits=Budget(
            wall_seconds=len(artifacts) * (timeout_seconds + EVAL_WALL_MARGIN_SECONDS),
            evaluations=len(artifacts),
        ),
        git_commit=current_git_commit(Path.cwd()),
        requires_predictions=False,
    )
    try:
        with run:
            for artifact in artifacts:
                outcome = evaluate_source(
                    _request(
                        artifact.source,
                        reference_config,
                        dataset=dataset,
                        pool_spec=pool_spec,
                        foundation_models=foundation_models,
                        interface=condition.interface,
                        cache_root=cache_root,
                        data_root=data_root,
                        manifest_dir=manifest_dir,
                        foundation_root=foundation_root,
                    ),
                    timeout_seconds=timeout_seconds,
                    memory_gib=memory_gib,
                )
                run.append(
                    "baseline",
                    Baseline(
                        updater_id=artifact.updater_id,
                        hyperparams={},
                        pool="dev",
                        status=outcome.status,
                        metrics=_metrics(outcome),
                        per_task_blob=run.put_blob(
                            {
                                "series_wall_seconds": list(outcome.series_wall_seconds),
                                "result": outcome.result_obj,
                                "lineage_run_id": from_run,
                                "proposal_record_id": artifact.proposal_record_id,
                                "generation": artifact.generation,
                            }
                        ),
                        error=outcome.error,
                    ),
                    budget=Budget(evaluations=1, wall_seconds=outcome.wall_seconds),
                )
            run.end(
                "completed",
                {
                    "from_run": from_run,
                    "dataset": dataset,
                    "interface": condition.interface,
                    "artifacts": len(artifacts),
                    "accepted": len(artifacts) - 1,
                    "budget_spent": {field: getattr(run.spent, field) for field in _BUDGET_SUMS},
                    "wall_seconds_total": time.monotonic() - started,
                },
            )
        return 0
    except Exception:
        # The Run context has already ended the run aborted with the
        # exception's name and message in its summary: the crash is in the
        # ledger, and the caller learns it through run_end, not a traceback.
        return 2
