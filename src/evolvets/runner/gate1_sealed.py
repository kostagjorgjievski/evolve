"""Gate 1 on sealed: the plan for a dev tune run, and its recorded execution.

:func:`plan_from_tune_run` recomputes a completed dev tune run's selection
from its own records (verifying them first) and maps every dev pool to its
sealed counterpart: the oracles of every pool, then one evaluation of every
dev-selected baseline configuration. :func:`sealed_run` executes that plan
inside one recorded run whose main stream holds only ``run_start`` and
``run_end``; every evaluation record is the sealed server's, charged by the
server and learned by the caller as a receipt alone. The caller stays blind
(TASK_008): this module imports only the sealed client, opens no sealed
file, salt or cache, and never reads a run's sealed stream — its own
monotonic elapsed time is what bounds the run's wall-clock, because the
sealed budgets are not the caller's to read. A configuration is evaluated
on sealed once per tune run: a tune run that already has a completed
sealed run is refused before anything is written.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from evolvets.arena.sealed.client import (
    SERIES_TIMEOUT_SECONDS,
    SealedOracleRequest,
    SealedPaths,
    SealedRequest,
    evaluate_sealed,
    evaluate_sealed_oracle,
)
from evolvets.arena.versions import protocol_versions
from evolvets.ledger.blobs import BlobError, BlobStore
from evolvets.ledger.payloads import RunEnd, RunStart
from evolvets.ledger.schema import Budget
from evolvets.ledger.store import LedgerError, LedgerReader
from evolvets.runner.gate1_grids import select_configs
from evolvets.runner.gate1_tune import TUNE_CONDITION
from evolvets.runner.runs import Run, RunError
from evolvets.updater.baselines import BASELINE_IDS
from evolvets.updater.interfaces import ConfigValue

SEALED_CONDITION = "gate1-sealed"
SEALED_DATASET_OF: Mapping[str, str] = MappingProxyType(
    {
        "etth1": "etth1",
        "ecl": "ecl",
        "synthetic-dev": "synthetic-sealed",
    }
)
#: The inverse mapping: sealed dataset -> dev dataset.
_DEV_DATASET_OF: Mapping[str, str] = MappingProxyType(
    {sealed: dev for dev, sealed in SEALED_DATASET_OF.items()}
)
_UPDATER_REF = "evolvets.updater.baselines:"


@dataclass(frozen=True)
class SealedPlanItem:
    """One sealed evaluation to request: an oracle or one dev-selected baseline."""

    dataset: str  # sealed dataset name
    pool_spec: str
    updater_id: str  # a BASELINE_IDS entry, or "oracle"
    config: dict[str, ConfigValue]  # {} for the oracle
    foundation_models: tuple[str, ...] | None  # None for pool_spec "simple"


@dataclass(frozen=True)
class SealedPlan:
    """Everything the sealed run needs, derived from one completed dev tune run."""

    tune_run_id: str
    tune_run_end_record_id: str
    tune_git_commit: str
    seed: int  # the tune run's seed
    items: tuple[SealedPlanItem, ...]  # all oracles first, then baselines
    skipped: tuple[tuple[str, str, str], ...]  # (dataset, pool_spec, baseline) left to None
    previous_sealed_runs: tuple[str, ...]


def _config_of(ledger_root: Path, run_id: str, record: RunStart) -> dict[str, object] | None:
    """The parsed config blob of one run's ``run_start``; ``None`` if not a JSON object."""
    raw = BlobStore(Path(ledger_root) / run_id / "blobs").get(record.config_blob)
    parsed: object = json.loads(raw.decode("utf-8"))
    if not isinstance(parsed, dict):
        return None
    return cast("dict[str, object]", parsed)


def _blob_reader(ledger_root: Path, run_id: str) -> Callable[[str], object]:
    """A ``read_blob`` over one run's blob store, as ``select_configs`` takes it."""
    blobs = BlobStore(Path(ledger_root) / run_id / "blobs")

    def read(ref: str) -> object:
        parsed: object = json.loads(blobs.get(ref).decode("utf-8"))
        return parsed

    return read


def _tune_axes(config: Mapping[str, object], run_id: str) -> tuple[list[str], list[str]]:
    """The tune config's datasets and pool specs, in run order; ``RunError`` if malformed."""
    datasets = config.get("datasets")
    pool_specs = config.get("pool_specs")
    if (
        not isinstance(datasets, list)
        or not datasets
        or not all(isinstance(name, str) for name in datasets)
    ):
        raise RunError(f"tune run {run_id}: config datasets must be a non-empty list of names")
    if (
        not isinstance(pool_specs, list)
        or not pool_specs
        or not all(isinstance(name, str) for name in pool_specs)
    ):
        raise RunError(f"tune run {run_id}: config pool_specs must be a non-empty list of names")
    return datasets, pool_specs


def _foundation_models(
    config: Mapping[str, object], run_id: str, dataset: str, pool_spec: str
) -> tuple[str, ...] | None:
    """The foundation models the tune run joined for one pool; ``None`` for ``simple``."""
    if pool_spec == "simple":
        return None
    by_dataset = config.get("foundation_models")
    if not isinstance(by_dataset, dict):
        raise RunError(f"tune run {run_id}: config foundation_models must be a mapping")
    by_pool = by_dataset.get(dataset)
    if not isinstance(by_pool, dict) or not isinstance(by_pool.get(pool_spec), list):
        raise RunError(
            f"tune run {run_id}: config foundation_models has no model list for "
            f"{dataset}/{pool_spec}"
        )
    models = by_pool[pool_spec]
    if not all(isinstance(name, str) for name in models):
        raise RunError(f"tune run {run_id}: foundation model names must be strings")
    return tuple(models)


def _previous_sealed_runs(ledger_root: Path, tune_run_id: str) -> tuple[str, ...]:
    """Earlier sealed runs of this tune run; refuse one that already ended ``completed``."""
    previous: list[str] = []
    for run_dir in sorted(Path(ledger_root).iterdir()):
        if not (run_dir / "main.jsonl").exists():
            continue
        run_id = run_dir.name
        try:
            records = list(LedgerReader(Path(ledger_root), run_id).records(["main"]))
            if not records:
                continue
            first = records[0]
            if first.kind != "run_start" or not isinstance(first.payload, RunStart):
                continue
            if first.payload.condition != SEALED_CONDITION:
                continue
            config = _config_of(Path(ledger_root), run_id, first.payload)
        except (BlobError, LedgerError, OSError, ValueError):
            continue  # a run that fails to read is skipped
        if config is None or config.get("tune_run_id") != tune_run_id:
            continue
        ended = any(
            isinstance(record.payload, RunEnd) and record.payload.status == "completed"
            for record in records
            if record.kind == "run_end"
        )
        if ended:
            raise RunError(f"tune run {tune_run_id} already has a completed sealed run {run_id}")
        previous.append(run_id)
    return tuple(sorted(previous))


def plan_from_tune_run(ledger_root: Path, tune_run_id: str) -> SealedPlan:
    """Derive the sealed plan from one completed dev tune run; ``RunError`` on any refusal.

    Every refusal happens before anything is written: the tune run must verify
    (blobs included), hold only a main stream, start as a ``gate1-dev-tune``
    run of experiment ``gate1`` under the current protocol versions, end
    ``completed``, and carry a selection equal to the one recomputed from its
    own records.
    """
    root = Path(ledger_root)
    chain = LedgerReader(root, tune_run_id).verify(check_blobs=True)
    if not chain.ok:
        raise RunError(f"tune run {tune_run_id} does not verify: {chain.reason}")
    if (root / tune_run_id / "sealed.jsonl").exists():
        raise RunError(f"tune run {tune_run_id} has a sealed stream; a tune run must not")
    records = list(LedgerReader(root, tune_run_id).records(["main"]))
    if not records:
        raise RunError(f"tune run {tune_run_id}: the main stream is empty; no run_start")
    first = records[0]
    if first.kind != "run_start" or not isinstance(first.payload, RunStart):
        raise RunError(f"tune run {tune_run_id}: the first record is not a run_start")
    start = first.payload
    if start.condition != TUNE_CONDITION:
        raise RunError(
            f"tune run {tune_run_id}: condition is {start.condition!r}, not {TUNE_CONDITION!r}"
        )
    if start.protocol_versions != protocol_versions():
        raise RunError(f"tune run {tune_run_id} pins different protocol versions than this code")
    config = _config_of(root, tune_run_id, start)
    if config is None:
        raise RunError(f"tune run {tune_run_id}: the run_start config blob is not a JSON object")
    if config.get("experiment") != "gate1" or config.get("stage") != "dev-tune":
        raise RunError(
            f"tune run {tune_run_id}: config is not a gate1 dev-tune run "
            f"(experiment {config.get('experiment')!r}, stage {config.get('stage')!r})"
        )
    last = records[-1]
    if last.kind != "run_end" or not isinstance(last.payload, RunEnd):
        raise RunError(f"tune run {tune_run_id} has no run_end; the selection would be incomplete")
    if last.payload.status != "completed":
        raise RunError(
            f"tune run {tune_run_id} ended {last.payload.status}; the selection would be incomplete"
        )
    selection = select_configs(records, _blob_reader(root, tune_run_id))
    if last.payload.summary.get("selected") != selection:
        raise RunError(
            f"tune run {tune_run_id}: the run_end summary's selection differs from the one "
            "recomputed from its records"
        )
    datasets, pool_specs = _tune_axes(config, tune_run_id)
    sealed_of: list[str] = []
    for dataset in datasets:
        if dataset not in SEALED_DATASET_OF:
            raise RunError(f"tune run {tune_run_id}: dataset {dataset!r} has no sealed counterpart")
        sealed_of.append(SEALED_DATASET_OF[dataset])
    items: list[SealedPlanItem] = []
    skipped: list[tuple[str, str, str]] = []
    for dataset, sealed in zip(datasets, sealed_of, strict=True):
        for pool_spec in pool_specs:
            items.append(
                SealedPlanItem(
                    dataset=sealed,
                    pool_spec=pool_spec,
                    updater_id="oracle",
                    config={},
                    foundation_models=_foundation_models(config, tune_run_id, dataset, pool_spec),
                )
            )
    for dataset, sealed in zip(datasets, sealed_of, strict=True):
        for pool_spec in pool_specs:
            for baseline_id in BASELINE_IDS:
                pick = selection.get(dataset, {}).get(pool_spec, {}).get(baseline_id)
                if pick is None:
                    skipped.append((sealed, pool_spec, baseline_id))
                    continue
                chosen = pick.get("config")
                if not isinstance(chosen, dict):
                    raise RunError(
                        f"tune run {tune_run_id}: the selection of {baseline_id} on "
                        f"{dataset}/{pool_spec} has no config"
                    )
                items.append(
                    SealedPlanItem(
                        dataset=sealed,
                        pool_spec=pool_spec,
                        updater_id=baseline_id,
                        config=cast("dict[str, ConfigValue]", dict(chosen)),
                        foundation_models=_foundation_models(
                            config, tune_run_id, dataset, pool_spec
                        ),
                    )
                )
    return SealedPlan(
        tune_run_id=tune_run_id,
        tune_run_end_record_id=last.record_id,
        tune_git_commit=start.git_commit,
        seed=start.seed,
        items=tuple(items),
        skipped=tuple(skipped),
        previous_sealed_runs=_previous_sealed_runs(root, tune_run_id),
    )


def _validate_run(plan: SealedPlan, max_wall_seconds: float) -> None:
    """Refuse a run that cannot be honoured, before anything is written."""
    if (
        isinstance(max_wall_seconds, bool)
        or not isinstance(max_wall_seconds, (int, float))
        or not math.isfinite(max_wall_seconds)
        or max_wall_seconds <= 0
    ):
        raise ValueError(f"max_wall_seconds must be a finite number > 0, got {max_wall_seconds!r}")
    if not plan.items:
        raise ValueError("the plan has no items; nothing to evaluate on sealed")
    for item in plan.items:
        if item.dataset not in _DEV_DATASET_OF:
            raise ValueError(f"sealed dataset {item.dataset!r} has no dev counterpart")


def _run_config(plan: SealedPlan, max_wall_seconds: float) -> dict[str, object]:
    """Build the sealed run's ``run_start`` config blob from the plan alone.

    Axes keep plan order: datasets and pool specs in the order their oracle
    items first name them (the plan lists every pool's oracle first), and
    every baseline of the selection in ``BASELINE_IDS`` order, a ``None``
    kept for the skipped ones.
    """
    sealed_datasets: list[str] = []
    pool_specs: list[str] = []
    foundation_models: dict[str, dict[str, list[str]]] = {}
    picks: dict[tuple[str, str], dict[str, dict[str, ConfigValue] | None]] = {}
    for dataset, pool_spec, baseline_id in plan.skipped:
        picks.setdefault((dataset, pool_spec), {})[baseline_id] = None
    for item in plan.items:
        if item.updater_id == "oracle":
            continue
        picks.setdefault((item.dataset, item.pool_spec), {})[item.updater_id] = dict(item.config)
    for item in plan.items:
        if item.dataset not in sealed_datasets:
            sealed_datasets.append(item.dataset)
        if item.pool_spec not in pool_specs:
            pool_specs.append(item.pool_spec)
        foundation_models.setdefault(item.dataset, {}).setdefault(
            item.pool_spec,
            list(item.foundation_models) if item.foundation_models is not None else [],
        )
    selected: dict[str, dict[str, dict[str, dict[str, ConfigValue] | None]]] = {}
    for dataset in sealed_datasets:
        for pool_spec in pool_specs:
            bucket = picks.get((dataset, pool_spec), {})
            ordered = {
                baseline_id: bucket[baseline_id]
                for baseline_id in BASELINE_IDS
                if baseline_id in bucket
            }
            if ordered:
                selected.setdefault(dataset, {})[pool_spec] = ordered
    planned_ids = {item.updater_id for item in plan.items if item.updater_id != "oracle"}
    planned_ids.update(entry[2] for entry in plan.skipped)
    return {
        "experiment": "gate1",
        "stage": "sealed",
        "tune_run_id": plan.tune_run_id,
        "tune_run_end_record_id": plan.tune_run_end_record_id,
        "tune_git_commit": plan.tune_git_commit,
        "datasets": sealed_datasets,
        "dev_datasets": [_DEV_DATASET_OF[dataset] for dataset in sealed_datasets],
        "pool_specs": pool_specs,
        "foundation_models": foundation_models,
        "selected": selected,
        "updater_refs": {
            baseline_id: f"{_UPDATER_REF}{baseline_id}"
            for baseline_id in BASELINE_IDS
            if baseline_id in planned_ids
        },
        "seed": plan.seed,
        "max_wall_seconds": max_wall_seconds,
        "series_timeout_seconds": SERIES_TIMEOUT_SECONDS,
        "previous_sealed_runs": list(plan.previous_sealed_runs),
    }


def _evaluate_item(
    item: SealedPlanItem, plan: SealedPlan, paths: SealedPaths, ledger_root: Path, run_id: str
) -> tuple[str, str]:
    """Send one item to the sealed server; return its ``(record_id, status)`` receipt."""
    if item.updater_id == "oracle":
        receipt = evaluate_sealed_oracle(
            SealedOracleRequest(
                dataset=item.dataset,
                pool_spec=item.pool_spec,
                foundation_models=item.foundation_models,
            ),
            ledger_root=ledger_root,
            run_id=run_id,
            paths=paths,
        )
    else:
        receipt = evaluate_sealed(
            SealedRequest(
                dataset=item.dataset,
                updater_id=item.updater_id,
                updater_ref=f"{_UPDATER_REF}{item.updater_id}",
                config=dict(item.config),
                seed=plan.seed,
                pool_spec=item.pool_spec,
                foundation_models=item.foundation_models,
            ),
            ledger_root=ledger_root,
            run_id=run_id,
            paths=paths,
        )
    return receipt.record_id, receipt.status


def sealed_run(
    plan: SealedPlan,
    paths: SealedPaths,
    *,
    ledger_root: Path,
    run_id: str,
    git_commit: str,
    max_wall_seconds: float,
) -> str:
    """Execute one sealed plan inside a recorded run; return the ``run_end`` status.

    The main stream holds only ``run_start`` and ``run_end``: every evaluation
    record is the sealed server's, in the sealed stream, and the server is the
    only budget charge (TASK_008). The caller learns each outcome as a receipt
    and never reads the sealed stream, so the loop times itself against the
    monotonic clock — the first item always runs, and the elapsed time is
    checked only after an item, bounding the sealed charges from above. The
    status is ``completed`` when every item ran, else ``budget_exhausted``; a
    ``failed``, ``timeout`` or ``invalid_weights`` updater receipt is a result,
    and the run continues. Any exception — a :class:`SealedError` refusal or
    server failure included — aborts the run through the
    :class:`~evolvets.runner.runs.Run` context and propagates; records the
    server already wrote stay in the sealed stream.
    """
    _validate_run(plan, max_wall_seconds)
    config = _run_config(plan, max_wall_seconds)
    limits = Budget(
        wall_seconds=max_wall_seconds,
        evaluations=sum(1 for item in plan.items if item.updater_id != "oracle"),
    )
    root = Path(ledger_root)
    with Run.start(
        root,
        run_id,
        condition=SEALED_CONDITION,
        seed=plan.seed,
        config=config,
        budget_limits=limits,
        git_commit=git_commit,
    ) as run:
        started = time.monotonic()
        receipts: list[dict[str, object]] = []
        counts: dict[str, int] = {"oracles": 0, "evaluations": 0}
        statuses: dict[str, int] = {}
        for index, item in enumerate(plan.items):
            if index and time.monotonic() - started >= max_wall_seconds:
                break
            record_id, receipt_status = _evaluate_item(item, plan, paths, root, run_id)
            receipts.append(
                {
                    "dataset": item.dataset,
                    "pool_spec": item.pool_spec,
                    "updater_id": item.updater_id,
                    "record_id": record_id,
                    "status": receipt_status,
                }
            )
            if item.updater_id == "oracle":
                counts["oracles"] += 1
            else:
                counts["evaluations"] += 1
                statuses[receipt_status] = statuses.get(receipt_status, 0) + 1
        status = "completed" if len(receipts) == len(plan.items) else "budget_exhausted"
        run.end(
            status,
            {
                "receipts": receipts,
                "skipped": [
                    {"dataset": dataset, "pool_spec": pool_spec, "updater_id": baseline_id}
                    for dataset, pool_spec, baseline_id in plan.skipped
                ],
                "oracles": counts["oracles"],
                "evaluations": counts["evaluations"],
                "statuses": {key: statuses[key] for key in sorted(statuses)},
            },
        )
    return status
