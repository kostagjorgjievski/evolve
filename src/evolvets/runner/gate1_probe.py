"""The learned-updater probe: one recorded dev run pinned to a tune run (TASK_014 PR C).

:func:`probe` replays every grid point of every learned updater
(:mod:`evolvets.updater.learned`) over the dev pools a completed dev tune
run covers, inside one recorded run pinned to that tune run: its config
names the tune run and its ``run_end`` record, it replays only (dataset,
pool spec) pairs the tune run's config covers, refuses any pool whose
caches are not the tune run's, and requires the tune run's protocol
versions to equal this code's, so numbers from different protocol versions
are never mixed (AGENTS.md 2.4). There are no oracle records — the oracles
are the tune run's. Each replay is one dev ``baseline`` record charged one
evaluation and its measured wall-clock, exactly like a tune replay; a
failed updater is a result, recorded and counted, never hidden
(AGENTS.md 2.8). Every refusal before start raises :class:`ProbeError`
(or ``ValueError`` for a spec or grid this code cannot honour) with
nothing written.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from evolvets.arena.dev import DevPool
from evolvets.arena.pool import POOL_SPECS
from evolvets.arena.synthetic.holdout import HOLDOUT_DATASETS
from evolvets.arena.versions import protocol_versions
from evolvets.cache.datasets import DEV_DATASETS
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import RunEnd, RunStart
from evolvets.ledger.schema import Budget
from evolvets.ledger.store import LedgerReader
from evolvets.runner.gate1_grids import SELECTION_METRIC, Selection, select_configs
from evolvets.runner.gate1_tune import (
    TUNE_CONDITION,
    DevPaths,
    TuneSpec,
    _load_pools,
    _record_replay,
)
from evolvets.runner.runs import Run
from evolvets.updater.interfaces import ConfigValue
from evolvets.updater.learned import LEARNED, LEARNED_IDS

PROBE_CONDITION = "gate1-dev-probe"
_UPDATER_REF = "evolvets.updater.learned:"
_SEED_LIMIT = 2**63

#: The grids the probe replays, fixed in code before any real dev run (as
#: ``GATE1_GRIDS`` is, TASK_011): three forgetting half-lives crossed with two
#: shrinkages for ``ridge_stack``, and three gate widths crossed with two
#: half-lives for ``loss_gate`` — six points each, the outer loop varying
#: slower. Each ``LEARNED_DEFAULT_CONFIGS`` entry is inside its grid.
PROBE_GRIDS: Mapping[str, tuple[Mapping[str, ConfigValue], ...]] = MappingProxyType(
    {
        "ridge_stack": tuple(
            {"half_life": half_life, "shrink": shrink, "refit_every": 24}
            for half_life in (48.0, 336.0, 2016.0)
            for shrink in (0.1, 1.0)
        ),
        "loss_gate": tuple(
            {"beta": beta, "half_life": half_life, "ridge": 0.01, "refit_every": 24}
            for beta in (4.0, 16.0, 64.0)
            for half_life in (336.0, 2016.0)
        ),
    }
)


class ProbeError(Exception):
    """Raised when the tune run a probe would build on is unusable."""


@dataclass(frozen=True)
class ProbeSpec:
    """What the probe replays, in run order, and the budget it may spend."""

    tune_run_id: str
    datasets: tuple[str, ...]  # each in DEV_DATASETS, no duplicates, in run order
    pool_specs: tuple[str, ...]  # each in POOL_SPECS, no duplicates, in run order
    foundation_models: tuple[str, ...] | None  # None: the registry's built models
    seed: int
    max_wall_seconds: float  # > 0, finite


@dataclass(frozen=True)
class _TuneRun:
    """What the probe needs from its tune run: its config and ``run_end`` id."""

    config: dict[str, object]
    run_end_record_id: str


def _planned_replays(
    spec: ProbeSpec, grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]]
) -> int:
    """Return the number of replays the probe plans: one per grid point per pool."""
    points = 0
    for grid in grids.values():
        points += len(grid)
    return len(spec.datasets) * len(spec.pool_specs) * points


def _validate_spec(
    spec: ProbeSpec, grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]]
) -> None:
    """Refuse a spec or grid the run cannot honour, before anything is read."""
    if not spec.datasets:
        raise ValueError("datasets must name at least one dataset")
    if len(set(spec.datasets)) != len(spec.datasets):
        raise ValueError(f"datasets must not repeat: {list(spec.datasets)}")
    for dataset in spec.datasets:
        if dataset not in DEV_DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}; expected one of {list(DEV_DATASETS)}")
        if dataset in HOLDOUT_DATASETS:
            raise ValueError(
                f"{dataset} is a holdout dataset: holdouts score accepted edits only "
                "and are never tuned on"
            )
    if not spec.pool_specs:
        raise ValueError("pool_specs must name at least one pool spec")
    if len(set(spec.pool_specs)) != len(spec.pool_specs):
        raise ValueError(f"pool_specs must not repeat: {list(spec.pool_specs)}")
    for pool_spec in spec.pool_specs:
        if pool_spec not in POOL_SPECS:
            raise ValueError(f"unknown pool spec {pool_spec!r}; expected one of {list(POOL_SPECS)}")
    if isinstance(spec.seed, bool) or not isinstance(spec.seed, int):
        raise ValueError(f"seed must be an int in [0, {_SEED_LIMIT}), got {spec.seed!r}")
    if spec.seed < 0 or spec.seed >= _SEED_LIMIT:
        raise ValueError(f"seed must be an int in [0, {_SEED_LIMIT}), got {spec.seed}")
    if (
        isinstance(spec.max_wall_seconds, bool)
        or not isinstance(spec.max_wall_seconds, (int, float))
        or not math.isfinite(spec.max_wall_seconds)
        or spec.max_wall_seconds <= 0
    ):
        raise ValueError(
            f"max_wall_seconds must be a finite number > 0, got {spec.max_wall_seconds!r}"
        )
    if not grids:
        raise ValueError("grids must map at least one learned id to its grid")
    for learned_id, grid in grids.items():
        if learned_id not in LEARNED_IDS:
            raise ValueError(f"grids keys must be learned ids, got {learned_id!r}")
        if not grid:
            raise ValueError(f"the grid of {learned_id!r} must hold at least one configuration")
        for config in grid:
            if not isinstance(config, Mapping):
                raise ValueError(
                    f"every {learned_id!r} grid point must be a mapping, got {config!r}"
                )
            for key, value in config.items():
                if not isinstance(key, str):
                    raise ValueError(f"grid point keys must be strings, got {key!r}")
                if isinstance(value, bool) or not isinstance(value, (float, int, str)):
                    raise ValueError(f"grid point values must be float, int or str, got {value!r}")


def _config_of(ledger_root: Path, run_id: str, record: RunStart) -> dict[str, object]:
    """The parsed config blob of ``run_id``'s ``run_start``; ``ProbeError`` if unusable."""
    raw = BlobStore(Path(ledger_root) / run_id / "blobs").get(record.config_blob)
    try:
        parsed: object = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise ProbeError(
            f"tune run {run_id}: the run_start config blob is not JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ProbeError(f"tune run {run_id}: the run_start config blob is not a JSON object")
    return cast("dict[str, object]", parsed)


def _tune_axes(config: Mapping[str, object], tune_run_id: str) -> tuple[list[str], list[str]]:
    """The tune config's datasets and pool specs, in run order; ``ProbeError`` if malformed."""
    datasets = config.get("datasets")
    pool_specs = config.get("pool_specs")
    if (
        not isinstance(datasets, list)
        or not datasets
        or not all(isinstance(name, str) for name in datasets)
    ):
        raise ProbeError(
            f"tune run {tune_run_id}: config datasets must be a non-empty list of names"
        )
    if (
        not isinstance(pool_specs, list)
        or not pool_specs
        or not all(isinstance(name, str) for name in pool_specs)
    ):
        raise ProbeError(
            f"tune run {tune_run_id}: config pool_specs must be a non-empty list of names"
        )
    return datasets, pool_specs


def _read_tune_run(ledger_root: Path, spec: ProbeSpec) -> _TuneRun:
    """Verify the tune run and read what pins the probe to it; ``ProbeError`` on any refusal.

    The tune run must verify (blobs included), open with a ``run_start`` of
    the tune condition and experiment ``gate1``, end ``completed``, and pin
    the protocol versions this code runs; every requested (dataset, pool
    spec) must be covered by its config.
    """
    root = Path(ledger_root)
    chain = LedgerReader(root, spec.tune_run_id).verify(check_blobs=True)
    if not chain.ok:
        raise ProbeError(f"tune run {spec.tune_run_id} does not verify: {chain.reason}")
    records = list(LedgerReader(root, spec.tune_run_id).records(["main"]))
    if not records:
        raise ProbeError(f"tune run {spec.tune_run_id}: the main stream is empty; no run_start")
    first = records[0]
    if first.kind != "run_start" or not isinstance(first.payload, RunStart):
        raise ProbeError(f"tune run {spec.tune_run_id}: the first record is not a run_start")
    start = first.payload
    if start.condition != TUNE_CONDITION:
        raise ProbeError(
            f"tune run {spec.tune_run_id}: condition is {start.condition!r}, not {TUNE_CONDITION!r}"
        )
    config = _config_of(root, spec.tune_run_id, start)
    if config.get("experiment") != "gate1":
        raise ProbeError(
            f"tune run {spec.tune_run_id}: config experiment is "
            f"{config.get('experiment')!r}, not 'gate1'"
        )
    last = records[-1]
    if last.kind != "run_end" or not isinstance(last.payload, RunEnd):
        raise ProbeError(f"tune run {spec.tune_run_id} has no run_end; its pools are not settled")
    if last.payload.status != "completed":
        raise ProbeError(
            f"tune run {spec.tune_run_id} ended {last.payload.status}; "
            "only a completed tune run pins its pools"
        )
    if start.protocol_versions != protocol_versions():
        raise ProbeError(
            f"tune run {spec.tune_run_id} pins different protocol versions than this code"
        )
    datasets, pool_specs = _tune_axes(config, spec.tune_run_id)
    for dataset in spec.datasets:
        if dataset not in datasets:
            raise ProbeError(
                f"tune run {spec.tune_run_id} does not cover dataset {dataset!r} "
                f"(it covers {datasets})"
            )
    for pool_spec in spec.pool_specs:
        if pool_spec not in pool_specs:
            raise ProbeError(
                f"tune run {spec.tune_run_id} does not cover pool spec {pool_spec!r} "
                f"(it covers {pool_specs})"
            )
    return _TuneRun(config=config, run_end_record_id=last.record_id)


def _pin_pools(pools: Mapping[tuple[str, str], DevPool], tune: _TuneRun, tune_run_id: str) -> None:
    """Refuse any pool whose caches are not the ones the tune run replayed."""
    simple_ids = tune.config.get("simple_cache_ids")
    foundation_ids = tune.config.get("foundation_cache_ids")
    if not isinstance(simple_ids, dict) or not isinstance(foundation_ids, dict):
        raise ProbeError(
            f"tune run {tune_run_id}: config simple_cache_ids and foundation_cache_ids "
            "must be mappings"
        )
    for (dataset, pool_spec), pool in pools.items():
        expected_simple = simple_ids.get(dataset)
        if not isinstance(expected_simple, str):
            raise ProbeError(
                f"tune run {tune_run_id}: config simple_cache_ids has no id for {dataset!r}"
            )
        if pool.cache_id != expected_simple:
            raise ProbeError(
                f"{dataset}/{pool_spec}: replays simple cache {pool.cache_id}, but the tune "
                f"run replayed {expected_simple}; a probe replays the tune run's caches"
            )
        by_dataset = foundation_ids.get(dataset)
        if not isinstance(by_dataset, dict):
            raise ProbeError(
                f"tune run {tune_run_id}: config foundation_cache_ids has no entry for {dataset!r}"
            )
        expected_foundation = by_dataset.get(pool_spec)
        if not isinstance(expected_foundation, list):
            raise ProbeError(
                f"tune run {tune_run_id}: config foundation_cache_ids has no list for "
                f"{dataset}/{pool_spec}"
            )
        joined = [cache.cache_id for cache in pool.foundation]
        if joined != expected_foundation:
            raise ProbeError(
                f"{dataset}/{pool_spec}: joins foundation caches {joined}, but the tune "
                f"run joined {expected_foundation}"
            )


def _run_config(
    spec: ProbeSpec,
    grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]],
    pools: Mapping[tuple[str, str], DevPool],
    tune: _TuneRun,
) -> dict[str, object]:
    """Build the probe's ``run_start`` config blob: the grids, caches and tune run it pins."""
    simple_cache_ids: dict[str, str] = {}
    foundation_cache_ids: dict[str, dict[str, list[str]]] = {}
    foundation_models: dict[str, dict[str, list[str]]] = {}
    for dataset in spec.datasets:
        ids: dict[str, list[str]] = {}
        models: dict[str, list[str]] = {}
        for pool_spec in spec.pool_specs:
            pool = pools[(dataset, pool_spec)]
            simple_cache_ids.setdefault(dataset, pool.cache_id)
            ids[pool_spec] = [cache.cache_id for cache in pool.foundation]
            models[pool_spec] = [cache.model for cache in pool.foundation]
        foundation_cache_ids[dataset] = ids
        foundation_models[dataset] = models
    return {
        "experiment": "gate1",
        "stage": "dev-probe",
        "tune_run_id": spec.tune_run_id,
        "tune_run_end_record_id": tune.run_end_record_id,
        "datasets": list(spec.datasets),
        "pool_specs": list(spec.pool_specs),
        "simple_cache_ids": simple_cache_ids,
        "foundation_cache_ids": foundation_cache_ids,
        "foundation_models": foundation_models,
        "grids": {
            learned_id: [dict(config) for config in grid] for learned_id, grid in grids.items()
        },
        "selection_metric": SELECTION_METRIC,
        "seed": spec.seed,
        "max_wall_seconds": spec.max_wall_seconds,
        "updater_refs": {learned_id: f"{_UPDATER_REF}{learned_id}" for learned_id in grids},
    }


def _select_this_run(ledger_root: Path, run_id: str) -> Selection:
    """Recompute the learned selection from the probe run's own records and blobs."""
    blobs = BlobStore(Path(ledger_root) / run_id / "blobs")

    def read_blob(ref: str) -> object:
        parsed: object = json.loads(blobs.get(ref).decode("utf-8"))
        return parsed

    records = list(LedgerReader(Path(ledger_root), run_id).records())
    return select_configs(records, read_blob, updater_ids=LEARNED_IDS)


def _probe_loop(
    run: Run,
    spec: ProbeSpec,
    pools: Mapping[tuple[str, str], DevPool],
    grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]],
    counts: dict[str, int],
) -> None:
    """Sweep every pool, grid configuration by grid configuration; no oracle is recomputed.

    The loop stops (a plain ``return``) as soon as spent wall-clock reaches
    the spec's limit, exactly as the tune loop does; the record whose charge
    crosses the limit has already been written as measured, never clipped.
    """
    for dataset in spec.datasets:
        for pool_spec in spec.pool_specs:
            if run.spent.wall_seconds >= spec.max_wall_seconds:
                return
            pool = pools[(dataset, pool_spec)]
            for learned_id in LEARNED_IDS:
                if learned_id not in grids:
                    continue
                for config in grids[learned_id]:
                    if run.spent.wall_seconds >= spec.max_wall_seconds:
                        return
                    _record_replay(
                        run,
                        pool,
                        learned_id,
                        config,
                        spec.seed,
                        counts,
                        factories=LEARNED,
                        ref_prefix=_UPDATER_REF,
                    )


def probe(
    spec: ProbeSpec,
    paths: DevPaths,
    *,
    ledger_root: Path,
    run_id: str,
    git_commit: str,
    grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]] = PROBE_GRIDS,
) -> str:
    """Run the learned grids' replay inside one recorded run; return the ``run_end`` status.

    The status is ``completed`` when every planned replay ran, else
    ``budget_exhausted``. Any exception inside the loop aborts the run
    through the :class:`~evolvets.runner.runs.Run` context and propagates;
    every refusal before start — a bad spec or grid (``ValueError``), an
    unusable tune run (:class:`ProbeError`), a pool that cannot load
    (:class:`~evolvets.arena.errors.ArenaError`) — raises with nothing
    written.
    """
    _validate_spec(spec, grids)
    tune_run = _read_tune_run(Path(ledger_root), spec)
    load_spec = TuneSpec(
        datasets=spec.datasets,
        pool_specs=spec.pool_specs,
        foundation_models=spec.foundation_models,
        seed=spec.seed,
        max_wall_seconds=spec.max_wall_seconds,
    )
    pools = _load_pools(load_spec, paths)
    _pin_pools(pools, tune_run, spec.tune_run_id)
    config = _run_config(spec, grids, pools, tune_run)
    limits = Budget(
        wall_seconds=spec.max_wall_seconds,
        evaluations=_planned_replays(spec, grids),
    )
    counts = {"replays": 0, "ok": 0, "failed": 0, "invalid_weights": 0}
    with Run.start(
        Path(ledger_root),
        run_id,
        condition=PROBE_CONDITION,
        seed=spec.seed,
        config=config,
        budget_limits=limits,
        git_commit=git_commit,
    ) as run:
        _probe_loop(run, spec, pools, grids, counts)
        status = (
            "completed"
            if counts["replays"] == _planned_replays(spec, grids)
            else "budget_exhausted"
        )
        summary: dict[str, object] = dict(counts)
        summary["selected"] = _select_this_run(Path(ledger_root), run_id)
        run.end(status, summary)
    return status
