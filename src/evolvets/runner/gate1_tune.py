"""Gate 1 tuning on dev: every grid configuration replayed inside one recorded run.

The grids (``gate1_grids``) are fixed in code before any real dev run, so
the run's ``run_start`` pins them together with its git commit (TASK_011).
One run holds, for each dev dataset and pool spec in order, the hindsight
oracles and then every configuration of every baseline's grid: one dev
``baseline`` record per replay, a failed updater included as a result and
counted, never hidden (AGENTS.md 2.8). Each replay charges one evaluation
and its measured wall-clock; the oracle charges wall-clock and no
evaluation, the sealed task's convention for the instrument itself. Every
pool loads before the run starts, so a pool that cannot load leaves nothing
in the ledger, and the tuned configurations are selected by the fixed rule
of ``gate1_grids.select_configs``, recomputed from the run's own records.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolvets.arena.dev import DevPool, load_dev_pool
from evolvets.arena.errors import ArenaError
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.arena.oracles import oracle_dataset, oracle_to_obj
from evolvets.arena.pool import POOL_SPECS
from evolvets.arena.synthetic.holdout import HOLDOUT_DATASETS
from evolvets.cache.datasets import DEV_DATASETS
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import Baseline
from evolvets.ledger.schema import Budget
from evolvets.ledger.store import LedgerReader
from evolvets.runner.gate1_grids import GATE1_GRIDS, SELECTION_METRIC, Selection, select_configs
from evolvets.runner.runs import Run
from evolvets.updater.baselines import BASELINE_IDS, BASELINES
from evolvets.updater.interfaces import ConfigValue, Updater

TUNE_CONDITION = "gate1-dev-tune"
_SEED_LIMIT = 2**63
_UPDATER_REF = "evolvets.updater.baselines:"


@dataclass(frozen=True)
class TuneSpec:
    """What the tuning run sweeps, in run order, and the budget it may spend."""

    datasets: tuple[str, ...]  # each in DEV_DATASETS, no duplicates, in run order
    pool_specs: tuple[str, ...]  # each in POOL_SPECS, no duplicates, in run order
    foundation_models: tuple[str, ...] | None  # None: the registry's built models
    seed: int
    max_wall_seconds: float  # > 0, finite


@dataclass(frozen=True)
class DevPaths:
    """The roots a dev pool loads its cache and public data from."""

    cache_root: Path
    foundation_root: Path
    data_root: Path
    manifest_dir: Path


def planned_replays(
    spec: TuneSpec, grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]]
) -> int:
    """Return the number of updater replays the sweep plans: one per grid point per pool."""
    points = 0
    for grid in grids.values():
        points += len(grid)
    return len(spec.datasets) * len(spec.pool_specs) * points


def _validate_spec(
    spec: TuneSpec, grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]]
) -> None:
    """Refuse a spec or grid the run cannot honour, before anything is loaded."""
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
        raise ValueError("grids must map at least one baseline id to its grid")
    for baseline_id, grid in grids.items():
        if baseline_id not in BASELINE_IDS:
            raise ValueError(f"grids keys must be baseline ids, got {baseline_id!r}")
        if not grid:
            raise ValueError(f"the grid of {baseline_id!r} must hold at least one configuration")
        for config in grid:
            if not isinstance(config, Mapping):
                raise ValueError(
                    f"every {baseline_id!r} grid point must be a mapping, got {config!r}"
                )
            for key, value in config.items():
                if not isinstance(key, str):
                    raise ValueError(f"grid point keys must be strings, got {key!r}")
                if isinstance(value, bool) or not isinstance(value, (float, int, str)):
                    raise ValueError(
                        f"grid point values must be float, int, str or bool, got {value!r}"
                    )


def _load_pools(spec: TuneSpec, paths: DevPaths) -> dict[tuple[str, str], DevPool]:
    """Load every (dataset, pool spec) pool, in run order, before the run starts."""
    pools: dict[tuple[str, str], DevPool] = {}
    for dataset in spec.datasets:
        for pool_spec in spec.pool_specs:
            pools[(dataset, pool_spec)] = load_dev_pool(
                dataset,
                paths.cache_root,
                paths.data_root,
                paths.manifest_dir,
                pool_spec=pool_spec,
                foundation_root=(
                    paths.foundation_root if pool_spec == "simple+foundation" else None
                ),
                foundation_models=(
                    spec.foundation_models if pool_spec == "simple+foundation" else None
                ),
            )
    return pools


def _run_config(
    spec: TuneSpec,
    grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]],
    pools: Mapping[tuple[str, str], DevPool],
) -> dict[str, object]:
    """Build the ``run_start`` config blob: what runs, from which caches, under which grids."""
    simple_cache_ids: dict[str, str] = {}
    foundation_cache_ids: dict[str, dict[str, list[str]]] = {}
    foundation_models: dict[str, dict[str, list[str]]] = {}
    for dataset in spec.datasets:
        ids: dict[str, list[str]] = {}
        models: dict[str, list[str]] = {}
        for pool_spec in spec.pool_specs:
            pool = pools[(dataset, pool_spec)]
            if dataset in simple_cache_ids and simple_cache_ids[dataset] != pool.cache_id:
                raise ArenaError(
                    f"{dataset}: pool spec {pool_spec!r} replays simple cache {pool.cache_id}, "
                    f"but {simple_cache_ids[dataset]} was loaded first; a dataset's pool specs "
                    "must share one simple cache"
                )
            simple_cache_ids.setdefault(dataset, pool.cache_id)
            ids[pool_spec] = [cache.cache_id for cache in pool.foundation]
            models[pool_spec] = [cache.model for cache in pool.foundation]
        foundation_cache_ids[dataset] = ids
        foundation_models[dataset] = models
    return {
        "experiment": "gate1",
        "stage": "dev-tune",
        "datasets": list(spec.datasets),
        "pool_specs": list(spec.pool_specs),
        "simple_cache_ids": simple_cache_ids,
        "foundation_cache_ids": foundation_cache_ids,
        "foundation_models": foundation_models,
        "grids": {
            baseline_id: [dict(config) for config in grid] for baseline_id, grid in grids.items()
        },
        "selection_metric": SELECTION_METRIC,
        "seed": spec.seed,
        "max_wall_seconds": spec.max_wall_seconds,
        "updater_refs": {baseline_id: f"{_UPDATER_REF}{baseline_id}" for baseline_id in grids},
    }


class _TimedFactory:
    """A baseline factory that timestamps each call; replay calls it once per series."""

    def __init__(self, make: Callable[[], Updater]) -> None:
        self._make = make
        self._calls: list[float] = []

    def __call__(self) -> Updater:
        self._calls.append(time.monotonic())
        return self._make()

    def series_wall_seconds(self, end: float) -> list[float]:
        """Time from each call to the next; the last entry runs to ``end``."""
        stamps = [*self._calls, end]
        return [stamps[index + 1] - stamps[index] for index in range(len(self._calls))]


def _record_oracle(run: Run, pool: DevPool, counts: dict[str, int]) -> None:
    """Compute one pool's hindsight oracles and append their baseline record."""
    started = time.monotonic()
    oracle = oracle_dataset(pool)
    run.append(
        "baseline",
        Baseline(
            updater_id="oracle",
            hyperparams={},
            pool="dev",
            status="ok",
            metrics=dict(oracle.metrics),
            per_task_blob=run.put_blob({"result": oracle_to_obj(oracle)}),
            error=None,
        ),
        budget=Budget(wall_seconds=time.monotonic() - started),
    )
    counts["oracles"] += 1


def _record_replay(
    run: Run,
    pool: DevPool,
    baseline_id: str,
    config: Mapping[str, ConfigValue],
    seed: int,
    counts: dict[str, int],
    *,
    factories: Mapping[str, Callable[[], Updater]] = BASELINES,
    ref_prefix: str = _UPDATER_REF,
) -> None:
    """Replay one grid configuration over one pool and append its baseline record.

    ``factories`` names the updater factories by id and ``ref_prefix`` the
    module prefix of the blob's ``updater_ref``: the baselines and their
    prefix by default, the learned updaters and ``evolvets.updater.learned:``
    for the probe (TASK_014 PR C).
    """
    started = time.monotonic()
    timed = _TimedFactory(factories[baseline_id])
    result = replay_dataset(timed, config, seed, pool)
    ended = time.monotonic()
    run.append(
        "baseline",
        Baseline(
            updater_id=baseline_id,
            hyperparams=dict(config),
            pool="dev",
            status=result.status,
            metrics=dict(result.metrics),
            per_task_blob=run.put_blob(
                {
                    "updater_ref": f"{ref_prefix}{baseline_id}",
                    "series_wall_seconds": timed.series_wall_seconds(ended),
                    "result": result_to_obj(result),
                }
            ),
            error=result.error,
        ),
        budget=Budget(wall_seconds=ended - started, evaluations=1),
    )
    counts["replays"] += 1
    counts[result.status] += 1


def _tune_loop(
    run: Run,
    spec: TuneSpec,
    pools: Mapping[tuple[str, str], DevPool],
    grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]],
    counts: dict[str, int],
) -> None:
    """Sweep every pool, oracle first, grid configuration by grid configuration.

    The loop stops (a plain ``return``) as soon as spent wall-clock reaches
    the spec's limit; the record whose charge crosses the limit has already
    been written as measured, never clipped.
    """
    for dataset in spec.datasets:
        for pool_spec in spec.pool_specs:
            if run.spent.wall_seconds >= spec.max_wall_seconds:
                return
            pool = pools[(dataset, pool_spec)]
            _record_oracle(run, pool, counts)
            for baseline_id in BASELINE_IDS:
                if baseline_id not in grids:
                    continue
                for config in grids[baseline_id]:
                    if run.spent.wall_seconds >= spec.max_wall_seconds:
                        return
                    # The module mapping, read here at call time (not the
                    # default, bound at import): a test that patches
                    # ``gate1_tune.BASELINES`` still reaches the replay.
                    _record_replay(
                        run, pool, baseline_id, config, spec.seed, counts, factories=BASELINES
                    )


def _select_this_run(ledger_root: Path, run_id: str) -> Selection:
    """Recompute the selection from the run's own records and blobs."""
    blobs = BlobStore(Path(ledger_root) / run_id / "blobs")

    def read_blob(ref: str) -> object:
        parsed: object = json.loads(blobs.get(ref).decode("utf-8"))
        return parsed

    records = list(LedgerReader(Path(ledger_root), run_id).records())
    return select_configs(records, read_blob)


def tune(
    spec: TuneSpec,
    paths: DevPaths,
    *,
    ledger_root: Path,
    run_id: str,
    git_commit: str,
    grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]] = GATE1_GRIDS,
) -> str:
    """Run the whole dev tuning sweep inside one recorded run; return the ``run_end`` status.

    The status is ``completed`` when every planned oracle and replay ran, else
    ``budget_exhausted``. Any exception inside the loop aborts the run through
    the :class:`~evolvets.runner.runs.Run` context and propagates.
    """
    _validate_spec(spec, grids)
    pools = _load_pools(spec, paths)
    config = _run_config(spec, grids, pools)
    limits = Budget(
        wall_seconds=spec.max_wall_seconds,
        evaluations=planned_replays(spec, grids),
    )
    counts = {"replays": 0, "oracles": 0, "ok": 0, "failed": 0, "invalid_weights": 0}
    with Run.start(
        Path(ledger_root),
        run_id,
        condition=TUNE_CONDITION,
        seed=spec.seed,
        config=config,
        budget_limits=limits,
        git_commit=git_commit,
    ) as run:
        _tune_loop(run, spec, pools, grids, counts)
        status = (
            "completed"
            if counts["oracles"] == len(spec.datasets) * len(spec.pool_specs)
            and counts["replays"] == planned_replays(spec, grids)
            else "budget_exhausted"
        )
        summary: dict[str, object] = dict(counts)
        summary["selected"] = _select_this_run(Path(ledger_root), run_id)
        run.end(status, summary)
    return status
