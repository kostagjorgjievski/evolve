"""Public series inputs the forecast cache is built from.

``cache_store/`` is readable by any process, which is why only public data
may go into it: the real datasets contribute their public segment only,
loaded through :func:`evolvets.arena.realdata.prepare.load_public`, which
never opens the sealed or the raw file, and the synthetic pools are
regenerable by anyone from the repository — the unsalted ``dev`` pool, the
seeded scan presets (TASK_015) and the holdout replicates (TASK_016). Sealed
forecasts are served by the arena's sealed process, never from this store.
"""

from __future__ import annotations

from pathlib import Path

from evolvets.arena.realdata.prepare import load_public, read_split_manifest
from evolvets.arena.synthetic import GENERATOR_VERSION
from evolvets.arena.synthetic.grammar import FAMILIES
from evolvets.arena.synthetic.holdout import (
    HOLDOUT_DATASETS,
    HOLDOUT_INDEX_START,
    HOLDOUT_INDEX_STOP,
    HOLDOUT_PRESETS,
    holdout_series_id,
    holdout_tasks,
)
from evolvets.arena.synthetic.presets import SCAN_DATASETS, SCAN_PRESETS, preset_sha256, scan_tasks
from evolvets.arena.synthetic.seeds import dev_tasks
from evolvets.cache.interfaces import SeriesInput
from evolvets.cache.store import CacheError

CACHE_DATASETS = ("etth1", "ecl", "synthetic-dev")
#: Every dataset a dev pool can load: the cache datasets, the scan presets
#: (TASK_015) and the holdout replicates (TASK_016 PR C). ``all`` on a command
#: line expands to ``CACHE_DATASETS`` only, so the existing cache and tune
#: commands keep their meaning, and the tune and probe commands refuse the
#: holdouts outright.
DEV_DATASETS: tuple[str, ...] = (*CACHE_DATASETS, *SCAN_DATASETS, *HOLDOUT_DATASETS)

_REAL_PERIOD = 24


def public_inputs(
    dataset: str, data_root: Path, manifest_dir: Path, per_family: int | None = None
) -> tuple[dict[str, str], tuple[SeriesInput, ...]]:
    """Return the ``(source, series)`` a cache for ``dataset`` is built from.

    Real datasets yield one series per manifest entry (public values only,
    ``period = 24``); the synthetic datasets yield ``per_family`` series per
    family — in ``FAMILIES`` order for ``synthetic-dev``, in the preset's
    families order for a scan dataset and for a holdout replicate — taking
    only ``world.y`` and ``world.spec.period`` from each task. ``per_family``
    is required for the synthetic datasets and ignored for the real ones.
    """
    if dataset in ("etth1", "ecl"):
        spec = read_split_manifest(manifest_dir, dataset)
        series = tuple(
            SeriesInput(entry.series_id, entry.values, _REAL_PERIOD)
            for entry in load_public(dataset, data_root, manifest_dir)
        )
        source = {
            "kind": "realdata",
            "dataset": dataset,
            "realdata_version": spec.realdata_version,
            "public_sha256": spec.public_sha256,
        }
        return source, series
    if dataset == "synthetic-dev":
        count = _check_per_family(per_family)
        series = tuple(
            SeriesInput(f"{family}/{i}", task.world.y, task.world.spec.period)
            for family in FAMILIES
            for i, task in enumerate(dev_tasks(family, count))
        )
        source = {
            "kind": "synthetic",
            "pool": "dev",
            "generator_version": GENERATOR_VERSION,
            "per_family": str(count),
        }
        return source, series
    if dataset in SCAN_DATASETS:
        preset = SCAN_PRESETS[dataset]
        count = _check_per_family(per_family)
        if count > HOLDOUT_INDEX_START:
            raise CacheError(
                f"{dataset}: per_family must be <= {HOLDOUT_INDEX_START} for a scan dataset; "
                f"scan indexes {HOLDOUT_INDEX_START} and up are the holdout range"
            )
        series = tuple(
            SeriesInput(f"{family}/{i}", task.world.y, task.world.spec.period)
            for family in preset.families
            for i, task in enumerate(scan_tasks(preset.name, family, count))
        )
        source = {
            "kind": "synthetic",
            "pool": "scan",
            "preset": preset.name,
            "preset_sha256": preset_sha256(preset),
            "generator_version": GENERATOR_VERSION,
            "per_family": str(count),
        }
        return source, series
    if dataset in HOLDOUT_DATASETS:
        holdout = HOLDOUT_PRESETS[dataset]
        base = SCAN_PRESETS[holdout.base]
        count = _check_per_family(per_family)
        if count > HOLDOUT_INDEX_STOP - HOLDOUT_INDEX_START:
            raise CacheError(
                f"{dataset}: per_family must be <= {HOLDOUT_INDEX_STOP - HOLDOUT_INDEX_START}, "
                f"the holdout index range"
            )
        series = tuple(
            SeriesInput(holdout_series_id(family, i), task.world.y, task.world.spec.period)
            for family in base.families
            for i, task in enumerate(holdout_tasks(dataset, family, count))
        )
        source = {
            "kind": "synthetic",
            "pool": "holdout",
            "preset": holdout.base,
            "preset_sha256": preset_sha256(base),
            "index_start": str(holdout.index_start),
            "generator_version": GENERATOR_VERSION,
            "per_family": str(count),
        }
        return source, series
    raise CacheError(f"unknown dataset {dataset!r}; expected one of {list(DEV_DATASETS)}")


def _check_per_family(per_family: int | None) -> int:
    """Return ``per_family`` as an int >= 1 (bools rejected), else raise CacheError."""
    if per_family is None:
        raise CacheError("per_family is required for the synthetic datasets")
    if isinstance(per_family, bool) or not isinstance(per_family, int):
        raise CacheError(f"per_family must be an int >= 1, got {per_family!r}")
    if per_family < 1:
        raise CacheError(f"per_family must be an int >= 1, got {per_family}")
    return per_family
