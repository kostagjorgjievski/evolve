"""Public series inputs the forecast cache is built from.

``cache_store/`` is readable by any process, which is why only public data
may go into it: the real datasets contribute their public segment only,
loaded through :func:`evolvets.arena.realdata.prepare.load_public`, which
never opens the sealed or the raw file, and the synthetic pool is the
unsalted ``dev`` pool, regenerable by anyone from the repository. Sealed
forecasts are served by the arena's sealed process, never from this store.
"""

from __future__ import annotations

from pathlib import Path

from evolvets.arena.realdata.prepare import load_public, read_split_manifest
from evolvets.arena.synthetic import GENERATOR_VERSION
from evolvets.arena.synthetic.grammar import FAMILIES
from evolvets.arena.synthetic.seeds import dev_tasks
from evolvets.cache.interfaces import SeriesInput
from evolvets.cache.store import CacheError

CACHE_DATASETS = ("etth1", "ecl", "synthetic-dev")

_REAL_PERIOD = 24


def public_inputs(
    dataset: str, data_root: Path, manifest_dir: Path, per_family: int | None = None
) -> tuple[dict[str, str], tuple[SeriesInput, ...]]:
    """Return the ``(source, series)`` a cache for ``dataset`` is built from.

    Real datasets yield one series per manifest entry (public values only,
    ``period = 24``); ``synthetic-dev`` yields ``per_family`` series per
    family in ``FAMILIES`` order, taking only ``world.y`` and
    ``world.spec.period`` from each task. ``per_family`` is required for
    ``synthetic-dev`` and ignored for the real datasets.
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
    raise CacheError(f"unknown dataset {dataset!r}; expected one of {list(CACHE_DATASETS)}")


def _check_per_family(per_family: int | None) -> int:
    """Return ``per_family`` as an int >= 1 (bools rejected), else raise CacheError."""
    if per_family is None:
        raise CacheError("per_family is required for the synthetic-dev dataset")
    if isinstance(per_family, bool) or not isinstance(per_family, int):
        raise CacheError(f"per_family must be an int >= 1, got {per_family!r}")
    if per_family < 1:
        raise CacheError(f"per_family must be an int >= 1, got {per_family}")
    return per_family
