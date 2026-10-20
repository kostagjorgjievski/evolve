"""Dev pools: the public series and scoring boundaries a replay is driven over.

A dev pool ties a cache to the public values it was built from, plus the
warmup/dev split boundaries that decide which origins are scored and which
window the MASE scale is read from. Synthetic pools also carry their change
points and the true-best forecasts the oracle metrics need. Nothing here holds
a value past the last scored target: ``values`` is truncated to
``[: score_end + horizon]`` after the hash check.

The sealed pools (``arena/sealed/pools.py``) reuse these types and helpers
unchanged, so a replay cannot tell a sealed pool from a dev pool apart from
its boundaries; the type names are historical.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from evolvets.arena.errors import ArenaError
from evolvets.arena.pool import FoundationCache, check_pool_spec, load_foundation_caches
from evolvets.arena.realdata.prepare import PrepareError, load_public
from evolvets.arena.synthetic import GENERATOR_VERSION
from evolvets.arena.synthetic.grammar import FAMILIES, GrammarConfig, SyntheticTask
from evolvets.arena.synthetic.holdout import (
    HOLDOUT_DATASETS,
    HOLDOUT_INDEX_START,
    HOLDOUT_INDEX_STOP,
    HOLDOUT_PRESETS,
    holdout_series_id,
    holdout_tasks,
)
from evolvets.arena.synthetic.oracle import true_best_forecast
from evolvets.arena.synthetic.presets import SCAN_DATASETS, SCAN_PRESETS, preset_sha256, scan_tasks
from evolvets.arena.synthetic.seeds import dev_tasks
from evolvets.cache.datasets import DEV_DATASETS
from evolvets.cache.interfaces import CachedSeries, CacheManifest
from evolvets.cache.store import CacheError, cache_id, read_manifest
from evolvets.ledger.canonical import canonical_bytes, sha256_hex

_REAL_PERIOD = 24


@dataclass(frozen=True)
class DevSeries:
    """One public series with the boundaries its replay and scoring use."""

    series_id: str
    period: int
    values: tuple[float, ...]  # exactly values[: score_end + horizon]
    scale_end: int  # MASE scale uses y_0 .. y_{scale_end - 1}
    score_start: int  # first scored origin
    score_end: int  # exclusive; also the replay's end_origin
    change_points: tuple[int, ...]  # synthetic only, else ()
    oracle: (
        tuple[tuple[float, ...], ...] | None
    )  # true-best forecasts, [origin - score_start][h-1]; synthetic only


@dataclass(frozen=True)
class DevPool:
    """A cache plus the public series to replay over, in cache manifest order."""

    dataset: str  # one of DEV_DATASETS (dev) or SEALED_DATASETS (sealed)
    cache_root: Path
    cache_id: str
    manifest: CacheManifest
    series: tuple[DevSeries, ...]  # cache manifest order
    pool_spec: str = "simple"  # which experts the pool replays (arena/pool.py)
    foundation: tuple[FoundationCache, ...] = ()  # () unless pool_spec is "simple+foundation"


def load_dev_pool(
    dataset: str,
    cache_root: Path,
    data_root: Path,
    manifest_dir: Path,
    *,
    pool_spec: str = "simple",
    foundation_root: Path | None = None,
    foundation_models: Sequence[str] | None = None,
) -> DevPool:
    """Load the dev pool for ``dataset`` from its cache and its public data.

    With ``pool_spec="simple+foundation"`` the foundation caches under
    ``foundation_root`` are loaded and joined onto the simple cache (the
    pinned registry models by default, ``foundation_models`` to name others,
    as tests name the fake). Every failure — an unknown dataset, a missing
    or tampered cache, missing or tampered public data, a foundation cache
    that does not join — surfaces as :class:`ArenaError`, so no caller needs
    to know the cache's and the real-data layer's own error types.
    """
    if dataset not in DEV_DATASETS:
        raise ArenaError(f"unknown dataset {dataset!r}; expected one of {list(DEV_DATASETS)}")
    spec = check_pool_spec(pool_spec)
    if spec == "simple" and (foundation_root is not None or foundation_models is not None):
        raise ArenaError("pool_spec 'simple' takes no foundation caches")
    try:
        manifest = read_manifest(cache_root, dataset)
        ident = cache_id(cache_root, dataset)
    except CacheError as exc:
        raise ArenaError(str(exc)) from exc
    foundation: tuple[FoundationCache, ...] = ()
    if spec == "simple+foundation":
        if foundation_root is None:
            raise ArenaError("pool_spec 'simple+foundation' needs a foundation cache root")
        foundation = load_foundation_caches(
            dataset, ident, manifest, foundation_root, foundation_models
        )
    if dataset in ("etth1", "ecl"):
        series = _real_series(dataset, manifest, data_root, manifest_dir)
    elif dataset == "synthetic-dev":
        series = _synthetic_series(manifest)
    elif dataset in SCAN_DATASETS:
        series = _scan_series(manifest)
    elif dataset in HOLDOUT_DATASETS:
        series = _holdout_series(manifest)
    else:  # unreachable: every DEV_DATASETS entry is dispatched above
        raise ArenaError(f"unknown dataset {dataset!r}; expected one of {list(DEV_DATASETS)}")
    return DevPool(
        dataset=dataset,
        cache_root=Path(cache_root),
        cache_id=ident,
        manifest=manifest,
        series=series,
        pool_spec=spec,
        foundation=foundation,
    )


def _real_series(
    dataset: str, manifest: CacheManifest, data_root: Path, manifest_dir: Path
) -> tuple[DevSeries, ...]:
    """Build the dev series of a real dataset from its public values and split."""
    try:
        public = load_public(dataset, data_root, manifest_dir)
    except PrepareError as exc:
        raise ArenaError(str(exc)) from exc
    if len(public) != len(manifest.series):
        raise ArenaError(
            f"{manifest.dataset}: the cache holds {len(manifest.series)} series but the "
            f"public data has {len(public)}"
        )
    out: list[DevSeries] = []
    for entry, loaded in zip(manifest.series, public, strict=True):
        if loaded.series_id != entry.series_id:
            raise ArenaError(
                f"{manifest.dataset}: cache series {entry.series_id!r} does not match the "
                f"public series {loaded.series_id!r} at the same position"
            )
        if entry.period != _REAL_PERIOD:
            raise ArenaError(
                f"{entry.series_id}: the cache records period {entry.period}, not {_REAL_PERIOD}"
            )
        match_values(entry, loaded.values)
        warmup, dev, _sealed, _reserve = loaded.split.segments
        out.append(
            checked_series(
                entry,
                loaded.values,
                manifest.horizon,
                scale_end=warmup.end,
                score_start=dev.start,
                score_end=dev.end,
                change_points=(),
                oracle=None,
            )
        )
    return tuple(out)


def _check_generator_version(dataset: str, source: Mapping[str, str]) -> None:
    """Refuse a synthetic cache whose source pins another generator version."""
    if source.get("generator_version") != GENERATOR_VERSION:
        raise ArenaError(
            f"{dataset}: cache generator_version "
            f"{source.get('generator_version')!r} != current {GENERATOR_VERSION!r}; unknown "
            "generator versions are refused, not reinterpreted"
        )


def _source_per_family(dataset: str, source: Mapping[str, str]) -> int:
    """Read ``source["per_family"]`` as an int >= 1, else refuse."""
    try:
        per_family = int(source["per_family"])
    except (KeyError, ValueError) as exc:
        raise ArenaError(
            f"{dataset}: source.per_family must be an int written as a string: {exc}"
        ) from exc
    if per_family < 1:
        raise ArenaError(f"{dataset}: source.per_family must be >= 1, got {per_family}")
    return per_family


def _synthetic_series(manifest: CacheManifest) -> tuple[DevSeries, ...]:
    """Build the synthetic dev series: worlds, change points and the oracle."""
    source = manifest.source
    _check_generator_version(manifest.dataset, source)
    per_family = _source_per_family(manifest.dataset, source)
    expected: list[tuple[str, SyntheticTask]] = []
    for family in FAMILIES:
        for index, task in enumerate(dev_tasks(family, per_family)):
            expected.append((f"{family}/{index}", task))
    if len(expected) != len(manifest.series):
        raise ArenaError(
            f"{manifest.dataset}: {per_family} series per family is {len(expected)} series, "
            f"but the cache holds {len(manifest.series)}"
        )
    return checked_synthetic_series(manifest, expected)


def _scan_series(manifest: CacheManifest) -> tuple[DevSeries, ...]:
    """Build one scan preset's series: worlds, change points and the oracle.

    The cache must have been built from the very preset this module pins
    today — same name, same ``preset_sha256``, current generator — so a
    preset edited after the cache was built is refused, not reinterpreted
    (TASK_015's rule: the presets were fixed before any scan ran).
    """
    source = manifest.source
    preset = SCAN_PRESETS[manifest.dataset]
    if source.get("pool") != "scan":
        raise ArenaError(
            f"{manifest.dataset}: source.pool must be 'scan' for a scan dataset, got "
            f"{source.get('pool')!r}"
        )
    if source.get("preset") != manifest.dataset:
        raise ArenaError(
            f"{manifest.dataset}: source.preset names {source.get('preset')!r}, not the "
            "dataset being loaded"
        )
    if source.get("preset_sha256") != preset_sha256(preset):
        raise ArenaError(
            f"{manifest.dataset}: cache preset_sha256 {source.get('preset_sha256')!r} != "
            f"current {preset_sha256(preset)!r}; a preset edited after the cache was built "
            "is refused, not reinterpreted"
        )
    _check_generator_version(manifest.dataset, source)
    per_family = _source_per_family(manifest.dataset, source)
    if per_family > HOLDOUT_INDEX_START:
        raise ArenaError(
            f"{manifest.dataset}: source.per_family {per_family} reaches scan index "
            f"{HOLDOUT_INDEX_START} or beyond, into the holdout range; a scan cache holds "
            "dev scan worlds only"
        )
    expected: list[tuple[str, SyntheticTask]] = []
    for family in preset.families:
        for index, task in enumerate(scan_tasks(preset.name, family, per_family)):
            expected.append((f"{family}/{index}", task))
    if len(expected) != len(manifest.series):
        raise ArenaError(
            f"{manifest.dataset}: {per_family} series over each of {len(preset.families)} "
            f"families is {len(expected)} series, but the cache holds {len(manifest.series)}"
        )
    return checked_synthetic_series(manifest, expected)


def _holdout_series(manifest: CacheManifest) -> tuple[DevSeries, ...]:
    """Build one holdout replicate's series: worlds, change points and the oracle.

    The cache must have been built from the very base preset, index range and
    generator this module pins today (TASK_016 PR C), so a preset edited after
    the cache was built, or a cache drawn from the wrong index range, is
    refused, not reinterpreted. Holdout pools score accepted edits only and
    are never tuned on; that later rule is enforced by the Phase 1 runner.
    """
    source = manifest.source
    preset = HOLDOUT_PRESETS[manifest.dataset]
    base = SCAN_PRESETS[preset.base]
    if source.get("pool") != "holdout":
        raise ArenaError(
            f"{manifest.dataset}: source.pool must be 'holdout' for a holdout dataset, got "
            f"{source.get('pool')!r}"
        )
    if source.get("preset") != preset.base:
        raise ArenaError(
            f"{manifest.dataset}: source.preset names {source.get('preset')!r}, not this "
            f"holdout's base preset {preset.base!r}"
        )
    if source.get("preset_sha256") != preset_sha256(base):
        raise ArenaError(
            f"{manifest.dataset}: cache preset_sha256 {source.get('preset_sha256')!r} != "
            f"current {preset_sha256(base)!r}; a preset edited after the cache was built "
            "is refused, not reinterpreted"
        )
    if source.get("index_start") != str(HOLDOUT_INDEX_START):
        raise ArenaError(
            f"{manifest.dataset}: source.index_start must be {HOLDOUT_INDEX_START}, got "
            f"{source.get('index_start')!r}; the holdout range is fixed"
        )
    _check_generator_version(manifest.dataset, source)
    per_family = _source_per_family(manifest.dataset, source)
    if per_family > HOLDOUT_INDEX_STOP - HOLDOUT_INDEX_START:
        raise ArenaError(
            f"{manifest.dataset}: source.per_family {per_family} exceeds the holdout index "
            f"range [{HOLDOUT_INDEX_START}, {HOLDOUT_INDEX_STOP})"
        )
    expected: list[tuple[str, SyntheticTask]] = []
    for family in base.families:
        for index, task in enumerate(holdout_tasks(manifest.dataset, family, per_family)):
            expected.append((holdout_series_id(family, index), task))
    if len(expected) != len(manifest.series):
        raise ArenaError(
            f"{manifest.dataset}: {per_family} series over each of {len(base.families)} "
            f"families is {len(expected)} series, but the cache holds {len(manifest.series)}"
        )
    return checked_synthetic_series(manifest, expected)


def checked_synthetic_series(
    manifest: CacheManifest, tasks: Sequence[tuple[str, SyntheticTask]]
) -> tuple[DevSeries, ...]:
    """Check ``tasks`` against a synthetic ``manifest`` and build the scored series.

    Shared by the dev pool and the sealed synthetic pool: ids in manifest
    order, each world's period, the values' hash, warmup boundaries, change
    points and the oracle from :func:`true_best_forecast`.
    """
    if len(tasks) != len(manifest.series):
        raise ArenaError(
            f"{manifest.dataset}: {len(tasks)} synthetic tasks, but the cache holds "
            f"{len(manifest.series)} series"
        )
    warmup = GrammarConfig().warmup
    out: list[DevSeries] = []
    for entry, (expected_id, task) in zip(manifest.series, tasks, strict=True):
        if entry.series_id != expected_id:
            raise ArenaError(
                f"{manifest.dataset}: cache series {entry.series_id!r} where task "
                f"{expected_id!r} was expected"
            )
        world = task.world
        if entry.period != world.spec.period:
            raise ArenaError(
                f"{entry.series_id}: the cache records period {entry.period}, but the world "
                f"has {world.spec.period}"
            )
        match_values(entry, world.y)
        score_end = entry.length - manifest.horizon
        series = checked_series(
            entry,
            world.y,
            manifest.horizon,
            scale_end=warmup,
            score_start=warmup,
            score_end=score_end,
            change_points=world.change_points,
            oracle=None,
        )
        oracle = tuple(
            true_best_forecast(world, origin, manifest.horizon)
            for origin in range(warmup, score_end)
        )
        out.append(replace(series, oracle=oracle))
    return tuple(out)


def match_values(entry: CachedSeries, values: tuple[float, ...]) -> None:
    """Check the public values against the cache's hash and length."""
    digest = sha256_hex(canonical_bytes(list(values)))
    if digest != entry.values_sha256:
        raise ArenaError(
            f"{entry.series_id}: the public values hash to {digest}, but the cache recorded "
            f"{entry.values_sha256}"
        )
    if len(values) != entry.length:
        raise ArenaError(
            f"{entry.series_id}: {len(values)} public values, but the cache recorded {entry.length}"
        )


def checked_series(
    entry: CachedSeries,
    values: tuple[float, ...],
    horizon: int,
    *,
    scale_end: int,
    score_start: int,
    score_end: int,
    change_points: tuple[int, ...],
    oracle: tuple[tuple[float, ...], ...] | None,
) -> DevSeries:
    """Check the scoring boundaries against the cache, then truncate the values."""
    if score_start < entry.first_origin:
        raise ArenaError(
            f"{entry.series_id}: scoring starts at origin {score_start}, before the cache's "
            f"first origin {entry.first_origin}"
        )
    if score_start >= score_end:
        raise ArenaError(f"{entry.series_id}: empty scoring range [{score_start}, {score_end})")
    if scale_end < entry.period + 1:
        raise ArenaError(
            f"{entry.series_id}: the MASE scale window [0, {scale_end}) holds no seasonal "
            f"difference for period {entry.period}"
        )
    if scale_end > score_start:
        raise ArenaError(
            f"{entry.series_id}: the MASE scale window ends at {scale_end}, inside the "
            f"scored range from {score_start}"
        )
    if score_end + horizon > len(values):
        raise ArenaError(
            f"{entry.series_id}: scoring [{score_start}, {score_end}) with horizon {horizon} "
            f"needs a target at index {score_end + horizon - 1}, but only {len(values)} "
            f"public values exist"
        )
    return DevSeries(
        series_id=entry.series_id,
        period=entry.period,
        values=values[: score_end + horizon],
        scale_end=scale_end,
        score_start=score_start,
        score_end=score_end,
        change_points=change_points,
        oracle=oracle,
    )
