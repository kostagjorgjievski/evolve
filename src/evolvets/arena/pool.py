"""The expert pool a replay drives: the simple five, plus cached foundation points.

A pool spec names the experts a replay uses: ``"simple"`` (the five
statistical experts, arena version 1's pool) or ``"simple+foundation"``
(those five, then the pinned foundation models in registry order). The
foundation models never run at replay time: their 0.5-quantile rows are read
from caches built offline (TASK_009) and joined onto the simple forecasts
origin for origin, after each cache has proved it was built against the very
simple cache the pool replays. Updaters see one flat expert list, so a pool
spec changes only how many experts there are, never what an expert is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from evolvets.arena.errors import ArenaError
from evolvets.cache.interfaces import CacheManifest
from evolvets.cache.store import CacheError, SeriesForecasts, load_series
from evolvets.experts.pool import EXPERT_NAMES
from evolvets.foundation.errors import FoundationError
from evolvets.foundation.interfaces import FoundationManifest, FoundationSeries
from evolvets.foundation.registry import MODELS
from evolvets.foundation.store import (
    foundation_cache_id,
    load_foundation_series,
    read_foundation_manifest,
)

if TYPE_CHECKING:
    from evolvets.arena.dev import DevPool

#: The pool specs a run config may name. ``simple`` replays exactly version
#: 1's pool; ``simple+foundation`` appends the pinned foundation models.
POOL_SPECS = ("simple", "simple+foundation")


def check_pool_spec(pool_spec: object) -> str:
    """Return ``pool_spec`` when it names a pool this code replays, else raise."""
    if not isinstance(pool_spec, str) or pool_spec not in POOL_SPECS:
        raise ArenaError(f"pool_spec must be one of {list(POOL_SPECS)}, got {pool_spec!r}")
    return pool_spec


def built_model_names() -> tuple[str, ...]:
    """Return the pinned foundation models in registry order.

    A model is in the ``simple+foundation`` pool only if its registry entry
    pins a commit hash (``hf_revision != "unpinned"``); the dropped
    placeholders (``toto``, ``moirai2``) never qualify.
    """
    return tuple(name for name, spec in MODELS.items() if spec.hf_revision != "unpinned")


@dataclass(frozen=True)
class FoundationCache:
    """One model's foundation cache, checked against the simple cache it joins."""

    model: str  # registry key, e.g. "chronos2"; also the cache directory name
    root: Path  # the foundation cache root holding <dataset>/<model>/
    cache_id: str  # sha256 of the manifest bytes
    manifest: FoundationManifest


def load_foundation_caches(
    dataset: str,
    simple_id: str,
    simple: CacheManifest,
    root: Path,
    models: Sequence[str] | None = None,
) -> tuple[FoundationCache, ...]:
    """Load every ``models`` cache under ``root`` and prove each joins the simple cache.

    Each cache must exist, name the model it was asked for, carry the same
    horizon, record the ``simple_cache_id`` of the simple cache in use, and
    match it series for series and origin for origin. ``models`` defaults to
    the pinned registry models; an explicit list (tests name the fake) keeps
    the same per-model checks.
    """
    names = tuple(models) if models is not None else built_model_names()
    if not names:
        raise ArenaError("the foundation pool is empty: no registry model has a pinned revision")
    seen: set[str] = set()
    out: list[FoundationCache] = []
    for name in names:
        if name in seen:
            raise ArenaError(f"foundation model {name!r} appears twice in the pool")
        seen.add(name)
        try:
            manifest = read_foundation_manifest(Path(root), dataset, name)
            ident = foundation_cache_id(Path(root), dataset, name)
        except FoundationError as exc:
            raise ArenaError(f"foundation cache {dataset!r}/{name!r}: {exc}") from exc
        if manifest.model.name != name:
            raise ArenaError(
                f"foundation cache {dataset!r}/{name!r} holds model {manifest.model.name!r}"
            )
        if manifest.simple_cache_id != simple_id:
            raise ArenaError(
                f"foundation cache {dataset!r}/{name!r} was built against simple cache "
                f"{manifest.simple_cache_id}, but this pool replays {simple_id}; rebuild it "
                "from the same simple cache first"
            )
        if manifest.horizon != simple.horizon:
            raise ArenaError(
                f"foundation cache {dataset!r}/{name!r} records horizon {manifest.horizon}, "
                f"but the simple cache records {simple.horizon}"
            )
        _match_simple_series(dataset, name, manifest, simple)
        out.append(FoundationCache(model=name, root=Path(root), cache_id=ident, manifest=manifest))
    return tuple(out)


def _match_simple_series(
    dataset: str, model: str, manifest: FoundationManifest, simple: CacheManifest
) -> None:
    """Refuse a foundation cache whose series do not join the simple cache's row for row."""
    foundation_ids = [entry.series_id for entry in manifest.series]
    simple_ids = [entry.series_id for entry in simple.series]
    if foundation_ids != simple_ids:
        raise ArenaError(
            f"foundation cache {dataset!r}/{model!r}: its series {foundation_ids} do not match "
            f"the simple cache's {simple_ids}, so the two cannot be joined origin for origin"
        )
    for entry, cached in zip(manifest.series, simple.series, strict=True):
        problems = []
        if entry.period != cached.period:
            problems.append(f"period {entry.period} != {cached.period}")
        if entry.length != cached.length:
            problems.append(f"length {entry.length} != {cached.length}")
        if entry.first_origin != cached.first_origin:
            problems.append(f"first_origin {entry.first_origin} != {cached.first_origin}")
        if entry.n_origins != cached.n_origins:
            problems.append(f"n_origins {entry.n_origins} != {cached.n_origins}")
        if entry.values_sha256 != cached.values_sha256:
            problems.append("values differ (values_sha256)")
        if problems:
            raise ArenaError(
                f"foundation cache {dataset!r}/{model!r}: series {entry.series_id!r} does not "
                f"match the simple cache's entry ({'; '.join(problems)})"
            )


def pool_expert_names(pool: DevPool) -> tuple[str, ...]:
    """Return every expert the pool replays: the simple five, then its foundation models."""
    return EXPERT_NAMES + tuple(cache.model for cache in pool.foundation)


def pool_leakage_risk(dataset: str, foundation: tuple[FoundationCache, ...]) -> bool:
    """Return whether a pooled model may have seen ``dataset`` in pretraining.

    True exactly when the dataset is real and some pooled model's published
    pretraining corpus includes it (``"yes"``) or cannot be decided
    (``"unknown"``). Synthetic datasets are no model's pretraining data, so
    they never carry the flag.
    """
    return any(
        cache.manifest.model.pretraining_overlap.get(dataset) in ("yes", "unknown")
        for cache in foundation
    )


class PoolForecasts:
    """One series' forecasts over the whole pool, in pool order.

    The simple rows come straight from the simple cache; each foundation
    model contributes its cached 0.5-quantile row per origin — the point
    forecast updaters receive (TASK_009). No value is copied or recomputed:
    ``at`` slices the two caches it was built from.
    """

    series_id: str
    first_origin: int
    last_origin: int

    def __init__(self, simple: SeriesForecasts, foundation: tuple[FoundationSeries, ...]) -> None:
        self.series_id = simple.series_id
        self.first_origin = simple.first_origin
        self.last_origin = simple.last_origin
        self._simple = simple
        self._foundation = foundation
        for joined in foundation:
            if joined.series_id != simple.series_id:
                raise ArenaError(
                    f"foundation series {joined.series_id!r} cannot join simple series "
                    f"{simple.series_id!r}"
                )
            if joined.first_origin != simple.first_origin or (
                joined.last_origin != simple.last_origin
            ):
                raise ArenaError(
                    f"{simple.series_id}: foundation origins [{joined.first_origin}, "
                    f"{joined.last_origin}] do not cover the simple cache's "
                    f"[{simple.first_origin}, {simple.last_origin}]"
                )

    def at(self, origin: int) -> tuple[tuple[float, ...], ...]:
        """Return ``[expert][h - 1]`` for ``origin``: five simple rows, then point rows."""
        return self._simple.at(origin) + tuple(
            joined.point(origin - joined.first_origin) for joined in self._foundation
        )


def load_pool_forecasts(pool: DevPool, series_id: str) -> PoolForecasts:
    """Load one series' simple forecasts and every pooled model's foundation series.

    Every cache failure — a missing or tampered file — surfaces as
    :class:`ArenaError`, as it does for the simple cache alone.
    """
    try:
        simple = load_series(pool.cache_root, pool.manifest, series_id)
    except CacheError as exc:
        raise ArenaError(str(exc)) from exc
    foundation: list[FoundationSeries] = []
    for cache in pool.foundation:
        try:
            foundation.append(load_foundation_series(cache.root, cache.manifest, series_id))
        except FoundationError as exc:
            raise ArenaError(str(exc)) from exc
    return PoolForecasts(simple, tuple(foundation))
