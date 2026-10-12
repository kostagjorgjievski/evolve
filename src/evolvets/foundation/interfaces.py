"""Interfaces shared by the foundation cache builder and every adapter."""

from __future__ import annotations

from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from evolvets.foundation.errors import FoundationError
from evolvets.foundation.registry import ModelSpec

#: The quantile levels every adapter must return, in order. The 0.5 quantile
#: (index 4) is the point forecast the updaters receive.
QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class Adapter(Protocol):
    """A batch forecaster: many variable-length contexts in, quantile forecasts out.

    Implementations live in ``foundation/models/`` (one per pretrained library)
    plus the torch-free :class:`evolvets.foundation.fake.FakeAdapter`. Contexts
    within one batch must not affect each other's forecasts.
    """

    name: str  # registry key, e.g. "chronos2"

    def forecast(
        self, contexts: Sequence[Sequence[float]], horizon: int
    ) -> list[list[list[float]]]: ...  # [context][quantile][h], len(QUANTILES) quantiles


@dataclass(frozen=True)
class SeriesForecast:
    """One series' forecasts for every origin, and how many steps were monotone-sorted.

    ``values`` is ``[origin][quantile][h]`` in origin order, with quantile
    crossing already fixed: at any step where a model returned quantiles that
    were not non-decreasing, they were sorted ascending (monotone
    rearrangement). ``sorted_steps`` counts the (origin, step) pairs sorted.
    """

    values: tuple[tuple[tuple[float, ...], ...], ...]
    sorted_steps: int


@dataclass(frozen=True)
class FoundationSeriesEntry:
    """Where one cached foundation series' forecasts live and what they were built from."""

    series_id: str
    length: int
    period: int
    first_origin: int
    n_origins: int  # length - first_origin
    file: str  # "series/<index:04d>.f64"
    sha256: str  # of the file bytes
    values_sha256: str  # sha256_hex(canonical_bytes(list(values)))
    sorted_steps: int  # (origin, step) pairs the builder monotone-sorted


@dataclass(frozen=True)
class FoundationManifest:
    """Everything a reader needs to check a foundation cache before trusting its forecasts.

    ``source`` is copied from :func:`evolvets.cache.datasets.public_inputs`;
    ``model`` is the spec the forecasts were computed with; ``device``,
    ``dtype`` and ``library_versions`` record how the adapter that produced
    them was equipped; ``simple_cache_id`` ties the cache to the simple cache
    it joins origin for origin. GPU forecasts are not bit-reproducible, so a
    cache's reproducibility unit is its cache id (the manifest hash) plus the
    hash-verified files — never a recomputation.
    """

    foundation_cache_version: str
    dataset: str
    source: dict[str, str]
    horizon: int
    quantiles: tuple[float, ...]
    model: ModelSpec
    library_versions: dict[str, str]
    device: str
    dtype: str
    batch_size: int
    build_seconds: float
    simple_cache_id: str
    series: tuple[FoundationSeriesEntry, ...]


@dataclass(frozen=True)
class FoundationSeries:
    """One cached foundation series: nine quantile forecasts for every origin.

    ``data`` is flat ``[origin][quantile][h]``, origin-major, exactly as
    stored in the series file; ``origin_index`` counts origins from
    ``first_origin``, so origin ``t`` is index ``t - first_origin``. The
    array is never mutated.
    """

    series_id: str
    first_origin: int
    horizon: int
    data: array[float]  # n_origins * len(QUANTILES) * horizon little-endian float64
    sorted_steps: int

    @property
    def n_origins(self) -> int:
        """Return the number of origins the series holds forecasts for."""
        return len(self.data) // (len(QUANTILES) * self.horizon)

    @property
    def last_origin(self) -> int:
        """Return the last origin with a forecast: ``first_origin + n_origins - 1``."""
        return self.first_origin + self.n_origins - 1

    def row(self, origin_index: int) -> tuple[tuple[float, ...], ...]:
        """Return ``[quantile][h]`` for origin ``first_origin + origin_index``."""
        start = self._element(origin_index, 0)
        return tuple(
            tuple(self.data[start + q * self.horizon : start + (q + 1) * self.horizon])
            for q in range(len(QUANTILES))
        )

    def point(self, origin_index: int) -> tuple[float, ...]:
        """Return the 0.5-quantile row (index 4) for origin ``first_origin + origin_index``."""
        start = self._element(origin_index, QUANTILES.index(0.5))
        return tuple(self.data[start : start + self.horizon])

    def _element(self, origin_index: int, quantile: int) -> int:
        """Validate ``origin_index`` and return the flat offset of (origin, quantile, step 1)."""
        if isinstance(origin_index, bool) or not isinstance(origin_index, int):
            raise FoundationError(f"origin_index must be an int, got {origin_index!r}")
        if not 0 <= origin_index < self.n_origins:
            raise FoundationError(
                f"origin_index {origin_index} is outside [0, {self.n_origins - 1}] "
                f"for series {self.series_id!r}"
            )
        return (origin_index * len(QUANTILES) + quantile) * self.horizon


@dataclass(frozen=True)
class RebuildReport:
    """What :func:`evolvets.foundation.store.compare_rebuild` found on one cache.

    GPU forecasts are not bit-reproducible, so a recompute is expected to
    differ slightly from the stored bytes; the report quantifies that drift
    over the sampled origins. It is a diagnostic: it never edits the cache,
    and it says nothing about integrity — file bytes are verified against the
    manifest by :func:`evolvets.foundation.store.verify_foundation_cache`.
    A compared value that is not finite is the worst kind of drift and never
    reads as a match: it is counted in ``n_nonfinite`` and drives both maxima
    to ``inf``.
    """

    dataset: str
    model: str
    seed: int
    n_compared: int  # origins recomputed
    n_total: int  # every (series, origin) pair the cache holds
    n_nonfinite: int  # compared values (stored or recomputed) that were NaN or ±inf
    # max |recomputed - stored| over the 0.5-quantile rows, and that diff divided by
    # max(|stored|, 1e-12). The 1e-12 floor makes a stored 0.0 against any nonzero
    # recompute an enormous relative diff, which a diagnostic wants. Both fields are
    # inf whenever a compared value (stored or recomputed) is NaN or ±inf.
    max_abs_diff: float
    max_rel_diff: float
