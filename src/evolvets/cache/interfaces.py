"""Data types shared by the prediction cache and its readers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeriesInput:
    """One series to cache: a unique id, its finite values and its period."""

    series_id: str
    values: tuple[float, ...]  # finite
    period: int  # >= 2


@dataclass(frozen=True)
class CachedSeries:
    """Where one cached series' forecasts live and what they were built from."""

    series_id: str
    length: int
    period: int
    first_origin: int
    n_origins: int  # length - first_origin
    file: str  # "series/<index:04d>.f64"
    sha256: str  # of the file bytes
    values_sha256: str  # sha256_hex(canonical_bytes(list(values)))


@dataclass(frozen=True)
class CacheManifest:
    """Everything a reader needs to check a cache before trusting its forecasts.

    ``experts`` is ``(name, params)`` in pool order for the one period every
    series of the dataset shares; ``source`` records where the values came
    from so a cache can be tied back to the data that produced it.
    """

    cache_version: str
    experts_version: str
    dataset: str
    source: dict[str, str]
    horizon: int
    experts: tuple[tuple[str, dict[str, float | int]], ...]  # (name, params) in pool order
    series: tuple[CachedSeries, ...]
