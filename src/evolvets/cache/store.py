"""The content-addressed prediction cache: build, inspect, verify, load.

Layout: ``<cache_root>/<dataset>/manifest.json`` plus
``<cache_root>/<dataset>/series/<index:04d>.f64`` per series, holding the
forecasts of :func:`evolvets.experts.pool.run_pool` as little-endian IEEE-754
float64, row-major ``[origin][expert][h]``. The manifest pins the experts and
cache protocol versions, the source of the values and the SHA-256 of every
file, so a reader can prove the bytes it is about to replay are the bytes
that were built.

Building never touches an existing cache: everything is computed in a fresh
temporary directory and either renamed into place (first build) or compared
against what is already there (rebuild), so an existing cache is never
overwritten, edited or deleted by a build.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolvets.cache import CACHE_VERSION
from evolvets.cache.interfaces import CachedSeries, CacheManifest, SeriesInput
from evolvets.experts import EXPERTS_VERSION
from evolvets.experts.pool import EXPERT_NAMES, first_origin, make_pool, run_pool
from evolvets.ledger.canonical import canonical_bytes, sha256_hex

DEFAULT_HORIZON = 24

_N_EXPERTS = len(EXPERT_NAMES)
_MANIFEST_KEYS = (
    "cache_version",
    "experts_version",
    "dataset",
    "source",
    "horizon",
    "experts",
    "series",
)
_SERIES_KEYS = (
    "series_id",
    "length",
    "period",
    "first_origin",
    "n_origins",
    "file",
    "sha256",
    "values_sha256",
)
_EXPERT_KEYS = ("name", "params")
_LOWER_ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789"
_NAME_CHARS = set(_LOWER_ALNUM + "_-")
_HEX = set("0123456789abcdef")


class CacheError(Exception):
    """Raised when a cache is built from invalid inputs or is missing or tampered."""


def cache_dir(cache_root: Path, dataset: str) -> Path:
    """Return ``<cache_root>/<dataset>`` after checking the dataset name."""
    return Path(cache_root) / _check_dataset_name(dataset)


def _check_dataset_name(dataset: object) -> str:
    """Return ``dataset`` if it matches ``^[a-z0-9][a-z0-9_-]*$``, else raise CacheError."""
    if (
        not isinstance(dataset, str)
        or not dataset
        or not set(dataset) <= _NAME_CHARS
        or dataset[0] not in _LOWER_ALNUM
    ):
        raise CacheError(f"dataset name must match ^[a-z0-9][a-z0-9_-]*$, got {dataset!r}")
    return dataset


def build_cache(
    dataset: str,
    source: Mapping[str, str],
    series: Sequence[SeriesInput],
    cache_root: Path,
    horizon: int = DEFAULT_HORIZON,
) -> CacheManifest:
    """Compute the pool's forecasts for ``series`` and install them as a cache.

    Validated inputs are computed into a fresh temporary directory, the
    manifest is written last, and the result is either renamed into place or
    — when ``<cache_root>/<dataset>`` already exists — compared with it:
    equal manifests make the build a no-op that returns the existing
    manifest, different ones raise :class:`CacheError` naming both cache ids
    and leave the existing cache untouched.
    """
    dataset = _check_dataset_name(dataset)
    checked_horizon = _check_horizon(horizon)
    source_map = _check_source(source)
    entries = _check_series(series)
    period = entries[0].period
    start = first_origin(period)

    root = Path(cache_root)
    final = root / dataset
    tmp = root / f".{dataset}.tmp-{os.getpid()}-{os.urandom(6).hex()}"
    try:
        (tmp / "series").mkdir(parents=True)
        pool = make_pool(period)
        cached: list[CachedSeries] = []
        for index, entry in enumerate(entries):
            n_origins = len(entry.values) - start
            forecasts = run_pool(entry.values, entry.period, checked_horizon)
            flat: array[float] = array("d")
            for per_expert in forecasts:
                for one_expert in per_expert:
                    flat.extend(one_expert)
            if len(flat) != n_origins * _N_EXPERTS * checked_horizon:
                raise CacheError(
                    f"{entry.series_id}: run_pool returned {len(flat)} values, "
                    f"expected {n_origins * _N_EXPERTS * checked_horizon}"
                )
            if sys.byteorder == "big":
                flat.byteswap()
            blob = flat.tobytes()
            name = f"series/{index:04d}.f64"
            _write_file(tmp / name, blob)
            cached.append(
                CachedSeries(
                    series_id=entry.series_id,
                    length=len(entry.values),
                    period=entry.period,
                    first_origin=start,
                    n_origins=n_origins,
                    file=name,
                    sha256=sha256_hex(blob),
                    values_sha256=sha256_hex(canonical_bytes(list(entry.values))),
                )
            )
        manifest = CacheManifest(
            cache_version=CACHE_VERSION,
            experts_version=EXPERTS_VERSION,
            dataset=dataset,
            source=source_map,
            horizon=checked_horizon,
            experts=tuple((e.name, dict(e.params())) for e in pool),
            series=tuple(cached),
        )
        manifest_bytes = canonical_bytes(_manifest_obj(manifest))
        _write_file(tmp / "manifest.json", manifest_bytes)
        _fsync_dir(tmp / "series")
        _fsync_dir(tmp)

        if not final.exists():
            try:
                os.rename(tmp, final)
                return manifest
            except OSError:
                if not final.exists():
                    raise  # a genuine failure, not a cache appearing concurrently
        existing = final / "manifest.json"
        try:
            existing_bytes = existing.read_bytes()
        except OSError as exc:
            raise CacheError(f"{existing} exists but cannot be read: {exc}") from exc
        if existing_bytes == manifest_bytes:
            shutil.rmtree(tmp, ignore_errors=True)
            return _parse_manifest(existing_bytes, str(existing), dataset)
        raise CacheError(
            f"cache {final} already exists with different content: existing cache id "
            f"{sha256_hex(existing_bytes)}, new cache id {sha256_hex(manifest_bytes)}; "
            "remove the old cache directory by hand to rebuild"
        )
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def read_manifest(cache_root: Path, dataset: str) -> CacheManifest:
    """Read and validate ``<cache_root>/<dataset>/manifest.json``.

    Unknown cache or experts versions are refused, not reinterpreted; every
    structural fault raises :class:`CacheError`.
    """
    path = cache_dir(cache_root, dataset) / "manifest.json"
    if not path.exists():
        raise CacheError(f"{path} is missing; build the cache first")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CacheError(f"{path}: cannot be read: {exc}") from exc
    return _parse_manifest(raw, str(path), dataset)


def cache_id(cache_root: Path, dataset: str) -> str:
    """Return the SHA-256 of the dataset's ``manifest.json`` bytes."""
    path = cache_dir(cache_root, dataset) / "manifest.json"
    if not path.exists():
        raise CacheError(f"{path} is missing; build the cache first")
    try:
        return sha256_hex(path.read_bytes())
    except OSError as exc:
        raise CacheError(f"{path}: cannot be read: {exc}") from exc


def verify_cache(cache_root: Path, dataset: str) -> CacheManifest:
    """Read the manifest and check every series file's size and SHA-256.

    No forecast is recomputed: verification proves the bytes on disk are the
    bytes the manifest was built from, nothing more.
    """
    manifest = read_manifest(cache_root, dataset)
    root = cache_dir(cache_root, dataset)
    for entry in manifest.series:
        raw = _read_series_file(root, entry, manifest.horizon)
        if sha256_hex(raw) != entry.sha256:
            raise CacheError(
                f"{root / entry.file}: sha256 {sha256_hex(raw)} does not match the "
                f"manifest's {entry.sha256}"
            )
    return manifest


def load_series(cache_root: Path, manifest: CacheManifest, series_id: str) -> SeriesForecasts:
    """Load one series' forecasts, checking size and SHA-256 against ``manifest``."""
    entry = next((s for s in manifest.series if s.series_id == series_id), None)
    if entry is None:
        raise CacheError(f"unknown series_id {series_id!r} in cache {manifest.dataset!r}")
    root = cache_dir(cache_root, manifest.dataset)
    raw = _read_series_file(root, entry, manifest.horizon)
    if sha256_hex(raw) != entry.sha256:
        raise CacheError(
            f"{root / entry.file}: sha256 {sha256_hex(raw)} does not match the "
            f"manifest's {entry.sha256}"
        )
    data: array[float] = array("d")
    if data.itemsize != 8:
        raise CacheError("array('d') is not 8 bytes wide on this platform")
    data.frombytes(raw)
    if sys.byteorder == "big":
        data.byteswap()
    return SeriesForecasts(
        series_id=entry.series_id,
        first_origin=entry.first_origin,
        last_origin=entry.length - 1,
        horizon=manifest.horizon,
        data=data,
    )


class SeriesForecasts:
    """One cached series: every expert's forecasts for every origin."""

    series_id: str
    first_origin: int
    last_origin: int  # length - 1

    def __init__(
        self, series_id: str, first_origin: int, last_origin: int, horizon: int, data: array[float]
    ) -> None:
        self.series_id = series_id
        self.first_origin = first_origin
        self.last_origin = last_origin
        self._horizon = horizon
        self._data = data

    def at(self, origin: int) -> tuple[tuple[float, ...], ...]:
        """Return the forecasts for ``origin`` as ``[expert][h - 1]`` in pool order."""
        index = self._origin_index(origin)
        return tuple(
            tuple(self._data[index + e * self._horizon : index + (e + 1) * self._horizon])
            for e in range(_N_EXPERTS)
        )

    def expert_forecast(self, origin: int, expert: str) -> tuple[float, ...]:
        """Return one expert's forecasts for ``origin``, by pool-order name."""
        if expert not in EXPERT_NAMES:
            raise CacheError(f"unknown expert {expert!r}; expected one of {list(EXPERT_NAMES)}")
        index = self._origin_index(origin) + EXPERT_NAMES.index(expert) * self._horizon
        return tuple(self._data[index : index + self._horizon])

    def _origin_index(self, origin: int) -> int:
        """Validate ``origin`` and return its first element offset in the flat data."""
        if isinstance(origin, bool) or not isinstance(origin, int):
            raise CacheError(f"origin must be an int, got {origin!r}")
        if not self.first_origin <= origin <= self.last_origin:
            raise CacheError(
                f"origin {origin} is outside [{self.first_origin}, {self.last_origin}] "
                f"for series {self.series_id!r}"
            )
        return (origin - self.first_origin) * _N_EXPERTS * self._horizon


def _check_horizon(horizon: object) -> int:
    """Return ``horizon`` as an int >= 1 (bools rejected), else raise CacheError."""
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise CacheError(f"horizon must be an int, got {horizon!r}")
    if horizon < 1:
        raise CacheError(f"horizon must be >= 1, got {horizon}")
    return horizon


def _check_source(source: Mapping[str, str]) -> dict[str, str]:
    """Copy ``source`` after checking every key and value is a string."""
    out: dict[str, str] = {}
    for key, value in source.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise CacheError(f"source keys and values must be strings, got {key!r}: {value!r}")
        out[key] = value
    return out


def _check_series(series: Sequence[SeriesInput]) -> tuple[SeriesInput, ...]:
    """Validate the series list and return it as a tuple, in order."""
    if not series:
        raise CacheError("at least one series is required")
    seen: set[str] = set()
    out: list[SeriesInput] = []
    for index, entry in enumerate(series):
        if not isinstance(entry, SeriesInput):
            raise CacheError(f"series[{index}] must be a SeriesInput, got {entry!r}")
        sid = entry.series_id
        if not isinstance(sid, str) or not sid:
            raise CacheError(f"series[{index}]: series_id must be a non-empty string, got {sid!r}")
        if sid in seen:
            raise CacheError(f"duplicate series_id {sid!r}")
        seen.add(sid)
        if isinstance(entry.period, bool) or not isinstance(entry.period, int) or entry.period < 2:
            raise CacheError(f"{sid}: period must be an int >= 2, got {entry.period!r}")
        _check_values(sid, entry.values)
        start = first_origin(entry.period)
        if len(entry.values) <= start:
            raise CacheError(
                f"{sid}: {len(entry.values)} values leave no origin to forecast from "
                f"(needs more than {start} for period {entry.period})"
            )
        out.append(entry)
    periods = {entry.period for entry in out}
    if len(periods) != 1:
        raise CacheError(
            f"all series of a cache must share one period (the manifest records one "
            f"expert pool), got {sorted(periods)}"
        )
    return tuple(out)


def _check_values(series_id: str, values: Sequence[float]) -> None:
    """Check every value is a finite float (bools rejected)."""
    for i, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CacheError(f"{series_id}: values[{i}] must be a finite float, got {value!r}")
        try:
            as_float = float(value)
        except OverflowError:  # an int too large for a float
            raise CacheError(
                f"{series_id}: values[{i}] must be a finite float, got {value!r}"
            ) from None
        if not math.isfinite(as_float):
            raise CacheError(f"{series_id}: values[{i}] must be a finite float, got {value!r}")


def _write_file(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` and fsync it, so a rename never outlives its data."""
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    """Fsync a directory so its entries survive a crash across the rename."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_series_file(root: Path, entry: CachedSeries, horizon: int) -> bytes:
    """Read one series file, checking its size against the manifest."""
    path = root / entry.file
    expected = entry.n_origins * _N_EXPERTS * horizon * 8
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CacheError(f"{path}: cannot be read: {exc}") from exc
    if len(raw) != expected:
        raise CacheError(f"{path}: expected {expected} bytes, got {len(raw)}")
    return raw


def _manifest_obj(manifest: CacheManifest) -> dict[str, object]:
    """Return the JSON-ready object ``manifest.json`` holds."""
    return {
        "cache_version": manifest.cache_version,
        "experts_version": manifest.experts_version,
        "dataset": manifest.dataset,
        "source": dict(manifest.source),
        "horizon": manifest.horizon,
        "experts": [{"name": name, "params": dict(params)} for name, params in manifest.experts],
        "series": [
            {
                "series_id": s.series_id,
                "length": s.length,
                "period": s.period,
                "first_origin": s.first_origin,
                "n_origins": s.n_origins,
                "file": s.file,
                "sha256": s.sha256,
                "values_sha256": s.values_sha256,
            }
            for s in manifest.series
        ],
    }


def _parse_manifest(raw: bytes, where: str, dataset: str) -> CacheManifest:
    """Parse and fully validate manifest bytes; raise CacheError on any fault."""
    try:
        obj: object = json.loads(raw)
    except ValueError as exc:
        raise CacheError(f"{where}: not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise CacheError(f"{where}: must be a JSON object")
    obj = {str(key): value for key, value in obj.items()}
    if set(obj) != set(_MANIFEST_KEYS):
        raise CacheError(f"{where}: keys must be exactly {sorted(_MANIFEST_KEYS)}")
    if obj["cache_version"] != CACHE_VERSION:
        raise CacheError(
            f"{where}: cache_version {obj['cache_version']!r} != current {CACHE_VERSION!r}; "
            "unknown cache versions are refused, not reinterpreted"
        )
    if obj["experts_version"] != EXPERTS_VERSION:
        raise CacheError(
            f"{where}: experts_version {obj['experts_version']!r} != current "
            f"{EXPERTS_VERSION!r}; unknown experts versions are refused, not reinterpreted"
        )
    if obj["dataset"] != dataset:
        raise CacheError(f"{where}: dataset must be {dataset!r}, got {obj['dataset']!r}")
    horizon = obj["horizon"]
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise CacheError(f"{where}: horizon must be an int >= 1, got {horizon!r}")
    source_obj = obj["source"]
    if not isinstance(source_obj, dict):
        raise CacheError(f"{where}: source must be a JSON object")
    source: dict[str, str] = {}
    for key, value in source_obj.items():
        if not isinstance(value, str):
            raise CacheError(f"{where}: source[{key!r}] must be a string, got {value!r}")
        source[str(key)] = value
    experts_obj = obj["experts"]
    if not isinstance(experts_obj, list) or len(experts_obj) != _N_EXPERTS:
        raise CacheError(f"{where}: experts must be a list of {_N_EXPERTS} entries")
    experts: list[tuple[str, dict[str, float | int]]] = []
    for index, entry in enumerate(experts_obj):
        if not isinstance(entry, dict):
            raise CacheError(f"{where}: experts[{index}] must be a JSON object")
        entry = {str(key): value for key, value in entry.items()}
        if set(entry) != set(_EXPERT_KEYS):
            raise CacheError(
                f"{where}: experts[{index}]: keys must be exactly {sorted(_EXPERT_KEYS)}"
            )
        name = entry["name"]
        if not isinstance(name, str):
            raise CacheError(f"{where}: experts[{index}]: name must be a string, got {name!r}")
        if name != EXPERT_NAMES[index]:
            raise CacheError(
                f"{where}: experts[{index}]: name must be {EXPERT_NAMES[index]!r} "
                f"in pool order, got {name!r}"
            )
        params_obj = entry["params"]
        if not isinstance(params_obj, dict):
            raise CacheError(f"{where}: experts[{index}]: params must be a JSON object")
        params: dict[str, float | int] = {}
        for key, value in params_obj.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CacheError(
                    f"{where}: experts[{index}]: params[{key!r}] must be a number, got {value!r}"
                )
            params[str(key)] = value
        experts.append((name, params))
    series_obj = obj["series"]
    if not isinstance(series_obj, list) or not series_obj:
        raise CacheError(f"{where}: series must be a non-empty list")
    seen: set[str] = set()
    cached: list[CachedSeries] = []
    for index, entry in enumerate(series_obj):
        cached.append(_parse_series_entry(entry, index, where, seen))
    return CacheManifest(
        cache_version=CACHE_VERSION,
        experts_version=EXPERTS_VERSION,
        dataset=dataset,
        source=source,
        horizon=horizon,
        experts=tuple(experts),
        series=tuple(cached),
    )


def _parse_series_entry(entry: object, index: int, where: str, seen: set[str]) -> CachedSeries:
    """Validate one ``series`` entry of a manifest against its own fields."""
    spot = f"{where}: series[{index}]: "
    if not isinstance(entry, dict):
        raise CacheError(f"{spot}must be a JSON object")
    entry = {str(key): value for key, value in entry.items()}
    if set(entry) != set(_SERIES_KEYS):
        raise CacheError(f"{spot}keys must be exactly {sorted(_SERIES_KEYS)}")
    series_id = entry["series_id"]
    if not isinstance(series_id, str) or not series_id:
        raise CacheError(f"{spot}series_id must be a non-empty string, got {series_id!r}")
    if series_id in seen:
        raise CacheError(f"{spot}duplicate series_id {series_id!r}")
    seen.add(series_id)
    numbers: dict[str, int] = {}
    for key in ("length", "period", "first_origin", "n_origins"):
        value = entry[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise CacheError(f"{spot}{key} must be an int, got {value!r}")
        numbers[key] = value
    if numbers["period"] < 2:
        raise CacheError(f"{spot}period must be >= 2, got {numbers['period']}")
    if numbers["n_origins"] < 1:
        raise CacheError(f"{spot}n_origins must be >= 1, got {numbers['n_origins']}")
    if numbers["n_origins"] != numbers["length"] - numbers["first_origin"]:
        raise CacheError(
            f"{spot}n_origins {numbers['n_origins']} != length - first_origin "
            f"({numbers['length']} - {numbers['first_origin']})"
        )
    if numbers["first_origin"] != first_origin(numbers["period"]):
        raise CacheError(
            f"{spot}first_origin must be {first_origin(numbers['period'])} for period "
            f"{numbers['period']}, got {numbers['first_origin']}"
        )
    expected_file = f"series/{index:04d}.f64"
    if entry["file"] != expected_file:
        raise CacheError(f"{spot}file must be {expected_file!r}, got {entry['file']!r}")
    for key in ("sha256", "values_sha256"):
        value = entry[key]
        if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX:
            raise CacheError(f"{spot}{key} must be 64 lowercase hex chars, got {value!r}")
    return CachedSeries(
        series_id=series_id,
        length=numbers["length"],
        period=numbers["period"],
        first_origin=numbers["first_origin"],
        n_origins=numbers["n_origins"],
        file=entry["file"],
        sha256=entry["sha256"],
        values_sha256=entry["values_sha256"],
    )
