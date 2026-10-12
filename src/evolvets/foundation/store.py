"""The on-disk foundation cache for one (dataset, model) pair.

Layout: ``<root>/<dataset>/<model>/manifest.json`` plus
``<root>/<dataset>/<model>/series/<index:04d>.f64`` per series, holding the
adapter's quantile forecasts as little-endian IEEE-754 float64, row-major
``[origin][quantile][h]``. The default root is ``foundation_cache/`` and,
like ``cache_store/``, holds public data only; sealed foundation caches
(TASK_009 PR D) live under the sealed cache root instead.

Building never overwrites: forecasts go into
``<root>/<dataset>/.<model>.partial-<params_hash>/`` — resumable across
crashes, reusing every completed series file — and the finished directory is
renamed into place only once the manifest (written last) is complete. GPU
forecasts are not bit-reproducible, so an existing final directory always
refuses the build, naming its cache id, instead of being compared with a
rebuild (TASK_006's rule with "rebuild and compare equal" replaced by
"built once, content-hashed, verified by hash"). :func:`compare_rebuild`
is the read-only diagnostic that quantifies recompute drift on a sample of
origins; it never edits the cache.
"""

from __future__ import annotations

import json
import math
import os
import random
import struct
import sys
import time
from array import array
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolvets.cache.datasets import public_inputs
from evolvets.cache.interfaces import CacheManifest, SeriesInput
from evolvets.cache.store import DEFAULT_HORIZON, CacheError, cache_id, read_manifest
from evolvets.experts.pool import first_origin
from evolvets.foundation import FOUNDATION_CACHE_VERSION
from evolvets.foundation.build import _checked_row, _fix_crossing, forecast_series
from evolvets.foundation.errors import FoundationError
from evolvets.foundation.interfaces import (
    QUANTILES,
    Adapter,
    FoundationManifest,
    FoundationSeries,
    FoundationSeriesEntry,
    RebuildReport,
    SeriesForecast,
)
from evolvets.foundation.registry import ModelSpec
from evolvets.ledger.canonical import canonical_bytes, sha256_hex

_N_QUANTILES = len(QUANTILES)
_MANIFEST_KEYS = (
    "foundation_cache_version",
    "dataset",
    "source",
    "horizon",
    "quantiles",
    "model",
    "library_versions",
    "device",
    "dtype",
    "batch_size",
    "build_seconds",
    "simple_cache_id",
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
    "sorted_steps",
)
_MODEL_KEYS = (
    "name",
    "hf_model_id",
    "hf_revision",
    "pip_package",
    "dependency_group",
    "max_context",
    "quantile_method",
    "license",
    "pretraining_overlap",
    "overlap_source",
)
_OVERLAP_KEYS = ("etth1", "ecl")
_OVERLAP_VALUES = ("yes", "no", "unknown")
_STATE_FILE = "sorted-steps.json"
_LOWER_ALNUM = "abcdefghijklmnopqrstuvwxyz0123456789"
_NAME_CHARS = set(_LOWER_ALNUM + "_-")
_HEX = set("0123456789abcdef")


def build_foundation_cache(
    dataset: str,
    adapter: Adapter,
    spec: ModelSpec,
    root: Path,
    cache_root: Path,
    data_root: Path,
    manifest_dir: Path,
    batch_size: int,
    per_family: int | None = None,
) -> FoundationManifest:
    """Compute one model's forecasts over a dataset's public series and install them as a cache.

    The inputs are exactly the series the dataset's simple cache was built
    from (:func:`evolvets.cache.datasets.public_inputs`, never raw or sealed
    files) and must match it series for series and origin for origin, checked
    before the adapter is called once. The horizon is the simple cache's
    :data:`DEFAULT_HORIZON`. A crash leaves the partial directory behind, and
    a rerun with the same dataset, model spec, horizon, batch size and source
    (the ``params_hash``) resumes from the first series file that is not
    complete. An installed cache is never overwritten or compared with a
    rebuild: the build refuses, naming the existing cache id.
    """
    started = time.monotonic()
    dataset_name = _check_name("dataset", dataset)
    model_name = _check_name("model", spec.name)
    if adapter.name != spec.name:
        raise FoundationError(
            f"adapter name {adapter.name!r} does not match the model spec name {spec.name!r}"
        )
    batch = _int_at_least("batch_size", batch_size, 1)
    if spec.max_context < 1:
        raise FoundationError(
            f"model {spec.name!r}: max_context must be >= 1, got {spec.max_context}"
        )
    horizon = DEFAULT_HORIZON

    try:
        source, series = public_inputs(
            dataset_name, Path(data_root), Path(manifest_dir), per_family
        )
    except CacheError as exc:
        raise FoundationError(f"dataset {dataset_name!r}: {exc}") from exc
    try:
        simple = read_manifest(Path(cache_root), dataset_name)
        simple_id = cache_id(Path(cache_root), dataset_name)
    except CacheError as exc:
        raise FoundationError(
            f"the simple cache for dataset {dataset_name!r} is missing or unusable: {exc}"
        ) from exc
    _match_simple_cache(dataset_name, series, simple)

    base = Path(root) / dataset_name
    final = base / model_name
    if final.exists():
        raise FoundationError(
            f"foundation cache {final} already exists (cache id {_existing_id(final)}); "
            "remove the directory by hand to rebuild it"
        )
    params_hash = sha256_hex(
        canonical_bytes(
            {
                "dataset": dataset_name,
                "model": _spec_obj(spec),
                "horizon": horizon,
                "batch_size": batch,
                "source": dict(source),
            }
        )
    )[:16]
    partial = base / f".{model_name}.partial-{params_hash}"
    (partial / "series").mkdir(parents=True, exist_ok=True)
    state = _read_state(partial)

    entries: list[FoundationSeriesEntry] = []
    for index, entry in enumerate(series):
        start = first_origin(entry.period)
        n_origins = len(entry.values) - start
        expected = n_origins * _N_QUANTILES * horizon * 8
        name = f"series/{index:04d}.f64"
        path = partial / name
        raw = _reusable_bytes(path, expected)
        sorted_steps = state.get(str(index))
        if raw is None or sorted_steps is None:
            forecast = forecast_series(
                adapter, entry.values, start, spec.max_context, horizon, batch
            )
            raw = _series_bytes(forecast)
            if len(raw) != expected:
                raise FoundationError(
                    f"{spec.name}: series {entry.series_id!r}: {len(raw) // 8} forecast values, "
                    f"expected {expected // 8}"
                )
            sorted_steps = forecast.sorted_steps
            _update_state(partial, state, index, sorted_steps)  # recorded before its file
            _write_renamed(path.with_name(path.name + ".tmp"), path, raw)
        entries.append(
            FoundationSeriesEntry(
                series_id=entry.series_id,
                length=len(entry.values),
                period=entry.period,
                first_origin=start,
                n_origins=n_origins,
                file=name,
                sha256=sha256_hex(raw),
                values_sha256=sha256_hex(canonical_bytes(list(entry.values))),
                sorted_steps=sorted_steps,
            )
        )

    manifest = FoundationManifest(
        foundation_cache_version=FOUNDATION_CACHE_VERSION,
        dataset=dataset_name,
        source=dict(source),
        horizon=horizon,
        quantiles=QUANTILES,
        model=spec,
        library_versions=_adapter_versions(adapter),
        device=_adapter_str_meta(adapter, "device"),
        dtype=_adapter_str_meta(adapter, "dtype"),
        batch_size=batch,
        build_seconds=time.monotonic() - started,
        simple_cache_id=simple_id,
        series=tuple(entries),
    )
    _write_file(partial / "manifest.json", canonical_bytes(_manifest_obj(manifest)))
    _fsync_dir(partial / "series")
    _finalize_cleanup(partial)
    _fsync_dir(partial)
    try:
        os.rename(partial, final)
    except OSError:
        if final.exists():
            raise FoundationError(
                f"foundation cache {final} appeared while building; the built cache is still in "
                f"{partial} — remove one of the two by hand"
            ) from None
        raise
    return manifest


def read_foundation_manifest(root: Path, dataset: str, model: str) -> FoundationManifest:
    """Read and validate ``<root>/<dataset>/<model>/manifest.json``.

    Unknown cache versions are refused, not reinterpreted; every structural
    fault raises :class:`FoundationError`.
    """
    path = _manifest_path(root, dataset, model)
    if not path.exists():
        raise FoundationError(f"{path} is missing; build the cache first")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FoundationError(f"{path}: cannot be read: {exc}") from exc
    return _parse_manifest(raw, str(path), _check_name("dataset", dataset), model)


def foundation_cache_id(root: Path, dataset: str, model: str) -> str:
    """Return the SHA-256 of the (dataset, model) cache's ``manifest.json`` bytes."""
    path = _manifest_path(root, dataset, model)
    if not path.exists():
        raise FoundationError(f"{path} is missing; build the cache first")
    try:
        return sha256_hex(path.read_bytes())
    except OSError as exc:
        raise FoundationError(f"{path}: cannot be read: {exc}") from exc


def verify_foundation_cache(root: Path, dataset: str, model: str) -> FoundationManifest:
    """Read the manifest and check every series file's size and SHA-256 against it.

    No forecast is recomputed: verification proves the bytes on disk are the
    bytes the manifest was built from, nothing more.
    """
    manifest = read_foundation_manifest(root, dataset, model)
    base = _cache_dir(root, manifest.dataset, manifest.model.name)
    for entry in manifest.series:
        raw = _read_series_file(base, entry, manifest.horizon)
        if sha256_hex(raw) != entry.sha256:
            raise FoundationError(
                f"{base / entry.file}: sha256 {sha256_hex(raw)} does not match the "
                f"manifest's {entry.sha256}"
            )
    return manifest


def load_foundation_series(
    root: Path, manifest: FoundationManifest, series_id: str
) -> FoundationSeries:
    """Load one series' quantile forecasts, checking size and SHA-256 against ``manifest``."""
    entry = next((s for s in manifest.series if s.series_id == series_id), None)
    if entry is None:
        raise FoundationError(
            f"unknown series_id {series_id!r} in foundation cache "
            f"{manifest.dataset!r}/{manifest.model.name!r}"
        )
    base = _cache_dir(root, manifest.dataset, manifest.model.name)
    raw = _read_series_file(base, entry, manifest.horizon)
    if sha256_hex(raw) != entry.sha256:
        raise FoundationError(
            f"{base / entry.file}: sha256 {sha256_hex(raw)} does not match the "
            f"manifest's {entry.sha256}"
        )
    data: array[float] = array("d")
    if data.itemsize != 8:
        raise FoundationError("array('d') is not 8 bytes wide on this platform")
    data.frombytes(raw)
    if sys.byteorder == "big":
        data.byteswap()
    return FoundationSeries(
        series_id=entry.series_id,
        first_origin=entry.first_origin,
        horizon=manifest.horizon,
        data=data,
        sorted_steps=entry.sorted_steps,
    )


def compare_rebuild(
    root: Path,
    dataset: str,
    model: str,
    adapter: Adapter,
    sample: int,
    seed: int,
    data_root: Path = Path("data"),
    manifest_dir: Path = Path("data/manifests"),
) -> RebuildReport:
    """Recompute a seeded sample of origins and diff the 0.5 quantile against the cache.

    The sample is drawn without replacement from every ``(series, origin)``
    pair the cache holds, so it spreads over the series in proportion to
    their origin counts; a ``sample`` at least that large compares them all.
    Each sampled origin's context is rebuilt exactly as the build built it —
    ``values[max(0, t + 1 - L) : t + 1]`` with the manifest's ``max_context``
    — recomputed through ``adapter`` one context at a time, validated and
    monotone-fixed exactly like the builder's output, and its 0.5-quantile
    row is compared with the stored one. Stored bytes are read as they are,
    without the manifest's SHA-256 gate: integrity is
    :func:`verify_foundation_cache`'s job, and this diagnostic must be able
    to describe a cache whose numbers drifted. A compared value that is not
    finite is counted in :attr:`RebuildReport.n_nonfinite` and drives both
    maxima to ``inf`` — non-finite drift never reads as a match. The cache
    is never written to.
    """
    manifest = read_foundation_manifest(root, dataset, model)
    if adapter.name != manifest.model.name:
        raise FoundationError(
            f"adapter name {adapter.name!r} does not match the cached model {manifest.model.name!r}"
        )
    sample_ok = _int_at_least("sample", sample, 1)
    seed_ok = _int_at_least("seed", seed, 0)
    max_context = manifest.model.max_context
    if max_context < 1:
        raise FoundationError(
            f"model {manifest.model.name!r}: max_context must be >= 1 to recompute, "
            f"got {max_context}"
        )
    horizon = manifest.horizon
    point_quantile = QUANTILES.index(0.5)

    source, series = public_inputs(
        manifest.dataset, Path(data_root), Path(manifest_dir), _per_family_from_source(manifest)
    )
    _match_manifest_inputs(manifest, series, source)

    base = _cache_dir(root, manifest.dataset, manifest.model.name)
    chosen = _sampled_origins(manifest, sample_ok, seed_ok)
    stored: dict[int, bytes] = {}
    max_abs = 0.0
    max_rel = 0.0
    nonfinite = 0
    for index, origin_index in chosen:
        entry = manifest.series[index]
        if index not in stored:
            stored[index] = _read_series_file(base, entry, horizon)
        old = struct.unpack_from(
            f"<{horizon}d",
            stored[index],
            (origin_index * _N_QUANTILES + point_quantile) * horizon * 8,
        )
        fresh = _recomputed_point(
            adapter, series[index].values, entry, origin_index, max_context, horizon
        )
        for new, was in zip(fresh[point_quantile], old, strict=True):
            if math.isfinite(new) and math.isfinite(was):
                diff = abs(new - was)
                max_abs = max(max_abs, diff)
                max_rel = max(max_rel, diff / max(abs(was), 1e-12))
            else:  # non-finite drift must never read as a match (NaN vanishes in max())
                nonfinite += 1
                max_abs = math.inf
                max_rel = math.inf
    return RebuildReport(
        dataset=manifest.dataset,
        model=manifest.model.name,
        seed=seed_ok,
        n_compared=len(chosen),
        n_total=sum(entry.n_origins for entry in manifest.series),
        n_nonfinite=nonfinite,
        max_abs_diff=max_abs,
        max_rel_diff=max_rel,
    )


# --- rebuild comparison -----------------------------------------------------------------


def _per_family_from_source(manifest: FoundationManifest) -> int | None:
    """Return the ``per_family`` a synthetic cache was built with (None for real data)."""
    if manifest.source.get("kind") != "synthetic":
        return None
    raw = manifest.source.get("per_family")
    try:
        count = int(raw) if isinstance(raw, str) else 0
    except ValueError:
        count = 0
    if count < 1:
        raise FoundationError(
            f"manifest source records per_family {raw!r}; a synthetic cache can only be "
            "compared against the per-family count it was built with"
        )
    return count


def _match_manifest_inputs(
    manifest: FoundationManifest, series: Sequence[SeriesInput], source: Mapping[str, str]
) -> None:
    """Refuse public inputs that no longer match the cache, before recomputing anything."""
    if dict(source) != manifest.source:
        raise FoundationError(
            f"dataset {manifest.dataset!r}: the public inputs' source {dict(source)} no longer "
            f"matches the manifest's {manifest.source}; the cache was built from different inputs"
        )
    inputs_ids = [entry.series_id for entry in series]
    cached_ids = [entry.series_id for entry in manifest.series]
    if inputs_ids != cached_ids:
        raise FoundationError(
            f"dataset {manifest.dataset!r}: the public inputs' series {inputs_ids} no longer "
            f"match the cache's {cached_ids}, so they cannot be compared origin for origin"
        )
    for entry, cached in zip(series, manifest.series, strict=True):
        problems = []
        if entry.period != cached.period:
            problems.append(f"period {entry.period} != {cached.period}")
        if len(entry.values) != cached.length:
            problems.append(f"length {len(entry.values)} != {cached.length}")
        if sha256_hex(canonical_bytes(list(entry.values))) != cached.values_sha256:
            problems.append("values differ (values_sha256)")
        if problems:
            raise FoundationError(
                f"dataset {manifest.dataset!r}: series {entry.series_id!r} no longer matches the "
                f"cached inputs ({'; '.join(problems)}); the cache was built from different "
                "public values"
            )


def _sampled_origins(manifest: FoundationManifest, sample: int, seed: int) -> list[tuple[int, int]]:
    """Return ``sample`` seeded ``(series index, origin index)`` pairs, or all of them."""
    pairs = [
        (index, origin)
        for index, entry in enumerate(manifest.series)
        for origin in range(entry.n_origins)
    ]
    if sample >= len(pairs):
        return pairs
    return sorted(random.Random(seed).sample(pairs, sample))


def _recomputed_point(
    adapter: Adapter,
    values: Sequence[float],
    entry: FoundationSeriesEntry,
    origin_index: int,
    max_context: int,
    horizon: int,
) -> list[list[float]]:
    """Forecast one origin through ``adapter`` on the builder's exact context, and fix crossing."""
    t = entry.first_origin + origin_index
    context = values[max(0, t + 1 - max_context) : t + 1]
    output = adapter.forecast([context], horizon)
    if len(output) != 1:
        raise FoundationError(
            f"{adapter.name}: origin {t}: 1 context sent, {len(output)} forecast rows returned"
        )
    row = _checked_row(adapter.name, t, output[0], horizon)
    _fix_crossing([row], horizon)
    return row


# --- input and simple-cache checks -----------------------------------------------------


def _match_simple_cache(dataset: str, series: Sequence[SeriesInput], simple: CacheManifest) -> None:
    """Refuse public inputs that do not match the simple cache series for series."""
    inputs_ids = [entry.series_id for entry in series]
    simple_ids = [entry.series_id for entry in simple.series]
    if inputs_ids != simple_ids:
        raise FoundationError(
            f"dataset {dataset!r}: the public inputs' series do not match the simple cache's "
            f"series, so the two caches cannot be joined origin for origin: inputs {inputs_ids}, "
            f"simple cache {simple_ids}"
        )
    for entry, cached in zip(series, simple.series, strict=True):
        problems = []
        if entry.period != cached.period:
            problems.append(f"period {entry.period} != {cached.period}")
        if len(entry.values) != cached.length:
            problems.append(f"length {len(entry.values)} != {cached.length}")
        start = first_origin(entry.period)
        if start != cached.first_origin:
            problems.append(f"first_origin {start} != {cached.first_origin}")
        if len(entry.values) - start != cached.n_origins:
            problems.append(f"n_origins {len(entry.values) - start} != {cached.n_origins}")
        values_sha256 = sha256_hex(canonical_bytes(list(entry.values)))
        if values_sha256 != cached.values_sha256:
            problems.append("values differ (values_sha256)")
        if problems:
            raise FoundationError(
                f"dataset {dataset!r}: series {entry.series_id!r} does not match the simple "
                f"cache's entry ({'; '.join(problems)}); rebuild the simple cache from the same "
                "inputs first"
            )


# --- adapter-run metadata --------------------------------------------------------------


def _adapter_str_meta(adapter: Adapter, key: str) -> str:
    """Read a string attribute the adapter must expose about its own run (``device``, ``dtype``)."""
    value = getattr(adapter, key, None)
    if not isinstance(value, str) or not value:
        raise FoundationError(
            f"adapter {adapter.name!r} must expose a non-empty {key!r} attribute for the "
            "manifest (see FakeAdapter)"
        )
    return value


def _adapter_versions(adapter: Adapter) -> dict[str, str]:
    """Read the adapter's ``library_versions`` mapping (``{}`` for the fake adapter)."""
    value = getattr(adapter, "library_versions", None)
    if not isinstance(value, Mapping):
        raise FoundationError(
            f"adapter {adapter.name!r} must expose a 'library_versions' mapping for the "
            "manifest (see FakeAdapter)"
        )
    out: dict[str, str] = {}
    for package, version in value.items():
        if not isinstance(package, str) or not isinstance(version, str) or not version:
            raise FoundationError(
                f"adapter {adapter.name!r}: library_versions entries must be non-empty "
                f"strings, got {package!r}: {version!r}"
            )
        out[package] = version
    return out


# --- serialization and validation ------------------------------------------------------


def _spec_obj(spec: ModelSpec) -> dict[str, object]:
    """Return the JSON-ready object the manifest's ``model`` field holds."""
    return {
        "name": spec.name,
        "hf_model_id": spec.hf_model_id,
        "hf_revision": spec.hf_revision,
        "pip_package": spec.pip_package,
        "dependency_group": spec.dependency_group,
        "max_context": spec.max_context,
        "quantile_method": spec.quantile_method,
        "license": spec.license,
        "pretraining_overlap": dict(spec.pretraining_overlap),
        "overlap_source": spec.overlap_source,
    }


def _manifest_obj(manifest: FoundationManifest) -> dict[str, object]:
    """Return the JSON-ready object ``manifest.json`` holds."""
    return {
        "foundation_cache_version": manifest.foundation_cache_version,
        "dataset": manifest.dataset,
        "source": dict(manifest.source),
        "horizon": manifest.horizon,
        "quantiles": list(manifest.quantiles),
        "model": _spec_obj(manifest.model),
        "library_versions": dict(manifest.library_versions),
        "device": manifest.device,
        "dtype": manifest.dtype,
        "batch_size": manifest.batch_size,
        "build_seconds": manifest.build_seconds,
        "simple_cache_id": manifest.simple_cache_id,
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
                "sorted_steps": s.sorted_steps,
            }
            for s in manifest.series
        ],
    }


def _parse_manifest(raw: bytes, where: str, dataset: str, model: str) -> FoundationManifest:
    """Parse and fully validate manifest bytes; raise FoundationError on any fault."""
    try:
        obj: object = json.loads(raw)
    except ValueError as exc:
        raise FoundationError(f"{where}: not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise FoundationError(f"{where}: must be a JSON object")
    obj = {str(key): value for key, value in obj.items()}
    if set(obj) != set(_MANIFEST_KEYS):
        raise FoundationError(f"{where}: keys must be exactly {sorted(_MANIFEST_KEYS)}")
    if obj["foundation_cache_version"] != FOUNDATION_CACHE_VERSION:
        raise FoundationError(
            f"{where}: foundation_cache_version {obj['foundation_cache_version']!r} != current "
            f"{FOUNDATION_CACHE_VERSION!r}; unknown cache versions are refused, not reinterpreted"
        )
    if obj["dataset"] != dataset:
        raise FoundationError(f"{where}: dataset must be {dataset!r}, got {obj['dataset']!r}")
    source = _parse_str_map(obj["source"], f"{where}: source")
    horizon = _int_at_least(f"{where}: horizon", obj["horizon"], 1)
    quantiles_obj = obj["quantiles"]
    if not isinstance(quantiles_obj, list) or quantiles_obj != list(QUANTILES):
        raise FoundationError(
            f"{where}: quantiles must be exactly {list(QUANTILES)}, got {quantiles_obj!r}"
        )
    quantiles = tuple(float(q) for q in quantiles_obj)
    spec = _parse_spec(obj["model"], f"{where}: model")
    if spec.name != model:
        raise FoundationError(f"{where}: model name must be {model!r}, got {spec.name!r}")
    library_versions = _parse_str_map(obj["library_versions"], f"{where}: library_versions")
    device = _str_value(obj["device"], f"{where}: device")
    dtype = _str_value(obj["dtype"], f"{where}: dtype")
    batch_size = _int_at_least(f"{where}: batch_size", obj["batch_size"], 1)
    seconds = obj["build_seconds"]
    if (
        isinstance(seconds, bool)
        or not isinstance(seconds, (int, float))
        or not math.isfinite(seconds)
        or seconds < 0
    ):
        raise FoundationError(
            f"{where}: build_seconds must be a finite number >= 0, got {seconds!r}"
        )
    simple_id = obj["simple_cache_id"]
    if not isinstance(simple_id, str) or len(simple_id) != 64 or not set(simple_id) <= _HEX:
        raise FoundationError(
            f"{where}: simple_cache_id must be 64 lowercase hex chars, got {simple_id!r}"
        )
    series_obj = obj["series"]
    if not isinstance(series_obj, list) or not series_obj:
        raise FoundationError(f"{where}: series must be a non-empty list")
    seen: set[str] = set()
    entries = []
    for index, entry in enumerate(series_obj):
        entries.append(_parse_series_entry(entry, index, where, seen))
    return FoundationManifest(
        foundation_cache_version=FOUNDATION_CACHE_VERSION,
        dataset=dataset,
        source=source,
        horizon=horizon,
        quantiles=quantiles,
        model=spec,
        library_versions=library_versions,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        build_seconds=float(seconds),
        simple_cache_id=simple_id,
        series=tuple(entries),
    )


def _parse_spec(obj: object, where: str) -> ModelSpec:
    """Validate the manifest's ``model`` object and rebuild the ModelSpec from it."""
    if not isinstance(obj, dict):
        raise FoundationError(f"{where} must be a JSON object")
    obj = {str(key): value for key, value in obj.items()}
    if set(obj) != set(_MODEL_KEYS):
        raise FoundationError(f"{where}: keys must be exactly {sorted(_MODEL_KEYS)}")
    spec = ModelSpec(
        name=_str_value(obj["name"], f"{where}: name"),
        hf_model_id=_str_value(obj["hf_model_id"], f"{where}: hf_model_id"),
        hf_revision=_str_value(obj["hf_revision"], f"{where}: hf_revision"),
        pip_package=_str_value(obj["pip_package"], f"{where}: pip_package"),
        dependency_group=_str_value(obj["dependency_group"], f"{where}: dependency_group"),
        max_context=_int_at_least(f"{where}: max_context", obj["max_context"], 0),
        quantile_method=_str_value(obj["quantile_method"], f"{where}: quantile_method"),
        license=_str_value(obj["license"], f"{where}: license"),
        pretraining_overlap=_parse_overlap(obj["pretraining_overlap"], where),
        overlap_source=_str_value(obj["overlap_source"], f"{where}: overlap_source"),
    )
    return spec


def _parse_overlap(obj: object, where: str) -> dict[str, str]:
    """Return ``obj`` as the pretraining-overlap map, else raise FoundationError."""
    if not isinstance(obj, dict):
        raise FoundationError(f"{where}: pretraining_overlap must be a JSON object")
    out: dict[str, str] = {}
    for key, value in obj.items():
        if str(key) not in _OVERLAP_KEYS:
            raise FoundationError(
                f"{where}: pretraining_overlap keys must be exactly {sorted(_OVERLAP_KEYS)}"
            )
        if not isinstance(value, str) or value not in _OVERLAP_VALUES:
            raise FoundationError(
                f"{where}: pretraining_overlap[{key!r}] must be one of "
                f"{list(_OVERLAP_VALUES)}, got {value!r}"
            )
        out[str(key)] = value
    if set(out) != set(_OVERLAP_KEYS):
        raise FoundationError(
            f"{where}: pretraining_overlap keys must be exactly {sorted(_OVERLAP_KEYS)}"
        )
    return out


def _parse_series_entry(
    entry: object, index: int, where: str, seen: set[str]
) -> FoundationSeriesEntry:
    """Validate one ``series`` entry of a manifest against its own fields."""
    spot = f"{where}: series[{index}]: "
    if not isinstance(entry, dict):
        raise FoundationError(f"{spot}must be a JSON object")
    entry = {str(key): value for key, value in entry.items()}
    if set(entry) != set(_SERIES_KEYS):
        raise FoundationError(f"{spot}keys must be exactly {sorted(_SERIES_KEYS)}")
    series_id = _str_value(entry["series_id"], f"{spot}series_id")
    if series_id in seen:
        raise FoundationError(f"{spot}duplicate series_id {series_id!r}")
    seen.add(series_id)
    minimums = {"length": 1, "period": 2, "first_origin": 0, "n_origins": 1}
    numbers = {
        key: _int_at_least(f"{spot}{key}", entry[key], minimum) for key, minimum in minimums.items()
    }
    if numbers["n_origins"] != numbers["length"] - numbers["first_origin"]:
        raise FoundationError(
            f"{spot}n_origins {numbers['n_origins']} != length - first_origin "
            f"({numbers['length']} - {numbers['first_origin']})"
        )
    if numbers["first_origin"] != first_origin(numbers["period"]):
        raise FoundationError(
            f"{spot}first_origin must be {first_origin(numbers['period'])} for period "
            f"{numbers['period']}, got {numbers['first_origin']}"
        )
    expected_file = f"series/{index:04d}.f64"
    if entry["file"] != expected_file:
        raise FoundationError(f"{spot}file must be {expected_file!r}, got {entry['file']!r}")
    for key in ("sha256", "values_sha256"):
        value = entry[key]
        if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX:
            raise FoundationError(f"{spot}{key} must be 64 lowercase hex chars, got {value!r}")
    return FoundationSeriesEntry(
        series_id=series_id,
        length=numbers["length"],
        period=numbers["period"],
        first_origin=numbers["first_origin"],
        n_origins=numbers["n_origins"],
        file=expected_file,
        sha256=entry["sha256"],
        values_sha256=entry["values_sha256"],
        sorted_steps=_int_at_least(f"{spot}sorted_steps", entry["sorted_steps"], 0),
    )


def _str_value(value: object, where: str) -> str:
    """Return ``value`` as a non-empty string, else raise FoundationError."""
    if not isinstance(value, str) or not value:
        raise FoundationError(f"{where} must be a non-empty string, got {value!r}")
    return value


def _parse_str_map(obj: object, where: str) -> dict[str, str]:
    """Return ``obj`` as a string-to-string mapping, else raise FoundationError."""
    if not isinstance(obj, dict):
        raise FoundationError(f"{where} must be a JSON object")
    out: dict[str, str] = {}
    for key, value in obj.items():
        if not isinstance(value, str):
            raise FoundationError(f"{where}[{key!r}] must be a string, got {value!r}")
        out[str(key)] = value
    return out


# --- paths, files and the resume state -------------------------------------------------


def _cache_dir(root: Path, dataset: str, model: str) -> Path:
    """Return ``<root>/<dataset>/<model>`` after checking both names."""
    return Path(root) / _check_name("dataset", dataset) / _check_name("model", model)


def _manifest_path(root: Path, dataset: str, model: str) -> Path:
    """Return the manifest path of the (dataset, model) cache."""
    return _cache_dir(root, dataset, model) / "manifest.json"


def _check_name(kind: str, value: object) -> str:
    """Return ``value`` if it matches ``^[a-z0-9][a-z0-9_-]*$``, else raise FoundationError."""
    if (
        not isinstance(value, str)
        or not value
        or not set(value) <= _NAME_CHARS
        or value[0] not in _LOWER_ALNUM
    ):
        raise FoundationError(f"{kind} name must match ^[a-z0-9][a-z0-9_-]*$, got {value!r}")
    return value


def _int_at_least(name: str, value: object, minimum: int) -> int:
    """Return ``value`` as an int >= ``minimum`` (bools rejected), else raise FoundationError."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise FoundationError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise FoundationError(f"{name} must be >= {minimum}, got {value}")
    return value


def _existing_id(final: Path) -> str:
    """Return the cache id of an already-installed cache, or ``<unreadable>``."""
    try:
        return sha256_hex((final / "manifest.json").read_bytes())
    except OSError:
        return "<unreadable>"


def _reusable_bytes(path: Path, expected: int) -> bytes | None:
    """Return the file's bytes if it exists with exactly ``expected`` bytes, else None."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) != expected:
        return None
    return raw


def _series_bytes(forecast: SeriesForecast) -> bytes:
    """Flatten one :class:`SeriesForecast` to little-endian float64 bytes.

    The layout is row-major ``[origin][quantile][h]``, exactly as stored.
    """
    flat: array[float] = array("d")
    for row in forecast.values:
        for quantile in row:
            flat.extend(quantile)
    if sys.byteorder == "big":
        flat.byteswap()
    return flat.tobytes()


def _read_series_file(base: Path, entry: FoundationSeriesEntry, horizon: int) -> bytes:
    """Read one series file, checking its size against the manifest."""
    path = base / entry.file
    expected = entry.n_origins * _N_QUANTILES * horizon * 8
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FoundationError(f"{path}: cannot be read: {exc}") from exc
    if len(raw) != expected:
        raise FoundationError(f"{path}: expected {expected} bytes, got {len(raw)}")
    return raw


def _write_file(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` and fsync it, so a rename never outlives its data."""
    with open(path, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _write_renamed(tmp: Path, final: Path, data: bytes) -> None:
    """Write ``data`` to ``tmp``, fsync it, and rename it onto ``final``."""
    _write_file(tmp, data)
    os.replace(tmp, final)


def _fsync_dir(path: Path) -> None:
    """Fsync a directory so its entries survive a crash across the rename."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _state_path(partial: Path) -> Path:
    """Return the partial directory's sorted-steps bookkeeping file."""
    return partial / _STATE_FILE


def _read_state(partial: Path) -> dict[str, int]:
    """Read the partial directory's ``{series index: sorted_steps}`` map, or ``{}`` if unusable.

    The state is resume bookkeeping, not a verified artefact: anything unreadable
    or malformed forces recomputation of the series it described, never a wrong
    count in the manifest.
    """
    try:
        raw = _state_path(partial).read_bytes()
    except OSError:
        return {}
    try:
        obj: object = json.loads(raw)
    except ValueError:
        return {}
    if not isinstance(obj, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in obj.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return {}
        out[str(key)] = value
    return out


def _update_state(partial: Path, state: dict[str, int], index: int, sorted_steps: int) -> None:
    """Record one series' sorted_steps before that series' file is written (crash order)."""
    state[str(index)] = sorted_steps
    path = _state_path(partial)
    _write_renamed(path.with_name(_STATE_FILE + ".tmp"), path, canonical_bytes(state))


def _finalize_cleanup(partial: Path) -> None:
    """Drop the partial directory's bookkeeping so the final cache holds only its spec'd files."""
    for leftover in (partial / "series").glob("*.tmp"):
        leftover.unlink()
    for leftover in partial.glob("*.tmp"):
        leftover.unlink()
    _state_path(partial).unlink(missing_ok=True)
