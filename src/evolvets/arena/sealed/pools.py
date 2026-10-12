"""Sealed pools: the salt, the sealed inputs, and the caches and pools built from them.

Everything here runs in trusted, human-started processes (the ``evolvets
sealed`` CLI, and from PR C on the sealed server). The synthetic worlds are
drawn from a salt only a human creates (:func:`init_salt`); the salt bytes
never appear in an exception message, a public return value or a log. The
real sealed datasets are read from ``<dataset>.sealed.json`` and tied to the
public data by the prefix check, so a sealed file that does not extend the
public one is refused. Sealed caches live outside the public cache root and
hold forecasts cut at ``sealed.end + horizon``: nothing from the reserve
segment ever enters one. A sealed pool's foundation caches (TASK_009 PR D)
live under the same sealed cache root, one directory per model beside the
simple cache's series, and are built and joined by exactly the code that
builds and joins the public ones.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evolvets.arena.dev import (
    DevPool,
    DevSeries,
    checked_series,
    checked_synthetic_series,
    match_values,
)
from evolvets.arena.errors import ArenaError
from evolvets.arena.pool import FoundationCache, check_pool_spec, load_foundation_caches
from evolvets.arena.realdata import REALDATA_VERSION
from evolvets.arena.realdata.fetch import file_sha256
from evolvets.arena.realdata.manifest import manifest_text, write_atomic
from evolvets.arena.realdata.prepare import (
    PrepareError,
    SplitManifest,
    load_public,
    read_split_manifest,
)
from evolvets.arena.realdata.splits import Split
from evolvets.arena.sealed.names import SEALED_DATASETS
from evolvets.arena.synthetic import GENERATOR_VERSION
from evolvets.arena.synthetic.grammar import FAMILIES, SyntheticTask, make_task
from evolvets.arena.synthetic.seeds import world_seed
from evolvets.cache.interfaces import CacheManifest, SeriesInput
from evolvets.cache.store import (
    DEFAULT_HORIZON,
    CacheError,
    build_cache,
    cache_id,
    read_manifest,
)
from evolvets.foundation.errors import FoundationError
from evolvets.foundation.interfaces import Adapter, FoundationManifest
from evolvets.foundation.registry import ModelSpec
from evolvets.foundation.store import build_foundation_cache_from_inputs
from evolvets.ledger.canonical import sha256_hex

SALT_BYTES = 32
SYNTHETIC_MANIFEST_VERSION = "1"
_REAL_PERIOD = 24
#: Directory names that mark a path as one of the two public cache roots; a
#: sealed cache root that is or sits under either is refused.
_PUBLIC_ROOT_NAMES = ("cache_store", "foundation_cache")

_HEX = set("0123456789abcdef")
_MANIFEST_KEYS = (
    "manifest_version",
    "pool",
    "generator_version",
    "salt_sha256",
    "salt_bytes",
    "per_family",
)


@dataclass(frozen=True)
class SyntheticSealedManifest:
    """The parsed contents of ``synthetic-sealed.yaml``."""

    generator_version: str
    salt_sha256: str  # 64 lowercase hex chars
    per_family: int  # >= 1


def salt_path(data_root: Path) -> Path:
    """Return ``<data_root>/sealed/synthetic.salt``."""
    return Path(data_root) / "sealed" / "synthetic.salt"


def synthetic_manifest_path(manifest_dir: Path) -> Path:
    """Return ``<manifest_dir>/synthetic-sealed.yaml``."""
    return Path(manifest_dir) / "synthetic-sealed.yaml"


def init_salt(data_root: Path, manifest_dir: Path, per_family: int) -> str:
    """Create the sealed synthetic salt and its manifest once; return the salt's SHA-256.

    Run once by a human (``evolvets sealed init-salt``); agents only ever run
    it against ``tmp_path``. Neither file is ever overwritten or deleted: a
    salt that survived a crash before its manifest was written is recovered
    into a fresh manifest, and every other combination — a missing salt, a
    different salt, a different ``per_family`` — is an error that changes
    nothing.
    """
    per_family = _check_per_family(per_family)
    salt_file = salt_path(data_root)
    manifest_file = synthetic_manifest_path(manifest_dir)
    if not salt_file.exists() and not manifest_file.exists():
        salt_file.parent.mkdir(parents=True, exist_ok=True)
        _create_salt(salt_file)
    elif not salt_file.exists():
        raise ArenaError(
            f"{manifest_file} exists but {salt_file} is missing: restore the salt from "
            "backup and never regenerate it"
        )
    salt = _read_salt(salt_file)
    digest = sha256_hex(salt)
    if manifest_file.exists():
        spec = read_synthetic_manifest(manifest_dir)
        if spec.salt_sha256 != digest:
            raise ArenaError(
                f"{salt_file} no longer matches {manifest_file}: the file hashes to "
                f"{digest}, the manifest pins {spec.salt_sha256}"
            )
        if spec.per_family != per_family:
            raise ArenaError(
                f"{manifest_file} records per_family {spec.per_family}, not {per_family}; "
                "the sealed pool size is fixed once the salt exists"
            )
    else:
        write_atomic(manifest_file, _manifest_bytes(digest, per_family))
    return digest


def read_synthetic_manifest(manifest_dir: Path) -> SyntheticSealedManifest:
    """Read and validate ``synthetic-sealed.yaml``; raise ArenaError on any fault.

    Unknown generator versions are refused, not reinterpreted.
    """
    path = synthetic_manifest_path(manifest_dir)
    if not path.exists():
        raise ArenaError(f"{path} is missing; run 'evolvets sealed init-salt --per-family N'")
    try:
        obj: object = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ArenaError(f"{path}: cannot be read: {exc}") from exc
    if not isinstance(obj, dict):
        raise ArenaError(f"{path}: manifest must be a JSON object")
    where = f"{path}: "
    obj = {str(key): value for key, value in obj.items()}
    if set(obj) != set(_MANIFEST_KEYS):
        raise ArenaError(f"{where}keys must be exactly {sorted(_MANIFEST_KEYS)}")
    if obj["manifest_version"] != SYNTHETIC_MANIFEST_VERSION:
        raise ArenaError(
            f"{where}manifest_version must be {SYNTHETIC_MANIFEST_VERSION!r}, "
            f"got {obj['manifest_version']!r}"
        )
    if obj["pool"] != "synthetic-sealed":
        raise ArenaError(f"{where}pool must be 'synthetic-sealed', got {obj['pool']!r}")
    if obj["generator_version"] != GENERATOR_VERSION:
        raise ArenaError(
            f"{where}generator_version {obj['generator_version']!r} != current "
            f"{GENERATOR_VERSION!r}; unknown generator versions are refused, not reinterpreted"
        )
    if isinstance(obj["salt_bytes"], bool) or obj["salt_bytes"] != SALT_BYTES:
        raise ArenaError(f"{where}salt_bytes must be {SALT_BYTES}, got {obj['salt_bytes']!r}")
    salt_sha256 = obj["salt_sha256"]
    if not isinstance(salt_sha256, str) or len(salt_sha256) != 64 or not set(salt_sha256) <= _HEX:
        raise ArenaError(f"{where}salt_sha256 must be 64 lowercase hex chars, got {salt_sha256!r}")
    per_family = obj["per_family"]
    if isinstance(per_family, bool) or not isinstance(per_family, int) or per_family < 1:
        raise ArenaError(f"{where}per_family must be an int >= 1, got {per_family!r}")
    return SyntheticSealedManifest(
        generator_version=GENERATOR_VERSION,
        salt_sha256=salt_sha256,
        per_family=per_family,
    )


def sealed_inputs(
    dataset: str, data_root: Path, manifest_dir: Path, horizon: int
) -> tuple[dict[str, str], tuple[SeriesInput, ...]]:
    """Return the ``(source, series)`` a sealed cache for ``dataset`` is built from.

    The real datasets read their sealed file and keep only
    ``values[: sealed.end + horizon]``; ``synthetic-sealed`` derives its
    worlds from the salt. Nothing from a reserve segment is returned.
    """
    if dataset in ("etth1", "ecl"):
        source, series, _splits = _real_sealed(dataset, data_root, manifest_dir, horizon)
        return source, series
    if dataset == "synthetic-sealed":
        source, series, _tasks, _spec = _synthetic_sealed_inputs(data_root, manifest_dir)
        return source, series
    raise ArenaError(f"unknown dataset {dataset!r}; expected one of {list(SEALED_DATASETS)}")


def build_sealed_cache(
    dataset: str,
    data_root: Path,
    manifest_dir: Path,
    sealed_cache_root: Path,
    horizon: int = DEFAULT_HORIZON,
) -> CacheManifest:
    """Build the sealed forecast cache for ``dataset`` under ``sealed_cache_root``.

    This only adds the sealed guarantees to
    :func:`evolvets.cache.store.build_cache` — which builds into a temporary
    directory, never overwrites, and treats an equal rebuild as a no-op — and
    refuses a cache root that could be the public one.
    """
    root = _checked_cache_root(sealed_cache_root)
    if dataset not in SEALED_DATASETS:
        raise ArenaError(f"unknown dataset {dataset!r}; expected one of {list(SEALED_DATASETS)}")
    source, series = sealed_inputs(dataset, data_root, manifest_dir, horizon)
    try:
        return build_cache(dataset, source, series, root, horizon=horizon)
    except CacheError as exc:
        raise ArenaError(str(exc)) from exc


def build_sealed_foundation_cache(
    dataset: str,
    adapter: Adapter,
    spec: ModelSpec,
    data_root: Path,
    manifest_dir: Path,
    sealed_cache_root: Path,
    batch_size: int,
) -> FoundationManifest:
    """Build one model's foundation cache for a sealed pool, under the sealed cache root.

    The same adapter and builder as the public build
    (:func:`evolvets.foundation.store.build_foundation_cache`), run from
    inside the arena's sealed code path: the inputs are the sealed pool's own
    series (:func:`sealed_inputs`, cut at ``sealed.end + horizon``, so no
    value from a reserve segment is ever forecast), the simple cache the
    forecasts must join is the sealed simple cache, and the cache under
    construction is installed at ``<sealed_cache_root>/<dataset>/<model>/``
    beside it. A root that is or sits under ``cache_store/`` or
    ``foundation_cache/`` is refused, so no sealed forecast can land in a
    directory a public tool reads.
    """
    if dataset not in SEALED_DATASETS:
        raise ArenaError(f"unknown dataset {dataset!r}; expected one of {list(SEALED_DATASETS)}")
    root = _checked_cache_root(sealed_cache_root)
    source, series = sealed_inputs(dataset, Path(data_root), Path(manifest_dir), DEFAULT_HORIZON)
    try:
        return build_foundation_cache_from_inputs(
            dataset, adapter, spec, root, root, source, series, batch_size
        )
    except FoundationError as exc:
        raise ArenaError(str(exc)) from exc


def load_sealed_pool(
    dataset: str,
    data_root: Path,
    manifest_dir: Path,
    sealed_cache_root: Path,
    *,
    pool_spec: str = "simple",
    foundation_models: Sequence[str] | None = None,
) -> DevPool:
    """Load the sealed pool for ``dataset`` from its sealed cache and sealed data.

    The pool is the dev replay's own type, so :func:`replay_dataset` and its
    scoring run on sealed pools unchanged. With
    ``pool_spec="simple+foundation"`` the foundation caches built under
    ``sealed_cache_root`` by :func:`build_sealed_foundation_cache` are loaded
    and joined onto the sealed simple cache exactly as the dev pool joins its
    own — both caches share the root, so there is no separate foundation root
    to pass; ``foundation_models`` names the models (tests name the fake) and
    defaults to the pinned registry entries. Every failure — an unknown
    dataset, a bad pool spec, a missing or tampered sealed cache, a foundation
    cache that does not join, missing or tampered sealed data — surfaces as
    :class:`ArenaError`.
    """
    if dataset not in SEALED_DATASETS:
        raise ArenaError(f"unknown dataset {dataset!r}; expected one of {list(SEALED_DATASETS)}")
    spec = check_pool_spec(pool_spec)
    if spec == "simple" and foundation_models is not None:
        raise ArenaError("pool_spec 'simple' takes no foundation models")
    root = _checked_cache_root(sealed_cache_root)
    try:
        manifest = read_manifest(root, dataset)
        ident = cache_id(root, dataset)
    except CacheError as exc:
        raise ArenaError(str(exc)) from exc
    foundation: tuple[FoundationCache, ...] = ()
    if spec == "simple+foundation":
        foundation = load_foundation_caches(dataset, ident, manifest, root, foundation_models)
    if dataset in ("etth1", "ecl"):
        series = _real_pool_series(dataset, manifest, data_root, manifest_dir)
    else:
        series = _synthetic_pool_series(manifest, data_root, manifest_dir)
    return DevPool(
        dataset=dataset,
        cache_root=root,
        cache_id=ident,
        manifest=manifest,
        series=series,
        pool_spec=spec,
        foundation=foundation,
    )


def _real_pool_series(
    dataset: str, manifest: CacheManifest, data_root: Path, manifest_dir: Path
) -> tuple[DevSeries, ...]:
    """Build the real sealed series: sealed boundaries, no change points, no oracle."""
    source, inputs, splits = _real_sealed(dataset, data_root, manifest_dir, manifest.horizon)
    if len(inputs) != len(manifest.series):
        raise ArenaError(
            f"{manifest.dataset}: the cache holds {len(manifest.series)} series but the "
            f"sealed data has {len(inputs)}"
        )
    out: list[DevSeries] = []
    for entry, series_input, split in zip(manifest.series, inputs, splits, strict=True):
        if series_input.series_id != entry.series_id:
            raise ArenaError(
                f"{manifest.dataset}: cache series {entry.series_id!r} does not match the "
                f"sealed series {series_input.series_id!r} at the same position"
            )
        if entry.period != _REAL_PERIOD:
            raise ArenaError(
                f"{entry.series_id}: the cache records period {entry.period}, not {_REAL_PERIOD}"
            )
        match_values(entry, series_input.values)
        warmup, _dev, sealed, _reserve = split.segments
        out.append(
            checked_series(
                entry,
                series_input.values,
                manifest.horizon,
                scale_end=warmup.end,
                score_start=sealed.start,
                score_end=sealed.end,
                change_points=(),
                oracle=None,
            )
        )
    _check_source(manifest, source)
    return tuple(out)


def _synthetic_pool_series(
    manifest: CacheManifest, data_root: Path, manifest_dir: Path
) -> tuple[DevSeries, ...]:
    """Build the synthetic sealed series: salted worlds, change points and the oracle."""
    source, _series, tasks, _spec = _synthetic_sealed_inputs(data_root, manifest_dir)
    series = checked_synthetic_series(manifest, tasks)
    _check_source(manifest, source)
    return series


def _real_sealed(
    dataset: str, data_root: Path, manifest_dir: Path, horizon: int
) -> tuple[dict[str, str], tuple[SeriesInput, ...], tuple[Split, ...]]:
    """Derive the real sealed source, series and splits from the sealed file."""
    try:
        spec = read_split_manifest(manifest_dir, dataset)
        public = load_public(dataset, data_root, manifest_dir)
    except PrepareError as exc:
        raise ArenaError(str(exc)) from exc
    sealed_file = Path(data_root) / "processed" / f"{dataset}.sealed.json"
    values = _sealed_file_values(sealed_file, dataset, spec)
    source = {
        "kind": "realdata-sealed",
        "dataset": dataset,
        "realdata_version": spec.realdata_version,
        "sealed_sha256": spec.sealed_sha256,
        "public_sha256": spec.public_sha256,
    }
    series: list[SeriesInput] = []
    splits: list[Split] = []
    for (series_id, _start, split), loaded, full in zip(spec.series, public, values, strict=True):
        _warmup, _dev, sealed, reserve = split.segments
        if loaded.values != full[: sealed.start]:
            raise ArenaError(
                f"{sealed_file}: series {series_id!r} does not extend the public data: its "
                f"first {sealed.start} values differ from the public file's"
            )
        if sealed.end + horizon > reserve.start:
            raise ArenaError(
                f"{series_id}: horizon {horizon} needs the target at index "
                f"{sealed.end + horizon - 1}, inside the reserve segment from {reserve.start}"
            )
        series.append(SeriesInput(series_id, full[: sealed.end + horizon], _REAL_PERIOD))
        splits.append(split)
    return source, tuple(series), tuple(splits)


def _sealed_file_values(
    sealed_file: Path, dataset: str, spec: SplitManifest
) -> tuple[tuple[float, ...], ...]:
    """Read the sealed file with ``load_public``'s checks, at full length.

    The file's SHA-256 must equal the split manifest's ``sealed_sha256``, the
    keys, dataset, ``realdata_version``, series ids and starts must match the
    manifest in order, every value must be a plain number, and each series'
    length must equal its split's full length.
    """
    if not sealed_file.exists():
        raise ArenaError(f"{sealed_file} is missing; run 'evolvets data prepare {dataset}'")
    digest, _size = file_sha256(sealed_file)
    if digest != spec.sealed_sha256:
        raise ArenaError(
            f"{sealed_file} does not match the split manifest: file sha256 {digest}, "
            f"manifest sha256 {spec.sealed_sha256}"
        )
    try:
        obj: object = json.loads(sealed_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise ArenaError(f"{sealed_file}: cannot be read: {exc}") from exc
    if not isinstance(obj, dict):
        raise ArenaError(f"{sealed_file}: must be a JSON object")
    obj = {str(key): value for key, value in obj.items()}
    if set(obj) != {"dataset", "realdata_version", "series"}:
        raise ArenaError(f"{sealed_file}: keys must be exactly dataset, realdata_version, series")
    if obj["dataset"] != dataset:
        raise ArenaError(f"{sealed_file}: dataset must be {dataset!r}, got {obj['dataset']!r}")
    if obj["realdata_version"] != REALDATA_VERSION:
        raise ArenaError(
            f"{sealed_file}: realdata_version must be {REALDATA_VERSION!r}, "
            f"got {obj['realdata_version']!r}"
        )
    entries = obj["series"]
    if not isinstance(entries, list) or len(entries) != len(spec.series):
        raise ArenaError(
            f"{sealed_file}: series count does not match the manifest's {len(spec.series)}"
        )
    out: list[tuple[float, ...]] = []
    for index, (entry, (series_id, start, split)) in enumerate(
        zip(entries, spec.series, strict=True)
    ):
        where = f"{sealed_file}: series[{index}]: "
        if not isinstance(entry, dict):
            raise ArenaError(f"{where}must be a JSON object")
        entry = {str(key): value for key, value in entry.items()}
        if set(entry) != {"series_id", "start", "values"}:
            raise ArenaError(f"{where}keys must be exactly series_id, start, values")
        if entry["series_id"] != series_id or entry["start"] != start:
            raise ArenaError(
                f"{where}must be {series_id!r} starting {start!r}, got "
                f"{entry['series_id']!r} starting {entry['start']!r}"
            )
        raw_values = entry["values"]
        if not isinstance(raw_values, list):
            raise ArenaError(f"{where}values must be a list")
        checked: list[float] = []
        for value in raw_values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ArenaError(f"{where}values must be numbers, got {value!r}")
            checked.append(float(value))
        if len(checked) != split.length:
            raise ArenaError(f"{where}length {len(checked)} != the split's length {split.length}")
        out.append(tuple(checked))
    return tuple(out)


def _checked_cache_root(sealed_cache_root: Path) -> Path:
    """Return ``sealed_cache_root`` unless it is or sits under a public cache root.

    Any path component named ``cache_store`` or ``foundation_cache`` marks
    the root as the public simple-cache or public foundation-cache root, or a
    directory inside one; both hold public data only, so the sealed cache
    must live somewhere no public tool reads.
    """
    root = Path(sealed_cache_root)
    for part in root.parts:
        if part in _PUBLIC_ROOT_NAMES:
            raise ArenaError(
                f"{root}: this path is or sits under a public cache root ({part}); the sealed "
                "cache must live in a directory no public tool reads"
            )
    return root


def _check_source(manifest: CacheManifest, source: dict[str, str]) -> None:
    """Refuse a cache whose source does not match the one the sealed data recomputes."""
    if manifest.source != source:
        raise ArenaError(
            f"{manifest.dataset}: the cache was built from {manifest.source}, but the sealed "
            f"data now recomputes {source}"
        )


def _synthetic_sealed_inputs(
    data_root: Path, manifest_dir: Path
) -> tuple[
    dict[str, str],
    tuple[SeriesInput, ...],
    list[tuple[str, SyntheticTask]],
    SyntheticSealedManifest,
]:
    """Derive the sealed synthetic source, series, tasks and manifest from the salt."""
    spec = read_synthetic_manifest(manifest_dir)
    salt = _load_salt(data_root, spec)
    tasks = _sealed_tasks(spec, salt)
    source = {
        "kind": "synthetic",
        "pool": "sealed",
        "generator_version": GENERATOR_VERSION,
        "per_family": str(spec.per_family),
        "salt_sha256": spec.salt_sha256,
    }
    series = tuple(
        SeriesInput(series_id, task.world.y, task.world.spec.period) for series_id, task in tasks
    )
    return source, series, tasks, spec


def _sealed_tasks(spec: SyntheticSealedManifest, salt: bytes) -> list[tuple[str, SyntheticTask]]:
    """Return ``(series_id, task)`` for every sealed synthetic world, in pool order."""
    return [
        (f"{family}/{index}", make_task(family, world_seed("sealed", family, index, salt)))
        for family in FAMILIES
        for index in range(spec.per_family)
    ]


def _create_salt(salt_file: Path) -> None:
    """Write a fresh salt exclusively (O_CREAT | O_EXCL, mode 0o600), then fsync it."""
    fd = os.open(salt_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, secrets.token_bytes(SALT_BYTES))
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_salt(salt_file: Path) -> bytes:
    """Read the salt file, requiring exactly ``SALT_BYTES`` bytes."""
    try:
        salt = salt_file.read_bytes()
    except OSError as exc:
        raise ArenaError(f"{salt_file}: cannot be read: {exc}") from exc
    if len(salt) != SALT_BYTES:
        raise ArenaError(
            f"{salt_file}: the salt must be exactly {SALT_BYTES} bytes, got {len(salt)}"
        )
    return salt


def _load_salt(data_root: Path, spec: SyntheticSealedManifest) -> bytes:
    """Read the salt and check it against ``spec``; the bytes stay inside this package."""
    salt_file = salt_path(data_root)
    if not salt_file.exists():
        raise ArenaError(f"{salt_file} is missing: restore it from backup and never regenerate it")
    salt = _read_salt(salt_file)
    if sha256_hex(salt) != spec.salt_sha256:
        raise ArenaError(f"{salt_file} does not match the hash pinned by the synthetic manifest")
    return salt


def _manifest_bytes(digest: str, per_family: int) -> bytes:
    """Return the canonical manifest bytes pinning the salt ``digest`` and ``per_family``."""
    return manifest_text(
        {
            "manifest_version": SYNTHETIC_MANIFEST_VERSION,
            "pool": "synthetic-sealed",
            "generator_version": GENERATOR_VERSION,
            "salt_sha256": digest,
            "salt_bytes": SALT_BYTES,
            "per_family": per_family,
        }
    ).encode("utf-8")


def _check_per_family(per_family: int) -> int:
    """Return ``per_family`` as an int >= 1 (bools rejected), else raise ArenaError."""
    if isinstance(per_family, bool) or not isinstance(per_family, int):
        raise ArenaError(f"per_family must be an int >= 1, got {per_family!r}")
    if per_family < 1:
        raise ArenaError(f"per_family must be >= 1, got {per_family}")
    return per_family
