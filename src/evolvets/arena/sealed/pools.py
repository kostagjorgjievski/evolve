"""The sealed synthetic pool: its salt, its manifest and its inputs.

Everything here runs in trusted, human-started processes (the ``evolvets
sealed`` CLI, and from PR C on the sealed server). The synthetic worlds are
drawn from a salt only a human creates (:func:`init_salt`); the salt bytes
never appear in an exception message, a public return value or a log. The
real sealed datasets, the sealed caches and the sealed pools land in the
next PR of this stack.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

from evolvets.arena.errors import ArenaError
from evolvets.arena.realdata.manifest import manifest_text, write_atomic
from evolvets.arena.synthetic import GENERATOR_VERSION
from evolvets.arena.synthetic.grammar import FAMILIES, SyntheticTask, make_task
from evolvets.arena.synthetic.seeds import world_seed
from evolvets.cache.interfaces import SeriesInput
from evolvets.ledger.canonical import sha256_hex

SALT_BYTES = 32
SYNTHETIC_MANIFEST_VERSION = "1"

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

    ``synthetic-sealed`` derives its worlds from the salt; the real sealed
    datasets join in the next PR of this stack. ``horizon`` is part of the
    sealed-inputs interface and is used by the real datasets.
    """
    if dataset != "synthetic-sealed":
        raise ArenaError(f"unknown dataset {dataset!r}; expected 'synthetic-sealed'")
    source, series, _tasks, _spec = _synthetic_sealed_inputs(data_root, manifest_dir)
    return source, series


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
