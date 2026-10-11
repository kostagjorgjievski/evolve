"""Raw-data manifests: JSON written to ``.yaml`` files.

JSON is a subset of YAML 1.2, so the manifests stay valid YAML without a
YAML dependency. Every write is atomic: a temporary file in the same
directory, flushed and ``fsync``ed, then moved into place with
``os.replace``, so no reader ever sees a partial file.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path

from evolvets.arena.realdata.sources import Source

MANIFEST_VERSION = "1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """A manifest is malformed or does not match its source."""


@dataclass(frozen=True)
class RawManifest:
    """A pinned raw data file: content hash, size and provenance."""

    dataset: str
    url: str
    filename: str
    license: str
    sha256: str  # 64 lowercase hex chars
    size_bytes: int  # > 0
    downloaded_on: str  # UTC date, "YYYY-MM-DD"
    manifest_version: str = MANIFEST_VERSION

    def __post_init__(self) -> None:
        for name in (
            "dataset",
            "url",
            "filename",
            "license",
            "sha256",
            "downloaded_on",
            "manifest_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ManifestError(f"{name} must be a non-empty string, got {value!r}")
        if _SHA256_RE.fullmatch(self.sha256) is None:
            raise ManifestError(f"sha256 must be 64 lowercase hex chars, got {self.sha256!r}")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise ManifestError(f"size_bytes must be an int, got {self.size_bytes!r}")
        if self.size_bytes <= 0:
            raise ManifestError(f"size_bytes must be > 0, got {self.size_bytes}")
        if len(self.downloaded_on) != 10:
            raise ManifestError(f"downloaded_on must be YYYY-MM-DD, got {self.downloaded_on!r}")
        try:
            date.fromisoformat(self.downloaded_on)
        except ValueError as exc:
            raise ManifestError(
                f"downloaded_on is not a valid date: {self.downloaded_on!r}"
            ) from exc
        if self.manifest_version != MANIFEST_VERSION:
            raise ManifestError(
                f"manifest_version must be {MANIFEST_VERSION!r}, got {self.manifest_version!r}"
            )


def raw_manifest_path(manifest_dir: Path, dataset: str) -> Path:
    """Return ``<manifest_dir>/<dataset>.yaml``."""
    return manifest_dir / f"{dataset}.yaml"


def manifest_text(obj: dict[str, object]) -> str:
    """Return the canonical JSON text (with trailing newline) for a manifest."""
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def write_raw_manifest(path: Path, manifest: RawManifest) -> None:
    """Write ``manifest`` as JSON (in a ``.yaml`` file), atomically."""
    text = manifest_text(asdict(manifest))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_raw_manifest(path: Path, source: Source) -> RawManifest:
    """Read a raw manifest and check it against ``source``.

    Raises :class:`ManifestError` if the file is not a JSON object with
    exactly the manifest keys, if any value is invalid, or if the dataset,
    url, filename or license differ from ``source`` (a registry edit that
    was not accompanied by a new manifest).
    """
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ManifestError(f"{path}: not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ManifestError(f"{path}: manifest must be a JSON object, got {type(obj).__name__}")
    expected = {field.name for field in fields(RawManifest)}
    if set(obj) != expected:
        raise ManifestError(
            f"{path}: manifest keys must be exactly {sorted(expected)}, got {sorted(obj)}"
        )
    try:
        manifest = RawManifest(**obj)
    except ManifestError as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    if (
        manifest.dataset,
        manifest.url,
        manifest.filename,
        manifest.license,
    ) != (source.name, source.url, source.filename, source.license):
        raise ManifestError(f"{path}: manifest is for a different source")
    return manifest
