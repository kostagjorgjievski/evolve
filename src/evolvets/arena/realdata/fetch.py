"""Download and verify the pinned raw data files.

An existing target file is never deleted or overwritten: if it does not
match its manifest, the user resolves it by hand. Downloads go to
``<target>.part`` and are moved into place with ``os.replace`` only after
the content hash has been checked, so a crash can never leave a half
written file under the pinned name.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from datetime import UTC, date, datetime
from pathlib import Path

from evolvets.arena.realdata.manifest import (
    RawManifest,
    raw_manifest_path,
    read_raw_manifest,
    write_raw_manifest,
)
from evolvets.arena.realdata.sources import Source

_CHUNK_BYTES = 1 << 20  # 1 MiB


class FetchError(Exception):
    """A download or verification of a raw data file failed."""


def raw_path(data_root: Path, source: Source) -> Path:
    """Return ``<data_root>/raw/<name>/<filename>``."""
    return data_root / "raw" / source.name / source.filename


def file_sha256(path: Path) -> tuple[str, int]:
    """Return ``(hex digest, size)`` of ``path``, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def fetch(
    source: Source,
    data_root: Path,
    manifest_dir: Path,
    *,
    today: date | None = None,
    timeout: float = 60.0,
) -> RawManifest:
    """Fetch ``source`` into ``data_root`` and return its manifest.

    With both manifest and target present and matching, no network access
    happens. With a manifest but no target, the file is re-downloaded and
    must still match, or the source is reported as changed. With no
    manifest, the downloaded file is pinned by a newly written manifest.
    """
    manifest_file = raw_manifest_path(manifest_dir, source.name)
    target = raw_path(data_root, source)
    part = target.parent / (target.name + ".part")
    manifest: RawManifest | None = None
    if manifest_file.exists():
        manifest = read_raw_manifest(manifest_file, source)
    if target.exists():
        digest, size = file_sha256(target)
        if manifest is None:
            raise FetchError(
                f"{target} exists but no manifest pins it; remove it by hand to proceed"
            )
        if digest != manifest.sha256 or size != manifest.size_bytes:
            raise FetchError(
                f"{target} does not match its manifest: file sha256 {digest}, "
                f"manifest sha256 {manifest.sha256}"
            )
        return manifest
    target.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    size = 0
    try:
        with (
            urllib.request.urlopen(source.url, timeout=timeout) as response,
            open(part, "wb") as handle,
        ):
            while chunk := response.read(_CHUNK_BYTES):
                hasher.update(chunk)
                size += len(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise FetchError(f"downloading {source.url}: {exc}") from exc
    hexdigest = hasher.hexdigest()
    if size == 0:
        part.unlink(missing_ok=True)
        raise FetchError(f"{source.url} downloaded zero bytes")
    if manifest is not None and (hexdigest != manifest.sha256 or size != manifest.size_bytes):
        part.unlink(missing_ok=True)
        raise FetchError(
            "source content changed: downloaded sha256 "
            f"{hexdigest} ({size} bytes), manifest sha256 {manifest.sha256} "
            f"({manifest.size_bytes} bytes)"
        )
    os.replace(part, target)
    if manifest is None:
        manifest = RawManifest(
            dataset=source.name,
            url=source.url,
            filename=source.filename,
            license=source.license,
            sha256=hexdigest,
            size_bytes=size,
            downloaded_on=(today or datetime.now(UTC).date()).isoformat(),
        )
        write_raw_manifest(manifest_file, manifest)
    return manifest


def verify_raw(source: Source, data_root: Path, manifest_dir: Path) -> RawManifest:
    """Verify the target file against its manifest; no network access."""
    manifest = read_raw_manifest(raw_manifest_path(manifest_dir, source.name), source)
    target = raw_path(data_root, source)
    if not target.exists():
        raise FetchError(f"{target} is missing; run 'evolvets data fetch {source.name}'")
    digest, size = file_sha256(target)
    if digest != manifest.sha256 or size != manifest.size_bytes:
        raise FetchError(
            f"{target} does not match its manifest: file sha256 {digest}, "
            f"manifest sha256 {manifest.sha256}"
        )
    return manifest
