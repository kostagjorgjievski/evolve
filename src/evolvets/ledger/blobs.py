"""Content-addressed blob store for large ledger payload content.

Code, arrays, traces and prompts live as blobs; payloads hold only small
fields and ``sha256:<64 hex>`` references. Content is written atomically
(temperature file, fsync, rename) under ``blobs/<first 2 hex>/<hash>`` and
never modified or removed afterwards. ``get`` re-hashes the content and
raises on any mismatch.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

BLOB_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class BlobError(Exception):
    """A blob reference is malformed, missing, or does not hash to itself."""


def is_blob_ref(value: object) -> bool:
    """Whether ``value`` has the ``sha256:<64 hex>`` blob-reference format."""
    return isinstance(value, str) and BLOB_RE.match(value) is not None


def find_blob_refs(obj: Any) -> Iterator[str]:
    """Yield every blob reference found anywhere inside a payload dict."""
    if is_blob_ref(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from find_blob_refs(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from find_blob_refs(item)


class BlobStore:
    """Stores immutable content keyed by its SHA-256 hash."""

    def __init__(self, blobs_dir: str | Path) -> None:
        self._dir = Path(blobs_dir)

    def put(self, data: bytes) -> str:
        """Store ``data`` atomically and return its ``sha256:...`` reference.

        Writing content that already exists is a no-op that returns the same
        reference.
        """
        digest = hashlib.sha256(data).hexdigest()
        ref = f"sha256:{digest}"
        path = self._dir / digest[:2] / digest
        if path.exists():
            return ref
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_name(f"{digest}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
        try:
            with temp.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()
        return ref

    def get(self, ref: str) -> bytes:
        """Read the blob for ``ref``, re-hashing it to prove integrity."""
        if not is_blob_ref(ref):
            raise BlobError(f"malformed blob reference {ref!r}")
        digest = ref[len("sha256:") :]
        path = self._dir / digest[:2] / digest
        try:
            data = path.read_bytes()
        except FileNotFoundError as exc:
            raise BlobError(f"missing blob {ref}") from exc
        if hashlib.sha256(data).hexdigest() != digest:
            raise BlobError(f"blob {ref} does not hash to its reference")
        return data
