"""Canonical JSON serialization and hashing for ledger records.

The same record must always serialize to the same bytes, so hashes are
reproducible across processes and Python versions.
"""

import hashlib
import json


def canonical_json(obj: object) -> str:
    """Serialize ``obj`` to canonical JSON: sorted keys, no extra whitespace."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def canonical_bytes(obj: object) -> bytes:
    """Canonical JSON encoded as UTF-8 bytes."""
    return canonical_json(obj).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data).hexdigest()
