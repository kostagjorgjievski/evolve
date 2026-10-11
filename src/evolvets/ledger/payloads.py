"""Kind-specific payload dataclasses and their validation.

Every payload field is required: unknown or missing fields are rejected, and
non-finite floats are rejected anywhere inside a payload, at construction
time rather than at write time. TASK_002 lands the kinds in two steps: this
module ships the framework plus ``run_start`` and ``run_end``; the remaining
kinds (proposal, prediction, allocation, evaluation, acceptance, baseline,
commitment, correction) follow in ``feat/ledger-payloads``.
"""

import dataclasses
import math
import re
from typing import Any

from evolvets.ledger.schema import Budget

PROTOCOL_KEYS = frozenset({"ledger", "arena", "generator", "cache_manifest"})
RUN_END_STATUSES = frozenset({"completed", "aborted", "budget_exhausted"})

_BLOB_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class PayloadError(ValueError):
    """Raised when a payload does not match its kind's schema."""


def _require_str(name: str, value: Any) -> None:
    """Check that ``value`` is a str."""
    if not isinstance(value, str):
        raise PayloadError(f"{name} must be a string, got {type(value).__name__}")


def _require_nonempty_str(name: str, value: Any) -> None:
    """Check that ``value`` is a non-empty str."""
    _require_str(name, value)
    if not value:
        raise PayloadError(f"{name} must not be empty")


def _require_blob_ref(name: str, value: Any) -> None:
    """Check that ``value`` is a ``sha256:<64 hex>`` blob reference."""
    if not isinstance(value, str) or not _BLOB_RE.match(value):
        raise PayloadError(f"{name} must be a blob ref 'sha256:<64 hex>', got {value!r}")


def _require_nullable_blob_ref(name: str, value: Any) -> None:
    """Check a blob reference that may be null until blob support lands."""
    if value is not None:
        _require_blob_ref(name, value)


def _require_literal(name: str, value: Any, options: frozenset[str]) -> None:
    """Check that ``value`` is one of ``options``."""
    if value not in options:
        raise PayloadError(f"{name} must be one of {sorted(options)}, got {value!r}")


def _require_int(name: str, value: Any) -> None:
    """Check that ``value`` is an int (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise PayloadError(f"{name} must be an int, got {value!r}")


def _require_exact_keys(name: str, value: Any, keys: frozenset[str]) -> None:
    """Check that ``value`` is a dict with exactly ``keys``."""
    if not isinstance(value, dict) or set(value) != set(keys):
        raise PayloadError(f"{name} must be a dict with exactly the keys {sorted(keys)}")


def _require_str_keys(name: str, value: Any) -> None:
    """Check that ``value`` is a dict whose keys are strings."""
    if not isinstance(value, dict):
        raise PayloadError(f"{name} must be a dict, got {type(value).__name__}")
    for key in value:
        if not isinstance(key, str):
            raise PayloadError(f"{name} keys must be strings, got {key!r}")


def _check_finite(name: str, obj: Any) -> None:
    """Reject NaN and infinities anywhere inside a JSON-like structure."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise PayloadError(f"{name} must not contain non-finite floats")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _check_finite(f"{name}.{key}", value)
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _check_finite(f"{name}[{i}]", item)


@dataclasses.dataclass(frozen=True)
class RunStart:
    """First record of a run: condition, config and protocol versions."""

    condition: str
    seed: int
    config_blob: str
    git_commit: str
    protocol_versions: dict[str, str]
    budget_limits: Budget

    def __post_init__(self) -> None:
        _require_nonempty_str("condition", self.condition)
        _require_int("seed", self.seed)
        _require_blob_ref("config_blob", self.config_blob)
        _require_nonempty_str("git_commit", self.git_commit)
        _require_exact_keys("protocol_versions", self.protocol_versions, PROTOCOL_KEYS)
        for key, value in self.protocol_versions.items():
            _require_str(f"protocol_versions[{key}]", value)
        if not isinstance(self.budget_limits, Budget):
            raise PayloadError("budget_limits must be a Budget instance")
        _check_finite("run_start", dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class RunEnd:
    """Last record of a run: how it ended and a free-form summary."""

    status: str
    summary: dict[str, Any]

    def __post_init__(self) -> None:
        _require_literal("status", self.status, RUN_END_STATUSES)
        _require_str_keys("summary", self.summary)
        _check_finite("run_end", dataclasses.asdict(self))


Payload = RunStart | RunEnd
PayloadClass = type[RunStart] | type[RunEnd]

PAYLOAD_KINDS: dict[str, PayloadClass] = {
    "run_start": RunStart,
    "run_end": RunEnd,
}


def validate_kind_payload(kind: str, payload: object) -> None:
    """Check that ``payload`` is the dataclass registered for ``kind``."""
    cls = PAYLOAD_KINDS.get(kind)
    if cls is None:
        raise PayloadError(f"unknown kind {kind!r}")
    if not isinstance(payload, cls):
        raise PayloadError(f"payload for kind {kind!r} must be {cls.__name__}")


def payload_from_dict(kind: str, data: dict[str, Any]) -> Payload:
    """Rebuild the payload for ``kind``, rejecting unknown or missing fields."""
    cls = PAYLOAD_KINDS.get(kind)
    if cls is None:
        raise PayloadError(f"unknown kind {kind!r}")
    names = {field.name for field in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise PayloadError(f"unknown fields for kind {kind!r}: {sorted(unknown)}")
    missing = names - set(data)
    if missing:
        raise PayloadError(f"missing fields for kind {kind!r}: {sorted(missing)}")
    kwargs = dict(data)
    for field in dataclasses.fields(cls):
        if field.type is Budget and isinstance(kwargs[field.name], dict):
            kwargs[field.name] = Budget(**kwargs[field.name])
    return cls(**kwargs)


def payload_to_dict(payload: Payload) -> dict[str, Any]:
    """Return the JSON-ready dict form of ``payload``."""
    return dataclasses.asdict(payload)
