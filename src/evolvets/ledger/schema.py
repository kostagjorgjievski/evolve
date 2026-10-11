"""The record envelope: one hash-chained line per ledger record.

Every line in ``<stream>.jsonl`` is the canonical JSON of one :class:`Record`.
``record_hash`` is the SHA-256 of the canonical JSON of all other fields, and
``prev_hash`` links each record to its predecessor, so any edit, deletion,
reordering or forged append is detectable by :meth:`LedgerReader.verify`.
"""

from __future__ import annotations

import dataclasses
import math
import re
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from evolvets.ledger.canonical import canonical_bytes, sha256_hex

if TYPE_CHECKING:  # pragma: no cover - import cycle avoided at runtime
    from evolvets.ledger.payloads import Payload

SCHEMA_VERSION = "1.0"
ZERO_HASH = "0" * 64
STREAMS = ("main", "sealed")
ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "record_id",
        "stream",
        "seq",
        "prev_hash",
        "record_hash",
        "created_at",
        "run_id",
        "kind",
        "refers_to",
        "budget",
        "payload",
    }
)

_BUDGET_FIELDS = (
    "llm_input_tokens",
    "llm_output_tokens",
    "wall_seconds",
    "gpu_seconds",
    "evaluations",
)
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_CREATED_AT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


class RecordError(ValueError):
    """Raised when a record or budget violates the schema."""


@dataclasses.dataclass(frozen=True)
class Budget:
    """Compute cost charged by one record; all fields non-negative."""

    llm_input_tokens: float = 0
    llm_output_tokens: float = 0
    wall_seconds: float = 0
    gpu_seconds: float = 0
    evaluations: float = 0

    def __post_init__(self) -> None:
        for name in _BUDGET_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RecordError(f"budget.{name} must be a number, got {type(value).__name__}")
            if not math.isfinite(value):
                raise RecordError(f"budget.{name} must be finite")
            if value < 0:
                raise RecordError(f"budget.{name} must be non-negative")


@dataclasses.dataclass(frozen=True)
class Record:
    """One append-only ledger line; see the module docstring for the chain."""

    schema_version: str
    record_id: str
    stream: str
    seq: int
    prev_hash: str
    record_hash: str
    created_at: str
    run_id: str
    kind: str
    refers_to: list[str]
    budget: Budget
    payload: Payload

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise RecordError(f"unsupported schema_version {self.schema_version!r}")
        if self.stream not in STREAMS:
            raise RecordError(f"stream must be one of {STREAMS}, got {self.stream!r}")
        if isinstance(self.seq, bool) or not isinstance(self.seq, int) or self.seq < 0:
            raise RecordError(f"seq must be a non-negative int, got {self.seq!r}")
        for name in ("prev_hash", "record_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _HEX64_RE.match(value):
                raise RecordError(f"{name} must be 64 lowercase hex chars")
        if self.seq == 0 and self.prev_hash != ZERO_HASH:
            raise RecordError("the first record of a stream must use prev_hash = 64 zeros")
        if not isinstance(self.created_at, str) or not _CREATED_AT_RE.match(self.created_at):
            raise RecordError(f"created_at must be UTC ISO-8601 with Z, got {self.created_at!r}")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise RecordError("run_id must be a non-empty string")
        if not isinstance(self.kind, str) or not self.kind:
            raise RecordError("kind must be a non-empty string")
        if not isinstance(self.refers_to, list):
            raise RecordError("refers_to must be a list of record ids")
        for ref in self.refers_to:
            _require_record_id(ref)
        if not isinstance(self.budget, Budget):
            raise RecordError("budget must be a Budget instance")
        if not dataclasses.is_dataclass(self.payload) or isinstance(self.payload, type):
            raise RecordError("payload must be a payload dataclass instance")
        pool = getattr(self.payload, "pool", None)
        if pool is not None and pool != "dev" and self.stream != "sealed":
            raise RecordError(f"{self.kind} with pool {pool!r} is valid only in the sealed stream")

    def to_dict(self) -> dict[str, Any]:
        """Return the full record as a JSON-ready dict, payload and budget included."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Record:
        """Rebuild a record from its dict form, validating every field."""
        unknown = set(data) - ENVELOPE_FIELDS
        if unknown:
            raise RecordError(f"unknown envelope fields: {sorted(unknown)}")
        missing = ENVELOPE_FIELDS - set(data)
        if missing:
            raise RecordError(f"missing envelope fields: {sorted(missing)}")
        from evolvets.ledger.payloads import payload_from_dict  # avoids import cycle

        payload_data = data["payload"]
        if not isinstance(payload_data, dict):
            raise RecordError("payload must be a JSON object")
        budget_data = data["budget"]
        if not isinstance(budget_data, dict):
            raise RecordError("budget must be a JSON object")
        return cls(
            schema_version=data["schema_version"],
            record_id=data["record_id"],
            stream=data["stream"],
            seq=data["seq"],
            prev_hash=data["prev_hash"],
            record_hash=data["record_hash"],
            created_at=data["created_at"],
            run_id=data["run_id"],
            kind=data["kind"],
            refers_to=list(data["refers_to"]),
            budget=Budget(**budget_data),
            payload=payload_from_dict(str(data["kind"]), payload_data),
        )


def _require_record_id(value: object) -> None:
    """Check that ``value`` is a UUID string, as all record ids are."""
    if not isinstance(value, str):
        raise RecordError(f"record id must be a string, got {type(value).__name__}")
    try:
        uuid.UUID(value)
    except ValueError as exc:
        raise RecordError(f"record id {value!r} is not a UUID") from exc


def _utc_now() -> str:
    """Current UTC time in ISO-8601 with ``Z`` and microseconds."""
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def make_record(
    *,
    stream: str,
    seq: int,
    prev_hash: str,
    run_id: str,
    kind: str,
    payload: Payload,
    refers_to: Sequence[str] = (),
    budget: Budget | None = None,
    created_at: str | None = None,
    record_id: str | None = None,
) -> Record:
    """Build a record and compute its ``record_hash`` over the canonical form."""
    budget = budget if budget is not None else Budget()
    created = created_at if created_at is not None else _utc_now()
    ident = record_id if record_id is not None else str(uuid.uuid4())
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_id": ident,
        "stream": stream,
        "seq": seq,
        "prev_hash": prev_hash,
        "created_at": created,
        "run_id": run_id,
        "kind": kind,
        "refers_to": list(refers_to),
        "budget": dataclasses.asdict(budget),
        "payload": dataclasses.asdict(payload),
    }
    record_hash = sha256_hex(canonical_bytes(body))
    return Record(
        schema_version=SCHEMA_VERSION,
        record_id=ident,
        stream=stream,
        seq=seq,
        prev_hash=prev_hash,
        record_hash=record_hash,
        created_at=created,
        run_id=run_id,
        kind=kind,
        refers_to=list(refers_to),
        budget=budget,
        payload=payload,
    )
