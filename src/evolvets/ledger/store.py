"""Append-only ledger store: one writer per stream, verifiable hash chains.

The writer acquires an exclusive lock file per stream, so two processes never
append to the same ``.jsonl``. Every append is flushed and ``fsync`` ed before
returning, so a record that was acknowledged survives a later crash. There is
deliberately no API to modify or remove records.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, cast

from evolvets.ledger.canonical import canonical_bytes, canonical_json, sha256_hex
from evolvets.ledger.payloads import Payload, PayloadError, validate_kind_payload
from evolvets.ledger.schema import (
    STREAMS,
    ZERO_HASH,
    Budget,
    Record,
    RecordError,
    make_record,
)


class LedgerError(Exception):
    """Base class for ledger store errors."""


class LedgerLockedError(LedgerError):
    """Another writer already holds the lock for this stream."""


class LedgerCorruptError(LedgerError):
    """The existing stream file fails verification; refusing to append."""


@dataclasses.dataclass(frozen=True)
class VerifyResult:
    """Outcome of a full-chain verification of one run's ledger."""

    ok: bool
    stream: str | None
    seq: int | None
    reason: str
    records: int


def _scan_stream(path: Path) -> tuple[list[Record], str | None, int]:
    """Parse and verify one stream file.

    Returns the records read, a failure reason (``None`` when intact) and the
    0-based position of the first problem, so callers can report it.
    """
    records: list[Record] = []
    if not path.exists():
        return records, None, 0
    prev_hash = ZERO_HASH
    with path.open(encoding="utf-8") as stream_file:
        for position, raw_line in enumerate(stream_file):
            line = raw_line.rstrip("\n")
            if not line:
                return records, f"empty line at position {position}", position
            try:
                parsed: Any = json.loads(line)
            except json.JSONDecodeError as exc:
                return records, f"invalid JSON at position {position}: {exc}", position
            if not isinstance(parsed, dict):
                return records, f"line {position} is not a JSON object", position
            stored_hash = parsed.get("record_hash")
            body = {key: value for key, value in parsed.items() if key != "record_hash"}
            if not isinstance(stored_hash, str) or sha256_hex(canonical_bytes(body)) != stored_hash:
                return records, f"record_hash mismatch at position {position}", position
            try:
                record = Record.from_dict(cast(dict[str, Any], parsed))
            except (RecordError, PayloadError) as exc:
                return records, f"invalid record at position {position}: {exc}", position
            if record.seq != position or record.prev_hash != prev_hash:
                return records, f"chain break at position {position}", position
            prev_hash = record.record_hash
            records.append(record)
    return records, None, len(records)


class LedgerWriter:
    """Appends records to one stream of one run's ledger."""

    def __init__(self, root: str | Path, run_id: str, stream: str) -> None:
        if stream not in STREAMS:
            raise ValueError(f"stream must be one of {STREAMS}, got {stream!r}")
        self._run_id = run_id
        self._stream = stream
        self._dir = Path(root) / run_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock_path = self._dir / f"{stream}.lock"
        try:
            self._lock_fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise LedgerLockedError(
                f"another writer holds {self._lock_path}; close it before reopening"
            ) from exc
        try:
            self._path = self._dir / f"{stream}.jsonl"
            records, problem, position = _scan_stream(self._path)
            if problem is not None:
                raise LedgerCorruptError(f"{self._path}: {problem} (position {position})")
            self._next_seq = len(records)
            self._prev_hash = records[-1].record_hash if records else ZERO_HASH
            self._file = self._path.open("a", encoding="utf-8")
            self._closed = False
        except Exception:
            self._release_lock()
            raise

    @property
    def stream(self) -> str:
        """The stream this writer appends to."""
        return self._stream

    def append(
        self,
        kind: str,
        payload: Payload,
        refers_to: Sequence[str] = (),
        budget: Budget | None = None,
    ) -> Record:
        """Validate, hash, write, flush and fsync one record; return it."""
        if self._closed:
            raise LedgerError("writer is closed")
        validate_kind_payload(kind, payload)
        record = make_record(
            stream=self._stream,
            seq=self._next_seq,
            prev_hash=self._prev_hash,
            run_id=self._run_id,
            kind=kind,
            payload=payload,
            refers_to=refers_to,
            budget=budget,
        )
        self._file.write(canonical_json(record.to_dict()) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._next_seq += 1
        self._prev_hash = record.record_hash
        return record

    def close(self) -> None:
        """Close the stream file and release the lock; safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._file.close()
        self._release_lock()

    def _release_lock(self) -> None:
        os.close(self._lock_fd)
        self._lock_path.unlink()

    def __enter__(self) -> LedgerWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class LedgerReader:
    """Reads and verifies the records of one run's ledger."""

    def __init__(self, root: str | Path, run_id: str) -> None:
        self._dir = Path(root) / run_id

    def records(self, streams: Sequence[str] = STREAMS) -> Iterator[Record]:
        """Iterate records of the given streams, in stream order."""
        for stream in streams:
            if stream not in STREAMS:
                raise ValueError(f"stream must be one of {STREAMS}, got {stream!r}")
            read, problem, _ = _scan_stream(self._dir / f"{stream}.jsonl")
            if problem is not None:
                raise LedgerCorruptError(f"{self._dir / (stream + '.jsonl')}: {problem}")
            yield from read

    def verify(self) -> VerifyResult:
        """Check hashes, chain links and seq continuity; report the first break."""
        total = 0
        for stream in STREAMS:
            read, problem, position = _scan_stream(self._dir / f"{stream}.jsonl")
            if problem is not None:
                return VerifyResult(False, stream, position, problem, total)
            total += len(read)
        return VerifyResult(True, None, None, "ok", total)
