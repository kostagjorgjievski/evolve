"""Append-only ledger store: one writer per stream, verifiable hash chains.

The writer acquires an exclusive lock file per stream, so two processes never
append to the same ``.jsonl``. The lock records the owning pid, host and
creation time; a lock whose pid is dead on this host can be broken explicitly
with :meth:`LedgerWriter.break_stale_lock` — never automatically. Every append
is flushed and ``fsync`` ed before returning, so a record that was
acknowledged survives a later crash. There is deliberately no API to modify
or remove records.
"""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import time
from collections.abc import Iterator, Sequence
from datetime import UTC
from pathlib import Path
from typing import Any, cast

from evolvets.ledger.blobs import BlobError, BlobStore, find_blob_refs
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

# An empty lock file this old means its writer died before writing metadata.
STALE_EMPTY_LOCK_SECONDS = 600


class LedgerError(Exception):
    """Base class for ledger store errors."""


class LedgerLockedError(LedgerError):
    """Another writer already holds the lock for this stream."""


class LedgerCorruptError(LedgerError):
    """The existing stream file fails verification; refusing to append."""


def _write_lock_metadata(fd: int) -> None:
    """Record the owning pid, host and creation time inside the lock file."""
    from datetime import datetime

    metadata = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    os.write(fd, json.dumps(metadata).encode("utf-8"))


def _pid_alive(pid: object) -> bool:
    """Whether a pid exists on this host (unparseable pids count as dead)."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stale_reason(path: Path) -> str | None:
    """Why this lock is stale, or ``None`` when live, foreign or unreadable.

    An empty lock file means its writer died between creating the lock and
    writing the metadata; it counts as stale once it is older than
    ``STALE_EMPTY_LOCK_SECONDS`` (the writer fills in the metadata within
    microseconds, so a fresh empty lock is treated as live).
    """
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not content.strip():
        age = time.time() - path.stat().st_mtime
        if age < STALE_EMPTY_LOCK_SECONDS:
            return None
        return (
            f"stale lock: empty lock file, age {age:.0f}s >= "
            f"{STALE_EMPTY_LOCK_SECONDS}s (writer died before writing metadata); "
            f"recover with LedgerWriter.break_stale_lock"
        )
    try:
        metadata: Any = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(metadata, dict):
        return None
    host = metadata.get("host")
    pid = metadata.get("pid")
    if host != socket.gethostname() or _pid_alive(pid):
        return None
    return (
        f"stale lock: pid {pid} on {host} is dead "
        f"(created {metadata.get('created_at')}); recover with "
        f"LedgerWriter.break_stale_lock"
    )


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
            if not raw_line.endswith("\n"):
                return records, f"torn tail at position {position}", position
            line = raw_line[:-1]
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
            stale = _stale_reason(self._lock_path)
            message = (
                f"{self._lock_path}: {stale}"
                if stale is not None
                else f"another writer holds {self._lock_path}; close it before reopening"
            )
            raise LedgerLockedError(message) from exc
        _write_lock_metadata(self._lock_fd)
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

    @staticmethod
    def break_stale_lock(root: str | Path, run_id: str, stream: str) -> None:
        """Break a lock whose owning process is dead on this host.

        The explicit recovery path for a crashed writer: it succeeds only when
        the lock file records this host and a pid that no longer exists.
        """
        if stream not in STREAMS:
            raise ValueError(f"stream must be one of {STREAMS}, got {stream!r}")
        lock_path = Path(root) / run_id / f"{stream}.lock"
        stale = _stale_reason(lock_path)
        if stale is None:
            raise LedgerLockedError(
                f"refusing to break {lock_path}: the lock is live, foreign or unreadable"
            )
        lock_path.unlink()

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

    def verify(self, check_blobs: bool = False) -> VerifyResult:
        """Check hashes, chain links, seq continuity and blob references.

        With ``check_blobs`` every ``sha256:...`` reference in every payload
        must exist in the run's blob store and hash to itself. The first
        broken position is reported; the chain is checked before the blobs.
        """
        main_path = self._dir / "main.jsonl"
        if not main_path.exists():
            return VerifyResult(False, "main", None, f"missing ledger stream file {main_path}", 0)
        total = 0
        for stream in STREAMS:
            read, problem, position = _scan_stream(self._dir / f"{stream}.jsonl")
            if problem is not None:
                return VerifyResult(False, stream, position, problem, total)
            if check_blobs:
                blob_result = self._verify_blobs(read, stream)
                if blob_result is not None:
                    return blob_result
            total += len(read)
        return VerifyResult(True, None, None, "ok", total)

    def _verify_blobs(self, records: list[Record], stream: str) -> VerifyResult | None:
        """Check every blob reference of ``records``; ``None`` when all exist."""
        store = BlobStore(self._dir / "blobs")
        for record in records:
            for ref in find_blob_refs(dataclasses.asdict(record.payload)):
                try:
                    store.get(ref)
                except BlobError as exc:
                    return VerifyResult(False, stream, record.seq, f"blob check failed: {exc}", 0)
        return None
