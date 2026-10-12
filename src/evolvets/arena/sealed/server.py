"""The sealed server process: check the ledger, replay, record, reply once.

Spawned by :func:`evolvets.arena.sealed.client.evaluate_sealed`, this is the
only code besides the trusted CLI that imports
:mod:`evolvets.arena.sealed.pools`. It checks the run's ledger before opening
any sealed file, opens the sealed stream's writer before evaluating (so a
concurrent request on the same run fails fast), replays the requested updater
over the sealed pool through one fresh worker process per series, and writes
the per-task blob and the ``baseline`` record before replying — whatever the
outcome was. A failure before the record is written appends a traceback to
``<data_root>/sealed/server.log`` and exits 2; the caller turns that into a
:class:`~evolvets.arena.sealed.client.SealedError` naming only the log.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import UTC, datetime
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, cast

from evolvets.arena import ARENA_VERSION
from evolvets.arena.dev import DevPool
from evolvets.arena.errors import ArenaError
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.arena.sealed.pools import load_sealed_pool
from evolvets.arena.sealed.remote import RemoteUpdaterFactory, SeriesTimeout
from evolvets.arena.versions import protocol_versions
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.canonical import canonical_bytes, canonical_json
from evolvets.ledger.payloads import Baseline, RunStart
from evolvets.ledger.schema import Budget
from evolvets.ledger.store import (
    LedgerCorruptError,
    LedgerLockedError,
    LedgerReader,
    LedgerWriter,
)

_REQUEST_KEYS = frozenset({"dataset", "updater_id", "updater_ref", "config", "seed"})
_PATHS_KEYS = frozenset({"data_root", "manifest_dir", "sealed_cache_root"})


def _serve(
    conn: Connection,
    request_json: bytes,
    ledger_root: Path,
    run_id: str,
    paths_json: bytes,
    timeout: float,
) -> None:
    """Serve exactly one request; anything unexpected is logged and exits 2."""
    request: dict[str, Any] | None = None
    paths: dict[str, Any] | None = None
    try:
        request = _decode(request_json, _REQUEST_KEYS)
        paths = _decode(paths_json, _PATHS_KEYS)
        _run(conn, request, paths, Path(ledger_root), run_id, timeout)
    except SystemExit:
        raise
    except BaseException:
        # A SeriesTimeout is handled inside _run; anything reaching here
        # happened before the record was written. Log it and die quietly: the
        # caller sees only the exit code and the log path, never a traceback.
        if paths is not None:
            updater_id = str(request["updater_id"]) if request is not None else "unknown"
            _append_server_log(Path(str(paths["data_root"])), run_id, updater_id)
        raise SystemExit(2) from None


def _run(
    conn: Connection,
    request: dict[str, Any],
    paths: dict[str, Any],
    ledger_root: Path,
    run_id: str,
    timeout: float,
) -> None:
    """Refuse, or open the sealed writer first and only then evaluate and record."""
    reason = _ledger_refusal(ledger_root, run_id)
    if reason is not None:
        _reply(conn, {"refused": reason})
        return
    try:
        writer = LedgerWriter(ledger_root, run_id, "sealed")
    except LedgerLockedError:
        _reply(conn, {"refused": "sealed stream is locked"})
        return
    try:
        _evaluate_and_record(conn, writer, request, paths, ledger_root, run_id, timeout)
    finally:
        writer.close()


def _ledger_refusal(ledger_root: Path, run_id: str) -> str | None:
    """Why this run cannot take a sealed evaluation, or ``None`` when it can.

    These checks read the main stream only, so no sealed file is opened before
    the run is proven ready to record the result.
    """
    try:
        records = list(LedgerReader(ledger_root, run_id).records(["main"]))
    except LedgerCorruptError:
        return "the main stream does not read cleanly"
    if not records:
        return "the main stream is empty; a sealed evaluation needs a run_start first"
    first = records[0]
    if first.kind != "run_start" or not isinstance(first.payload, RunStart):
        return "the first main record is not a run_start"
    if first.payload.protocol_versions != protocol_versions():
        return "the run_start pins different protocol versions than this code"
    if any(record.kind == "run_end" for record in records):
        return "the main stream already ends with a run_end"
    return None


def _evaluate_and_record(
    conn: Connection,
    writer: LedgerWriter,
    request: dict[str, Any],
    paths: dict[str, Any],
    ledger_root: Path,
    run_id: str,
    timeout: float,
) -> None:
    """Load the sealed pool, replay the updater through fresh workers, record, reply."""
    pool = load_sealed_pool(
        str(request["dataset"]),
        Path(str(paths["data_root"])),
        Path(str(paths["manifest_dir"])),
        Path(str(paths["sealed_cache_root"])),
    )
    factory = RemoteUpdaterFactory(
        [series.series_id for series in pool.series], str(request["updater_ref"]), timeout
    )
    started = time.monotonic()
    try:
        status, error, metrics, result_obj = _replay(factory, request, pool, timeout)
    except Exception as exc:
        # Not an UpdaterFailure (replay_dataset turns those into results) and
        # not a SeriesTimeout (_replay turns that into a timeout): the server
        # itself failed mid-run. A started sealed run is always in the ledger,
        # so record it, log the traceback and reply normally.
        _append_server_log(Path(str(paths["data_root"])), run_id, str(request["updater_id"]))
        status = "failed"
        error = f"sealed server error: {type(exc).__name__}"
        metrics = {}
        result_obj = None
    finally:
        factory.shutdown()
        elapsed = time.monotonic() - started
    per_task_blob: str | None = None
    if result_obj is not None:
        blob_obj = {
            "updater_ref": request["updater_ref"],
            "series_timeout_seconds": timeout,
            "result": result_obj,
        }
        per_task_blob = BlobStore(ledger_root / run_id / "blobs").put(canonical_bytes(blob_obj))
    record = writer.append(
        "baseline",
        Baseline(
            updater_id=str(request["updater_id"]),
            hyperparams=dict(request["config"]),
            pool="sealed",
            status=status,
            metrics=metrics,
            per_task_blob=per_task_blob,
            error=error,
        ),
        budget=Budget(wall_seconds=elapsed, evaluations=1),
    )
    _reply(conn, {"record_id": record.record_id, "status": status})


def _replay(
    factory: RemoteUpdaterFactory,
    request: dict[str, Any],
    pool: DevPool,
    timeout: float,
) -> tuple[str, str | None, dict[str, float], dict[str, object] | None]:
    """Replay one sealed evaluation into ``(status, error, metrics, result_obj)``.

    A :class:`SeriesTimeout` — a ``BaseException`` — becomes the timeout
    outcome: ``result_to_obj``'s key set with ``status = "timeout"``, the
    failing series and step, and no metrics or per-series scores.
    """
    try:
        result = replay_dataset(factory, request["config"], request["seed"], pool)
    except SeriesTimeout as exc:
        error = f"series {exc.series_id} exceeded {timeout}s at step {exc.step}"
        result_obj: dict[str, object] = {
            "dataset": pool.dataset,
            "cache_id": pool.cache_id,
            "arena_version": ARENA_VERSION,
            "status": "timeout",
            "error": error,
            "failed_series": exc.series_id,
            "failed_step": exc.step,
            "metrics": {},
            "series": [],
        }
        return "timeout", error, {}, result_obj
    return result.status, result.error, dict(result.metrics), result_to_obj(result)


def _reply(conn: Connection, message: dict[str, object]) -> None:
    """Send the one reply as canonical JSON bytes and close the pipe."""
    conn.send_bytes(canonical_json(message).encode("utf-8"))
    conn.close()


def _append_server_log(data_root: Path, run_id: str, updater_id: str) -> None:
    """Append a UTC timestamp, the run, the updater and the traceback to the log.

    Best effort: a failure to log must not mask the exit it accompanies.
    """
    entry = (
        datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
        + f" run_id={run_id} updater_id={updater_id}\n"
        + traceback.format_exc()
    )
    try:
        log_dir = data_root / "sealed"
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "server.log").open("a", encoding="utf-8") as handle:
            handle.write(entry)
    except OSError:
        pass


def _decode(raw: bytes, keys: frozenset[str]) -> dict[str, Any]:
    """Decode one spawn argument (the trusted direction) and check its shape."""
    obj: object = json.loads(raw.decode("utf-8"))
    if not isinstance(obj, dict) or set(obj) != keys:
        raise ArenaError(f"malformed sealed server argument; expected exactly {sorted(keys)}")
    return cast("dict[str, Any]", obj)
