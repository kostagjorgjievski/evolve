"""The caller's side of a sealed evaluation: one request in, one receipt out.

Callers (the entry-10 baseline runner, the entry-11 experiment runner, Phase 1)
import only this module and :mod:`evolvets.arena.sealed.names`. Nothing here
can read data: the dataset names come from ``names``, and the server module is
imported inside the spawned child by :func:`_server_entry`, so even loading
this module loads no sealed code path. The caller receives exactly one reply —
a receipt or a refusal — and never a traceback, an exception message or a
sealed value from the server.
"""

from __future__ import annotations

import dataclasses
import json
import math
import multiprocessing
import re
from multiprocessing.connection import Connection
from pathlib import Path

from evolvets.arena.sealed.names import SEALED_DATASETS
from evolvets.ledger.canonical import canonical_json
from evolvets.ledger.payloads import EVALUATION_STATUSES
from evolvets.updater.interfaces import ConfigValue

SERIES_TIMEOUT_SECONDS = 600.0

_REPLY_MAX_BYTES = 4096
_UPDATER_REF_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")
_SEED_LIMIT = 2**63


@dataclasses.dataclass(frozen=True)
class SealedRequest:
    """One sealed evaluation: which updater to replay over which sealed pool."""

    dataset: str  # one of SEALED_DATASETS
    updater_id: str  # non-empty; recorded as Baseline.updater_id
    updater_ref: str  # "package.module:attr"; attr is a zero-argument factory returning an Updater
    config: dict[str, ConfigValue]  # recorded as Baseline.hyperparams
    seed: int


@dataclasses.dataclass(frozen=True)
class SealedPaths:
    """Where the sealed data, the manifests and the sealed caches live."""

    data_root: Path
    manifest_dir: Path
    sealed_cache_root: Path


@dataclasses.dataclass(frozen=True)
class SealedReceipt:
    """All the caller learns about a sealed evaluation: which record, how it ended."""

    record_id: str  # the sealed-stream baseline record
    status: str  # one of EVALUATION_STATUSES


class SealedError(Exception):
    """The server refused the request or failed before writing a record."""


def evaluate_sealed(
    request: SealedRequest,
    *,
    ledger_root: Path,
    run_id: str,
    paths: SealedPaths,
    series_timeout_seconds: float = SERIES_TIMEOUT_SECONDS,
) -> SealedReceipt:
    """Evaluate ``request`` in a spawned sealed server and return its receipt.

    The whole evaluation runs in the server process, which writes the
    per-task blob and the ``baseline`` record to the run's sealed stream and
    is the only budget charge for it. A refusal (bad ledger preconditions, a
    locked sealed stream) or a server failure raises :class:`SealedError`
    without leaking any traceback, exception message or sealed value.
    """
    _validate(request, run_id=run_id, series_timeout_seconds=series_timeout_seconds)
    paths_json = _paths_json(paths)
    context = multiprocessing.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=True)
    process = context.Process(
        target=_server_entry,
        args=(
            child_conn,
            _request_json(request),
            Path(ledger_root),
            run_id,
            paths_json,
            float(series_timeout_seconds),
        ),
        daemon=False,  # the server must spawn one worker per series itself
    )
    process.start()
    child_conn.close()
    try:
        try:
            raw: bytes | None = parent_conn.recv_bytes(_REPLY_MAX_BYTES)
        except (EOFError, OSError):
            raw = None
    finally:
        parent_conn.close()
        process.join()
    return _receipt(raw, process.exitcode, paths)


def _server_entry(
    conn: Connection,
    request_json: bytes,
    ledger_root: Path,
    run_id: str,
    paths_json: bytes,
    timeout: float,
) -> None:
    """Spawn target: import the server inside the child, so the caller never loads it."""
    from evolvets.arena.sealed import server

    server._serve(conn, request_json, ledger_root, run_id, paths_json, timeout)


def _validate(request: SealedRequest, *, run_id: str, series_timeout_seconds: float) -> None:
    """Refuse malformed requests in the caller, spawning nothing."""
    if request.dataset not in SEALED_DATASETS:
        raise SealedError(
            f"dataset must be one of {list(SEALED_DATASETS)}, got {request.dataset!r}"
        )
    if not isinstance(request.updater_id, str) or not request.updater_id:
        raise SealedError(f"updater_id must be a non-empty str, got {request.updater_id!r}")
    if not isinstance(request.updater_ref, str) or not _UPDATER_REF_RE.match(request.updater_ref):
        raise SealedError(f"updater_ref must be 'package.module:attr', got {request.updater_ref!r}")
    for key, value in request.config.items():
        if not isinstance(key, str):
            raise SealedError(f"config keys must be strings, got {key!r}")
        if type(value) not in (bool, int, str, float):
            raise SealedError(f"config[{key!r}] must be a bool, int, str or float, got {value!r}")
        if isinstance(value, float) and not math.isfinite(value):
            raise SealedError(f"config[{key!r}] must be finite, got {value!r}")
    if isinstance(request.seed, bool) or not isinstance(request.seed, int):
        raise SealedError(f"seed must be an int, got {request.seed!r}")
    if request.seed < 0 or request.seed >= _SEED_LIMIT:
        raise SealedError(f"seed must be in [0, {_SEED_LIMIT}), got {request.seed}")
    if isinstance(series_timeout_seconds, bool) or not isinstance(
        series_timeout_seconds, (int, float)
    ):
        raise SealedError(
            f"series_timeout_seconds must be a number, got {series_timeout_seconds!r}"
        )
    if not math.isfinite(series_timeout_seconds) or series_timeout_seconds <= 0:
        raise SealedError(
            f"series_timeout_seconds must be finite and > 0, got {series_timeout_seconds!r}"
        )
    if not isinstance(run_id, str) or not run_id:
        raise SealedError(f"run_id must be a non-empty str, got {run_id!r}")


def _request_json(request: SealedRequest) -> bytes:
    """The request as canonical JSON bytes — spawn pickles the args, the trusted direction."""
    return canonical_json(
        {
            "dataset": request.dataset,
            "updater_id": request.updater_id,
            "updater_ref": request.updater_ref,
            "config": dict(request.config),
            "seed": request.seed,
        }
    ).encode("utf-8")


def _paths_json(paths: SealedPaths) -> bytes:
    """The paths as canonical JSON bytes, so the server never unpickles a Path."""
    return canonical_json(
        {
            "data_root": str(paths.data_root),
            "manifest_dir": str(paths.manifest_dir),
            "sealed_cache_root": str(paths.sealed_cache_root),
        }
    ).encode("utf-8")


def _receipt(raw: bytes | None, exitcode: int | None, paths: SealedPaths) -> SealedReceipt:
    """Turn the server's one reply into a receipt, or a SealedError naming only the log."""
    if raw is None or exitcode != 0:
        raise SealedError(_failed(paths, exitcode))
    try:
        reply: object = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise SealedError(_failed(paths, exitcode)) from None
    if not isinstance(reply, dict):
        raise SealedError(_failed(paths, exitcode))
    if set(reply) == {"refused"}:
        reason = reply["refused"]
        if isinstance(reason, str):
            raise SealedError(reason)
        raise SealedError(_failed(paths, exitcode))
    record_id, status = reply.get("record_id"), reply.get("status")
    if (
        set(reply) != {"record_id", "status"}
        or not isinstance(record_id, str)
        or not isinstance(status, str)
        or status not in EVALUATION_STATUSES
    ):
        raise SealedError(_failed(paths, exitcode))
    return SealedReceipt(record_id=record_id, status=status)


def _failed(paths: SealedPaths, exitcode: int | None) -> str:
    """The caller's only diagnostic when the server dies: the exit code and the log path."""
    return (
        f"sealed server failed (exit code {exitcode}); "
        f"see {paths.data_root / 'sealed' / 'server.log'}"
    )
