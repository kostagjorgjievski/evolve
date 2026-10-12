"""The server side of the worker protocol: spawn, message, deadline, shutdown.

The factory hands :func:`replay_dataset` one :class:`RemoteUpdater` per
series, each bound to a freshly spawned worker process, so module-level and
process-global updater state cannot carry over between series (the gap
TASK_007 left open). Pipes carry JSON bytes only — ``send_bytes`` and
``recv_bytes``, never ``recv`` — and :exc:`SeriesTimeout` subclasses
``BaseException`` on purpose, so the replay loop (which wraps only
``Exception``) lets a timed-out series stop the whole sealed evaluation.
"""

from __future__ import annotations

import contextlib
import json
import multiprocessing
import multiprocessing.process
import time
from collections.abc import Mapping, Sequence
from multiprocessing.connection import Connection
from typing import cast

from evolvets.arena.sealed.worker import _worker_main
from evolvets.updater.interfaces import ConfigValue, Context, Matured, Updater, Weights

MAX_REPLY_BYTES = 1 << 22  # 4 MiB; a bigger reply is a failed updater, not a protocol to support
CLOSE_JOIN_SECONDS = 5.0
_REPLY_KEYS = ("ok", "weights", "error")


class WorkerError(Exception):
    """The worker died, replied unintelligibly, or reported an updater failure."""


class SeriesTimeout(BaseException):
    """One series exceeded its deadline; ends the sealed evaluation as a timeout.

    Deliberately a ``BaseException``: ``replay_series`` wraps only ``Exception``
    to convert updater faults into failed results, so a timeout must not be
    catchable there — it has to reach the server untouched, where it becomes a
    ``timeout`` baseline rather than a ``failed`` one.
    """

    def __init__(self, series_id: str, step: int | None) -> None:
        super().__init__(series_id, step)
        self.series_id = series_id
        self.step = step


class RemoteUpdater:
    """The server-side proxy of one series' updater, running in a worker process."""

    def __init__(
        self,
        conn: Connection,
        process: multiprocessing.process.BaseProcess,
        series_id: str,
        deadline: float,
    ) -> None:
        self._conn = conn
        self._process = process
        self._series_id = series_id
        self._deadline = deadline
        self._step: int | None = None

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> object:
        """Forward ``init`` to the worker; the state stays in the worker process."""
        self._step = None
        self._send(
            {
                "op": "init",
                "n_experts": n_experts,
                "horizon": horizon,
                "config": dict(config),
                "seed": seed,
            }
        )
        reply = self._reply()
        if set(reply) != {"ok"} or reply["ok"] is not True:
            raise WorkerError(f"series {self._series_id}: the worker replied {reply!r} to init")
        return None

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Weights, object]:
        """Forward one ``update`` to the worker and return its weights, unvalidated."""
        self._step = context.step
        self._send(
            {
                "op": "update",
                "feedback": [_matured_obj(item) for item in feedback],
                "context": _context_obj(context),
            }
        )
        reply = self._reply()
        if set(reply) != {"weights"}:
            raise WorkerError(f"series {self._series_id}: the worker replied {reply!r} to update")
        return cast("Weights", reply["weights"]), None

    def close(self) -> None:
        """Shut the worker down: close, join with a limit, then kill and join."""
        with contextlib.suppress(OSError, ValueError):
            # the worker may already be gone; the join below reaps it either way
            self._conn.send_bytes(b'{"op": "close"}')
        self._process.join(CLOSE_JOIN_SECONDS)
        self._process.kill()
        self._process.join()
        with contextlib.suppress(OSError):
            self._conn.close()

    def _send(self, message: dict[str, object]) -> None:
        """Send one message as JSON bytes; floats round-trip exactly through repr."""
        try:
            self._conn.send_bytes(json.dumps(message, allow_nan=False).encode("utf-8"))
        except (OSError, ValueError) as exc:
            raise WorkerError(f"series {self._series_id}: cannot reach the worker: {exc}") from None

    def _reply(self) -> dict[str, object]:
        """Wait for the worker's reply until the series deadline, then parse and check it."""
        if not self._conn.poll(self._deadline - time.monotonic()):
            raise SeriesTimeout(self._series_id, self._step)
        try:
            raw = self._conn.recv_bytes(MAX_REPLY_BYTES)
        except (EOFError, OSError) as exc:
            self._process.join(CLOSE_JOIN_SECONDS)
            raise WorkerError(
                f"series {self._series_id}: the worker failed ({exc}); "
                f"worker exit code {self._process.exitcode}"
            ) from None
        reply = _parse_reply(raw)
        if "error" in reply:
            raise WorkerError(f"series {self._series_id}: {reply['error']}")
        return reply


class RemoteUpdaterFactory:
    """Updater factory that spawns a fresh worker process per series, in pool order."""

    def __init__(self, series_ids: Sequence[str], updater_ref: str, timeout_seconds: float) -> None:
        self._series_ids = tuple(series_ids)
        self._updater_ref = updater_ref
        self._timeout_seconds = timeout_seconds
        self._next = 0
        self._current: RemoteUpdater | None = None
        self._context = multiprocessing.get_context("spawn")

    def __call__(self) -> Updater:
        """Shut the previous worker down, spawn a fresh one, bind it to its series."""
        self.shutdown()
        if self._next >= len(self._series_ids):
            raise WorkerError(
                "the replay asked for more updater instances than the pool has series"
            )
        series_id = self._series_ids[self._next]
        self._next += 1
        deadline = time.monotonic() + self._timeout_seconds
        parent_conn, child_conn = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=_worker_main, args=(child_conn, self._updater_ref), daemon=True
        )
        process.start()
        child_conn.close()
        updater = RemoteUpdater(
            conn=parent_conn,
            process=process,
            series_id=series_id,
            deadline=deadline,
        )
        self._current = updater
        return updater

    def shutdown(self) -> None:
        """Shut the current worker down, if any; safe to call repeatedly."""
        if self._current is not None:
            self._current.close()
            self._current = None


def _parse_reply(raw: bytes) -> dict[str, object]:
    """Decode one worker reply and check it is an object with exactly one reply key."""
    try:
        reply: object = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise WorkerError(f"the worker sent invalid JSON: {exc}") from None
    if not isinstance(reply, dict) or len(reply) != 1 or next(iter(reply)) not in _REPLY_KEYS:
        raise WorkerError(f"the worker replied {raw[:100]!r}")
    return dict(reply)


def _matured_obj(matured: Matured) -> dict[str, object]:
    """The JSON form of one Matured feedback row."""
    return {
        "step": matured.step,
        "errors": [list(row) for row in matured.errors],
        "combined_errors": list(matured.combined_errors),
    }


def _context_obj(context: Context) -> dict[str, object]:
    """The JSON form of one Context."""
    return {
        "step": context.step,
        "season_index": context.season_index,
        "period": context.period,
        "horizon": context.horizon,
        "expert_names": list(context.expert_names),
        "forecasts": [list(row) for row in context.forecasts],
    }
