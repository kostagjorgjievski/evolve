"""Evaluating one Phase 1 artifact in a fresh subprocess under hard limits.

The artifact validator (``improver/artifacts.py``) is a grammar, not a CPU or
memory sandbox: ``**`` is banned but a grammatical ``range(1000000000000)``
can still burn either (TASK_017 PR A1's review note). This module is the
second wall every evaluated edit runs behind: the parent spawns one child
per evaluation (``python -m evolvets.runner.phase1_eval``) with a wall
deadline and an address-space cap set before ``exec``, so a pathological
artifact costs at most its deadline and its cap — never the parent, and
never the ledger: the worker holds no ledger path at all.

The protocol is one request, one reply. The parent writes the JSON request
to the child's stdin; the child loads the dev pool, replays the artifact
over it (a fresh adapter per series, per-series wall stamped like gate1's
timed factory) and writes one JSON reply stamped with ``WORKER_PROTOCOL``.
Every parent-side failure — a timeout, a nonzero exit, an empty, garbled,
oversized, wrongly-stamped or errorless reply — becomes an
:class:`EvalOutcome` with status ``"failed"`` or ``"timeout"`` and a
``worker: ...`` error line; no child failure ever raises out of
:func:`evaluate_source`.

Imports are the arena's dev side, the replay loop and the artifact loader:
``runner`` is instrument code, not agent-editable code, so the import-linter
contracts leave it untouched.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from evolvets.arena.dev import load_dev_pool
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.improver.artifacts import artifact_updater
from evolvets.updater.interfaces import ConfigValue, ForecastUpdater

#: Stamped on every worker reply; the parent refuses a reply without it.
WORKER_PROTOCOL = "1"

#: A reply larger than this is a failure, not a result.
MAX_REPLY_BYTES = 64 << 20

#: How the parent launches the worker; read at call time so tests can stub it.
_CHILD_COMMAND: tuple[str, ...] = (sys.executable, "-m", "evolvets.runner.phase1_eval")

#: The replay statuses a well-formed reply may carry (timeout is parent-only).
_WORKER_STATUSES = frozenset({"ok", "failed", "invalid_weights"})

#: Cap on the stderr tail folded into a ``worker: ...`` error line.
_STDERR_CAP = 2000


@dataclass(frozen=True)
class EvalOutcome:
    """One evaluated artifact: its replay status, scores and the parent-measured wall.

    ``status`` widens the ledger's evaluation statuses with ``"timeout"``: the
    child was SIGKILLed at its wall deadline. ``result_obj`` is
    ``result_to_obj(...)`` when ``ok`` and ``None`` otherwise; ``error`` is
    ``None`` iff ``ok``.
    """

    status: str  # "ok" | "failed" | "invalid_weights" | "timeout"
    result_obj: dict[str, object] | None  # result_to_obj(...) when ok
    series_wall_seconds: tuple[float, ...]
    error: str | None
    wall_seconds: float  # parent-measured, whole request


def evaluate_source(
    request: Mapping[str, object], *, timeout_seconds: float, memory_gib: float
) -> EvalOutcome:
    """Evaluate one artifact ``request`` in a fresh subprocess under hard limits.

    ``request`` is the JSON object the worker reads on stdin:
    ``{"source": str, "config": {...}, "seed": int, "interface":
    "forecast-v2" | "weights-v1", "dataset": str, "pool_spec": str,
    "foundation_models": [...], "cache_root": str, "data_root": str,
    "manifest_dir": str, "foundation_root": str}``. The child runs under
    ``RLIMIT_AS = memory_gib`` GiB and is killed, process group and all, at
    ``timeout_seconds`` of wall clock; its outcome never raises here. The two
    limits themselves are caller configuration, not worker behaviour, so a
    limit that is not finite and positive raises ``ValueError`` before any
    child is spawned: a negative address-space cap is ``RLIM_INFINITY`` (a
    silently uncapped child) and a zero one fails the exec in confusing ways.
    """
    for name, limit in (("timeout_seconds", timeout_seconds), ("memory_gib", memory_gib)):
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {limit!r}")
    cap = int(memory_gib * 2**30)

    def limit_address_space() -> None:
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    payload = json.dumps(dict(request)).encode("utf-8")
    started = time.monotonic()
    # preexec_fn is not thread-safe in general; it is safe here because the
    # runner is single-threaded.
    process = subprocess.Popen(
        list(_CHILD_COMMAND),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
        preexec_fn=limit_address_space,
    )
    try:
        stdout, stderr = process.communicate(payload, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        # start_new_session=True made the child its own process-group
        # leader, so the group kill takes the worker and anything it spawned.
        if process.pid is not None:
            # a race with the child's own death, not an error to report
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        return EvalOutcome(
            status="timeout",
            result_obj=None,
            series_wall_seconds=(),
            error=f"worker exceeded the {timeout_seconds:g} s wall deadline",
            wall_seconds=time.monotonic() - started,
        )
    ended = time.monotonic()
    return _parse_reply(stdout, stderr, process.returncode, ended - started)


def _failed_outcome(stderr: bytes, note: str, wall_seconds: float) -> EvalOutcome:
    """One parent-side failure: a ``worker: ...`` line with the stderr tail, else ``note``."""
    tail = stderr.decode("utf-8", "replace").strip()[-_STDERR_CAP:]
    detail = tail if tail else note
    return EvalOutcome(
        status="failed",
        result_obj=None,
        series_wall_seconds=(),
        error=f"worker: {detail}",
        wall_seconds=wall_seconds,
    )


def _parse_reply(
    stdout: bytes, stderr: bytes, returncode: int | None, wall_seconds: float
) -> EvalOutcome:
    """Turn one finished worker into an outcome; a malformed reply is a failed outcome."""
    if returncode != 0:
        return _failed_outcome(stderr, f"exit code {returncode}", wall_seconds)
    if len(stdout) > MAX_REPLY_BYTES:
        return _failed_outcome(
            stderr,
            f"reply of {len(stdout)} bytes exceeds the {MAX_REPLY_BYTES}-byte cap",
            wall_seconds,
        )
    try:
        reply = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return _failed_outcome(stderr, "the reply is not JSON", wall_seconds)
    if not isinstance(reply, dict) or reply.get("worker") != WORKER_PROTOCOL:
        return _failed_outcome(
            stderr, f"the reply does not stamp worker protocol {WORKER_PROTOCOL}", wall_seconds
        )
    status = reply.get("status")
    if status not in _WORKER_STATUSES:
        return _failed_outcome(stderr, f"unknown reply status {status!r}", wall_seconds)
    raw_walls = reply.get("series_wall_seconds")
    if not isinstance(raw_walls, list) or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for value in raw_walls
    ):
        return _failed_outcome(stderr, "the reply's series_wall_seconds is malformed", wall_seconds)
    result_obj = cast("dict[str, object] | None", reply.get("result"))
    error = reply.get("error")
    if status == "ok":
        if not isinstance(result_obj, dict) or error is not None:
            return _failed_outcome(
                stderr, "an ok reply must carry a result and no error", wall_seconds
            )
    elif result_obj is not None or not (isinstance(error, str) and error):
        return _failed_outcome(
            stderr, "a failed reply must carry an error and no result", wall_seconds
        )
    return EvalOutcome(
        status=cast(str, status),
        result_obj=result_obj if status == "ok" else None,
        series_wall_seconds=tuple(float(value) for value in raw_walls),
        error=error if status != "ok" else None,
        wall_seconds=wall_seconds,
    )


# --- the worker ----------------------------------------------------------------------


class _TimedFactory:
    """An artifact factory that timestamps each call; the replay calls it once per series."""

    def __init__(self, make: Callable[[], ForecastUpdater]) -> None:
        self._make = make
        self._calls: list[float] = []

    def __call__(self) -> ForecastUpdater:
        self._calls.append(time.monotonic())
        return self._make()

    def series_wall_seconds(self, end: float) -> list[float]:
        """Time from each call to the next; the last entry runs to ``end``."""
        stamps = [*self._calls, end]
        return [stamps[index + 1] - stamps[index] for index in range(len(self._calls))]


def _text(request: Mapping[str, object], key: str) -> str:
    """Return ``request[key]`` as a str, else raise ValueError naming the key."""
    value = request.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string, got {value!r}")
    return value


def _main() -> None:
    """Read one request from stdin, replay it over its dev pool, write one reply.

    Any failure — a malformed request, a bad cache or data path, an invalid
    artifact source — raises, so the worker exits nonzero with a traceback on
    stderr and the parent reports it as a failed outcome.
    """
    request = json.loads(sys.stdin.read())
    if not isinstance(request, dict):
        raise ValueError("the worker request must be a JSON object")
    pool_spec = _text(request, "pool_spec")
    foundation_root: Path | None = None
    foundation_models: tuple[str, ...] | None = None
    if pool_spec != "simple":
        foundation_root = Path(_text(request, "foundation_root"))
        raw_models = request.get("foundation_models")
        if not isinstance(raw_models, list) or not all(
            isinstance(name, str) for name in raw_models
        ):
            raise ValueError("foundation_models must be a list of strings")
        foundation_models = tuple(raw_models) if raw_models else None
    config = request.get("config")
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    seed = request.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    pool = load_dev_pool(
        _text(request, "dataset"),
        Path(_text(request, "cache_root")),
        Path(_text(request, "data_root")),
        Path(_text(request, "manifest_dir")),
        pool_spec=pool_spec,
        foundation_root=foundation_root,
        foundation_models=foundation_models,
    )
    timed = _TimedFactory(artifact_updater(_text(request, "source")))
    result = replay_dataset(
        timed,
        cast("Mapping[str, ConfigValue]", config),
        seed,
        pool,
        interface=_text(request, "interface"),
    )
    reply = {
        "worker": WORKER_PROTOCOL,
        "status": result.status,
        "result": result_to_obj(result) if result.status == "ok" else None,
        "series_wall_seconds": timed.series_wall_seconds(time.monotonic()),
        "error": result.error,
    }
    sys.stdout.write(json.dumps(reply))
    sys.stdout.flush()


if __name__ == "__main__":
    _main()
