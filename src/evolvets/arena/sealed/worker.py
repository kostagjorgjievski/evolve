"""The sealed worker: one series' updater, alone in its own process.

Imports from ``evolvets`` stop at ``evolvets.updater.interfaces``, so nothing
that can load data — cache, arena, ledger — is ever imported here, whatever
updater the worker is asked to resolve. File descriptors 1 and 2 are pointed
at ``os.devnull`` before anything else runs, so an updater that prints cannot
reach the caller's console or CI logs. Every message is JSON bytes through
``send_bytes``/``recv_bytes``; nothing is ever unpickled from the pipe.
"""

from __future__ import annotations

import importlib
import json
import os
from multiprocessing.connection import Connection
from typing import Any, cast

from evolvets.updater.interfaces import Context, Matured, Updater


def _worker_main(conn: Connection, updater_ref: str) -> None:
    """Serve one updater over the pipe: ``init``/``update`` until ``close`` or EOF."""
    _silence_output()
    session = _Session(updater_ref)
    while True:
        try:
            raw = conn.recv_bytes()
        except (EOFError, OSError):
            return
        try:
            reply = session.handle(raw)
            if reply is None:  # the close message
                return
            encoded = json.dumps(reply, allow_nan=True).encode("utf-8")
        except Exception as exc:
            encoded = json.dumps({"error": _failure(exc)}, allow_nan=True).encode("utf-8")
        conn.send_bytes(encoded)


class _Session:
    """One updater instance: rebuilds Context/Matured from JSON, keeps the state."""

    def __init__(self, updater_ref: str) -> None:
        self._updater: Updater | None = None
        self._start_error: BaseException | None = None
        self._state: object = None
        module_name, _, attr = updater_ref.partition(":")
        try:
            factory = getattr(importlib.import_module(module_name), attr)
            self._updater = cast(Updater, factory())
        except Exception as exc:
            self._start_error = exc  # reported on the first message that asks

    def handle(self, raw: bytes) -> dict[str, Any] | None:
        """Return the reply for one message, or ``None`` when it was ``close``."""
        message = json.loads(raw.decode("utf-8"))
        if not isinstance(message, dict):
            raise ValueError(f"the message must be a JSON object, got {type(message).__name__}")
        op = message.get("op")
        if op == "close":
            return None
        if self._updater is None:
            if self._start_error is not None:
                raise self._start_error
            raise ValueError("the updater was never created")
        if op == "init":
            self._state = self._updater.init(
                message["n_experts"], message["horizon"], message["config"], message["seed"]
            )
            return {"ok": True}
        if op == "update":
            feedback = tuple(_matured(item) for item in message["feedback"])
            weights, self._state = self._updater.update(
                self._state, feedback, _context(message["context"])
            )
            return {"weights": weights}
        raise ValueError(f"unknown op {op!r}")


def _matured(item: Any) -> Matured:
    """Rebuild one Matured feedback row from its JSON form."""
    return Matured(
        step=item["step"],
        errors=tuple(tuple(row) for row in item["errors"]),
        combined_errors=tuple(item["combined_errors"]),
    )


def _context(obj: Any) -> Context:
    """Rebuild one Context from its JSON form."""
    return Context(
        step=obj["step"],
        season_index=obj["season_index"],
        period=obj["period"],
        horizon=obj["horizon"],
        expert_names=tuple(obj["expert_names"]),
        forecasts=tuple(tuple(row) for row in obj["forecasts"]),
    )


def _silence_output() -> None:
    """Point file descriptors 1 and 2 at ``os.devnull`` before any updater code runs."""
    with open(os.devnull, "wb") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)


def _failure(exc: BaseException) -> str:
    """Format an exception for an error reply, truncated to 1000 characters."""
    return f"{type(exc).__name__}: {exc}"[:1000]
