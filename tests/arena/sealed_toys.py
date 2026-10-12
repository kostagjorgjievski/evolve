"""Toy updaters for the sealed-worker tests (``tests/arena/test_sealed_worker.py``).

Every toy is reachable as a zero-argument factory — the class itself when its
constructor takes no arguments, a module-level ``make_*``-style function
otherwise — so an ``updater_ref`` can name it. Toys that record append JSON
lines to ``config['spy_path']``; the same toy class records byte-identical
lines when replayed in-process, which is what the worker-isolation tests
compare against. ``FailNth`` toys count their series across worker processes
through that shared file, because a fresh process per series cannot remember
its predecessors any other way.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def equal_weights(n_experts: int, horizon: int) -> list[list[float]]:
    """Equal weight on every expert, per horizon step."""
    weight = 1.0 / n_experts
    return [[weight] * horizon for _ in range(n_experts)]


def _note(path_value: object, payload: dict[str, object]) -> int:
    """Append one JSON line to the shared note file; return its 0-based line index."""
    path = Path(str(path_value))
    ordinal = 0
    if path.exists():
        ordinal = len(path.read_text(encoding="utf-8").splitlines())
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return ordinal


class Spy:
    """Records its init args and every (feedback, context) as JSON lines on spy_path.

    The config is recorded sorted and without ``spy_path``, so a remote run
    and an in-process replay over the same pool write identical bytes.
    """

    def init(self, n_experts: int, horizon: int, config: Any, seed: int) -> None:
        self._path = Path(str(config["spy_path"]))
        recorded = {key: config[key] for key in sorted(config) if key != "spy_path"}
        self._record({"init": [n_experts, horizon, recorded, seed]})
        return None

    def update(self, state: Any, feedback: Any, context: Any) -> tuple[list[list[float]], None]:
        self._record(
            {"update": {"feedback": [_matured(m) for m in feedback], "context": _obj(context)}}
        )
        return (equal_weights(len(context.expert_names), context.horizon), None)

    def _record(self, obj: dict[str, object]) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(obj, allow_nan=False) + "\n")


class PidCounter:
    """Notes its pid and a module-level init counter on every init.

    One fresh worker process per series means one distinct pid per line and a
    counter that is 1 every time; an instance shared across series would show
    something else.
    """

    _INITS = 0

    def init(self, n_experts: int, horizon: int, config: Any, seed: int) -> None:
        type(self)._INITS += 1
        _note(
            config["spy_path"],
            {"pid": os.getpid(), "inits": type(self)._INITS},
        )
        return None

    def update(self, state: Any, feedback: Any, context: Any) -> tuple[list[list[float]], None]:
        return (equal_weights(len(context.expert_names), context.horizon), None)


class SleepAt:
    """Sleeps ``seconds`` at ``step`` of every series it serves; notes its pid at init."""

    def __init__(self, seconds: float, step: int) -> None:
        self.seconds = seconds
        self.step = step

    def init(self, n_experts: int, horizon: int, config: Any, seed: int) -> None:
        _note(config["spy_path"], {"pid": os.getpid(), "event": "init"})
        return None

    def update(self, state: Any, feedback: Any, context: Any) -> tuple[list[list[float]], None]:
        if context.step == self.step:
            time.sleep(self.seconds)
        return (equal_weights(len(context.expert_names), context.horizon), None)


class SleepInFactory:
    """Sleeps in its constructor, so the worker never answers the first init in time."""

    def __init__(self, seconds: float) -> None:
        time.sleep(seconds)

    def init(self, n_experts: int, horizon: int, config: Any, seed: int) -> None:
        return None

    def update(self, state: Any, feedback: Any, context: Any) -> tuple[list[list[float]], None]:
        return (equal_weights(len(context.expert_names), context.horizon), None)


class FailNth:
    """Misbehaves at ``step`` of the ``series``-th series (0-based, pool order).

    The series ordinal comes from counting init lines in the shared note
    file, so it survives the fresh process per series. ``mode`` picks the
    misbehaviour: raise in init, raise in update, non-simplex weights, NaN
    weights, unencodable weights, ``os._exit(3)``, or an oversized weights
    reply.
    """

    def __init__(self, mode: str, series: int, step: int) -> None:
        self.mode = mode
        self.series = series
        self.step = step
        self.ordinal = -1

    def init(self, n_experts: int, horizon: int, config: Any, seed: int) -> None:
        self.ordinal = _note(config["spy_path"], {"event": "init", "pid": os.getpid()})
        if self.mode == "init" and self.ordinal == self.series:
            raise ValueError("init boom")
        return None

    def update(self, state: Any, feedback: Any, context: Any) -> tuple[Any, None]:
        if self.ordinal == self.series and context.step == self.step:
            if self.mode == "update":
                raise RuntimeError("update boom")
            if self.mode == "invalid":
                return ([[2.0] * context.horizon for _ in context.expert_names], None)
            if self.mode == "nan":
                return ([[float("nan")] * context.horizon for _ in context.expert_names], None)
            if self.mode == "set":
                return ({1, 2}, None)
            if self.mode == "exit":
                os._exit(3)
            if self.mode == "big":
                return ([[0.2] * context.horizon for _ in range(80_000)], None)
        return (equal_weights(len(context.expert_names), context.horizon), None)


class PrintNoise:
    """Prints a marker to stdout and stderr on every call."""

    def init(self, n_experts: int, horizon: int, config: Any, seed: int) -> None:
        print("sealed-toy-marker init out")
        print("sealed-toy-marker init err", file=sys.stderr)
        return None

    def update(self, state: Any, feedback: Any, context: Any) -> tuple[list[list[float]], None]:
        print("sealed-toy-marker update out")
        print("sealed-toy-marker update err", file=sys.stderr)
        return (equal_weights(len(context.expert_names), context.horizon), None)


def sleeping_at_step3() -> SleepAt:
    """A toy that sleeps 30 s at step 3."""
    return SleepAt(seconds=30.0, step=3)


def sleeping_factory() -> SleepInFactory:
    """A toy whose factory sleeps 30 s."""
    return SleepInFactory(seconds=30.0)


def failing_init_series0() -> FailNth:
    return FailNth("init", 0, 0)


def failing_update_series2_step7() -> FailNth:
    return FailNth("update", 2, 7)


def invalid_weights_series0_step1() -> FailNth:
    return FailNth("invalid", 0, 1)


def nan_weights_series0_step1() -> FailNth:
    return FailNth("nan", 0, 1)


def set_weights_series0_step1() -> FailNth:
    return FailNth("set", 0, 1)


def exit3_series0_step5() -> FailNth:
    return FailNth("exit", 0, 5)


def big_weights_series0_step1() -> FailNth:
    return FailNth("big", 0, 1)


def _matured(matured: Any) -> dict[str, object]:
    """The JSON form of one Matured, as the worker-side Spy sees it."""
    return {
        "step": matured.step,
        "errors": [list(row) for row in matured.errors],
        "combined_errors": list(matured.combined_errors),
    }


def _obj(context: Any) -> dict[str, object]:
    """The JSON form of one Context, as the worker-side Spy sees it."""
    return {
        "step": context.step,
        "season_index": context.season_index,
        "period": context.period,
        "horizon": context.horizon,
        "expert_names": list(context.expert_names),
        "forecasts": [list(row) for row in context.forecasts],
    }
