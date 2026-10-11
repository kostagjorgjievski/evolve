"""The updater contract: everything an updater may know and may return.

This file is part of the arena protocol (``ARENA_VERSION``): any change that
alters what an updater receives or may return must bump it. It deliberately
imports nothing from ``evolvets`` — the direction is arena -> updater, never
the reverse — and it must stay agent-editable-code-free of any data path: an
updater sees expert forecasts and matured errors only, never a target, a
value, an absolute origin, a series id, a timestamp, the cache or an oracle.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

ConfigValue = float | int | str | bool
Weights = Sequence[Sequence[float]]  # [expert][h - 1]; each column h is a simplex over experts


@dataclass(frozen=True)
class Matured:
    """Errors for one origin whose full horizon has just passed."""

    step: int  # step index of the matured origin (see Context.step)
    errors: tuple[tuple[float, ...], ...]  # [expert][h - 1] = expert forecast - target
    combined_errors: tuple[float, ...]  # [h - 1] = this updater's combined forecast - target


@dataclass(frozen=True)
class Context:
    """Everything an updater may know at the current origin."""

    step: int  # 0 at the first replayed origin, +1 per origin
    season_index: int  # origin % period
    period: int
    horizon: int
    expert_names: tuple[str, ...]
    forecasts: tuple[tuple[float, ...], ...]  # [expert][h - 1] for targets origin+1 .. origin+H


class Updater(Protocol):
    """An online combiner of expert forecasts. A module defining init/update also satisfies it."""

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> object: ...

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Weights, object]: ...
