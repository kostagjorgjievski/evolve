"""The updater contract: everything an updater may know and may return.

This file is part of the arena protocol (``ARENA_VERSION``): any change that
alters what an updater receives or may return must bump it. It deliberately
imports nothing from ``evolvets`` — the direction is arena -> updater, never
the reverse — and it must stay agent-editable-code-free of any data path: an
updater sees expert forecasts and matured errors only, never a target, a
value, an absolute origin, a series id, a timestamp, the cache or an oracle.

``weights-v1`` is the original contract and ``forecast-v2`` its extension:
a forecast-v2 updater sees exactly what a weights-v1 updater sees — the same
``Context`` and the same matured errors — and may additionally answer with
its own point forecast instead of weights.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

ConfigValue = float | int | str | bool
Weights = Sequence[Sequence[float]]  # [expert][h - 1]; each column h is a simplex over experts

UPDATER_INTERFACES: tuple[str, ...] = ("weights-v1", "forecast-v2")
FORECAST_ABS_LIMIT = 1e12  # bound on |forecast value|; keeps every metric finite


@dataclass(frozen=True)
class Matured:
    """Errors for one origin whose full horizon has just passed."""

    step: int  # step index of the matured origin (see Context.step)
    errors: tuple[tuple[float, ...], ...]  # [expert][h - 1] = expert forecast - target
    # [h - 1] = this updater's output forecast - target (the weighted combination under weights-v1)
    combined_errors: tuple[float, ...]


@dataclass(frozen=True)
class Context:
    """Everything an updater may know at the current origin."""

    step: int  # 0 at the first replayed origin, +1 per origin
    season_index: int  # origin % period
    period: int
    horizon: int
    expert_names: tuple[str, ...]
    forecasts: tuple[tuple[float, ...], ...]  # [expert][h - 1] for targets origin+1 .. origin+H


@dataclass(frozen=True)
class ForecastOutput:
    """A forecast-v2 updater's own point forecast for targets origin+1 .. origin+H."""

    values: tuple[float, ...]  # [h - 1]


Output = Weights | ForecastOutput


class Updater(Protocol):
    """An online combiner of expert forecasts. A module defining init/update also satisfies it."""

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> object: ...

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Weights, object]: ...


class ForecastUpdater(Protocol):
    """A forecast-v2 updater: like Updater, but update may return a ForecastOutput."""

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> object: ...

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Output, object]: ...
