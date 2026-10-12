"""Interfaces shared by the foundation cache builder and every adapter."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

#: The quantile levels every adapter must return, in order. The 0.5 quantile
#: (index 4) is the point forecast the updaters receive.
QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class Adapter(Protocol):
    """A batch forecaster: many variable-length contexts in, quantile forecasts out.

    Implementations live in ``foundation/models/`` (one per pretrained library)
    plus the torch-free :class:`evolvets.foundation.fake.FakeAdapter`. Contexts
    within one batch must not affect each other's forecasts.
    """

    name: str  # registry key, e.g. "chronos2"

    def forecast(
        self, contexts: Sequence[Sequence[float]], horizon: int
    ) -> list[list[list[float]]]: ...  # [context][quantile][h], len(QUANTILES) quantiles


@dataclass(frozen=True)
class SeriesForecast:
    """One series' forecasts for every origin, and how many steps were monotone-sorted.

    ``values`` is ``[origin][quantile][h]`` in origin order, with quantile
    crossing already fixed: at any step where a model returned quantiles that
    were not non-decreasing, they were sorted ascending (monotone
    rearrangement). ``sorted_steps`` counts the (origin, step) pairs sorted.
    """

    values: tuple[tuple[tuple[float, ...], ...], ...]
    sorted_steps: int
