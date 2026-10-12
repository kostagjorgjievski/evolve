"""A deterministic torch-free adapter for tests, CI and smoke runs."""

from __future__ import annotations

from collections.abc import Sequence

from evolvets.foundation.interfaces import QUANTILES


class FakeAdapter:
    """Quantile forecasts from a closed form: ``last + (q - 0.5) * spread + h * slope``.

    ``last`` is the context's last value, ``spread`` its max minus min, and
    ``slope`` is ``(last - first) / len(context)`` for the context's first
    value; ``h`` counts 1 .. horizon and ``q`` runs over :data:`QUANTILES`.
    The result is deterministic in the context alone, so equal contexts give
    equal forecasts whatever the batching. Every context received is recorded,
    in order, in ``seen``.
    """

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.seen: list[tuple[float, ...]] = []

    def forecast(
        self, contexts: Sequence[Sequence[float]], horizon: int
    ) -> list[list[list[float]]]:
        """Return one ``[quantile][h]`` row per context, recording each context in ``seen``."""
        out: list[list[list[float]]] = []
        for context in contexts:
            values = tuple(float(v) for v in context)
            self.seen.append(values)
            last = values[-1]
            spread = max(values) - min(values)
            slope = (last - values[0]) / len(values)
            out.append(
                [
                    [last + (q - 0.5) * spread + h * slope for h in range(1, horizon + 1)]
                    for q in QUANTILES
                ]
            )
        return out
