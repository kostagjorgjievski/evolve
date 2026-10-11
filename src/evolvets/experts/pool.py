"""The standard expert pool and its in-memory evaluation over one series."""

from collections.abc import Sequence

from evolvets.experts.interfaces import Expert, ExpertError
from evolvets.experts.ridge import RidgeAR
from evolvets.experts.simple import DampedHoltWinters, Naive, SeasonalNaive, SimpleExpSmoothing

EXPERT_NAMES = ("naive", "seasonal_naive", "ses", "holt_winters", "ar_ridge")  # cache order


def make_pool(period: int) -> tuple[Expert, ...]:
    """Return fresh experts with default parameters, in :data:`EXPERT_NAMES` order."""
    return (
        Naive(),
        SeasonalNaive(period),
        SimpleExpSmoothing(),
        DampedHoltWinters(period),
        RidgeAR(period),
    )


def first_origin(period: int) -> int:
    """Return the first index at which the whole pool can forecast: ``max(min_history) - 1``."""
    return max(expert.min_history for expert in make_pool(period)) - 1


def run_pool(
    values: Sequence[float], period: int, horizon: int
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    """Run the pool over ``values`` and forecast at every origin from the first one on.

    Returns one entry per origin in origin order, each entry being the five
    experts' forecasts in :data:`EXPERT_NAMES` order. Every value is observed
    by every expert before any expert forecasts at that origin, so no expert
    ever sees a value beyond the origin it forecasts from. Targets beyond the
    end of ``values`` are fine: a forecast never needs its target.
    """
    start = first_origin(period)
    if len(values) <= start:
        raise ExpertError(
            f"run_pool needs more than {start} values for period {period}, got {len(values)}"
        )
    pool = make_pool(period)
    results: list[tuple[tuple[float, ...], ...]] = []
    for t, value in enumerate(values):
        for expert in pool:
            expert.observe(value)
        if t >= start:
            results.append(tuple(expert.forecast(horizon) for expert in pool))
    return tuple(results)
