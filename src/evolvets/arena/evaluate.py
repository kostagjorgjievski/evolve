"""Replay a whole dev pool into the result later PRs write to the ledger.

A fresh updater per series (AGENTS.md 2.2): the real pools are sibling columns
or clients over one shared timeline, and the replay is sequential, so an
instance that survives from one series into the next holds the earlier
series' matured feedback while the later series is still at its early origins
— the future of a correlated sibling. :func:`replay_dataset` therefore calls
the factory once per series and hands the new instance to that series only.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from evolvets.arena import ARENA_VERSION
from evolvets.arena.dev import DevPool, DevSeries
from evolvets.arena.errors import ArenaError, UpdaterFailure
from evolvets.arena.labels import LabelRelease
from evolvets.arena.metrics import (
    SeriesScore,
    aggregate,
    mase_scale,
    origin_losses,
    recovery_times,
)
from evolvets.arena.replay import SeriesTrace, replay_series
from evolvets.cache.store import CacheError, load_series
from evolvets.updater.interfaces import ConfigValue, Updater

ResultStatus = Literal["ok", "failed", "invalid_weights"]  # the ledger's minus timeout


@dataclass(frozen=True)
class ReplayResult:
    """The outcome of one updater replayed over one dev pool."""

    dataset: str
    cache_id: str
    arena_version: str
    status: ResultStatus
    error: str | None  # None iff ok
    failed_series: str | None
    failed_step: int | None
    metrics: dict[str, float]  # {} unless ok
    series: tuple[SeriesScore, ...]  # () unless ok


UpdaterFactory = Callable[[], Updater]  # zero-argument; called once per series


def replay_dataset(
    updater_factory: UpdaterFactory,
    config: Mapping[str, ConfigValue],
    seed: int,
    pool: DevPool,
) -> ReplayResult:
    """Replay a fresh updater over every series of ``pool``, in pool order, and score it.

    An :class:`UpdaterFailure` stops the replay as a failed result;
    :class:`ArenaError` (a bad pool or cache) propagates, since it is not the
    updater's fault. Nothing here charges a budget: the caller wraps it and
    charges every condition equally (AGENTS.md 2.5).
    """
    scores: list[SeriesScore] = []
    for series in pool.series:
        updater = updater_factory()
        try:
            forecasts = load_series(pool.cache_root, pool.manifest, series.series_id)
        except CacheError as exc:
            raise ArenaError(str(exc)) from exc
        try:
            trace = replay_series(
                updater,
                config,
                seed,
                series.series_id,
                forecasts,
                series.values,
                series.period,
                pool.manifest.horizon,
                series.score_end,
            )
        except UpdaterFailure as failure:
            return ReplayResult(
                dataset=pool.dataset,
                cache_id=pool.cache_id,
                arena_version=ARENA_VERSION,
                status=failure.kind,
                error=failure.message,
                failed_series=failure.series_id,
                failed_step=failure.step,
                metrics={},
                series=(),
            )
        scores.append(_score_series(series, trace))
    return ReplayResult(
        dataset=pool.dataset,
        cache_id=pool.cache_id,
        arena_version=ARENA_VERSION,
        status="ok",
        error=None,
        failed_series=None,
        failed_step=None,
        metrics=aggregate(scores),
        series=tuple(scores),
    )


def result_to_obj(result: ReplayResult) -> dict[str, object]:
    """Return the JSON-ready object of ``result``; ``canonical_bytes`` accepts it.

    This is the planned content of an evaluation's ``per_task_blob``; its
    ``metrics`` is the planned ``metrics`` field of ``evaluation`` and
    ``baseline`` records.
    """
    return {
        "dataset": result.dataset,
        "cache_id": result.cache_id,
        "arena_version": result.arena_version,
        "status": result.status,
        "error": result.error,
        "failed_series": result.failed_series,
        "failed_step": result.failed_step,
        "metrics": dict(result.metrics),
        "series": [
            {
                "series_id": score.series_id,
                "mase": score.mase,
                "oracle_mase": score.oracle_mase,
                "recovery": [[change_point, recovery] for change_point, recovery in score.recovery],
            }
            for score in result.series
        ],
    }


def _score_series(series: DevSeries, trace: SeriesTrace) -> SeriesScore:
    """Score one series' trace: MASE against its warmup scale, oracle and recovery."""
    release = LabelRelease(series.values, trace.horizon)
    scale = mase_scale(release, series.period, series.scale_end, series.score_start)
    first = trace.first_origin
    losses = origin_losses(trace.errors[series.score_start - first : series.score_end - first])
    mase = _mean(losses) / scale

    oracle_losses: tuple[float, ...] = ()
    oracle_mase: float | None = None
    if series.oracle is not None:
        rows = []
        for offset, forecast in enumerate(series.oracle):
            origin = series.score_start + offset
            targets = release.targets(origin, origin + trace.horizon)
            rows.append(tuple(forecast[h] - targets[h] for h in range(trace.horizon)))
        oracle_losses = origin_losses(rows)
        oracle_mase = _mean(oracle_losses) / scale

    recovery: tuple[tuple[int, int | None], ...] = ()
    if series.change_points:
        recovery = recovery_times(
            losses,
            oracle_losses,
            series.score_start,
            series.score_end,
            series.change_points,
            series.period,
        )
    return SeriesScore(
        series_id=series.series_id,
        mase=mase,
        oracle_mase=oracle_mase,
        recovery=recovery,
    )


def _mean(values: Sequence[float]) -> float:
    """Return the left-to-right loop mean of ``values``."""
    total = 0.0
    for value in values:
        total = total + value
    return total / len(values)
