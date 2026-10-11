"""Metrics that turn a replay into the numbers Gate 1 reports.

Float arithmetic sticks to ``+ - * /``, comparisons and ``abs``, accumulated
left-to-right in the order docs/tasks/TASK_007_arena_replay.md states, so a
score is bit-identical everywhere. Every value these functions read as a
target or a history reaches them through
:class:`~evolvets.arena.labels.LabelRelease`.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from evolvets.arena.errors import ArenaError
from evolvets.arena.labels import LabelRelease

RECOVERY_WINDOW_PERIODS = 2
RECOVERY_TOLERANCE = 0.25

_INF = float("inf")
_NEG_INF = float("-inf")


def _require_int(name: str, value: object, minimum: int) -> int:
    """Return ``value`` as an int >= ``minimum`` (bools rejected), else raise ArenaError."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArenaError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ArenaError(f"{name} must be >= {minimum}, got {value}")
    return value


def _is_finite(value: float) -> bool:
    """Return whether ``value`` is neither NaN nor an infinity, via comparisons only."""
    return value == value and value != _INF and value != _NEG_INF


@dataclass(frozen=True)
class SeriesScore:
    """One series' replay, scored."""

    series_id: str
    mase: float
    oracle_mase: float | None  # synthetic pools only
    recovery: tuple[tuple[int, int | None], ...]  # (change point, origins to recover or None)


def mase_scale(release: LabelRelease, period: int, scale_end: int, now: int) -> float:
    """Return the in-scale mean absolute seasonal difference of ``y_0 .. y_{scale_end - 1}``.

    The scale is the mean of ``|y[i] - y[i - period]|`` over ``i = period ..
    scale_end - 1`` ascending, read through :meth:`LabelRelease.history`. A
    zero or non-finite result is a pool problem, not an updater failure, so it
    raises :class:`ArenaError`.
    """
    period = _require_int("period", period, 1)
    scale_end = _require_int("scale_end", scale_end, period + 1)
    y = release.history(scale_end, now)
    total = 0.0
    for i in range(period, scale_end):
        total = total + abs(y[i] - y[i - period])
    scale = total / (scale_end - period)
    if scale == 0.0 or not _is_finite(scale):
        raise ArenaError(
            f"the MASE scale over y_0 .. y_{scale_end - 1} with period {period} is "
            f"degenerate ({scale!r})"
        )
    return scale


def origin_losses(errors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Per origin, the mean absolute error over the horizon, left to right."""
    out: list[float] = []
    for row in errors:
        if not row:
            raise ArenaError("an origin with no errors has no mean absolute error")
        total = 0.0
        for error in row:
            total = total + abs(error)
        out.append(total / len(row))
    return tuple(out)


def recovery_times(
    losses: Sequence[float],
    oracle_losses: Sequence[float],
    score_start: int,
    score_end: int,
    change_points: Sequence[int],
    period: int,
) -> tuple[tuple[int, int | None], ...]:
    """Origins each in-range change point needs to recover its oracle loss level.

    A change point ``c`` has recovered at the smallest ``k >= 0`` whose window
    ``[c + k, c + k + window)`` — with ``window = RECOVERY_WINDOW_PERIODS *
    period`` — lies inside the span up to the next change point, or the end of
    scoring, whichever comes first, and holds losses totalling at most ``(1 +
    RECOVERY_TOLERANCE)`` times the oracle's over the same window. ``None``
    means censored: no such ``k`` exists in that span. Change points outside
    ``[score_start, score_end)`` are skipped, not counted. ``losses`` and
    ``oracle_losses`` are indexed by ``origin - score_start`` and must hold
    exactly one entry per scored origin.
    """
    score_start = _require_int("score_start", score_start, 0)
    score_end = _require_int("score_end", score_end, score_start + 1)
    period = _require_int("period", period, 1)
    n = score_end - score_start
    if len(losses) != n or len(oracle_losses) != n:
        raise ArenaError(
            f"losses and oracle_losses must hold one entry per scored origin ({n}), "
            f"got {len(losses)} and {len(oracle_losses)}"
        )
    window = RECOVERY_WINDOW_PERIODS * period
    points = sorted(change_points)
    out: list[tuple[int, int | None]] = []
    for index, c in enumerate(points):
        if not score_start <= c < score_end:
            continue
        bound = score_end
        for later in points[index + 1 :]:
            if later > c:
                if later < bound:
                    bound = later
                break
        recovery: int | None = None
        k = 0
        while c + k + window <= bound:
            actual = 0.0
            expected = 0.0
            for i in range(c + k - score_start, c + k + window - score_start):
                actual = actual + losses[i]
                expected = expected + oracle_losses[i]
            if actual <= (1.0 + RECOVERY_TOLERANCE) * expected:
                recovery = k
                break
            k = k + 1
        out.append((c, recovery))
    return tuple(out)


def aggregate(scores: Sequence[SeriesScore]) -> dict[str, float]:
    """Pool per-series scores into the metrics a result carries, left to right.

    ``n_series`` and ``mase`` are always present. When any score has an
    oracle (synthetic pools), ``oracle_mase``, ``regret`` and the recovery
    counts join, with ``recovery_mean`` present only when some recovery was
    not censored. Every value is a float.
    """
    if not scores:
        raise ArenaError("cannot aggregate an empty pool of series scores")
    total = 0.0
    for score in scores:
        total = total + score.mase
    mase = total / len(scores)
    out: dict[str, float] = {"n_series": float(len(scores)), "mase": mase}
    oracle_total = 0.0
    n_oracle = 0
    for score in scores:
        if score.oracle_mase is not None:
            oracle_total = oracle_total + score.oracle_mase
            n_oracle = n_oracle + 1
    if n_oracle:
        oracle_mase = oracle_total / n_oracle
        out["oracle_mase"] = oracle_mase
        out["regret"] = mase - oracle_mase
        entries: list[int | None] = []
        for score in scores:
            for _change_point, recovery in score.recovery:
                entries.append(recovery)
        out["recovery_n"] = float(len(entries))
        censored = 0
        recovered_total = 0.0
        n_recovered = 0
        for entry in entries:
            if entry is None:
                censored = censored + 1
            else:
                recovered_total = recovered_total + entry
                n_recovered = n_recovered + 1
        out["recovery_censored"] = float(censored)
        if n_recovered:
            out["recovery_mean"] = recovered_total / n_recovered
    return out
