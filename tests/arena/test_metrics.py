"""Metrics tests: MASE scale, origin losses, series mase, recovery, aggregation (tests 4-7)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evolvets.arena.dev import DevPool, DevSeries
from evolvets.arena.errors import ArenaError
from evolvets.arena.evaluate import replay_dataset
from evolvets.arena.labels import LabelRelease
from evolvets.arena.metrics import (
    RECOVERY_TOLERANCE,
    RECOVERY_WINDOW_PERIODS,
    SeriesScore,
    aggregate,
    mase_scale,
    origin_losses,
    recovery_times,
)
from evolvets.cache.interfaces import SeriesInput
from evolvets.cache.store import build_cache, cache_id, read_manifest

# --- Test 4: mase_scale ----------------------------------------------------------------------


def test_mase_scale_matches_a_hand_computation(monkeypatch: pytest.MonkeyPatch) -> None:
    """|7-3| + |11-5| over i = 2..3, period 2: (4 + 6) / 2 = 5, read through history."""
    release = LabelRelease((3.0, 5.0, 7.0, 11.0, 13.0), 1)
    calls: list[tuple[int, int]] = []
    original = LabelRelease.history

    def recording(self: LabelRelease, end: int, now: int) -> tuple[float, ...]:
        calls.append((end, now))
        return original(self, end, now)

    monkeypatch.setattr(LabelRelease, "history", recording)
    assert mase_scale(release, 2, 4, now=10) == 5.0
    assert calls == [(4, 10)]


def test_mase_scale_rejects_a_degenerate_scale() -> None:
    """A constant series has a zero scale: a pool problem, not an updater failure."""
    release = LabelRelease((2.0, 2.0, 2.0, 2.0), 1)
    with pytest.raises(ArenaError, match="degenerate"):
        mase_scale(release, 2, 4, now=10)


@pytest.mark.parametrize("period, scale_end", [(1, 1), (2, 2), (2, 1)], ids=["p1", "p_eq", "under"])
def test_mase_scale_needs_a_seasonal_difference(period: int, scale_end: int) -> None:
    """A window shorter than period + 1 holds no seasonal difference."""
    release = LabelRelease((1.0, 2.0, 3.0, 4.0), 1)
    with pytest.raises(ArenaError):
        mase_scale(release, period, scale_end, now=10)


# --- Test 5: origin losses and series mase ----------------------------------------------------


def test_origin_losses_match_hand_values() -> None:
    """H = 2, four scored origins: (-1, -2) at each gives mean 1.5 per origin."""
    rows = [(-1.0, -2.0)] * 4
    assert origin_losses(rows) == (1.5, 1.5, 1.5, 1.5)


def test_origin_losses_rejects_an_empty_origin() -> None:
    with pytest.raises(ArenaError, match="no errors"):
        origin_losses([(), (1.0,)])


# A one-series pool over a ramp: the naive expert (persistence) forecasts
# y_t at every horizon, so its one-hot errors are (-1, -2) at every origin,
# the scale over [0, 8) with period 2 is 2.0, and mase is 1.5 / 2.0 = 0.75.
_PERIOD = 2
_H2 = 2
_N = 260  # >= score_end + H; first_origin(2) = 243 <= score_start
_VALUES = tuple(float(i) for i in range(_N))
_SCALE_END, _SCORE_START, _SCORE_END = 8, 246, 250


class OneHotNaive:
    """All weight on the naive expert at every step."""

    def init(self, n_experts, horizon, config, seed):
        return None

    def update(self, state, feedback, context):
        rows = []
        for e in range(len(context.expert_names)):
            rows.append([1.0 if e == 0 else 0.0] * context.horizon)
        return (rows, None)


@pytest.fixture()
def hand_pool(tmp_path: Path) -> DevPool:
    """A hand-built one-series pool over the ramp, with its cache."""
    cache_root = tmp_path / "cache_store"
    build_cache(
        "hand",
        {"kind": "hand"},
        (SeriesInput("hand/linear", _VALUES, _PERIOD),),
        cache_root,
        horizon=_H2,
    )
    series = DevSeries(
        series_id="hand/linear",
        period=_PERIOD,
        values=_VALUES[: _SCORE_END + _H2],
        scale_end=_SCALE_END,
        score_start=_SCORE_START,
        score_end=_SCORE_END,
        change_points=(),
        oracle=None,
    )
    return DevPool(
        dataset="hand",
        cache_root=cache_root,
        cache_id=cache_id(cache_root, "hand"),
        manifest=read_manifest(cache_root, "hand"),
        series=(series,),
    )


def test_series_mase_matches_hand_values(hand_pool: DevPool) -> None:
    """Replaying one-hot naive over the ramp scores mase 0.75 exactly."""
    result = replay_dataset(lambda: OneHotNaive(), {}, 0, hand_pool)
    assert result.status == "ok"
    assert result.metrics == {"n_series": 1.0, "mase": 0.75}
    assert [score.mase for score in result.series] == [0.75]
    assert result.series[0].oracle_mase is None
    assert result.series[0].recovery == ()


# --- Test 6: recovery_times -------------------------------------------------------------------

# Hand-built losses over origins 10..29 (period 2, so w = 4): oracle losses are
# 1.0 everywhere, so a full window recovers when its losses sum to at most
# (1 + RECOVERY_TOLERANCE) * 4.0 = 5.0.
START, END = 10, 30
ORACLE = [1.0] * (END - START)
BAD = 9.0


def _losses(bad_origins=()) -> list[float]:
    losses = [1.0] * (END - START)
    for origin in bad_origins:
        losses[origin - START] = BAD
    return losses


def test_recovery_constants() -> None:
    assert RECOVERY_WINDOW_PERIODS == 2
    assert RECOVERY_TOLERANCE == 0.25


def test_recovery_at_k_zero() -> None:
    assert recovery_times(_losses(), ORACLE, START, END, [12], 2) == ((12, 0),)


def test_recovery_at_the_tolerance_boundary() -> None:
    """A window summing to exactly (1 + tolerance) times the oracle's recovers."""
    losses = [1.0] * (END - START)
    for origin in (12, 13, 14, 15):
        losses[origin - START] = 1.25
    assert recovery_times(losses, ORACLE, START, END, [12], 2) == ((12, 0),)


def test_recovery_at_k_three() -> None:
    assert recovery_times(_losses((12, 13, 14)), ORACLE, START, END, [12], 2) == ((12, 3),)


def test_censored_because_the_loss_never_drops() -> None:
    losses = [BAD] * (END - START)
    assert recovery_times(losses, ORACLE, START, END, [12], 2) == ((12, None),)


def test_censored_because_the_window_does_not_fit() -> None:
    """c + w = 16 > B = 15, so no window fits before the next change point."""
    assert recovery_times(_losses(), ORACLE, START, END, [12, 15], 2) == (
        (12, None),
        (15, 0),
    )


def test_change_points_outside_the_scored_range_are_skipped() -> None:
    assert recovery_times(_losses(), ORACLE, START, END, [5, 12], 2) == ((12, 0),)
    assert recovery_times(_losses(), ORACLE, START, END, [30, 45], 2) == ()


def test_two_change_points_where_the_second_bounds_the_first() -> None:
    """c = 12 would recover at k = 8, but B = 20 cuts the search off at k = 4."""
    losses = _losses(range(12, 20))
    assert recovery_times(losses, ORACLE, START, END, [12, 20], 2) == ((12, None), (20, 0))


def test_recovery_times_requires_matching_lengths() -> None:
    with pytest.raises(ArenaError, match="one entry per scored origin"):
        recovery_times(_losses()[:-1], ORACLE, START, END, [12], 2)
    with pytest.raises(ArenaError, match="one entry per scored origin"):
        recovery_times(_losses(), ORACLE[:-2], START, END, [12], 2)


# --- Test 7: aggregate ------------------------------------------------------------------------


def test_aggregate_real_only_scores() -> None:
    scores = (
        SeriesScore("etth1/HUFL", 0.75, None, ()),
        SeriesScore("etth1/HULL", 1.25, None, ()),
    )
    assert aggregate(scores) == {"n_series": 2.0, "mase": 1.0}


def test_aggregate_synthetic_scores() -> None:
    scores = (
        SeriesScore("sudden/0", 2.0, 1.0, ((100, 3), (200, None))),
        SeriesScore("mixed/0", 4.0, 2.0, ((300, 5),)),
    )
    assert aggregate(scores) == {
        "n_series": 2.0,
        "mase": 3.0,
        "oracle_mase": 1.5,
        "regret": 1.5,
        "recovery_n": 3.0,
        "recovery_censored": 1.0,
        "recovery_mean": 4.0,
    }


def test_aggregate_omits_recovery_mean_when_all_censored() -> None:
    scores = (SeriesScore("stationary/0", 2.0, 1.0, ((100, None),)),)
    out = aggregate(scores)
    assert out["recovery_n"] == 1.0
    assert out["recovery_censored"] == 1.0
    assert "recovery_mean" not in out


def test_aggregate_refuses_an_empty_pool() -> None:
    with pytest.raises(ArenaError, match="empty"):
        aggregate(())
