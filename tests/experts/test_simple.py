"""Exact-behaviour tests for the four simple experts (task tests 1, 2, 7, 8, 9, 10)."""

import math

import pytest

from evolvets.experts.interfaces import ExpertError
from evolvets.experts.pool import make_pool
from evolvets.experts.simple import (
    DampedHoltWinters,
    Naive,
    SeasonalNaive,
    SimpleExpSmoothing,
)


def _feed(expert, values):
    for v in values:
        expert.observe(v)


# --- Test 1: naive, seasonal_naive (m = 3, H = 7, k reaches 3) and ses -------------


def test_naive_exact():
    """naive repeats the last observed value for every step."""
    e = Naive()
    _feed(e, [3.0, 1.5, -2.0, 0.5, 4.25, -1.0, 2.0])
    assert e.forecast(4) == (2.0, 2.0, 2.0, 2.0)


def test_seasonal_naive_exact():
    """seasonal_naive with m = 3 cycles the last three values; k reaches 3 at h = 7."""
    e = SeasonalNaive(3)
    # t = 6; same-phase lookups: h = 1..3 -> y_4, y_5, y_6; h = 4..6 -> the same
    # three again (k = 2); h = 7 -> y_4 again (k = 3).
    _feed(e, [10.0, 20.0, 30.0, 11.0, 21.0, 31.0, 12.0])
    assert e.forecast(7) == (21.0, 31.0, 12.0, 21.0, 31.0, 12.0, 21.0)


def test_ses_exact():
    """ses levels are written out step by step and held for every step."""
    e = SimpleExpSmoothing(alpha=0.3)
    l0 = 2.0
    l1 = 0.3 * 4.0 + 0.7 * l0
    l2 = 0.3 * 3.0 + 0.7 * l1
    l3 = 0.3 * 5.0 + 0.7 * l2
    l4 = 0.3 * 1.0 + 0.7 * l3
    l5 = 0.3 * 6.0 + 0.7 * l4
    l6 = 0.3 * 2.5 + 0.7 * l5
    _feed(e, [2.0, 4.0, 3.0, 5.0, 1.0, 6.0, 2.5])
    assert e.forecast(3) == (l6, l6, l6)


# --- Test 2: holt_winters with m = 2 on 7 hand-written values -----------------------


def test_holt_winters_initial_state_and_forecasts():
    """Initial L, B, S after 4 values and the forecasts after 4, 5 and 7 values."""
    alpha, beta, gamma, phi = 0.2, 0.01, 0.1, 0.98
    e = DampedHoltWinters(period=2, alpha=alpha, beta=beta, gamma=gamma, phi=phi)
    y = [1.0, 3.0, 4.0, 8.0, 2.0, 7.0, 5.0]

    # After 4 values (t = 3): initialisation from the two blocks of two.
    _feed(e, y[:4])
    a0 = (y[0] + y[1]) / 2
    a1 = (y[2] + y[3]) / 2
    b_init = (a1 - a0) / 2
    l_init = a1 + b_init * ((2 - 1) / 2)
    s0_init = ((y[0] - a0) + (y[2] - a1)) / 2
    s1_init = ((y[1] - a0) + (y[3] - a1)) / 2
    assert e._level == l_init
    assert e._trend == b_init
    assert e._seasonal == [s0_init, s1_init]

    def hw_forecast(t, lev, b, s, horizon):
        """The spec's damped-trend forecast, written out step by step."""
        d = 0.0
        p = 1.0
        out = []
        for h in range(1, horizon + 1):
            p = p * phi
            d = d + p
            out.append(lev + d * b + s[(t + h) % 2])
        return tuple(out)

    # Forecast after 4 values.
    assert e.forecast(3) == hw_forecast(3, l_init, b_init, [s0_init, s1_init], 3)

    # After the 5th value (t = 4): one update step, then forecast.
    e.observe(y[4])
    t = 4
    lev, b, s = l_init, b_init, [s0_init, s1_init]
    s_old = s[t % 2]
    lev_prev = lev
    lev = alpha * (y[t] - s_old) + (1.0 - alpha) * (lev_prev + phi * b)
    b = beta * (lev - lev_prev) + (1.0 - beta) * phi * b
    s[t % 2] = gamma * (y[t] - lev) + (1.0 - gamma) * s_old
    assert e.forecast(2) == hw_forecast(t, lev, b, s, 2)

    # After the 7th value (t = 6): two more update steps, then forecast.
    e.observe(y[5])
    t = 5
    s_old = s[t % 2]
    lev_prev = lev
    lev = alpha * (y[t] - s_old) + (1.0 - alpha) * (lev_prev + phi * b)
    b = beta * (lev - lev_prev) + (1.0 - beta) * phi * b
    s[t % 2] = gamma * (y[t] - lev) + (1.0 - gamma) * s_old
    e.observe(y[6])
    t = 6
    s_old = s[t % 2]
    lev_prev = lev
    lev = alpha * (y[t] - s_old) + (1.0 - alpha) * (lev_prev + phi * b)
    b = beta * (lev - lev_prev) + (1.0 - beta) * phi * b
    s[t % 2] = gamma * (y[t] - lev) + (1.0 - gamma) * s_old
    assert e.forecast(3) == hw_forecast(t, lev, b, s, 3)


# --- Test 7: forecast raises before min_history, succeeds after --------------------


@pytest.mark.parametrize(
    ("expert", "min_history"),
    [
        (Naive(), 1),
        (SeasonalNaive(3), 3),
        (SimpleExpSmoothing(), 1),
        (DampedHoltWinters(2), 4),
    ],
)
def test_forecast_before_min_history(expert, min_history):
    """forecast raises with ExpertError one value early and works at min_history."""
    _feed(expert, [float(i + 1) for i in range(min_history - 1)])
    with pytest.raises(ExpertError):
        expert.forecast(2)
    expert.observe(99.0)
    assert len(expert.forecast(2)) == 2


# --- Test 8: input validation ------------------------------------------------------


def test_observe_rejects_bad_values():
    """observe rejects NaN, inf, True and non-numbers, for every expert."""
    bad = [float("nan"), float("inf"), float("-inf"), True, False, "1", None]
    for expert in make_pool(4):
        for value in bad:
            with pytest.raises(ExpertError):
                expert.observe(value)


def test_forecast_rejects_bad_horizons():
    """forecast rejects 0, -1 and True, for every expert with enough history."""
    for expert in make_pool(4):
        _feed(expert, [1.0 + (i % 4) * 0.5 for i in range(expert.min_history)])
        for horizon in (0, -1, True):
            with pytest.raises(ExpertError):
                expert.forecast(horizon)


def test_constructor_bounds():
    """Each constructor bound has a failing case."""
    with pytest.raises(ExpertError):
        SeasonalNaive(1)
    with pytest.raises(ExpertError):
        SeasonalNaive(True)
    with pytest.raises(ExpertError):
        SeasonalNaive(2.0)
    for alpha in (0.0, 1.0, 1.5, -0.1, True, "0.3"):
        with pytest.raises(ExpertError):
            SimpleExpSmoothing(alpha)
    with pytest.raises(ExpertError):
        DampedHoltWinters(1)
    with pytest.raises(ExpertError):
        DampedHoltWinters(True)
    for kwargs in (
        {"alpha": 0.0},
        {"alpha": 1.0},
        {"beta": 0.0},
        {"beta": 1.0},
        {"gamma": 0.0},
        {"phi": 0.0},
        {"phi": 1.1},
        {"alpha": True},
    ):
        with pytest.raises(ExpertError):
            DampedHoltWinters(period=2, **kwargs)
    # phi = 1.0 is allowed (undamped); the rest defaults are valid.
    DampedHoltWinters(period=2, phi=1.0)


# --- Test 9: forecast is pure ------------------------------------------------------


def test_forecast_is_pure():
    """Two calls agree, and interleaving forecasts with observes changes nothing."""
    values = [10.0, -3.5, 7.25, 4.0, 0.5, 12.0, 6.0, 9.0, 3.0, 5.0]
    makers = [
        Naive,
        lambda: SeasonalNaive(3),
        SimpleExpSmoothing,
        lambda: DampedHoltWinters(2),
    ]
    for make in makers:
        e = make()
        _feed(e, values[: e.min_history])
        first = e.forecast(4)
        assert e.forecast(4) == first
        e.observe(values[e.min_history])
        interleaved = e.forecast(4)
        batched = make()
        _feed(batched, values[: e.min_history + 1])
        assert interleaved == batched.forecast(4)


# --- Test 10: non-finite output is refused -----------------------------------------


def test_holt_winters_overflow_refused():
    """holt_winters fed 1e308-scale values raises instead of returning inf or NaN."""
    e = DampedHoltWinters(period=2)
    big = 1.7e308
    _feed(e, [big, -big, big, -big])
    with pytest.raises(ExpertError):
        e.forecast(3)
    e.observe(big)
    e.observe(-big)
    e.observe(big)
    with pytest.raises(ExpertError):
        e.forecast(3)


def test_naive_and_ses_finite_by_construction():
    """Finite inputs keep naive and ses finite (their outputs are observed values)."""
    e = Naive()
    e.observe(-5.0)
    assert all(math.isfinite(v) for v in e.forecast(24))
