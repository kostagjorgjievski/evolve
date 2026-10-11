"""RidgeAR tests: independent-solver agreement, schedule independence, edge paths (3-8)."""

import math

import pytest

from evolvets.experts.interfaces import ExpertError
from evolvets.experts.ridge import RidgeAR

TWO_PI = 2.0 * math.pi


def _feed(expert, values):
    for v in values:
        expert.observe(v)


def _close(a, b):
    """Agreement within 1e-12 relative (with an absolute floor near zero)."""
    return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))


def _solve_gauss_jordan(a, b):
    """Independent 6x6 solver: reduce [A|b] to the identity."""
    n = len(b)
    m = [[*row, b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = col
        best = abs(m[col][col])
        for r in range(col + 1, n):
            if abs(m[r][col]) > best:
                best = abs(m[r][col])
                pivot = r
        m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        m[col] = [v / div for v in m[col]]
        for r in range(n):
            if r != col:
                factor = m[r][col]
                m[r] = [m[r][c] - factor * m[col][c] for c in range(n + 1)]
    return [m[i][n] for i in range(n)]


def _fit_reference(values, t, period, window, ridge):
    """Fit (mu, sd, w) exactly as the spec prescribes, but with the test's own code."""
    m, fit_len = period, window + 2 * period
    data = values[t - fit_len + 1 : t + 1]
    # Left-to-right loops, never builtin sum(): sum() switched to compensated
    # summation for floats in Python 3.12, so an exact == against the
    # implementation would depend on the interpreter version.
    mu = 0.0
    for v in data:
        mu = mu + v
    mu = mu / len(data)
    sq = 0.0
    for v in data:
        sq = sq + (v - mu) * (v - mu)
    sd = math.sqrt(sq / len(data))
    if sd == 0.0:
        sd = 1.0
    z = {(t - fit_len + 1 + i): (v - mu) / sd for i, v in enumerate(data)}

    def z_at(i):
        return z[i]

    rows = []
    targets = []
    for s in range(t - window + 1, t + 1):
        rows.append([1.0, z_at(s - 1), z_at(s - 2), z_at(s - 3), z_at(s - m), z_at(s - 2 * m)])
        targets.append(z_at(s))
    # Accumulate row by row, in the same order as the implementation.
    a = [[0.0] * 6 for _ in range(6)]
    b = [0.0] * 6
    for row, target in zip(rows, targets, strict=True):
        for i in range(6):
            b[i] = b[i] + row[i] * target
            for j in range(6):
                a[i][j] = a[i][j] + row[i] * row[j]
    for i in range(1, 6):
        a[i][i] += ridge
    return mu, sd, _solve_gauss_jordan(a, b)


def _forecast_reference(values, t, model, period, horizon):
    """The recursive multi-step forecast, using a model fitted earlier."""
    mu, sd, w = model
    m = period
    z = {}
    for i in range(t - 2 * m + 1, t + 1):
        z[i] = (values[i] - mu) / sd
    out = []
    for h in range(1, horizon + 1):
        target = t + h
        x = [
            1.0,
            z[target - 1],
            z[target - 2],
            z[target - 3],
            z[target - m],
            z[target - 2 * m],
        ]
        z_hat = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3] + w[4] * x[4] + w[5] * x[5]
        z[target] = z_hat
        out.append(mu + sd * z_hat)
    return tuple(out)


# --- Test 3: agreement with an independent solver, and the refit schedule ----------


def test_ridge_matches_independent_solver():
    """Fits and forecasts at t0, t0+1 and t0+3 agree with the test's own solver."""
    period, window, refit_every, ridge = 2, 6, 3, 1.0
    t0 = window + 2 * period - 1
    values = [3.0 * math.sin(0.7 * i) + (i % 4) * 0.8 + 0.05 * i for i in range(t0 + 4)]

    e = RidgeAR(period=period, window=window, refit_every=refit_every, ridge=ridge)
    _feed(e, values[: t0 + 1])
    mu0, sd0, w0 = _fit_reference(values, t0, period, window, ridge)
    assert e._mu == mu0 and e._sd == sd0
    assert all(_close(got, ref) for got, ref in zip(e._w, w0, strict=True))
    f = e.forecast(5)
    ref = _forecast_reference(values, t0, (mu0, sd0, w0), period, 5)
    assert all(_close(got, r) for got, r in zip(f, ref, strict=True))
    w_t0 = list(e._w)

    # t0 + 1: same model (no refit), fresh lag inputs.
    e.observe(values[t0 + 1])
    assert e._w == w_t0
    f = e.forecast(5)
    ref = _forecast_reference(values, t0 + 1, (mu0, sd0, w0), period, 5)
    assert all(_close(got, r) for got, r in zip(f, ref, strict=True))

    # t0 + 2: still the t0 fit; t0 + 3: a new fit on the schedule.
    e.observe(values[t0 + 2])
    assert e._w == w_t0
    e.observe(values[t0 + 3])
    assert e._w != w_t0
    mu3, sd3, w3 = _fit_reference(values, t0 + 3, period, window, ridge)
    assert e._mu == mu3 and e._sd == sd3
    assert all(_close(got, ref) for got, ref in zip(e._w, w3, strict=True))


# --- Test 4: refit schedule independence -------------------------------------------


def test_refit_schedule_independence():
    """Forecasting at earlier origins never changes a later forecast."""
    values = [math.sin(0.11 * i) * 4.0 + (i % 24) * 0.3 + i * 0.002 for i in range(400)]
    t0 = 240 + 2 * 24 - 1
    for t in range(t0, t0 + 61):
        chatty = RidgeAR(period=24)
        quiet = RidgeAR(period=24)
        for i in range(t + 1):
            chatty.observe(values[i])
            quiet.observe(values[i])
            if i >= t0:
                chatty.forecast(24)
        assert chatty.forecast(24) == quiet.forecast(24), f"origin {t}"


# --- Test 5: constant series (sd = 0 path) ------------------------------------------


def test_constant_series_forecasts_the_constant():
    """With sd = 0 the model degenerates to the mean, so forecasts equal the constant."""
    e = RidgeAR(period=2, window=6, refit_every=3)
    _feed(e, [12.5] * 12)
    assert e.forecast(4) == (12.5, 12.5, 12.5, 12.5)


# --- Test 6: recovers a noiseless linear recurrence ---------------------------------


def test_recovers_noiseless_recurrence():
    """One-step forecasts of a pure sinusoid via its AR(2) recurrence stay within 1e-4."""
    values = [math.sin(TWO_PI * s / 24.0) for s in range(600)]
    for t in range(549, 599):  # the last 50 origins whose targets exist in the series
        e = RidgeAR(period=24, ridge=1e-6)
        _feed(e, values[: t + 1])
        got = e.forecast(1)[0]
        assert abs(got - values[t + 1]) < 1e-4, f"origin {t}: {got} vs {values[t + 1]}"


# --- Tests 7 and 8: min_history and constructor bounds ------------------------------


def test_forecast_before_min_history():
    """forecast raises one value before min_history and succeeds at it."""
    e = RidgeAR(period=2, window=2)
    assert e.min_history == 2 + 2 * 2
    _feed(e, [float(i + 1) for i in range(e.min_history - 1)])
    with pytest.raises(ExpertError):
        e.forecast(2)
    e.observe(7.0)
    assert len(e.forecast(2)) == 2


def test_constructor_bounds():
    """Each RidgeAR bound has a failing case."""
    for period in (1, True, 2.0):
        with pytest.raises(ExpertError):
            RidgeAR(period=period)
    for kwargs in (
        {"window": 1},
        {"window": True},
        {"refit_every": 0},
        {"refit_every": True},
        {"ridge": 0.0},
        {"ridge": -1.0},
        {"ridge": float("inf")},
        {"ridge": True},
    ):
        with pytest.raises(ExpertError):
            RidgeAR(period=2, **kwargs)
