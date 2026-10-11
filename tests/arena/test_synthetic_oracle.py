"""Oracle tests: closed form, no look-ahead, boundaries, calibration, ceiling."""

import math

import pytest

from evolvets.arena.synthetic.oracle import true_best_error_variance, true_best_forecast
from evolvets.arena.synthetic.simulate import (
    simulate,
    simulate_from_innovations,
    standard_normals,
)
from evolvets.arena.synthetic.spec import TWO_PI, NoiseBurst, Regime, WorldSpec


def _m(spec, r, t):
    """Regime r's own mean function, mirroring the simulator's expression."""
    reg = spec.regimes[r]
    return (
        reg.level
        + reg.slope * (t - reg.start)
        + reg.season_amp * math.sin(TWO_PI * t / spec.period + reg.season_phase)
    )


def _regime_at(spec, t):
    index = 0
    for i, reg in enumerate(spec.regimes):
        if reg.start <= t:
            index = i
        else:
            break
    return index


def _sigma_at(spec, t):
    """Innovation std at t from the spec, bursts included."""
    reg = spec.regimes[_regime_at(spec, t)]
    for burst in spec.bursts:
        if burst.start <= t < burst.start + burst.length:
            return reg.sigma * burst.sigma_mult
    return reg.sigma


def _two_regime_spec():
    return WorldSpec(
        length=100,
        period=24,
        regimes=(
            Regime(
                start=0,
                level=1.0,
                slope=0.01,
                season_amp=1.5,
                season_phase=0.2,
                ar_phi=0.5,
                sigma=0.8,
                transition="sudden",
                ramp=0,
                recurs_from=None,
            ),
            Regime(
                start=40,
                level=3.0,
                slope=-0.02,
                season_amp=0.5,
                season_phase=1.0,
                ar_phi=0.2,
                sigma=1.2,
                transition="sudden",
                ramp=0,
                recurs_from=None,
            ),
        ),
        bursts=(NoiseBurst(start=60, length=10, sigma_mult=2.5),),
    )


def test_forecast_matches_closed_form():
    """The forecast is mu_{t+h} + prod(phi) e_t, including across a sudden change."""
    spec = _two_regime_spec()
    world = simulate(spec, 7)
    for origin in (0, 39, 70):
        horizon = 24
        got = true_best_forecast(world, origin, horizon)
        phi_prod = 1.0
        e_t = world.noise[origin]
        for h in range(1, horizon + 1):
            t_h = origin + h
            phi_prod *= spec.regimes[_regime_at(spec, t_h)].ar_phi
            expected = _m(spec, _regime_at(spec, t_h), t_h) + phi_prod * e_t
            assert got[h - 1] == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_forecast_crosses_sudden_change():
    """From an origin just before a change, later steps use the new regime's mean."""
    spec = _two_regime_spec()
    world = simulate(spec, 7)
    origin, horizon = 39, 10
    got = true_best_forecast(world, origin, horizon)
    # Step 40 is already in regime 1, so its mean and phi are the new regime's.
    assert got[0] == pytest.approx(_m(spec, 1, 40) + 0.2 * world.noise[39], rel=1e-12)
    assert got[9] == pytest.approx(_m(spec, 1, 49) + 0.2**10 * world.noise[39], rel=1e-12)


def test_error_variance_matches_closed_form():
    spec = _two_regime_spec()
    world = simulate(spec, 7)
    origin, horizon = 50, 24
    got = true_best_error_variance(world, origin, horizon)
    for h in range(1, horizon + 1):
        terms = []
        for j in range(1, h + 1):
            phi_sq = 1.0
            for k in range(j + 1, h + 1):
                phi_sq *= spec.regimes[_regime_at(spec, origin + k)].ar_phi ** 2
            terms.append(_sigma_at(spec, origin + j) ** 2 * phi_sq)
        assert got[h - 1] == pytest.approx(sum(terms), rel=1e-12)


def test_no_look_ahead_beyond_origin():
    """Replacing innovations after the origin leaves the forecast unchanged."""
    spec = _two_regime_spec()
    z = standard_normals(11, spec.length)
    world = simulate_from_innovations(spec, z)
    t = 45
    z_changed = tuple(z[: t + 1]) + tuple(v + 3.0 for v in z[t + 1 :])
    world_changed = simulate_from_innovations(spec, z_changed)
    assert world_changed.noise[t] == world.noise[t]
    assert true_best_forecast(world_changed, t, 24) == true_best_forecast(world, t, 24)
    assert true_best_error_variance(world_changed, t, 24) == true_best_error_variance(world, t, 24)


def test_origin_boundaries():
    spec = _two_regime_spec()
    world = simulate(spec, 3)
    horizon = 24
    assert len(true_best_forecast(world, 75, horizon)) == horizon
    with pytest.raises(ValueError, match="origin"):
        true_best_forecast(world, 76, horizon)
    with pytest.raises(ValueError, match="origin"):
        true_best_forecast(world, -1, horizon)
    with pytest.raises(ValueError, match="horizon"):
        true_best_forecast(world, 10, 0)
    with pytest.raises(ValueError, match="origin"):
        true_best_forecast(world, True, horizon)
    with pytest.raises(ValueError, match="horizon"):
        true_best_forecast(world, 10, True)


def _big_world():
    """Two regimes with persistent AR noise and one burst, 200k steps."""
    spec = WorldSpec(
        length=200_000,
        period=24,
        regimes=(
            Regime(
                start=0,
                level=0.0,
                slope=0.001,
                season_amp=1.0,
                season_phase=0.3,
                ar_phi=0.9,
                sigma=1.0,
                transition="sudden",
                ramp=0,
                recurs_from=None,
            ),
            Regime(
                start=100_000,
                level=5.0,
                slope=-0.001,
                season_amp=0.5,
                season_phase=1.0,
                ar_phi=0.9,
                sigma=1.0,
                transition="sudden",
                ramp=0,
                recurs_from=None,
            ),
        ),
        bursts=(NoiseBurst(start=50_000, length=50, sigma_mult=3.0),),
    )
    return simulate(spec, 99)


def test_calibration_of_error_variance():
    """Empirical oracle MSE per horizon matches the predicted variance."""
    world = _big_world()
    y = world.y
    horizon = 24
    last_origin = len(y) - 1 - horizon
    sq_sum = {h: 0.0 for h in (1, 6, 24)}
    var_sum = {h: 0.0 for h in (1, 6, 24)}
    for t in range(last_origin + 1):
        forecast = true_best_forecast(world, t, horizon)
        predicted = true_best_error_variance(world, t, horizon)
        for h in (1, 6, 24):
            sq_sum[h] += (y[t + h] - forecast[h - 1]) ** 2
            var_sum[h] += predicted[h - 1]
    for h in (1, 6, 24):
        empirical = sq_sum[h] / (last_origin + 1)
        predicted_mean = var_sum[h] / (last_origin + 1)
        assert empirical == pytest.approx(predicted_mean, rel=0.05), f"h = {h}"


def test_oracle_beats_naive_and_mean_only():
    """Oracle MAE is below naive and mean-only MAE at every horizon."""
    world = _big_world()
    y = world.y
    mean = world.mean
    horizon = 24
    last_origin = len(y) - 1 - horizon
    oracle_abs = [0.0] * horizon
    naive_abs = [0.0] * horizon
    mean_only_abs = [0.0] * horizon
    for t in range(last_origin + 1):
        forecast = true_best_forecast(world, t, horizon)
        y_t = y[t]
        for h in range(1, horizon + 1):
            actual = y[t + h]
            oracle_abs[h - 1] += abs(actual - forecast[h - 1])
            naive_abs[h - 1] += abs(actual - y_t)
            mean_only_abs[h - 1] += abs(actual - mean[t + h])
    for h in range(1, horizon + 1):
        assert oracle_abs[h - 1] < naive_abs[h - 1], f"h = {h}"
        assert oracle_abs[h - 1] < mean_only_abs[h - 1], f"h = {h}"
