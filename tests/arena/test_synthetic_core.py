"""Core generator tests: spec validation, normals, simulation, fingerprints."""

import math

import pytest

from evolvets.arena.synthetic.simulate import (
    simulate,
    simulate_from_innovations,
    standard_normals,
    world_fingerprint,
)
from evolvets.arena.synthetic.spec import NoiseBurst, Regime, WorldSpec, WorldSpecError


def _regime(
    start=0,
    level=0.0,
    slope=0.0,
    season_amp=0.0,
    season_phase=0.0,
    ar_phi=0.5,
    sigma=1.0,
    transition="sudden",
    ramp=0,
    recurs_from=None,
):
    return Regime(
        start=start,
        level=level,
        slope=slope,
        season_amp=season_amp,
        season_phase=season_phase,
        ar_phi=ar_phi,
        sigma=sigma,
        transition=transition,
        ramp=ramp,
        recurs_from=recurs_from,
    )


def _spec(regimes, bursts=(), length=100, period=24):
    return WorldSpec(length=length, period=period, regimes=regimes, bursts=tuple(bursts))


def test_valid_multi_regime_spec_constructs():
    """One spec with a gradual ramp, a recurring regime and two bursts is valid."""
    base = _regime(level=1.0, slope=0.01, season_amp=2.0, season_phase=0.3, ar_phi=0.5, sigma=0.8)
    recurring_level = 1.0 + 0.01 * (70 - 0)
    spec = _spec(
        [
            base,
            _regime(start=30, level=3.0, slope=-0.02, transition="gradual", ramp=10),
            _regime(
                start=70,
                level=recurring_level,
                slope=base.slope,
                season_amp=base.season_amp,
                season_phase=base.season_phase,
                ar_phi=base.ar_phi,
                sigma=base.sigma,
                recurs_from=0,
            ),
        ],
        bursts=[
            NoiseBurst(start=40, length=10, sigma_mult=2.0),
            NoiseBurst(start=85, length=5, sigma_mult=3.0),
        ],
        length=100,
    )
    assert spec.length == 100
    assert spec.bursts[1].start == 85


# --- rule 1: first regime anchors the world ---


def test_empty_regimes_rejected():
    with pytest.raises(WorldSpecError, match="non-empty"):
        _spec([])


def test_first_regime_start_must_be_zero():
    with pytest.raises(WorldSpecError, match="regimes\\[0\\]"):
        _spec([_regime(start=1)])


def test_first_regime_must_be_sudden():
    with pytest.raises(WorldSpecError, match="regimes\\[0\\]"):
        _spec([_regime(transition="gradual", ramp=10)])


def test_first_regime_ramp_must_be_zero():
    with pytest.raises(WorldSpecError, match="ramp"):
        _spec([_regime(ramp=3)])


def test_first_regime_recurs_from_must_be_none():
    with pytest.raises(WorldSpecError, match="recurs_from"):
        _spec([_regime(recurs_from=0)])


# --- rule 2: increasing starts inside the world ---


def test_regime_starts_must_increase():
    with pytest.raises(WorldSpecError, match="strictly increasing"):
        _spec([_regime(), _regime(start=30), _regime(start=30)])


def test_regime_start_must_be_inside_world():
    with pytest.raises(WorldSpecError, match="< length"):
        _spec([_regime(), _regime(start=100)], length=100)


# --- rule 3: ramps may not overlap the next change ---


def test_ramp_may_not_overlap_next_change():
    with pytest.raises(WorldSpecError, match="ramp"):
        _spec([_regime(), _regime(start=50, transition="gradual", ramp=60), _regime(start=100)])


def test_last_regime_ramp_must_end_inside_world():
    with pytest.raises(WorldSpecError, match="ramp"):
        _spec([_regime(), _regime(start=50, transition="gradual", ramp=100)])


# --- rule 4: recurring regimes copy exactly ---


def test_recurs_from_must_reference_earlier_regime():
    with pytest.raises(WorldSpecError, match="smaller index"):
        _spec([_regime(), _regime(start=30, recurs_from=1)])


def test_recurring_regime_parameter_must_match():
    base = _regime(slope=0.01, sigma=0.8)
    with pytest.raises(WorldSpecError, match="slope"):
        _spec([base, _regime(start=30, slope=0.02, sigma=0.8, recurs_from=0)])


def test_recurring_regime_level_must_be_on_trend_line():
    base = _regime(level=1.0, slope=0.01)
    with pytest.raises(WorldSpecError, match="level"):
        _spec([base, _regime(start=30, level=1.29, slope=0.01, recurs_from=0)])


# --- rule 5: bursts sorted, disjoint, inside the world ---


def test_bursts_must_be_sorted_and_disjoint():
    with pytest.raises(WorldSpecError, match="sorted"):
        _spec(
            [_regime()],
            bursts=[
                NoiseBurst(start=20, length=10, sigma_mult=2.0),
                NoiseBurst(start=15, length=5, sigma_mult=2.0),
            ],
        )


def test_bursts_must_not_overlap():
    with pytest.raises(WorldSpecError, match="sorted"):
        _spec(
            [_regime()],
            bursts=[
                NoiseBurst(start=20, length=10, sigma_mult=2.0),
                NoiseBurst(start=25, length=5, sigma_mult=2.0),
            ],
        )


def test_burst_must_be_inside_world():
    with pytest.raises(WorldSpecError, match="inside"):
        _spec([_regime()], bursts=[NoiseBurst(start=95, length=10, sigma_mult=2.0)], length=100)


# --- rule 6 and field-level checks ---


def test_bools_rejected_for_int_fields():
    """True is not a valid int anywhere an index or count is expected."""
    with pytest.raises(WorldSpecError, match="int"):
        _regime(start=True)
    with pytest.raises(WorldSpecError, match="int"):
        _regime(ramp=True)
    with pytest.raises(WorldSpecError, match="int"):
        WorldSpec(length=True, period=24, regimes=(_regime(),), bursts=())
    with pytest.raises(WorldSpecError, match="int"):
        NoiseBurst(start=True, length=5, sigma_mult=2.0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"level": math.nan}, "finite"),
        ({"level": math.inf}, "finite"),
        ({"sigma": 0.0}, "sigma"),
        ({"sigma": -1.0}, "sigma"),
        ({"ar_phi": 1.0}, "ar_phi"),
        ({"ar_phi": -1.0}, "ar_phi"),
        ({"season_amp": -0.1}, "season_amp"),
        ({"season_phase": 2 * math.pi}, "season_phase"),
        ({"season_phase": -0.1}, "season_phase"),
    ],
)
def test_bad_regime_fields_rejected(kwargs, match):
    with pytest.raises(WorldSpecError, match=match):
        _regime(**kwargs)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"length": 1}, "length"),
        ({"period": 1}, "period"),
    ],
)
def test_bad_world_fields_rejected(kwargs, match):
    with pytest.raises(WorldSpecError, match=match):
        _spec([_regime()], **kwargs)


def test_bad_burst_multiplier_rejected():
    with pytest.raises(WorldSpecError, match="sigma_mult"):
        NoiseBurst(start=10, length=5, sigma_mult=1.0)


# --- standard_normals ---


def test_normals_same_seed_identical():
    assert standard_normals(7, 16) == standard_normals(7, 16)


def test_normals_different_seeds_differ():
    assert standard_normals(7, 16) != standard_normals(8, 16)


@pytest.mark.parametrize("seed", [-1, 2**63])
def test_normals_seed_out_of_range(seed):
    with pytest.raises(ValueError, match="seed"):
        standard_normals(seed, 4)


def test_normals_bool_seed_rejected():
    with pytest.raises(ValueError, match="seed"):
        standard_normals(True, 4)


def test_normals_zero_n_returns_empty():
    assert standard_normals(0, 0) == ()


def test_normals_moments():
    """Over 200k draws the sample mean and variance match N(0, 1)."""
    values = standard_normals(12345, 200_000)
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    assert abs(mean) < 0.01
    assert abs(var - 1.0) < 0.02


# --- simulate_from_innovations input checks ---


def test_simulate_rejects_wrong_length_z():
    spec = _spec([_regime()], length=10)
    with pytest.raises(ValueError, match="length"):
        simulate_from_innovations(spec, (0.0,) * 9)


def test_simulate_rejects_nan_z():
    spec = _spec([_regime()], length=3)
    with pytest.raises(ValueError, match="finite"):
        simulate_from_innovations(spec, (0.0, math.nan, 0.0))


# --- mean path ---


TWO_REGIMES = [
    _regime(level=1.0, slope=0.01, season_amp=1.5, season_phase=0.2, ar_phi=0.5, sigma=0.8),
    _regime(
        start=40, level=3.0, slope=-0.02, season_amp=0.5, season_phase=1.0, ar_phi=0.2, sigma=1.2
    ),
]


def test_mean_path_sudden_change_is_exact():
    """At the boundary the mean equals the regime's own formula, exactly."""
    spec = _spec(TWO_REGIMES, length=100)
    world = simulate(spec, 99)
    r = 1
    t = spec.regimes[r].start
    m_prev = spec.mean_at(r - 1, t - 1)
    m_curr = spec.mean_at(r, t)
    assert world.mean[t - 1] == m_prev
    assert world.mean[t] == m_curr


def test_mean_path_gradual_ramp_alphas():
    """With ramp = 4 the blend factor rises 1/5..4/5, then the mean is m_r exactly."""
    spec = _spec(
        [
            _regime(level=1.0, slope=0.01),
            _regime(start=40, level=3.0, slope=-0.02, transition="gradual", ramp=4),
        ],
        length=100,
    )
    world = simulate(spec, 99)
    r = 1
    start = spec.regimes[r].start
    for j in range(4):
        t = start + j
        alpha = (j + 1) / 5
        blended = (1 - alpha) * spec.mean_at(r - 1, t) + alpha * spec.mean_at(r, t)
        assert world.mean[t] == blended
    assert world.mean[start + 4] == spec.mean_at(r, start + 4)


# --- noise ---


def test_noise_statistics_stationary():
    """A stationary AR(1) has the closed-form variance and lag-1 autocorrelation."""
    phi, sigma = 0.6, 0.5
    spec = _spec([_regime(ar_phi=phi, sigma=sigma)], length=200_000)
    world = simulate(spec, 2026)
    e = world.noise
    n = len(e)
    mean = sum(e) / n
    var = sum((v - mean) ** 2 for v in e) / n
    lag1 = sum((e[i] - mean) * (e[i + 1] - mean) for i in range(n - 1)) / (n - 1) / var
    assert var == pytest.approx(sigma**2 / (1 - phi**2), rel=0.05)
    assert lag1 == pytest.approx(phi, abs=0.02)


def test_bursts_change_only_noise():
    """A burst scales sigma inside its span and touches nothing else."""
    base = _regime(ar_phi=0.5, sigma=0.8)
    with_burst = _spec([base], bursts=[NoiseBurst(start=20, length=10, sigma_mult=3.0)], length=100)
    without = _spec([base], length=100)
    w1 = simulate(with_burst, 42)
    w0 = simulate(without, 42)
    assert w1.mean == w0.mean
    assert w1.noise[:20] == w0.noise[:20]
    mult = 3.0
    for t in range(100):
        if 20 <= t < 30:
            assert w1.sigma[t] == base.sigma * mult
        else:
            assert w1.sigma[t] == w0.sigma[t]
    assert w1.burst_spans == ((20, 30),)


# --- determinism and fingerprints ---


def test_simulate_deterministic():
    spec = _spec(TWO_REGIMES, length=100)
    w1 = simulate(spec, 5)
    w2 = simulate(spec, 5)
    assert w1 == w2
    assert world_fingerprint(w1) == world_fingerprint(w2)
    assert world_fingerprint(w1) != world_fingerprint(simulate(spec, 6))


def test_golden_fingerprint():
    """Pinned fingerprint of a fixed spec at seed 12345.

    Changing this value means the generator's output changed and
    GENERATOR_VERSION must be bumped.
    """
    spec = _spec(TWO_REGIMES, length=100)
    world = simulate(spec, 12345)
    assert world_fingerprint(world) == _GOLDEN


_GOLDEN = "a14f3c397c90e60acadd0cc02f90b1848cf3a533a7f2feebfc63f3c54aa1dd3b"


def test_world_metadata_matches_spec():
    spec = _spec(
        [TWO_REGIMES[0], TWO_REGIMES[1]],
        bursts=[NoiseBurst(start=60, length=5, sigma_mult=2.0)],
        length=100,
    )
    world = simulate(spec, 1)
    assert world.seed == 1
    assert world.change_points == (40,)
    assert world.burst_spans == ((60, 65),)
