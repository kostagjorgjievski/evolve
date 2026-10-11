"""Grammar tests: family structure, placement, change kinds, config validation."""

import json
import math
from itertools import pairwise

import pytest

from evolvets.arena.synthetic.grammar import (
    CHANGE_KINDS,
    FAMILIES,
    GrammarConfig,
    GrammarError,
    SyntheticTask,
    config_to_dict,
    make_task,
)
from evolvets.arena.synthetic.seeds import world_seed
from evolvets.arena.synthetic.simulate import world_fingerprint
from evolvets.arena.synthetic.spec import TWO_PI

DEFAULT = GrammarConfig()
N_SEEDS = 50

_EXPECTED_DIFFERENCES = {
    "level_shift": set(),
    "slope_change": {"slope"},
    "season_change": {"season_amp", "season_phase"},
    "noise_change": {"ar_phi", "sigma"},
}


def _tasks(family, n=N_SEEDS, config=DEFAULT):
    return [make_task(family, seed, config) for seed in range(n)]


def _trend(world, spec, t):
    """Mean minus the active regime's seasonal term at ``t``."""
    regime = spec.regimes[spec.regime_at(t)]
    seasonal = regime.season_amp * math.sin(TWO_PI * t / spec.period + regime.season_phase)
    return world.mean[t] - seasonal


# --- task group 1: every family samples valid tasks ---


@pytest.mark.parametrize("family", FAMILIES)
def test_every_seed_samples_a_valid_task(family):
    """For 200 seeds per family the sampled spec passes WorldSpec validation."""
    for seed in range(200):
        task = make_task(family, seed)
        assert len(task.change_kinds) == len(task.world.change_points)


# --- task group 2: family structure ---


def test_stationary_structure():
    """Stationary: one regime, no bursts, no changes."""
    for task in _tasks("stationary"):
        spec = task.world.spec
        assert len(spec.regimes) == 1
        assert spec.bursts == ()
        assert task.world.change_points == ()
        assert task.change_kinds == ()


def test_sudden_structure():
    """Sudden: 1..3 changes, all sudden transitions."""
    for task in _tasks("sudden"):
        spec = task.world.spec
        assert 1 <= len(task.world.change_points) <= DEFAULT.max_changes
        assert all(regime.transition == "sudden" for regime in spec.regimes)
        assert task.change_kinds == () or all(kind in CHANGE_KINDS for kind in task.change_kinds)


def test_gradual_structure():
    """Gradual: all transitions gradual with ramps in ramp_range."""
    for task in _tasks("gradual"):
        spec = task.world.spec
        assert 1 <= len(task.world.change_points) <= DEFAULT.max_changes
        for regime in spec.regimes[1:]:
            assert regime.transition == "gradual"
            assert DEFAULT.ramp_range[0] <= regime.ramp <= DEFAULT.ramp_range[1]


def test_recurring_structure():
    """Recurring: alternating A/B, exact copies, back on the referenced mean path."""
    for task in _tasks("recurring"):
        spec = task.world.spec
        regimes = spec.regimes
        assert 2 <= len(task.world.change_points) <= max(2, DEFAULT.max_changes)
        assert task.change_kinds[0] in CHANGE_KINDS
        assert all(kind == "recurrence" for kind in task.change_kinds[1:])
        for index, regime in enumerate(regimes):
            assert regime.transition == "sudden"
            if index < 2:
                assert regime.recurs_from is None
                continue
            source = 0 if index % 2 == 0 else 1
            assert regime.recurs_from == source
            ref = regimes[source]
            for name in ("slope", "season_amp", "season_phase", "ar_phi", "sigma"):
                assert getattr(regime, name) == getattr(ref, name)
            assert regime.level == ref.level + ref.slope * (regime.start - ref.start)
            end = regimes[index + 1].start if index + 1 < len(regimes) else spec.length
            for t in range(regime.start, end):
                assert abs(task.world.mean[t] - spec.mean_at(source, t)) <= 1e-9


def test_false_drift_structure():
    """False drift: one regime, 1..3 bursts, no changes."""
    for task in _tasks("false_drift"):
        spec = task.world.spec
        assert len(spec.regimes) == 1
        assert 1 <= len(spec.bursts) <= DEFAULT.max_bursts
        assert task.world.change_points == ()
        assert task.change_kinds == ()


def test_mixed_structure():
    """Mixed: 1..3 changes with sudden or gradual transitions, 0..3 bursts."""
    for task in _tasks("mixed"):
        spec = task.world.spec
        assert 1 <= len(task.world.change_points) <= DEFAULT.max_changes
        assert 0 <= len(spec.bursts) <= DEFAULT.max_bursts
        for regime in spec.regimes[1:]:
            if regime.transition == "gradual":
                assert DEFAULT.ramp_range[0] <= regime.ramp <= DEFAULT.ramp_range[1]
            else:
                assert regime.transition == "sudden" and regime.ramp == 0
        assert all(kind in CHANGE_KINDS for kind in task.change_kinds)


# --- task group 3: placement ---


@pytest.mark.parametrize("family", FAMILIES)
def test_events_are_placed_by_the_slotting_rule(family):
    """Events sit at or after warmup and min_segment apart within their kind."""
    for task in _tasks(family):
        points = task.world.change_points
        for point in points:
            assert point >= DEFAULT.warmup
        for earlier, later in pairwise(points):
            assert later - earlier >= DEFAULT.min_segment
        starts = [start for start, _ in task.world.burst_spans]
        for start in starts:
            assert start >= DEFAULT.warmup
        for earlier, later in pairwise(starts):
            assert later - earlier >= DEFAULT.min_segment


# --- task group 4: change kinds and the mean path ---


def test_change_kinds_move_exactly_the_named_parameters():
    """The trend is continuous except for level_shift; only named parameters differ."""
    for task in _tasks("sudden"):
        world = task.world
        spec = world.spec
        for index, kind in enumerate(task.change_kinds):
            regime, prev = spec.regimes[index + 1], spec.regimes[index]
            start = regime.start
            jump = _trend(world, spec, start) - _trend(world, spec, start - 1)
            if kind == "level_shift":
                unit = prev.sigma / math.sqrt(1.0 - prev.ar_phi**2)
                magnitude = abs(jump - prev.slope) / unit
                assert DEFAULT.shift_range[0] - 1e-9 <= magnitude
                assert magnitude <= DEFAULT.shift_range[1] + 1e-9
            else:
                assert jump == pytest.approx(prev.slope, abs=1e-9)
            # The re-anchored level: within 1e-9 of the previous trend line at
            # start, once a drawn level shift is subtracted. The subtraction
            # does not exactly invert the re-anchoring rounding.
            reanchored = regime.level - prev.slope * (start - prev.start)
            if kind == "level_shift":
                reanchored -= jump - prev.slope
            assert reanchored == pytest.approx(prev.level, abs=1e-9)
            differs = {
                name
                for name in ("slope", "season_amp", "season_phase", "ar_phi", "sigma")
                if getattr(regime, name) != getattr(prev, name)
            }
            assert differs == _EXPECTED_DIFFERENCES[kind]
            if kind == "slope_change":
                assert DEFAULT.new_slope_range[0] <= regime.slope <= DEFAULT.new_slope_range[1]
            if kind == "season_change":
                assert DEFAULT.season_amp_range[0] <= regime.season_amp
                assert regime.season_amp <= DEFAULT.season_amp_range[1]
                assert 0.0 <= regime.season_phase < TWO_PI
            if kind == "noise_change":
                assert DEFAULT.phi_range[0] <= regime.ar_phi <= DEFAULT.phi_range[1]
                assert DEFAULT.sigma_range[0] <= regime.sigma <= DEFAULT.sigma_range[1]


# --- task group 5: determinism ---


def test_make_task_deterministic_and_seed_sensitive():
    task = make_task("mixed", 123)
    same = make_task("mixed", 123)
    other = make_task("mixed", 124)
    assert task == same
    assert world_fingerprint(task.world) == world_fingerprint(same.world)
    assert world_fingerprint(task.world) != world_fingerprint(other.world)


# --- task group 6: golden fingerprints ---


@pytest.mark.parametrize("family", FAMILIES)
def test_golden_family_fingerprint(family):
    """Pinned fingerprint of the first dev task of each family.

    Changing any of these values means the sampler's output changed and
    GENERATOR_VERSION must be bumped.
    """
    task = make_task(family, world_seed("dev", family, 0))
    assert world_fingerprint(task.world) == _GOLDEN_FINGERPRINTS[family]


_GOLDEN_FINGERPRINTS = {
    "stationary": "6f5a2dcc4261d43177095bada9bdc71ae274a329c37640d71b42703bc57565b4",
    "sudden": "19b02db4b7a1b297a41eadbb2372cb270f664f337005c98453176079f5798a48",
    "gradual": "5af1d744577602a1073a846d62cb5c5a9fdad5a66b1e75cd556ad30230e18b1c",
    "recurring": "257b54ad4c4c281ab713c1f834b6267df5afaf6dac1bcf9f81185072a9a01e86",
    "false_drift": "08e5e622fef4b7d9fe9679f12b9c81d7fd6f81e86f2d11f0edd2597d3b35de84",
    "mixed": "5521cb9f8368404b9e40b3326a3b801b289fd5abb42d112db0959bc47625072b",
}


# --- task group 7: GrammarConfig validation ---


def test_default_config_constructs():
    assert GrammarConfig().length == 2000


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"level_range": (5.0, -5.0)}, "lo <= hi"),
        ({"ramp_range": (150, 20)}, "lo <= hi"),
        ({"ramp_range": (1, 150)}, r"ramp_range\[0\]"),
        ({"ramp_range": (20, 200)}, r"ramp_range\[1\]"),
        ({"max_changes": 0}, "max_changes"),
        ({"min_segment": 567}, "length - warmup"),
        ({"burst_length_range": (10, 200)}, "burst_length_range"),
        ({"burst_mult_range": (1.0, 4.0)}, "burst_mult_range"),
        ({"sigma_range": (0.0, 1.0)}, "sigma_range"),
        ({"phi_range": (0.0, 1.0)}, "phi_range"),
        ({"phi_range": (-1.0, 0.9)}, "phi_range"),
        ({"season_amp_range": (-0.1, 3.0)}, "season_amp_range"),
        ({"level_range": (math.nan, 5.0)}, "finite"),
        ({"level_range": (-math.inf, 5.0)}, "finite"),
    ],
)
def test_bad_configs_rejected(kwargs, match):
    with pytest.raises(GrammarError, match=match):
        GrammarConfig(**kwargs)


# --- task group 8: a non-default config ---


def test_non_default_config_works():
    """A shorter, tighter config samples valid tasks; over-fitting it raises.

    The task file gives this config as length=800, warmup=100,
    min_segment=150, max_changes=4, max_bursts=4 with the default
    ramp_range. That exact config violates the task's own rule
    ``ramp_range[1] < min_segment`` (default ramp_range[1] == 150), so
    ramp_range is narrowed to (20, 140) here; the slotting arithmetic the
    task cites is unchanged (700 // 4 = 175 >= 150, 700 // 5 = 140 < 150).
    """
    kwargs = dict(
        length=800,
        warmup=100,
        min_segment=150,
        max_changes=4,
        max_bursts=4,
        ramp_range=(20, 140),
    )
    config = GrammarConfig(**kwargs)
    for family in FAMILIES:
        for seed in (0, 1, 2):
            task = make_task(family, seed, config)
            assert len(task.change_kinds) == len(task.world.change_points)
            assert all(point >= config.warmup for point in task.world.change_points)
            assert all(start >= config.warmup for start, _ in task.world.burst_spans)
    with pytest.raises(GrammarError, match="length - warmup"):
        GrammarConfig(**{**kwargs, "max_changes": 5})


# --- dataclass validation and serialisation identity ---


def test_synthetic_task_validates_and_freezes():
    """Misaligned or unknown kinds raise; a list of kinds is frozen to a tuple."""
    task = make_task("sudden", 0)
    with pytest.raises(GrammarError, match="align"):
        SyntheticTask(family="sudden", world=task.world, change_kinds=())
    with pytest.raises(GrammarError, match="change kind"):
        SyntheticTask(family="sudden", world=task.world, change_kinds=("teleport",))
    with pytest.raises(GrammarError, match="family"):
        SyntheticTask(family="bogus", world=task.world, change_kinds=task.change_kinds)
    frozen = SyntheticTask(family="sudden", world=task.world, change_kinds=list(task.change_kinds))
    assert frozen.change_kinds == task.change_kinds


def test_config_to_dict_is_json_ready_and_normalised():
    """int and float spellings of one config serialise identically."""
    int_config = GrammarConfig(level_range=(-5, 5), sigma_range=(0.3, 1), burst_mult_range=(2, 4))
    float_config = GrammarConfig(
        level_range=(-5.0, 5.0), sigma_range=(0.3, 1.0), burst_mult_range=(2.0, 4.0)
    )
    assert int_config == float_config
    assert config_to_dict(int_config) == config_to_dict(float_config)
    assert json.dumps(config_to_dict(int_config), allow_nan=False)
