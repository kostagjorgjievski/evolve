"""Unit tests for the learned updaters (TASK_014 PR A tests 1-12, PR B tests 1-8).

Every test drives ``init``/``update`` directly with hand-built Context and
Matured objects, feeding the Matured for step ``k - 1`` at step ``k`` unless
it says otherwise; the replay tests wrap the updater in a recorder and run
``replay_series`` over hand-built forecasts, and the arena tests replay the
synthetic-dev fixtures of ``tests/updater/test_baselines_replay.py``. The
tests PR A states for one updater and PR B repeats for both are parametrised
over ``LEARNED_IDS``; the ridge_stack-specific ones (closed form, behaviour,
refit schedule, overflow) stay on ``RidgeStack`` alone.
"""

from __future__ import annotations

import ast
import importlib
import itertools
import math
import random
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.dev import load_dev_pool
from evolvets.arena.errors import UpdaterFailure
from evolvets.arena.evaluate import replay_dataset
from evolvets.arena.replay import replay_series
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.foundation.fake import FAKE_SPEC, FakeAdapter
from evolvets.foundation.store import build_foundation_cache
from evolvets.updater.interfaces import Context, Matured
from evolvets.updater.learned import (
    LEARNED,
    LEARNED_DEFAULT_CONFIGS,
    LEARNED_IDS,
    LossGate,
    RidgeStack,
    loss_gate,
    project_to_simplex,
    solve_linear,
)

SRC = Path(__file__).parents[2] / "src" / "evolvets"
CONFIG = dict(LEARNED_DEFAULT_CONFIGS["ridge_stack"])
CONFIGS: dict[str, dict[str, Any]] = {
    name: dict(values) for name, values in LEARNED_DEFAULT_CONFIGS.items()
}


def names(n_experts: int) -> tuple[str, ...]:
    """``n_experts`` placeholder expert names."""
    return tuple(f"e{e}" for e in range(n_experts))


def step_context(
    n_experts: int,
    horizon: int,
    step: int = 0,
    forecasts: tuple[tuple[float, ...], ...] | None = None,
    period: int = 7,
) -> Context:
    """A valid context: constant forecasts (or ``forecasts``), the init'ed horizon."""
    rows = (
        forecasts
        if forecasts is not None
        else tuple(tuple(0.5 for _ in range(horizon)) for _ in range(n_experts))
    )
    return Context(
        step=step,
        season_index=step % period,
        period=period,
        horizon=horizon,
        expert_names=names(n_experts),
        forecasts=rows,
    )


def matured(error_rows: list[list[float]], horizon: int, step: int = 0) -> Matured:
    """One Matured whose errors are ``error_rows`` ([expert][h])."""
    return Matured(
        step=step,
        errors=tuple(tuple(row) for row in error_rows),
        combined_errors=(0.0,) * horizon,
    )


def drive(
    updater: Any,
    config: dict[str, Any],
    error_steps: list[list[list[float]]],
    n_experts: int,
    horizon: int = 1,
    seed: int = 0,
    trailing: int = 1,
) -> list[tuple[tuple[float, ...], ...]]:
    """Feed the Matured for step k-1 at step k; return each call's [expert][h] weights."""
    state = updater.init(n_experts, horizon, config, seed)
    out = []
    for k in range(len(error_steps) + trailing):
        feedback = ()
        if 0 < k <= len(error_steps):
            feedback = (matured(error_steps[k - 1], horizon, step=k - 1),)
        weights, state = updater.update(state, feedback, step_context(n_experts, horizon, k))
        out.append(weights)
    return out


def col0(weights_steps: list[tuple[tuple[float, ...], ...]]) -> list[tuple[float, ...]]:
    """Reduce [step][expert][h] weights to per-step tuples of column 0."""
    return [tuple(row[0] for row in weights) for weights in weights_steps]


class Recording:
    """Wraps an updater, recording the weights every update call returns."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.weights: list[tuple[tuple[float, ...], ...]] = []

    def init(self, n_experts: int, horizon: int, config: object, seed: int) -> object:
        return self.inner.init(n_experts, horizon, config, seed)

    def update(self, state: object, feedback: object, context: object) -> object:
        weights, state = self.inner.update(state, feedback, context)
        self.weights.append(weights)
        return (weights, state)


class HandForecasts:
    """``[origin][expert][h]`` rows behind the replay loop's Forecasts protocol."""

    def __init__(self, rows: list[tuple[tuple[float, ...], ...]], series_id: str = "probe"):
        self.series_id = series_id
        self.first_origin = 0
        self.last_origin = len(rows) - 1
        self._rows = rows

    def at(self, origin: int) -> tuple[tuple[float, ...], ...]:
        return self._rows[origin]


def probe_fixture(
    rng: random.Random, n_experts: int, horizon: int, n_origins: int, worst: int | None = None
) -> tuple[list[tuple[tuple[float, ...], ...]], list[float]]:
    """Forecasts and targets from one rng stream; expert ``worst`` is ten times worse."""
    values = [rng.uniform(-2.0, 2.0) for _ in range(n_origins + horizon)]
    rows = []
    for o in range(n_origins):
        experts = []
        for e in range(n_experts):
            scale = 10.0 if e == worst else 1.0
            experts.append(
                tuple(values[o + 1 + h] + scale * rng.uniform(-1.0, 1.0) for h in range(horizon))
            )
        rows.append(tuple(experts))
    return rows, values


def replay_probe(
    updater: Any,
    rows: list[tuple[tuple[float, ...], ...]],
    values: list[float],
    n_experts: int,
    horizon: int,
    n_origins: int,
    config: dict[str, Any],
    seed: int = 0,
) -> Any:
    """replay_series over a hand-built fixture of ``n_origins`` origins."""
    return replay_series(
        updater,
        config,
        seed,
        "probe",
        HandForecasts(rows),
        values,
        7,
        horizon,
        n_origins,
        expert_names=names(n_experts),
    )


# --- Test 1: the registry -------------------------------------------------------------------------


def test_registry_shape() -> None:
    """LEARNED_IDS lists the registry keys in order; each id names its own factory."""
    assert LEARNED_IDS == ("ridge_stack", "loss_gate")
    assert tuple(LEARNED) == LEARNED_IDS
    assert set(LEARNED_DEFAULT_CONFIGS) == set(LEARNED_IDS)
    module = importlib.import_module("evolvets.updater.learned")
    assert module.ridge_stack is LEARNED["ridge_stack"]
    assert module.loss_gate is LEARNED["loss_gate"]
    for factory in LEARNED.values():
        assert factory() is not factory()


def test_default_config_is_accepted_and_read_only() -> None:
    """Every default config inits cleanly, and neither layer of them can be mutated."""
    for learned_id in LEARNED_IDS:
        assert LEARNED[learned_id]().init(4, 3, LEARNED_DEFAULT_CONFIGS[learned_id], 0)
    for mapping in (LEARNED_DEFAULT_CONFIGS, *LEARNED_DEFAULT_CONFIGS.values()):
        with pytest.raises(TypeError):
            mapping["x"] = 1  # type: ignore[index]


# --- Test 2: config validation --------------------------------------------------------------------


def _bad_configs() -> list[tuple[str, str, dict[str, Any], str]]:
    """(updater id, label, config, message needle) for every learned init refusal."""
    ridge, gate = CONFIGS["ridge_stack"], CONFIGS["loss_gate"]
    cases: list[tuple[str, str, dict[str, Any], str]] = [
        (
            "ridge_stack",
            "missing-key",
            {k: v for k, v in ridge.items() if k != "shrink"},
            "config keys",
        ),
        ("ridge_stack", "extra-key", ridge | {"eta": 1.0}, "config keys"),
        ("ridge_stack", "shrink-zero", ridge | {"shrink": 0}, "shrink"),
        ("ridge_stack", "shrink-nan", ridge | {"shrink": float("nan")}, "shrink"),
        ("loss_gate", "missing-key", {k: v for k, v in gate.items() if k != "beta"}, "config keys"),
        ("loss_gate", "extra-key", gate | {"shrink": 1.0}, "config keys"),
        ("loss_gate", "beta-zero", gate | {"beta": 0}, "beta"),
        ("loss_gate", "beta-negative", gate | {"beta": -1.0}, "beta"),
        ("loss_gate", "beta-nan", gate | {"beta": float("nan")}, "beta"),
        ("loss_gate", "beta-inf", gate | {"beta": float("inf")}, "beta"),
        ("loss_gate", "beta-bool", gate | {"beta": True}, "beta"),
        ("loss_gate", "ridge-zero", gate | {"ridge": 0}, "ridge"),
        ("loss_gate", "ridge-negative", gate | {"ridge": -0.5}, "ridge"),
        ("loss_gate", "ridge-nan", gate | {"ridge": float("nan")}, "ridge"),
    ]
    half_lives = [0, -1.0, float("nan"), float("inf"), True]
    refits = [0, 1.5, True]
    for value in half_lives:
        label = f"half-life-{value!r}"
        cases.append(("ridge_stack", label, ridge | {"half_life": value}, "half_life"))
        cases.append(("loss_gate", label, gate | {"half_life": value}, "half_life"))
    for value in refits:
        label = f"refit-{value!r}"
        cases.append(("ridge_stack", label, ridge | {"refit_every": value}, "refit_every"))
        cases.append(("loss_gate", label, gate | {"refit_every": value}, "refit_every"))
    return cases


_CASES = _bad_configs()
_CASE_IDS = [f"{learned_id}-{label}" for learned_id, label, _, _ in _CASES]


@pytest.mark.parametrize(("learned_id", "label", "config", "needle"), _CASES, ids=_CASE_IDS)
def test_init_rejects_bad_configs(
    learned_id: str, label: str, config: dict[str, Any], needle: str
) -> None:
    """An unknown or missing key, or a bad beta, half_life, ridge, shrink or refit_every, raises."""
    with pytest.raises(ValueError, match=needle):
        LEARNED[learned_id]().init(3, 2, config, 0)


@pytest.mark.parametrize("learned_id", LEARNED_IDS)
def test_init_rejects_bad_scalars(learned_id: str) -> None:
    """n_experts of 0, horizon 0 and a bool seed are refused; the seed value is ignored."""
    with pytest.raises(ValueError, match="n_experts"):
        LEARNED[learned_id]().init(0, 2, CONFIGS[learned_id], 0)
    with pytest.raises(ValueError, match="horizon"):
        LEARNED[learned_id]().init(3, 0, CONFIGS[learned_id], 0)
    with pytest.raises(ValueError, match="seed"):
        LEARNED[learned_id]().init(3, 2, CONFIGS[learned_id], True)


# --- Test 3: uniform first ------------------------------------------------------------------------


@pytest.mark.parametrize("learned_id", LEARNED_IDS)
def test_uniform_before_anything_matures(learned_id: str) -> None:
    """Every call before a Matured is absorbed returns exactly 1 / n per column."""
    updater = LEARNED[learned_id]()
    state = updater.init(4, 3, CONFIGS[learned_id], 0)
    for step in range(5):  # the arena's first `horizon` steps carry empty feedback
        weights, state = updater.update(state, (), step_context(4, 3, step))
        assert weights == tuple(tuple(0.25 for _ in range(3)) for _ in range(4))


# --- Test 4: valid weights over a long replay -----------------------------------------------------


@pytest.mark.parametrize("learned_id", LEARNED_IDS)
def test_valid_weights_over_a_long_replay(learned_id: str) -> None:
    """400 origins of random data replay without ever leaving the simplex."""
    rows, values = probe_fixture(random.Random(0), 4, 3, 400, worst=3)
    recorder = Recording(LEARNED[learned_id]())
    trace = replay_probe(recorder, rows, values, 4, 3, 400, CONFIGS[learned_id])
    assert len(trace.errors) == 400  # replay_series completed: every column passed the check
    assert len(recorder.weights) == 400
    for weights in recorder.weights:
        for row in weights:
            for weight in row:
                assert weight >= 0.0
        for h in range(3):
            total = 0.0
            for row in weights:
                total = total + row[h]
            assert abs(total - 1.0) <= 1e-9


# --- Test 5: the closed form ----------------------------------------------------------------------


def hand_projection(values: list[float]) -> list[float]:
    """Duchi et al. (2008) written out naively, independent of the module."""
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    cs = 0.0
    theta = 0.0
    for k, index in enumerate(order, start=1):
        cs = cs + values[index]
        if values[index] - (cs - 1.0) / k > 0.0:
            theta = (cs - 1.0) / k
    out = [max(values[i] - theta, 0.0) for i in range(len(values))]
    total = 0.0
    for value in out:
        total = total + value
    return [value / total for value in out]


def test_matches_the_closed_form() -> None:
    """After three matured origins the weights are the projected normalised A^-1 1."""
    half_life, shrink = 8.0, 0.5
    config = {"half_life": half_life, "shrink": shrink, "refit_every": 1}
    gamma = math.exp(-math.log(2.0) / half_life)
    steps = [
        [[0.5], [-0.3]],
        [[0.2], [0.4]],
        [[-0.6], [0.1]],
    ]
    weights = col0(drive(RidgeStack(), config, steps, 2))
    s = [[0.0, 0.0], [0.0, 0.0]]
    m = 0.0
    for step in steps:
        s = [[gamma * s[i][j] + step[i][0] * step[j][0] for j in range(2)] for i in range(2)]
        m = gamma * m + 1.0
    sbar = [[s[i][j] / m for j in range(2)] for i in range(2)]
    trace = sbar[0][0] / 2 + sbar[1][1] / 2
    a = [
        [sbar[0][0] + shrink * trace, sbar[0][1]],
        [sbar[1][0], sbar[1][1] + shrink * trace],
    ]
    det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
    z = [(a[1][1] - a[0][1]) / det, (a[0][0] - a[1][0]) / det]  # A^-1 1 for a 2x2 system
    total = z[0] + z[1]
    expected = hand_projection([z[0] / total, z[1] / total])
    for e in range(2):
        assert weights[3][e] == pytest.approx(expected[e], abs=1e-12)


# --- Test 6: behaviour ----------------------------------------------------------------------------


def test_symmetric_constant_errors_split_evenly() -> None:
    """Constant errors (+1, -1) give exactly equal weights."""
    steps = [[[1.0], [-1.0]] for _ in range(6)]
    for row in col0(drive(RidgeStack(), CONFIG, steps, 2))[1:]:
        assert row[0] == pytest.approx(0.5, abs=1e-12)
        assert row[1] == pytest.approx(0.5, abs=1e-12)


def test_dominant_constant_errors_concentrate() -> None:
    """Constant errors (1, 3) with a tiny shrink give the better expert almost everything."""
    steps = [[[1.0], [3.0]] for _ in range(6)]
    config = {"half_life": 336.0, "shrink": 1e-6, "refit_every": 24}
    weights = col0(drive(RidgeStack(), config, steps, 2))
    assert weights[-1][0] >= 0.999  # the projection clipped expert 1's negative weight


# --- Test 7: the refit schedule -------------------------------------------------------------------


def test_refits_only_on_the_schedule() -> None:
    """With refit_every = 24 the weights change only at multiples of 24 after the first fit."""
    rng = random.Random(3)
    steps = [[[rng.uniform(-1.0, 1.0)] for _ in range(3)] for _ in range(30)]
    config = {"half_life": 24.0, "shrink": 1.0, "refit_every": 24}
    weights = col0(drive(RidgeStack(), config, steps, 3))
    for k in range(2, 31):
        if k % 24 != 0:
            assert weights[k] == weights[k - 1]
    assert weights[24] != weights[23]


# --- Test 8: fallbacks ----------------------------------------------------------------------------


def test_all_zero_errors_stay_uniform() -> None:
    """A zero trace makes every refit return the uniform column."""
    steps = [[[0.0], [0.0]] for _ in range(5)]
    for row in col0(drive(RidgeStack(), CONFIG, steps, 2)):
        assert row == (0.5, 0.5)


def test_overflowing_errors_freeze_the_column() -> None:
    """Errors of 1e200 overflow S to inf; the column keeps its weights and nothing fails."""
    steps = [[[0.4], [-0.2]] for _ in range(3)] + [[[1e200], [1e200]]]
    steps += [[[0.3], [0.1]] for _ in range(2)]
    config = {"half_life": 24.0, "shrink": 1.0, "refit_every": 1}
    weights = col0(drive(RidgeStack(), config, steps, 2))
    assert weights[4] == weights[3]  # the overflowing refit kept the previous weights
    assert weights[5] == weights[4]  # S stays inf, so every later refit keeps them too


def test_solve_linear_singular_and_shapes() -> None:
    """A singular system returns None; bad shapes raise; arguments are never mutated."""
    assert solve_linear([[1.0, 2.0], [2.0, 4.0]], [1.0, 1.0]) is None
    assert solve_linear([[float("nan"), 1.0], [1.0, 1.0]], [1.0, 1.0]) is None
    with pytest.raises(ValueError, match="square"):
        solve_linear([[1.0, 2.0], [2.0]], [1.0, 1.0])
    with pytest.raises(ValueError, match="rhs"):
        solve_linear([[1.0, 2.0], [2.0, 4.0]], [1.0])
    with pytest.raises(ValueError, match="at least one row"):
        solve_linear([], [])
    matrix = [[1.0, 2.0], [3.0, 4.0]]
    rhs = [1.0, 1.0]
    assert solve_linear(matrix, rhs) == pytest.approx([-1.0, 1.0])  # (1*-1+2*1, 3*-1+4*1) = (1, 1)
    assert matrix == [[1.0, 2.0], [3.0, 4.0]]
    assert rhs == [1.0, 1.0]


# --- Test 9: project_to_simplex -------------------------------------------------------------------


def test_projection_keeps_simplex_points() -> None:
    """A point already on the simplex comes back unchanged within 1e-15."""
    point = [0.5, 0.3, 0.2]
    for ours, theirs in zip(project_to_simplex(point), point, strict=True):
        assert ours == pytest.approx(theirs, abs=1e-15)


def test_projection_of_a_vertex() -> None:
    """(2.0, 0.0) projects to (1.0, 0.0)."""
    assert project_to_simplex([2.0, 0.0]) == [1.0, 0.0]


def test_projection_of_equal_entries() -> None:
    """Three equal entries project to thirds."""
    for value in project_to_simplex([0.4, 0.4, 0.4]):
        assert value == pytest.approx(1.0 / 3.0, abs=1e-15)


def test_projection_clips_negative_entries() -> None:
    """A negative entry is clipped to zero and the rest renormalised."""
    assert project_to_simplex([1.5, -0.5]) == [1.0, 0.0]
    out = project_to_simplex([0.7, -0.2, 0.5])
    assert out[1] == 0.0
    total = 0.0
    for value in out:
        total = total + value
    assert total == pytest.approx(1.0)


def test_projection_is_order_invariant() -> None:
    """Every permutation of one input gives the same values at the matching positions."""
    values = [0.1, 0.8, -0.3, 0.6]
    direct = project_to_simplex(values)
    for permutation in itertools.permutations(range(4)):
        shuffled = [values[i] for i in permutation]
        out = project_to_simplex(shuffled)
        for position, index in enumerate(permutation):
            assert out[position] == direct[index]


def test_projection_rejects_bad_input() -> None:
    """An empty input, a NaN or an infinity raises ValueError."""
    with pytest.raises(ValueError, match="at least one entry"):
        project_to_simplex([])
    with pytest.raises(ValueError, match="finite"):
        project_to_simplex([0.5, float("nan")])
    with pytest.raises(ValueError, match="finite"):
        project_to_simplex([0.5, float("inf")])


# --- Test 10: no look-ahead -----------------------------------------------------------------------


@pytest.mark.parametrize("learned_id", LEARNED_IDS)
def test_no_look_ahead(learned_id: str) -> None:
    """Targets beyond step k never reach the weights at or before step k."""
    k = 50
    rows, values = probe_fixture(random.Random(7), 4, 3, 120)
    changed = list(values)
    for i in range(k + 1, len(changed)):
        changed[i] = changed[i] + 5.0
    first, second = Recording(LEARNED[learned_id]()), Recording(LEARNED[learned_id]())
    replay_probe(first, rows, values, 4, 3, 120, CONFIGS[learned_id])
    replay_probe(second, rows, changed, 4, 3, 120, CONFIGS[learned_id])
    for step in range(k + 1):
        assert first.weights[step] == second.weights[step]


# --- Test 11: determinism and isolation -----------------------------------------------------------


@pytest.mark.parametrize("learned_id", LEARNED_IDS)
def test_two_replays_are_bit_identical(learned_id: str) -> None:
    """Two replays of the same inputs return identical weights at every step."""
    rows, values = probe_fixture(random.Random(0), 4, 3, 100)
    first, second = Recording(LEARNED[learned_id]()), Recording(LEARNED[learned_id]())
    replay_probe(first, rows, values, 4, 3, 100, CONFIGS[learned_id])
    replay_probe(second, rows, values, 4, 3, 100, CONFIGS[learned_id])
    assert first.weights == second.weights


@pytest.mark.parametrize("learned_id", LEARNED_IDS)
def test_state_lives_in_the_state(learned_id: str) -> None:
    """Two states of one instance evolve exactly like two separate instances."""
    rng_a, rng_b = random.Random(1), random.Random(2)
    n_experts, horizon, steps = 4, 2, 8
    config = CONFIGS[learned_id]
    shared = LEARNED[learned_id]()
    state_a = shared.init(n_experts, horizon, config, 0)
    state_b = shared.init(n_experts, horizon, config, 1)
    alone_a, alone_b = LEARNED[learned_id](), LEARNED[learned_id]()
    alone_state_a = alone_a.init(n_experts, horizon, config, 0)
    alone_state_b = alone_b.init(n_experts, horizon, config, 1)
    # the arena's first call carries empty feedback; loss_gate stores step 0's features there
    _, state_a = shared.update(state_a, (), step_context(4, 2, 0))
    _, state_b = shared.update(state_b, (), step_context(4, 2, 0))
    _, alone_state_a = alone_a.update(alone_state_a, (), step_context(4, 2, 0))
    _, alone_state_b = alone_b.update(alone_state_b, (), step_context(4, 2, 0))
    for k in range(1, steps + 1):
        series_a = [[rng_a.uniform(-1.0, 1.0) for _ in range(horizon)] for _ in range(n_experts)]
        series_b = [[rng_b.uniform(-1.0, 1.0) for _ in range(horizon)] for _ in range(n_experts)]
        feedback_a = (matured(series_a, horizon, step=k - 1),)
        feedback_b = (matured(series_b, horizon, step=k - 1),)
        interleaved_a, state_a = shared.update(state_a, feedback_a, step_context(4, 2, k))
        interleaved_b, state_b = shared.update(state_b, feedback_b, step_context(4, 2, k))
        separate_a, alone_state_a = alone_a.update(alone_state_a, feedback_a, step_context(4, 2, k))
        separate_b, alone_state_b = alone_b.update(alone_state_b, feedback_b, step_context(4, 2, k))
        assert interleaved_a == separate_a
        assert interleaved_b == separate_b


# --- Test 12: the arena replay --------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_pool(tmp_path_factory: pytest.TempPathFactory):
    """The dev pool of a synthetic-dev cache with one series per family (TASK_007 fixture)."""
    base = tmp_path_factory.mktemp("learned-synthetic")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-dev", base, base, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    return load_dev_pool("synthetic-dev", cache_root, base, base)


@pytest.fixture(scope="module")
def pooled_env(tmp_path_factory: pytest.TempPathFactory):
    """A synthetic-dev cache plus a fake foundation cache built against it (TASK_009 helpers)."""
    base = tmp_path_factory.mktemp("learned-pooled")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-dev", base, base, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    foundation_root = base / "foundation_cache"
    build_foundation_cache(
        "synthetic-dev",
        FakeAdapter(FAKE_SPEC.name),
        FAKE_SPEC,
        foundation_root,
        cache_root,
        base,
        base,
        7,
        per_family=1,
    )
    return base, cache_root, foundation_root


@pytest.mark.parametrize("learned_id", LEARNED_IDS)
def test_replays_the_synthetic_pool(synthetic_pool, learned_id: str) -> None:
    """Each learned updater at its default config replays synthetic-dev to ok."""
    result = replay_dataset(
        LEARNED[learned_id], LEARNED_DEFAULT_CONFIGS[learned_id], 0, synthetic_pool
    )
    assert result.status == "ok"


@pytest.mark.parametrize("learned_id", LEARNED_IDS)
def test_replays_the_pooled_fixture(pooled_env, learned_id: str) -> None:
    """The same replay over the pooled fake-foundation pool is ok too."""
    base, cache_root, foundation_root = pooled_env
    pool = load_dev_pool(
        "synthetic-dev",
        cache_root,
        base,
        base,
        pool_spec="simple+foundation",
        foundation_root=foundation_root,
        foundation_models=("fake",),
    )
    result = replay_dataset(LEARNED[learned_id], LEARNED_DEFAULT_CONFIGS[learned_id], 0, pool)
    assert result.status == "ok"


# --- PR B test 2: an unknown matured step ---------------------------------------------------------


class StepRewriter:
    """Wraps an updater, shifting every Matured's step to one no update call stored."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def init(self, n_experts: int, horizon: int, config: object, seed: int) -> object:
        return self.inner.init(n_experts, horizon, config, seed)

    def update(self, state: object, feedback: object, context: object) -> object:
        shifted = tuple(
            Matured(
                step=matured_item.step - 1,
                errors=matured_item.errors,
                combined_errors=matured_item.combined_errors,
            )
            for matured_item in feedback
        )
        return self.inner.update(state, shifted, context)


def test_unknown_matured_step_is_refused() -> None:
    """A Matured whose step no update call of this state returned raises ValueError."""
    updater = loss_gate()
    state = updater.init(2, 1, CONFIGS["loss_gate"], 0)
    _, state = updater.update(state, (), step_context(2, 1, 0))  # stores step 0
    with pytest.raises(ValueError, match="step 3"):
        updater.update(state, (matured([[0.2], [-0.4]], 1, step=3),), step_context(2, 1, 1))


def test_unknown_step_fails_the_replay() -> None:
    """Inside replay_series the same refusal stops the series as a failed replay."""
    rows, values = probe_fixture(random.Random(4), 3, 2, 30)
    with pytest.raises(UpdaterFailure) as failure:
        replay_probe(StepRewriter(loss_gate()), rows, values, 3, 2, 30, CONFIGS["loss_gate"])
    assert failure.value.kind == "failed"
    assert "ValueError" in failure.value.message


# --- PR B test 3: a persistent best ---------------------------------------------------------------


def test_persistent_best_gets_the_weight() -> None:
    """With beta = 64, an expert whose error is always 0.1 against 1.0 reaches 0.9."""
    config = CONFIGS["loss_gate"] | {"beta": 64.0}
    steps = [[[1.0, 1.0], [0.1, 0.1], [1.0, 1.0]] for _ in range(200)]
    weights = drive(LossGate(), config, steps, 3, horizon=2)
    for h in range(2):
        assert weights[-1][1][h] >= 0.9


# --- PR B test 4: a switch after a regime change -------------------------------------------------


def test_switch_after_a_regime_change() -> None:
    """The best expert changes at origin 500; with half_life = 48 the gate follows it."""
    config = CONFIGS["loss_gate"] | {"half_life": 48.0}
    steps = [[[0.1, 0.1], [1.0, 1.0]] for _ in range(500)]
    steps += [[[1.0, 1.0], [0.1, 0.1]] for _ in range(500)]
    weights = drive(LossGate(), config, steps, 2, horizon=2)
    for h in range(2):
        assert weights[-1][1][h] > 0.5


# --- PR B test 5: a seasonal gate -----------------------------------------------------------------


def test_seasonal_gate_prefers_the_seasons_best() -> None:
    """Expert 0 wins only at season-0 origins; its mean weight there exceeds the rest."""
    updater = loss_gate()
    state = updater.init(2, 1, CONFIGS["loss_gate"], 0)
    forecasts = ((0.5,), (1.5,))
    weights, state = updater.update(state, (), step_context(2, 1, 0, forecasts=forecasts, period=4))
    season_0, other = [], []
    for k in range(1, 401):
        season = (k - 1) % 4
        errors = [[0.1], [1.0]] if season == 0 else [[1.0], [0.1]]
        feedback = (matured(errors, 1, step=k - 1),)
        weights, state = updater.update(
            state, feedback, step_context(2, 1, k, forecasts=forecasts, period=4)
        )
        (season_0 if k % 4 == 0 else other).append(weights[0][0])
    assert sum(season_0) / len(season_0) > sum(other) / len(other)


# --- PR B test 6: training rows use origin-time features -----------------------------------------


def test_training_rows_use_origin_time_features() -> None:
    """G and r after maturity hold the features stored at the origin, not today's."""
    half_life = 8.0
    config = CONFIGS["loss_gate"] | {"half_life": half_life, "refit_every": 1}
    gamma = math.exp(-math.log(2.0) / half_life)
    updater = loss_gate()
    state = updater.init(2, 1, config, 0)
    # origin 0: forecasts (0.0, 2.0) -> mean 1.0, deviations (1.0, 1.0), dev = (1.0, 1.0);
    # nothing has matured, so fast, slow and season are all 0.0
    _, state = updater.update(state, (), step_context(2, 1, 0, forecasts=((0.0,), (2.0,))))
    # origin 1 matures origin 0 and stores its own rows: its forecasts are constant, so
    # recomputing origin 0's features now would give dev = (0.0, 0.0), not the stored
    # (1.0, 1.0), and its rows see the EWMAs origin 0's losses set (season falls back
    # to slow: origin 1's season index was never seen yet)
    _, state = updater.update(
        state,
        (matured([[0.5], [-0.25]], 1, step=0),),
        step_context(2, 1, 1, forecasts=((1.0,), (1.0,))),
    )
    # origin 2 matures origin 1: a second origin, so G and r are discounted with a
    # non-zero history behind them
    _, state = updater.update(
        state,
        (matured([[0.2], [0.4]], 1, step=1),),
        step_context(2, 1, 2, forecasts=((1.0,), (1.0,))),
    )
    origin0_rows = [(1.0, 0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 0.0, 1.0)]
    origin0_losses = [1.0, 0.5]  # _loss_column of (0.5, -0.25): abs over the column max 0.5
    origin1_rows = [(1.0, 1.0, 1.0, 1.0, 0.0), (1.0, 0.5, 0.5, 0.5, 0.0)]
    origin1_losses = [0.5, 1.0]  # _loss_column of (0.2, 0.4)
    expected_g = [[0.0] * 5 for _ in range(5)]
    expected_r = [0.0] * 5
    expected_m = 0.0
    # one discount per origin, then every expert's row of that origin
    for rows, losses in ((origin0_rows, origin0_losses), (origin1_rows, origin1_losses)):
        for i in range(5):
            for j in range(5):
                gram_sum = 0.0
                for row in rows:
                    gram_sum = gram_sum + row[i] * row[j]
                expected_g[i][j] = gamma * expected_g[i][j] + gram_sum
            rhs_sum = 0.0
            for row, loss in zip(rows, losses, strict=True):
                rhs_sum = rhs_sum + row[i] * loss
            expected_r[i] = gamma * expected_r[i] + rhs_sum
        expected_m = gamma * expected_m + 2.0
    assert state.m[0] == pytest.approx(expected_m, abs=1e-12)
    for i in range(5):
        assert state.r[0][i] == pytest.approx(expected_r[i], abs=1e-12)
        for j in range(5):
            assert state.g[0][i][j] == pytest.approx(expected_g[i][j], abs=1e-12)
    late_rows = [(1.0, 1.0, 1.0, 1.0, 0.0), (1.0, 0.5, 0.5, 0.5, 0.0)]  # origin 0, recomputed now
    late_g = [[0.0] * 5 for _ in range(5)]
    for row, _ in zip(late_rows, origin0_losses, strict=True):
        for i in range(5):
            for j in range(5):
                late_g[i][j] = gamma * late_g[i][j] + row[i] * row[j]
    assert late_g != expected_g  # the origin-time and maturity-time designs really differ


# --- PR B: the discounted design ------------------------------------------------------------------


@pytest.mark.parametrize("n_experts", [2, 5])
def test_discounted_design_is_a_discounted_mean(n_experts: int) -> None:
    """G, r and m discount once per origin, so the intercept entry equals the count."""
    updater = loss_gate()
    state = updater.init(n_experts, 1, CONFIGS["loss_gate"], 0)
    errors = [[0.25] for _ in range(n_experts)]  # constant errors fix every feature
    _, state = updater.update(state, (), step_context(n_experts, 1, 0))
    for k in range(1, 3001):
        _, state = updater.update(
            state, (matured(errors, 1, step=k - 1),), step_context(n_experts, 1, k)
        )
        # each origin adds n intercept rows of 1.0 * 1.0 and n to m, with one
        # discount between origins: G[0][0] and m stay equal whatever n is
        assert abs(state.g[0][0][0] - state.m[0]) <= 1e-9
        assert abs(state.g[0][0][0] / state.m[0] - 1.0) <= 1e-9


# --- PR B test 7: the pending bound --------------------------------------------------------------


class PendingProbe:
    """Wraps the updater, recording how large pending ever grew after a call."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.max_pending = 0

    def init(self, n_experts: int, horizon: int, config: object, seed: int) -> object:
        return self.inner.init(n_experts, horizon, config, seed)

    def update(self, state: object, feedback: object, context: object) -> object:
        result = self.inner.update(state, feedback, context)
        self.max_pending = max(self.max_pending, len(result[1].pending))
        return result


def test_pending_never_exceeds_the_horizon() -> None:
    """Through a whole replay, pending holds at most the horizon's worth of origins."""
    rows, values = probe_fixture(random.Random(5), 3, 3, 80)
    probe = PendingProbe(loss_gate())
    trace = replay_probe(probe, rows, values, 3, 3, 80, CONFIGS["loss_gate"])
    assert len(trace.errors) == 80
    assert probe.max_pending <= 3


# --- PR B test 8: fallbacks ----------------------------------------------------------------------


def test_uniform_until_the_first_matured_origin() -> None:
    """beta_hat stays None before the first matured origin, so the column is uniform."""
    updater = loss_gate()
    state = updater.init(3, 2, CONFIGS["loss_gate"], 0)
    weights, state = updater.update(state, (), step_context(3, 2, 0))
    assert state.beta_hat == [None, None]
    assert weights == tuple(tuple(1.0 / 3 for _ in range(2)) for _ in range(3))


def test_singular_refit_keeps_the_previous_coefficients() -> None:
    """A design whose normalised system is exactly singular keeps the last beta_hat."""
    config = {"beta": 16.0, "half_life": 8.0, "ridge": 0.5, "refit_every": 1}
    updater = loss_gate()
    state = updater.init(2, 1, config, 0)
    _, state = updater.update(state, (), step_context(2, 1, 0))
    _, state = updater.update(state, (matured([[0.3], [-0.4]], 1, step=0),), step_context(2, 1, 1))
    assert state.beta_hat[0] is not None
    fitted = state.beta_hat[0]
    assert state.m[0] == 2.0  # exact, so the crafted design below reconstructs exactly
    singular = [
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.0, 0.0],  # a duplicate row: the system cannot have a pivot
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    count, ridge = state.m[0], 0.5
    state.g[0] = [
        [count * (singular[i][j] - (ridge if i == j and i > 0 else 0.0)) for j in range(5)]
        for i in range(5)
    ]
    _, state = updater.update(state, (), step_context(2, 1, 2))
    assert state.beta_hat[0] is fitted  # solve_linear returned None; nothing was replaced


# --- AST guards -----------------------------------------------------------------------------------


ALLOWED_IMPORTS = {
    "math",
    "types",
    "dataclasses",
    "collections.abc",
    "evolvets.updater.interfaces",
    "evolvets.updater.baselines",
}


def _tree() -> ast.Module:
    return ast.parse((SRC / "updater" / "learned.py").read_text(encoding="utf-8"))


def test_imports_are_exactly_the_allowed_ones() -> None:
    """learned.py imports only the standard library, the contract and the baselines."""
    imported: set[str | None] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= ALLOWED_IMPORTS


def test_no_banned_calls() -> None:
    """No builtin sum, no open, no math.fsum; math attributes are exp and log only."""
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"sum", "open"}, node.func.id
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr != "fsum"
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "math"
        ):
            assert node.attr in {"exp", "log"}, node.attr


def test_no_banned_math_imports() -> None:
    """``from math import ...`` may name only exp and log."""
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module == "math":
            for alias in node.names:
                assert alias.name in {"exp", "log"}, alias.name


def test_no_power_operator() -> None:
    """No ``**`` anywhere: squares and cubes are written as explicit products."""
    for node in ast.walk(_tree()):  # ``**=`` reuses ast.Pow as its AugAssign op
        assert not isinstance(node, ast.Pow)
