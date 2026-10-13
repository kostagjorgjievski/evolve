"""Unit tests for the five baseline updaters (TASK_010 PR A, tests 1-12 and 16).

Every test drives ``init``/``update`` directly with hand-built Context and
Matured objects, feeding the Matured for step ``k - 1`` at step ``k`` unless
it says otherwise. Raw error columns are written with a maximum absolute
value of 1 where a test wants the shared loss to equal the raw column.
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

from evolvets.updater.baselines import (
    BASELINE_IDS,
    BASELINES,
    DEFAULT_CONFIGS,
    AdaHedge,
    Equal,
    FixedShare,
    FollowTheLeader,
    Hedge,
)
from evolvets.updater.interfaces import Context, Matured

SRC = Path(__file__).parents[2] / "src" / "evolvets"


def names(n_experts: int) -> tuple[str, ...]:
    """``n_experts`` placeholder expert names."""
    return tuple(f"e{e}" for e in range(n_experts))


def step_context(n_experts: int, horizon: int, step: int = 0) -> Context:
    """A valid context: constant forecasts, the horizon the baselines were init'ed with."""
    return Context(
        step=step,
        season_index=step % 7,
        period=7,
        horizon=horizon,
        expert_names=names(n_experts),
        forecasts=tuple(tuple(0.5 for _ in range(horizon)) for _ in range(n_experts)),
    )


def matured(error_rows: list[list[float]], horizon: int, step: int = 0) -> Matured:
    """One Matured whose errors are ``error_rows`` ([expert][h])."""
    return Matured(
        step=step,
        errors=tuple(tuple(row) for row in error_rows),
        combined_errors=(0.0,) * horizon,
    )


def shared_losses(step: list[list[float]], column: int = 0) -> list[float]:
    """The shared loss of one column, as the task states it (|error| over the max)."""
    raw = [abs(row[column]) for row in step]
    top = raw[0]
    for value in raw[1:]:
        top = value if value > top else top
    if top > 0.0:
        return [value / top for value in raw]
    return [0.0] * len(raw)


def drive(
    updater: Any,
    config: dict[str, Any],
    error_steps: list[list[list[float]]],
    n_experts: int,
    horizon: int = 1,
    seed: int = 0,
    trailing: int = 1,
) -> list[tuple[tuple[float, ...], ...]]:
    """Feed the Matured for step k-1 at step k; return each call's [expert][h] weights.

    ``trailing`` extra update calls run after the last step's feedback, so a
    sequence of ``n`` steps yields ``n + trailing`` weight matrices and the
    weights after the final feedback sit at index ``n``.
    """
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


# --- Test 1: the contract ------------------------------------------------------------------------


@pytest.mark.parametrize("id_", BASELINE_IDS)
def test_contract(id_: str) -> None:
    """A fresh object per factory call; 20 updates return simplex columns of floats."""
    first = BASELINES[id_]()
    assert first is not BASELINES[id_]()
    rng = random.Random(0)
    n_experts, horizon = 5, 3
    state = first.init(n_experts, horizon, DEFAULT_CONFIGS[id_], 0)
    for step in range(20):
        feedback = ()
        if step > 0:
            errors = [[rng.uniform(-2.0, 2.0) for _ in range(horizon)] for _ in range(n_experts)]
            feedback = (matured(errors, horizon, step=step - 1),)
        context = Context(
            step=step,
            season_index=step % 7,
            period=7,
            horizon=horizon,
            expert_names=names(n_experts),
            forecasts=tuple(
                tuple(rng.uniform(-1.0, 1.0) for _ in range(horizon)) for _ in range(n_experts)
            ),
        )
        weights, state = first.update(state, feedback, context)
        assert type(weights) is tuple
        assert len(weights) == n_experts
        for row in weights:
            assert type(row) is tuple
            assert len(row) == horizon
            for weight in row:
                assert type(weight) is float
                assert weight >= 0.0
        for h in range(horizon):
            total = 0.0
            for e in range(n_experts):
                total = total + weights[e][h]
            assert abs(total - 1.0) <= 1e-9


def test_registry_shape() -> None:
    """BASELINE_IDS lists the registry keys in order; each id names its own factory."""
    assert BASELINE_IDS == ("equal", "ftl", "hedge", "fixed_share", "adahedge")
    assert tuple(BASELINES) == BASELINE_IDS
    assert set(DEFAULT_CONFIGS) == set(BASELINE_IDS)
    module = importlib.import_module("evolvets.updater.baselines")
    for id_ in BASELINE_IDS:
        assert getattr(module, id_) is BASELINES[id_]


def test_default_configs_are_read_only() -> None:
    """DEFAULT_CONFIGS and its inner configs cannot be mutated."""
    for mapping in (DEFAULT_CONFIGS, *DEFAULT_CONFIGS.values()):
        with pytest.raises(TypeError):
            mapping["x"] = 1  # type: ignore[index]


# --- Test 2: config validation -------------------------------------------------------------------


def _cases() -> list[tuple[str, str, str, str]]:
    """(id, label, config, message needle) for every init refusal."""
    return [
        ("equal", "unknown-key", {"eta": 1.0}, "config keys"),
        ("hedge", "missing-eta", {}, "config keys"),
        ("hedge", "unknown-alpha", {"eta": 1.0, "alpha": 0.1}, "config keys"),
        ("fixed_share", "missing-alpha", {"eta": 1.0}, "config keys"),
        ("hedge", "eta-zero", {"eta": 0}, "eta"),
        ("hedge", "eta-negative", {"eta": -1.0}, "eta"),
        ("hedge", "eta-nan", {"eta": float("nan")}, "eta"),
        ("hedge", "eta-inf", {"eta": float("inf")}, "eta"),
        ("hedge", "eta-bool", {"eta": True}, "eta"),
        ("hedge", "eta-str", {"eta": "1"}, "eta"),
        ("fixed_share", "alpha-negative", {"eta": 1.0, "alpha": -0.1}, "alpha"),
        ("fixed_share", "alpha-above-one", {"eta": 1.0, "alpha": 1.5}, "alpha"),
        ("fixed_share", "alpha-nan", {"eta": 1.0, "alpha": float("nan")}, "alpha"),
        ("fixed_share", "alpha-bool", {"eta": 1.0, "alpha": True}, "alpha"),
    ]


_CASE_IDS = [case[1] for case in _cases()]


@pytest.mark.parametrize(("id_", "label", "config", "needle"), _cases(), ids=_CASE_IDS)
def test_init_rejects_bad_configs(
    id_: str, label: str, config: dict[str, Any], needle: str
) -> None:
    """An unknown or missing key, or a bad eta or alpha, raises ValueError."""
    with pytest.raises(ValueError, match=needle):
        BASELINES[id_]().init(3, 2, config, 0)


@pytest.mark.parametrize(
    ("argument", "value"),
    [("n_experts", 0), ("n_experts", True), ("horizon", 0), ("seed", True)],
    ids=["experts-zero", "experts-bool", "horizon-zero", "seed-bool"],
)
def test_init_rejects_bad_scalars(argument: str, value: Any) -> None:
    """n_experts of 0 or True, horizon 0 and a bool seed are refused."""
    kwargs: dict[str, Any] = {"n_experts": 3, "horizon": 2, "config": {}, "seed": 0}
    kwargs[argument] = value
    with pytest.raises(ValueError, match=argument):
        Equal().init(**kwargs)


def test_init_accepts_edge_values() -> None:
    """alpha of 0 and 1 and an int eta are accepted."""
    FixedShare().init(3, 1, {"eta": 1.0, "alpha": 0}, 0)
    FixedShare().init(3, 1, {"eta": 1.0, "alpha": 1}, 0)
    Hedge().init(3, 1, {"eta": 3}, 0)


# --- Test 3: shape validation in update ----------------------------------------------------------


def _bad_update_cases(
    n_experts: int, horizon: int
) -> list[tuple[str, tuple[Matured, ...], Context, str]]:
    """(label, feedback, context, message needle) cases that update must refuse."""
    good_context = step_context(n_experts, horizon, step=1)
    good_feedback = (matured([[0.25] * horizon for _ in range(n_experts)], horizon, step=0),)
    few_rows = Context(
        step=1,
        season_index=1,
        period=7,
        horizon=horizon,
        expert_names=names(n_experts),
        forecasts=tuple(tuple(0.5 for _ in range(horizon)) for _ in range(n_experts - 1)),
    )
    short_row = Context(
        step=1,
        season_index=1,
        period=7,
        horizon=horizon,
        expert_names=names(n_experts),
        forecasts=tuple(
            tuple(0.5 for _ in range(horizon - 1)) if e == 1 else tuple(0.5 for _ in range(horizon))
            for e in range(n_experts)
        ),
    )
    wrong_context_horizon = Context(
        step=1,
        season_index=1,
        period=7,
        horizon=horizon + 1,
        expert_names=names(n_experts),
        forecasts=tuple(tuple(0.5 for _ in range(horizon + 1)) for _ in range(n_experts)),
    )
    return [
        ("forecasts-few-rows", good_feedback, few_rows, "forecasts must hold"),
        ("forecast-row-short", good_feedback, short_row, "row must hold"),
        ("context-horizon", good_feedback, wrong_context_horizon, "context.horizon"),
        (
            "matured-few-rows",
            (matured([[0.25] * horizon for _ in range(n_experts - 1)], horizon),),
            good_context,
            "errors must hold",
        ),
        (
            "matured-row-short",
            (matured([[0.25] * (horizon - 1) for _ in range(n_experts)], horizon),),
            good_context,
            "error rows",
        ),
        (
            "matured-nan-error",
            (
                matured(
                    [[float("nan") if e == 0 else 0.25] * horizon for e in range(n_experts)],
                    horizon,
                ),
            ),
            good_context,
            "finite",
        ),
    ]


@pytest.mark.parametrize("id_", BASELINE_IDS)
def test_update_rejects_bad_shapes(id_: str) -> None:
    """Every baseline refuses mismatched forecasts, a wrong context horizon and bad errors."""
    updater = BASELINES[id_]()
    n_experts, horizon = 3, 2
    state = updater.init(n_experts, horizon, DEFAULT_CONFIGS[id_], 0)
    for _label, feedback, context, needle in _bad_update_cases(n_experts, horizon):
        with pytest.raises(ValueError, match=needle):
            updater.update(state, feedback, context)


@pytest.mark.parametrize("id_", BASELINE_IDS)
def test_update_refuses_a_foreign_state(id_: str) -> None:
    """A state object another baseline's init returned is refused."""
    next_id = BASELINE_IDS[(BASELINE_IDS.index(id_) + 1) % len(BASELINE_IDS)]
    updater = BASELINES[id_]()
    foreign_state = BASELINES[next_id]().init(3, 2, DEFAULT_CONFIGS[next_id], 0)
    with pytest.raises(ValueError, match="state"):
        updater.update(foreign_state, (), step_context(3, 2))


# --- Test 4: the shared loss ---------------------------------------------------------------------


def test_shared_loss_through_ftl() -> None:
    """(2, -4, 0) scales to losses (0.5, 1, 0); all-zero errors keep every leader."""
    weights = col0(drive(FollowTheLeader(), {}, [[[2.0], [-4.0], [0.0]]], 3))
    assert weights[1] == (0.0, 0.0, 1.0)  # expert 2's loss is the only 0
    scaled = col0(drive(FollowTheLeader(), {}, [[[20.0], [-40.0], [0.0]]], 3))
    assert scaled[1] == weights[1]  # the loss is invariant to the column's scale
    zeros = col0(drive(FollowTheLeader(), {}, [[[0.0], [0.0], [0.0]]], 3))
    assert zeros[1] == (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)  # all-zero losses tie everybody


# --- Test 5: Equal --------------------------------------------------------------------------------


@pytest.mark.parametrize("n_experts", [1, 3, 8])
def test_equal_weights_are_exact(n_experts: int) -> None:
    """Every weight is exactly 1 / E at every step, feedback or not."""
    rng = random.Random(4)
    steps = [[[rng.uniform(-1.0, 1.0)] for _ in range(n_experts)] for _ in range(10)]
    for weights in drive(Equal(), {}, steps, n_experts):
        for row in weights:
            for weight in row:
                assert weight == 1.0 / n_experts


# --- Test 6: FollowTheLeader ----------------------------------------------------------------------


def test_ftl_uniform_then_leaders() -> None:
    """Uniform before feedback; the leaders track the cumulative loss; exact ties split."""
    # Every column's maximum absolute error is 1, so the shared loss equals the
    # raw column, and every value is dyadic, so the cumulative losses are exact.
    steps = [
        [[0.5], [1.0], [0.25]],  # losses (0.5, 1.0, 0.25): leader 2
        [[0.25], [1.0], [0.5]],  # L = (0.75, 2.0, 0.75): exact tie between 0 and 2
        [[1.0], [0.5], [0.75]],  # L = (1.75, 2.5, 1.5): leader 2 alone
    ]
    weights = col0(drive(FollowTheLeader(), {}, steps, 3))
    assert weights[0] == (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)  # no feedback yet: all leaders
    assert weights[1] == (0.0, 0.0, 1.0)
    assert weights[2] == (0.5, 0.0, 0.5)
    assert weights[3] == (0.0, 0.0, 1.0)


# --- Test 7: Hedge --------------------------------------------------------------------------------


HEDGE_STEPS = [
    [[0.9], [0.4], [0.7]],
    [[0.2], [0.8], [0.5]],
    [[0.6], [0.1], [0.3]],
    [[0.0], [1.0], [0.5]],
]


def test_hedge_matches_the_naive_reference() -> None:
    """The weights equal exp(-eta L) / sum exp(-eta L), written naively."""
    eta = 1.0
    weights = col0(drive(Hedge(), {"eta": eta}, HEDGE_STEPS, 3))
    assert weights[0] == (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    cumulative = [0.0, 0.0, 0.0]
    for k, step in enumerate(HEDGE_STEPS):
        losses = shared_losses(step)
        for e in range(3):
            cumulative[e] = cumulative[e] + losses[e]
        total = 0.0
        for value in cumulative:
            total = total + math.exp(-eta * value)
        for e in range(3):
            expected = math.exp(-eta * cumulative[e]) / total
            assert weights[k + 1][e] == pytest.approx(expected, rel=1e-12)


def test_hedge_with_huge_eta_equals_ftl() -> None:
    """eta = 1e6 concentrates exactly like FTL while the leader is unique."""
    # Well-separated dyadic losses with a unique leader at every step: every
    # non-leader's exp(-1e6 * gap) is exactly 0.0, so Hedge's weights are
    # FTL's bit for bit at every step.
    steps = [
        [[0.5], [1.0], [0.25]],  # L = (0.5, 1.0, 0.25)
        [[0.25], [0.5], [1.0]],  # L = (0.75, 1.5, 1.25)
        [[0.75], [0.25], [1.0]],  # L = (1.5, 1.75, 2.25)
        [[0.5], [1.0], [0.25]],  # L = (2.0, 2.75, 2.5)
        [[0.25], [1.0], [0.5]],  # L = (2.25, 3.75, 3.0)
    ]
    hedge_weights = col0(drive(Hedge(), {"eta": 1e6}, steps, 3))
    ftl_weights = col0(drive(FollowTheLeader(), {}, steps, 3))
    assert hedge_weights == ftl_weights


def test_hedge_survives_a_large_loss_gap() -> None:
    """A cumulative gap of 1000 at eta = 10 gives no NaN and no all-zero column."""
    steps = [[[0.0], [1.0], [1.0]] for _ in range(1000)]
    weights = col0(drive(Hedge(), {"eta": 10.0}, steps, 3))
    final = weights[1000]
    for weight in final:
        assert weight == weight  # not NaN
        assert weight != float("inf")
    assert final[0] == 1.0  # the leader's score is exactly 1
    assert any(weight > 0.0 for weight in final)


# --- Test 8: FixedShare ---------------------------------------------------------------------------


def test_fixed_share_alpha_zero_matches_hedge() -> None:
    """With alpha = 0 Fixed-Share is Hedge with incremental updates."""
    hedge_weights = col0(drive(Hedge(), {"eta": 1.0}, HEDGE_STEPS, 3))
    share_weights = col0(drive(FixedShare(), {"eta": 1.0, "alpha": 0.0}, HEDGE_STEPS, 3))
    for hedge_row, share_row in zip(hedge_weights, share_weights, strict=False):
        for hedge_weight, share_weight in zip(hedge_row, share_row, strict=False):
            assert share_weight == pytest.approx(hedge_weight, abs=1e-12)


def test_fixed_share_alpha_one_matches_equal() -> None:
    """With alpha = 1 every weight is exactly Equal's."""
    share_weights = drive(FixedShare(), {"eta": 1.0, "alpha": 1.0}, HEDGE_STEPS, 3)
    equal_weights = drive(Equal(), {}, HEDGE_STEPS, 3)
    assert share_weights == equal_weights


def test_fixed_share_keeps_the_floor() -> None:
    """Every weight stays at or above alpha / E at every step."""
    weights = col0(drive(FixedShare(), {"eta": 1.0, "alpha": 0.1}, HEDGE_STEPS, 3))
    floor = 0.1 / 3 - 1e-15
    for row in weights:
        for weight in row:
            assert weight >= floor


def test_fixed_share_switches_faster_than_hedge() -> None:
    """After a switch of the best expert, Fixed-Share follows the new best sooner."""
    steps = [[[0.0], [1.0], [1.0]] for _ in range(50)] + [[[1.0], [0.0], [1.0]] for _ in range(50)]
    share = col0(drive(FixedShare(), {"eta": 1.0, "alpha": 0.05}, steps, 3))
    hedge = col0(drive(Hedge(), {"eta": 1.0}, steps, 3))
    assert share[100][1] > hedge[100][1]  # expert 1 is the new best
    assert share[100][1] > 0.5


# --- Test 9: AdaHedge -----------------------------------------------------------------------------


def test_adahedge_uniform_and_zero_gap_under_identical_losses() -> None:
    """Identical losses keep the weights uniform and the mixability gap at exactly 0."""
    updater = AdaHedge()
    state = updater.init(3, 1, {}, 0)
    for k in range(10):
        feedback = ()
        if k > 0:
            feedback = (matured([[1.0], [1.0], [1.0]], 1, step=k - 1),)
        weights, state = updater.update(state, feedback, step_context(3, 1, k))
        assert tuple(row[0] for row in weights) == (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
        assert state.gaps == [0.0]  # the private state, read on purpose: Delta stays 0


def test_adahedge_locks_onto_a_dominant_expert() -> None:
    """One expert at loss 0, the rest at 1: its weight only grows and passes 0.99."""
    steps = [[[0.0], [1.0], [1.0]] for _ in range(21)]
    weights = col0(drive(AdaHedge(), {}, steps, 3))
    for before, after in itertools.pairwise(weights):
        assert after[0] >= before[0]
    assert weights[20][0] > 0.99


def adahedge_reference(loss_steps: list[list[float]], n_experts: int) -> list[tuple[float, ...]]:
    """AdaHedge straight from Algorithm 1 of de Rooij et al. (2014), naive form."""
    cumulative = [0.0] * n_experts
    delta = 0.0
    out = []
    for losses in loss_steps:
        if delta == 0.0:
            low = min(cumulative)
            n_leaders = len([value for value in cumulative if value == low])
            weights = [1.0 / n_leaders if value == low else 0.0 for value in cumulative]
        else:
            eta = math.log(n_experts) / delta
            scores = [math.exp(-eta * value) for value in cumulative]
            total = 0.0
            for score in scores:
                total = total + score
            weights = [score / total for score in scores]
        mixed = 0.0
        for e in range(n_experts):
            mixed = mixed + weights[e] * losses[e]
        if delta == 0.0:
            low = min(cumulative)
            mix = min(losses[e] for e in range(n_experts) if cumulative[e] == low)
        else:
            eta = math.log(n_experts) / delta
            total = 0.0
            for e in range(n_experts):
                total = total + weights[e] * math.exp(-eta * losses[e])
            mix = -math.log(total) / eta
        gap = mixed - mix
        if gap < 0.0:
            gap = 0.0
        delta = delta + gap
        cumulative = [cumulative[e] + losses[e] for e in range(n_experts)]
        out.append(tuple(weights))
    return out


def test_adahedge_matches_the_papers_algorithm() -> None:
    """2000 steps of random errors match the paper's pseudocode within 1e-9."""
    rng = random.Random(1)
    n_experts = 4
    error_steps = [[[rng.uniform(0.0, 1.0)] for _ in range(n_experts)] for _ in range(2000)]
    weights = col0(drive(AdaHedge(), {}, error_steps, n_experts, trailing=0))
    reference = adahedge_reference([shared_losses(step) for step in error_steps], n_experts)
    assert len(weights) == len(reference)
    for k, (ours, theirs) in enumerate(zip(weights, reference, strict=False)):
        for a, b in zip(ours, theirs, strict=False):
            assert abs(a - b) <= 1e-9, (k, a, b)


# --- Test 10: regret sanity -----------------------------------------------------------------------


def _regret(updater: Any, config: dict[str, Any], n_experts: int, steps: int) -> float:
    """The mixture loss of the returned column-0 weights minus the best expert's total."""
    rng = random.Random(2)
    state = updater.init(n_experts, 1, config, 0)
    mixture = 0.0
    totals = [0.0] * n_experts
    for k in range(steps):
        raw = [rng.uniform(0.05, 1.0) for _ in range(n_experts)]
        raw[0] = raw[0] * 0.5  # expert 0 is biased lower
        feedback = ()
        if k > 0:
            feedback = (matured([[value] for value in raw], 1, step=k - 1),)
        weights, state = updater.update(state, feedback, step_context(n_experts, 1, k))
        losses = shared_losses([[value] for value in raw])
        mixed = 0.0
        for e in range(n_experts):
            mixed = mixed + weights[e][0] * losses[e]
        mixture = mixture + mixed
        for e in range(n_experts):
            totals[e] = totals[e] + losses[e]
    best = totals[0]
    for total in totals[1:]:
        best = total if total < best else best
    return mixture - best


def test_hedge_regret_bound() -> None:
    """Hedge at eta = sqrt(8 ln E / T) stays within sqrt(T ln E / 2) regret."""
    steps, n_experts = 2000, 4
    eta = math.sqrt(8.0 * math.log(n_experts) / steps)
    regret = _regret(Hedge(), {"eta": eta}, n_experts, steps)
    assert regret <= math.sqrt(steps * math.log(n_experts) / 2)


def test_adahedge_regret_bound() -> None:
    """AdaHedge stays within 2 sqrt(T ln E) + 16/3 ln E + 2 regret."""
    steps, n_experts = 2000, 4
    regret = _regret(AdaHedge(), {}, n_experts, steps)
    bound = 2.0 * math.sqrt(steps * math.log(n_experts)) + 16.0 / 3.0 * math.log(n_experts) + 2.0
    assert regret <= bound


# --- Test 11: columns are independent -------------------------------------------------------------


@pytest.mark.parametrize("id_", ["ftl", "hedge", "fixed_share", "adahedge"])
def test_columns_are_independent(id_: str) -> None:
    """Column 0 favours expert 0 and column 1 expert 1; each column leans its own way."""
    steps = [[[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] for _ in range(5)]
    config = dict(DEFAULT_CONFIGS[id_])
    weights = drive(BASELINES[id_](), config, steps, 3, horizon=2)
    final = weights[-1]
    assert final[0][0] > final[1][0]  # column 0 leans towards expert 0
    assert final[1][1] > final[0][1]  # column 1 leans towards expert 1


# --- Test 12: the state lives in the state --------------------------------------------------------


@pytest.mark.parametrize("id_", BASELINE_IDS)
def test_state_lives_in_the_state(id_: str) -> None:
    """Two states of one instance evolve exactly like two separate instances."""
    rng_a, rng_b = random.Random(1), random.Random(2)
    n_experts, horizon = 4, 2
    steps = 8
    series_a = [
        [[rng_a.uniform(-1.0, 1.0) for _ in range(horizon)] for _ in range(n_experts)]
        for _ in range(steps)
    ]
    series_b = [
        [[rng_b.uniform(-1.0, 1.0) for _ in range(horizon)] for _ in range(n_experts)]
        for _ in range(steps)
    ]
    shared = BASELINES[id_]()
    config = DEFAULT_CONFIGS[id_]
    state_a = shared.init(n_experts, horizon, config, 0)
    state_b = shared.init(n_experts, horizon, config, 1)
    alone_a = BASELINES[id_]()
    alone_b = BASELINES[id_]()
    alone_state_a = alone_a.init(n_experts, horizon, config, 0)
    alone_state_b = alone_b.init(n_experts, horizon, config, 1)
    for k in range(1, steps + 1):
        feedback_a = (matured(series_a[k - 1], horizon, step=k - 1),)
        feedback_b = (matured(series_b[k - 1], horizon, step=k - 1),)
        interleaved_a, state_a = shared.update(
            state_a, feedback_a, step_context(n_experts, horizon, k)
        )
        interleaved_b, state_b = shared.update(
            state_b, feedback_b, step_context(n_experts, horizon, k)
        )
        separate_a, alone_state_a = alone_a.update(
            alone_state_a, feedback_a, step_context(n_experts, horizon, k)
        )
        separate_b, alone_state_b = alone_b.update(
            alone_state_b, feedback_b, step_context(n_experts, horizon, k)
        )
        assert interleaved_a == separate_a
        assert interleaved_b == separate_b


# --- Test 16: AST guards --------------------------------------------------------------------------


ALLOWED_IMPORTS = {
    "math",
    "types",
    "dataclasses",
    "collections.abc",
    "typing",
    "evolvets.updater.interfaces",
}


def _tree() -> ast.Module:
    return ast.parse((SRC / "updater" / "baselines.py").read_text(encoding="utf-8"))


def test_imports_are_exactly_the_allowed_ones() -> None:
    """baselines.py imports only math, types, dataclasses, collections.abc, typing, the contract."""
    imported: set[str | None] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= ALLOWED_IMPORTS


def test_no_banned_calls() -> None:
    """No builtin sum, no math.fsum, no open; math attributes are exp, log, isfinite only."""
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
            assert node.attr in {"exp", "log", "isfinite"}, node.attr


def test_no_banned_math_imports() -> None:
    """``from math import ...`` may name only exp, log and isfinite."""
    for node in ast.walk(_tree()):
        if isinstance(node, ast.ImportFrom) and node.module == "math":
            for alias in node.names:
                assert alias.name in {"exp", "log", "isfinite"}, alias.name
