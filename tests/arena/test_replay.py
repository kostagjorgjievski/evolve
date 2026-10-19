"""Replay tests: timing, isolation, validation, seeds, determinism, AST guards (tests 2-14)."""

import ast
import math
from array import array
from dataclasses import fields
from hashlib import sha256
from pathlib import Path

import pytest

from evolvets.arena.errors import ArenaError, UpdaterFailure
from evolvets.arena.labels import LabelRelease
from evolvets.arena.replay import SEED_LIMIT, replay_series, series_seed
from evolvets.cache.store import SeriesForecasts
from evolvets.experts.pool import EXPERT_NAMES, first_origin, run_pool
from evolvets.updater.interfaces import Context, ForecastOutput, Matured

SRC = Path(__file__).parents[2] / "src" / "evolvets"
E = len(EXPERT_NAMES)
H5 = 5
N_ORIGINS = 60


def synth_values(n):
    """A deterministic, varied series of ``n`` finite values."""
    return [0.3 * math.sin(i / 5.0) + (i % 24) * 0.25 + i * 0.001 for i in range(n)]


def hand_rows(n_origins, horizon):
    """Deterministic ``[origin][expert][h]`` forecasts, independent of any series."""
    return [
        [tuple(0.5 + e + 0.01 * (o % 13) + 0.25 * h for h in range(horizon)) for e in range(E)]
        for o in range(n_origins)
    ]


def to_forecasts(rows, series_id="synthetic", first_origin_=0):
    """Build a SeriesForecasts straight from ``[origin][expert][h]`` rows via an array('d')."""
    horizon = len(rows[0][0])
    data = array("d")
    for origin_rows in rows:
        for expert_row in origin_rows:
            data.extend(expert_row)
    return SeriesForecasts(series_id, first_origin_, first_origin_ + len(rows) - 1, horizon, data)


def one_hot(expert, horizon):
    """All weight on one expert, per horizon step: a valid simplex column."""
    return [[1.0 if e == expert else 0.0] * horizon for e in range(E)]


def mixed_weights(horizon):
    """A fixed mixed matrix whose columns sum to 1 only within float rounding."""
    return [[0.1] * horizon, [0.2] * horizon, [0.7] * horizon, [0.0] * horizon, [0.0] * horizon]


def equal_weights(horizon):
    """Equal weight on every expert."""
    return [[0.2] * horizon for _ in range(E)]


def combine(weights, rows, origin, h):
    """The arena's left-to-right combination, one horizon index at a time."""
    total = 0.0
    for e in range(E):
        total = total + weights[e][h] * rows[origin][e][h]
    return total


ROWS5 = hand_rows(N_ORIGINS, H5)
FC5 = to_forecasts(ROWS5)
VALS5 = synth_values(N_ORIGINS + H5)


def replay5(updater, **kwargs):
    """replay_series over the standard hand-built 60-origin, horizon-5 fixture."""
    params = {
        "config": {"lr": 0.1},
        "seed": 5,
        "series_id": "synthetic",
        "forecasts": FC5,
        "values": VALS5,
        "period": 7,
        "horizon": H5,
        "end_origin": N_ORIGINS,
    }
    params.update(kwargs)
    return replay_series(updater, **params)


class Spy:
    """Records init args and every (feedback, context); returns cycling one-hot weights."""

    def __init__(self):
        self.init_args = None
        self.calls = []

    def init(self, n_experts, horizon, config, seed):
        self.init_args = (n_experts, horizon, dict(config), seed)
        return None

    def update(self, state, feedback, context):
        self.calls.append((feedback, context))
        star = context.step % len(context.expert_names)
        return (one_hot(star, context.horizon), None)


class FixedWeights:
    """Returns a fresh copy of the same valid weights every step."""

    def __init__(self, weights):
        self.weights = weights

    def init(self, n_experts, horizon, config, seed):
        return None

    def update(self, state, feedback, context):
        return ([list(row) for row in self.weights], None)


class FailsAt:
    """Behaves until ``at``, then raises the given exception."""

    def __init__(self, at, exc):
        self.at = at
        self.exc = exc

    def init(self, n_experts, horizon, config, seed):
        return None

    def update(self, state, feedback, context):
        if context.step == self.at:
            raise self.exc
        return (equal_weights(context.horizon), None)


class BadWeights:
    """Returns valid weights until ``fail_step``, then the given object."""

    def __init__(self, bad, fail_step=3):
        self.bad = bad
        self.fail_step = fail_step

    def init(self, n_experts, horizon, config, seed):
        return None

    def update(self, state, feedback, context):
        if context.step < self.fail_step:
            return (one_hot(context.step % E, context.horizon), None)
        return (self.bad, None)


class ReturnsLiteral:
    """Returns a fixed object instead of a (weights, state) tuple."""

    def __init__(self, result):
        self.result = result

    def init(self, n_experts, horizon, config, seed):
        return None

    def update(self, state, feedback, context):
        return self.result


class MutatingUpdater:
    """Returns the same nested list object each step and rewrites it on the next call."""

    def __init__(self):
        self.returned = []

    def init(self, n_experts, horizon, config, seed):
        return []

    def update(self, state, feedback, context):
        if self.returned:
            for row in self.returned[-1]:
                for j in range(len(row)):
                    row[j] = 99.0
        state.append(context.step)
        rows = mixed_weights(context.horizon)
        self.returned.append(rows)
        return (rows, state)


# --- Test 2: feedback timing ------------------------------------------------------------


def test_feedback_timing():
    """Steps 0..4 see no feedback; every later step sees exactly one matured origin."""
    spy = Spy()
    replay5(spy)
    assert len(spy.calls) == N_ORIGINS
    for k, (feedback, _context) in enumerate(spy.calls):
        if k < H5:
            assert feedback == (), f"step {k}"
            continue
        assert len(feedback) == 1, f"step {k}"
        matured = feedback[0]
        assert isinstance(matured, Matured)
        assert matured.step == k - H5
        s = k - H5
        for e in range(E):
            for h in range(H5):
                assert matured.errors[e][h] == ROWS5[s][e][h] - VALS5[s + 1 + h]
        star = s % E
        weights = one_hot(star, H5)
        for h in range(H5):
            assert matured.combined_errors[h] == combine(weights, ROWS5, s, h) - VALS5[s + 1 + h]


# --- Test 3: no look-ahead through values ----------------------------------------------


def test_no_look_ahead_through_values():
    """Changing values after a cut cannot change anything recorded at or before it."""
    values = synth_values(70)
    fc = to_forecasts(hand_rows(40, 2))
    cut = 15
    changed = list(values)
    for i in range(cut + 1, len(changed)):
        changed[i] = changed[i] * 0.5 - 3.0
    spy_base, spy_changed = Spy(), Spy()
    replay_series(spy_base, {}, 3, "s", fc, values, 6, 2, 40)
    replay_series(spy_changed, {}, 3, "s", fc, changed, 6, 2, 40)
    assert spy_base.calls[: cut + 1] == spy_changed.calls[: cut + 1]


def test_no_look_ahead_prefix_property_with_run_pool():
    """The same holds when the forecasts themselves are rebuilt from the altered values."""
    period = 24
    start = first_origin(period)
    values = synth_values(400)
    cut = 320  # inside the replay [287, 347)
    changed = values[: cut + 1] + [v * 0.5 - 3.0 for v in values[cut + 1 :]]
    fc_base = to_forecasts(run_pool(values, period, 5)[:60], first_origin_=start)
    fc_changed = to_forecasts(run_pool(changed, period, 5)[:60], first_origin_=start)
    spy_base, spy_changed = Spy(), Spy()
    replay_series(spy_base, {}, 9, "s", fc_base, values, period, 5, start + 60)
    replay_series(spy_changed, {}, 9, "s", fc_changed, changed, period, 5, start + 60)
    n_recorded = cut - start + 1
    assert spy_base.calls[:n_recorded] == spy_changed.calls[:n_recorded]


# --- Test 4: no premature target request ------------------------------------------------


def test_targets_called_only_when_matured(monkeypatch):
    """Every targets() call happens exactly at maturity; every origin is matured once."""
    original = LabelRelease.targets
    seen = []

    def recording(self, origin, now):
        seen.append((origin, now))
        return original(self, origin, now)

    monkeypatch.setattr(LabelRelease, "targets", recording)
    spy = Spy()
    trace = replay5(spy)
    assert len(trace.errors) == N_ORIGINS
    assert seen
    for origin, now in seen:
        assert origin + H5 == now, (origin, now)
    assert sorted(origin for origin, _ in seen) == list(range(N_ORIGINS))


# --- Test 5: the last H origins ----------------------------------------------------------


def test_last_h_origins_get_no_feedback_but_are_scored():
    """The updater never sees feedback for the last H origins; the trace still scores them."""
    spy = Spy()
    trace = replay5(spy)
    seen_steps = [m.step for feedback, _ in spy.calls for m in feedback]
    assert seen_steps == list(range(N_ORIGINS - H5))
    assert len(trace.errors) == N_ORIGINS
    for row in trace.errors:
        assert len(row) == H5


# --- Test 6: Context and Matured ---------------------------------------------------------


def test_context_and_matured_fields_and_content():
    """Exact field sets, and every context carries only origin-relative knowledge."""
    assert [f.name for f in fields(Context)] == [
        "step",
        "season_index",
        "period",
        "horizon",
        "expert_names",
        "forecasts",
    ]
    assert [f.name for f in fields(Matured)] == ["step", "errors", "combined_errors"]
    spy = Spy()
    replay5(spy)
    for k, (_feedback, context) in enumerate(spy.calls):
        assert context.step == k
        assert context.season_index == k % 7
        assert context.period == 7
        assert context.horizon == H5
        assert context.expert_names == EXPERT_NAMES
        assert context.forecasts == FC5.at(k)


# --- Test 7: the combination -------------------------------------------------------------


@pytest.mark.parametrize("expert", range(E))
def test_one_hot_weights_equal_expert_errors(expert):
    """One-hot weights reproduce that expert's errors exactly."""
    trace = replay5(FixedWeights(one_hot(expert, H5)))
    for o in range(N_ORIGINS):
        for h in range(H5):
            assert trace.errors[o][h] == ROWS5[o][expert][h] - VALS5[o + 1 + h]


def test_mixed_weights_match_left_to_right_formula():
    """A fixed mixed matrix gives errors equal to the arena's formula, bit for bit."""
    weights = mixed_weights(H5)
    trace = replay5(FixedWeights(weights))
    for o in range(N_ORIGINS):
        for h in range(H5):
            assert trace.errors[o][h] == combine(weights, ROWS5, o, h) - VALS5[o + 1 + h]


# --- Test 8: isolation from mutating updaters -------------------------------------------


def test_mutating_updater_cannot_change_the_trace():
    """Mutating the lists it returned (or its state) leaves the trace untouched."""
    mutated = replay5(MutatingUpdater())
    honest = replay5(FixedWeights(mixed_weights(H5)))
    assert mutated == honest
    attacker = MutatingUpdater()
    replay5(attacker)
    assert attacker.returned[0][0][0] == 99.0  # the mutation really happened


# --- Test 9: invalid weights ------------------------------------------------------------


class LyingList(list):
    """A list subclass whose __iter__ yields valid rows first, garbage after."""

    def __init__(self, rows):
        super().__init__(rows)
        self.passes = 0

    def __iter__(self):
        self.passes += 1
        if self.passes <= 1:
            return super().__iter__()
        return iter([object()])


class LyingFloat(float):
    """A float subclass whose __mul__ lies."""

    def __mul__(self, other):
        return 42.0


def _bad_weights_cases():
    short_outer = equal_weights(H5)[:-1]
    wrong_inner = equal_weights(H5)
    wrong_inner[2] = [0.2] * (H5 - 1)
    negative = equal_weights(H5)
    negative[1][0] = -0.2
    nan = equal_weights(H5)
    nan[0][0] = float("nan")
    inf = equal_weights(H5)
    inf[0][0] = float("inf")
    true_entry = equal_weights(H5)
    true_entry[3][2] = True
    str_entry = equal_weights(H5)
    str_entry[3][2] = "0.2"
    off_by_1e6 = equal_weights(H5)
    off_by_1e6[0][0] = 0.200001
    lying_float_rows = equal_weights(H5)
    lying_float_rows[0] = [LyingFloat(0.2)] * H5
    huge_int_rows = equal_weights(H5)
    huge_int_rows[0][0] = 10**400
    return [
        ("wrong_outer", short_outer),
        ("wrong_inner", wrong_inner),
        ("negative", negative),
        ("nan", nan),
        ("inf", inf),
        ("bool", true_entry),
        ("str", str_entry),
        ("none", None),
        ("sum_off_by_1e_6", off_by_1e6),
        ("lying_list", LyingList(equal_weights(H5))),
        ("lying_float", lying_float_rows),
        ("huge_int", huge_int_rows),
    ]


@pytest.mark.parametrize("name,bad", _bad_weights_cases(), ids=[c[0] for c in _bad_weights_cases()])
def test_invalid_weights_rejected(name, bad):
    """Each malformed weights object fails as invalid_weights at the failing step."""
    with pytest.raises(UpdaterFailure) as exc:
        replay5(BadWeights(bad, fail_step=3))
    assert exc.value.kind == "invalid_weights"
    assert exc.value.step == 3


def test_column_summing_within_rounding_is_accepted():
    """(0.1, 0.2, 0.7, 0.0, 0.0) sums to 1 only approximately; it is accepted."""
    trace = replay5(FixedWeights(mixed_weights(H5)))
    assert len(trace.errors) == N_ORIGINS


# --- Test 10: updater errors -------------------------------------------------------------


class InitFails:
    def init(self, n_experts, horizon, config, seed):
        raise ValueError("boom")

    def update(self, state, feedback, context):  # pragma: no cover - never reached
        raise AssertionError


class InitInterrupts:
    def init(self, n_experts, horizon, config, seed):
        raise KeyboardInterrupt()

    def update(self, state, feedback, context):  # pragma: no cover - never reached
        raise AssertionError


def test_init_failure():
    with pytest.raises(UpdaterFailure) as exc:
        replay5(InitFails())
    assert exc.value.kind == "failed"
    assert exc.value.step is None
    assert exc.value.series_id == "synthetic"
    assert exc.value.message.startswith("ValueError")


def test_update_failure_reports_step():
    with pytest.raises(UpdaterFailure) as exc:
        replay5(FailsAt(7, RuntimeError("nope")))
    assert exc.value.kind == "failed"
    assert exc.value.step == 7
    assert exc.value.message.startswith("RuntimeError")


@pytest.mark.parametrize(
    "result", [[one_hot(0, H5), None], (one_hot(0, H5), None, None), "weights"]
)
def test_update_bad_return_shape(result):
    with pytest.raises(UpdaterFailure) as exc:
        replay5(ReturnsLiteral(result))
    assert exc.value.kind == "failed"
    assert "must return" in exc.value.message


def test_long_exception_message_truncated():
    with pytest.raises(UpdaterFailure) as exc:
        replay5(FailsAt(2, RuntimeError("a" * 5000)))
    assert len(exc.value.message) == 1000


def test_keyboard_interrupt_propagates():
    with pytest.raises(KeyboardInterrupt):
        replay5(InitInterrupts())


# --- Test 11: seeds ---------------------------------------------------------------------


def test_series_seed_matches_formula():
    digest = sha256(b"evolvets/replay/v1\x00" + str(0).encode() + b"\x00" + b"alpha").digest()
    expected = int.from_bytes(digest[:8], "big") & (2**63 - 1)
    assert SEED_LIMIT == 2**63
    assert series_seed(0, "alpha") == expected
    assert series_seed(0, "beta") != series_seed(0, "alpha")


@pytest.mark.parametrize("bad", [True, -1, 2**63])
def test_series_seed_rejects(bad):
    with pytest.raises(ArenaError):
        series_seed(bad, "s")


def test_init_receives_series_seed():
    spy = Spy()
    replay_series(spy, {}, 5, "synthetic", FC5, VALS5, 7, H5, N_ORIGINS)
    assert spy.init_args is not None
    n_experts, horizon, _config, seed = spy.init_args
    assert n_experts == E
    assert horizon == H5
    assert seed == series_seed(5, "synthetic")


# --- Test 12: argument validation -------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"end_origin": 0},  # == first_origin
        {"end_origin": 61},  # last_origin + 2
        {"values": VALS5[: N_ORIGINS + H5 - 1]},  # one target short
        {"horizon": 4},  # differs from the forecasts' rows
        {"period": 1},
    ],
    ids=["end_at_first", "end_past_cache", "short_values", "horizon_mismatch", "period_one"],
)
def test_argument_validation(kwargs):
    """Bad arguments raise ArenaError before init is ever called."""
    spy = Spy()
    with pytest.raises(ArenaError):
        replay5(spy, **kwargs)
    assert spy.init_args is None


# --- Test 13: determinism ---------------------------------------------------------------


def test_two_replays_are_equal():
    a = replay5(FixedWeights(one_hot(2, H5)))
    b = replay5(FixedWeights(one_hot(2, H5)))
    assert a == b


# --- Test 14: AST guards ----------------------------------------------------------------


def test_updater_interfaces_imports_nothing_from_evolvets():
    tree = ast.parse((SRC / "updater" / "interfaces.py").read_text(encoding="utf-8"))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            modules.append(node.module)
    for module in modules:
        assert not (module == "evolvets" or module.startswith("evolvets."))


@pytest.mark.parametrize("name", ["replay.py", "labels.py"])
def test_no_float_sum_or_statistics(name):
    """replay.py and labels.py never use builtin sum, math.fsum or statistics."""
    tree = ast.parse((SRC / "arena" / name).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "sum":
                raise AssertionError(f"{name}: builtin sum()")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "fsum":
                raise AssertionError(f"{name}: math.fsum()")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("statistics")
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not (node.module or "").startswith("statistics")


def test_replay_reads_values_only_through_label_release():
    """The name 'values' appears only in the length check and the LabelRelease call."""
    tree = ast.parse((SRC / "arena" / "replay.py").read_text(encoding="utf-8"))
    allowed = set()
    for node in ast.walk(tree):
        is_release_call = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "LabelRelease"
        )
        is_len_check = (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Name)
            and node.left.func.id == "len"
        )
        if is_release_call or is_len_check:
            call = node if is_release_call else node.left
            for arg in call.args:
                if isinstance(arg, ast.Name) and arg.id == "values":
                    allowed.add(id(arg))
    used = [n for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == "values"]
    assert len(used) == 2
    assert all(id(node) in allowed for node in used)


# --- TASK_016 PR A: the forecast-v2 updater interface ------------------------------------------


CONST_ROW = (0.5, -1.25, 2.0, 0.0, 0.75)


class ConstantForecast:
    """A forecast-v2 updater answering with one fixed row; records the feedback it gets."""

    def __init__(self, row):
        self.row = tuple(row)
        self.feedbacks = []

    def init(self, n_experts, horizon, config, seed):
        return None

    def update(self, state, feedback, context):
        self.feedbacks.append(feedback)
        return (ForecastOutput(tuple(self.row)), None)


class Alternating:
    """Equal weights on even steps, a fixed forecast row on odd steps (forecast-v2)."""

    def __init__(self, row):
        self.row = tuple(row)

    def init(self, n_experts, horizon, config, seed):
        return None

    def update(self, state, feedback, context):
        if context.step % 2 == 0:
            return (equal_weights(context.horizon), None)
        return (ForecastOutput(tuple(self.row)), None)


class SelfCombined:
    """Records every (feedback, context); answers with its own equal-weight combination.

    The combination follows the replay's own loop order, so under forecast-v2
    its trace is bit-identical to a weights-v1 equal-weights replay.
    """

    def __init__(self):
        self.calls = []

    def init(self, n_experts, horizon, config, seed):
        return None

    def update(self, state, feedback, context):
        self.calls.append((feedback, context))
        weight = 1.0 / len(context.expert_names)
        out = []
        for h in range(context.horizon):
            total = 0.0
            for row in context.forecasts:
                total = total + weight * row[h]
            out.append(total)
        return (ForecastOutput(tuple(out)), None)


class RecordingEqualWeights:
    """Records every (feedback, context); returns equal weights under weights-v1."""

    def __init__(self):
        self.calls = []

    def init(self, n_experts, horizon, config, seed):
        return None

    def update(self, state, feedback, context):
        self.calls.append((feedback, context))
        return (equal_weights(context.horizon), None)


@pytest.mark.parametrize("bad", ["weights-v2", "", None, 1])
def test_interface_checked_before_anything_runs(bad):
    """An unknown interface is refused before validation, cache reads or init."""
    spy = Spy()
    with pytest.raises(ArenaError, match="interface"):
        replay5(spy, interface=bad)
    assert spy.init_args is None


def test_forecast_output_refused_under_weights_v1():
    """A ForecastOutput under weights-v1 is invalid_weights: it is no weights container."""
    with pytest.raises(UpdaterFailure) as exc:
        replay5(BadWeights(ForecastOutput((0.2,) * H5), fail_step=3))
    assert exc.value.kind == "invalid_weights"
    assert exc.value.step == 3


def test_forecast_output_scores_as_given():
    """forecast-v2 scores a row exactly as given: trace and Matured errors are c[h] - target."""
    updater = ConstantForecast(CONST_ROW)
    trace = replay5(updater, interface="forecast-v2")
    assert len(trace.errors) == N_ORIGINS
    for o in range(N_ORIGINS):
        for h in range(H5):
            assert trace.errors[o][h] == CONST_ROW[h] - VALS5[o + 1 + h]
    # feedback for step s arrives exactly at s + H; its combined_errors are the same values
    for k, feedback in enumerate(updater.feedbacks):
        if k < H5:
            assert feedback == (), f"step {k}"
            continue
        assert len(feedback) == 1, f"step {k}"
        matured = feedback[0]
        assert matured.step == k - H5
        for h in range(H5):
            assert matured.combined_errors[h] == CONST_ROW[h] - VALS5[(k - H5) + 1 + h]


def test_mixed_outputs():
    """Weights and forecasts may alternate: each trace row is one or the other, per step."""
    row = tuple(0.3 + 0.1 * h for h in range(H5))
    trace = replay5(Alternating(row), interface="forecast-v2")
    weights = equal_weights(H5)
    for o in range(N_ORIGINS):
        for h in range(H5):
            if o % 2 == 0:
                assert trace.errors[o][h] == combine(weights, ROWS5, o, h) - VALS5[o + 1 + h]
            else:
                assert trace.errors[o][h] == row[h] - VALS5[o + 1 + h]


def test_forecast_v2_equivalent_to_weights_v1():
    """Self-combining equal weights under forecast-v2 replays bit-identically to weights-v1."""
    assert replay5(SelfCombined(), interface="forecast-v2") == replay5(
        FixedWeights(equal_weights(H5))
    )


class FloatSubclass(float):
    """A benign float subclass: still not an exact float."""


class RaisingIter:
    """An object whose __iter__ raises."""

    def __iter__(self):
        raise RuntimeError("iter boom")


class ForecastSubclass(ForecastOutput):
    """A ForecastOutput subclass: not an exact ForecastOutput."""


def _invalid_forecast_cases():
    nan_row = (float("nan"), 0.2, 0.2, 0.2, 0.2)
    inf_row = (float("inf"), 0.2, 0.2, 0.2, 0.2)
    neg_inf_row = (float("-inf"), 0.2, 0.2, 0.2, 0.2)
    bool_row = (True, 0.2, 0.2, 0.2, 0.2)
    huge_int_row = (10**400, 0.0, 0.0, 0.0, 0.0)
    subclass_row = (FloatSubclass(0.2), 0.2, 0.2, 0.2, 0.2)
    return [
        ("wrong_length", ForecastOutput((0.1, 0.2, 0.3))),
        ("nan", ForecastOutput(nan_row)),
        ("inf", ForecastOutput(inf_row)),
        ("neg_inf", ForecastOutput(neg_inf_row)),
        ("too_large", ForecastOutput((1e13, 0.0, 0.0, 0.0, 0.0))),
        ("huge_int", ForecastOutput(huge_int_row)),
        ("bool", ForecastOutput(bool_row)),
        ("float_subclass", ForecastOutput(subclass_row)),
        ("generator", ForecastOutput(v for v in (0.2,) * H5)),
        ("output_subclass", ForecastSubclass((0.2,) * H5)),
        ("iter_raises", ForecastOutput(RaisingIter())),
    ]


@pytest.mark.parametrize(
    "name,bad", _invalid_forecast_cases(), ids=[c[0] for c in _invalid_forecast_cases()]
)
def test_invalid_forecasts_rejected(name, bad):
    """Each malformed forecast output fails as failed/invalid_forecast at the failing step."""
    with pytest.raises(UpdaterFailure) as exc:
        replay5(BadWeights(bad, fail_step=3), interface="forecast-v2")
    assert exc.value.kind == "failed"
    assert exc.value.message.startswith("invalid_forecast:")
    assert exc.value.step == 3


def test_no_new_information_reaches_a_forecast_v2_updater():
    """A forecast-v2 updater receives exactly what a weights-v1 updater receives.

    Both return the same combination here — the forecast-v2 one through its
    own self-combination — so every Context and Matured is field-for-field
    equal, and feedback for step s still arrives exactly at s + H.
    """
    weights_updater = RecordingEqualWeights()
    forecast_updater = SelfCombined()
    replay5(weights_updater)
    replay5(forecast_updater, interface="forecast-v2")
    assert forecast_updater.calls == weights_updater.calls
    for k, (feedback, _context) in enumerate(forecast_updater.calls):
        if k < H5:
            assert feedback == (), f"step {k}"
        else:
            assert len(feedback) == 1 and feedback[0].step == k - H5, f"step {k}"
