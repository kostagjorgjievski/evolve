"""Tests for the forecast-output updaters (TASK_016 PR B, tests 1-9 and 11).

The unit tests drive ``init``/``update`` directly with hand-built Context and
Matured objects; the replay tests run ``replay_series`` over hand-built
forecasts (``tests/updater/test_learned.py``'s ``HandForecasts`` pattern), and
the arena tests replay the synthetic-dev fixture of
``tests/updater/test_baselines_replay.py``. Test 10 (the CLI) and the
``arena replay`` dispatch moved to ``feat/forecast-updater-cli`` under the
task's size split; they follow this PR in the stack.
"""

from __future__ import annotations

import ast
import importlib
import random
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.dev import load_dev_pool
from evolvets.arena.errors import UpdaterFailure
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.arena.replay import replay_series
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.ledger.canonical import canonical_bytes
from evolvets.updater.baselines import BASELINE_IDS, FixedShare, fixed_share
from evolvets.updater.forecast import (
    FORECAST_DEFAULT_CONFIGS,
    FORECAST_UPDATER_IDS,
    FORECAST_UPDATERS,
    BiasShare,
    bias_share,
)
from evolvets.updater.interfaces import Context, Matured
from evolvets.updater.learned import LEARNED_IDS

SRC = Path(__file__).parents[2] / "src" / "evolvets"
CONFIG: dict[str, Any] = dict(FORECAST_DEFAULT_CONFIGS["bias_share"])


def names(n_experts: int) -> tuple[str, ...]:
    """``n_experts`` placeholder expert names."""
    return tuple(f"e{e}" for e in range(n_experts))


def step_context(
    n_experts: int,
    horizon: int,
    step: int = 0,
    forecasts: tuple[tuple[float, ...], ...] | None = None,
) -> Context:
    """A valid context: constant forecasts (or ``forecasts``), the init'ed horizon."""
    rows = (
        forecasts
        if forecasts is not None
        else tuple(tuple(0.5 for _ in range(horizon)) for _ in range(n_experts))
    )
    return Context(
        step=step,
        season_index=step % 7,
        period=7,
        horizon=horizon,
        expert_names=names(n_experts),
        forecasts=rows,
    )


def matured(error_rows: list[list[float]], combined: list[float], step: int = 0) -> Matured:
    """One Matured with ``error_rows`` ([expert][h]) errors and ``combined`` combined errors."""
    return Matured(
        step=step,
        errors=tuple(tuple(row) for row in error_rows),
        combined_errors=tuple(combined),
    )


class HandForecasts:
    """``[origin][expert][h]`` rows behind the replay loop's Forecasts protocol."""

    def __init__(self, rows: list[tuple[tuple[float, ...], ...]], series_id: str = "probe"):
        self.series_id = series_id
        self.first_origin = 0
        self.last_origin = len(rows) - 1
        self._rows = rows

    def at(self, origin: int) -> tuple[tuple[float, ...], ...]:
        return self._rows[origin]


def hand_fixture(
    n_experts: int, horizon: int, n_origins: int, seed: int = 3
) -> tuple[list[tuple[tuple[float, ...], ...]], list[float]]:
    """Deterministic near-target forecasts and their targets, independent of any cache."""
    rng = random.Random(seed)
    values = [rng.uniform(-2.0, 2.0) for _ in range(n_origins + horizon)]
    rows = []
    for o in range(n_origins):
        experts = []
        for _ in range(n_experts):
            experts.append(
                tuple(values[o + 1 + h] + rng.uniform(-0.5, 0.5) for h in range(horizon))
            )
        rows.append(tuple(experts))
    return rows, values


def replay_forecast(
    updater: Any,
    rows: list[tuple[tuple[float, ...], ...]],
    values: list[float],
    n_experts: int,
    horizon: int,
    n_origins: int,
    config: dict[str, Any],
    seed: int = 0,
    interface: str = "forecast-v2",
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
        interface=interface,
    )


# --- Test 1: the registry -------------------------------------------------------------------------


def test_registry_shape() -> None:
    """FORECAST_UPDATER_IDS lists the registry keys in order; ids are disjoint from the others."""
    assert FORECAST_UPDATER_IDS == ("bias_share",)
    assert tuple(FORECAST_UPDATERS) == FORECAST_UPDATER_IDS
    assert set(FORECAST_DEFAULT_CONFIGS) == set(FORECAST_UPDATER_IDS)
    assert not (set(FORECAST_UPDATER_IDS) & set(BASELINE_IDS))
    assert not (set(FORECAST_UPDATER_IDS) & set(LEARNED_IDS))
    module = importlib.import_module("evolvets.updater.forecast")
    assert module.bias_share is FORECAST_UPDATERS["bias_share"]
    assert FORECAST_UPDATERS["bias_share"]() is not FORECAST_UPDATERS["bias_share"]()


def test_default_config_is_accepted_and_read_only() -> None:
    """The default config inits cleanly, and neither layer of it can be mutated."""
    assert bias_share().init(4, 3, FORECAST_DEFAULT_CONFIGS["bias_share"], 0)
    for mapping in (FORECAST_DEFAULT_CONFIGS, *FORECAST_DEFAULT_CONFIGS.values()):
        with pytest.raises(TypeError):
            mapping["x"] = 1  # type: ignore[index]


# --- Test 2: config validation --------------------------------------------------------------------


def _bad_configs() -> list[tuple[str, dict[str, Any], str]]:
    """(label, config, message needle) for every bias_share init refusal."""
    return [
        ("missing-key", {k: v for k, v in CONFIG.items() if k != "bias_rate"}, "config keys"),
        ("extra-key", CONFIG | {"extra": 1.0}, "config keys"),
        ("bias-rate-negative", CONFIG | {"bias_rate": -0.1}, "bias_rate"),
        ("bias-rate-above-one", CONFIG | {"bias_rate": 1.5}, "bias_rate"),
        ("bias-rate-nan", CONFIG | {"bias_rate": float("nan")}, "bias_rate"),
        ("bias-rate-inf", CONFIG | {"bias_rate": float("inf")}, "bias_rate"),
        ("bias-rate-bool", CONFIG | {"bias_rate": True}, "bias_rate"),
        ("bias-rate-str", CONFIG | {"bias_rate": "0.1"}, "bias_rate"),
        ("bad-eta", CONFIG | {"eta": 0.0}, "eta"),
        ("bad-alpha", CONFIG | {"alpha": 2.0}, "alpha"),
    ]


_CASES = _bad_configs()
_CASE_IDS = [label for label, _, _ in _CASES]


@pytest.mark.parametrize(("label", "config", "needle"), _CASES, ids=_CASE_IDS)
def test_init_rejects_bad_configs(label: str, config: dict[str, Any], needle: str) -> None:
    """A missing or extra key, a bad bias_rate, eta or alpha, raises ValueError."""
    with pytest.raises(ValueError, match=needle):
        bias_share().init(2, 2, config, 0)


def test_seed_is_validated_then_ignored() -> None:
    """A bool or str seed is refused; the seed value never reaches the output."""
    with pytest.raises(ValueError, match="seed"):
        bias_share().init(2, 2, CONFIG, True)
    with pytest.raises(ValueError, match="seed"):
        bias_share().init(2, 2, CONFIG, "0")
    first, second = bias_share(), bias_share()
    context = step_context(2, 2, 0)
    output_first, _ = first.update(first.init(2, 2, CONFIG, 0), (), context)
    output_second, _ = second.update(second.init(2, 2, CONFIG, 123456), (), context)
    assert output_first == output_second


# --- Test 3: reduces to Fixed-Share ---------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_pool(tmp_path_factory: pytest.TempPathFactory):
    """The dev pool of a synthetic-dev cache with one series per family (TASK_007 fixture)."""
    base = tmp_path_factory.mktemp("forecast-synthetic")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-dev", base, base, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    return load_dev_pool("synthetic-dev", cache_root, base, base)


def test_bias_rate_zero_reduces_to_fixed_share(synthetic_pool) -> None:
    """With bias_rate = 0 every per-series MASE equals Fixed-Share's, exactly."""
    forecast = replay_dataset(
        bias_share,
        CONFIG | {"bias_rate": 0.0},
        0,
        synthetic_pool,
        interface="forecast-v2",
    )
    baseline = replay_dataset(fixed_share, {"eta": 1.0, "alpha": 0.01}, 0, synthetic_pool)
    assert forecast.status == "ok"
    assert baseline.status == "ok"
    assert [score.mase for score in forecast.series] == [score.mase for score in baseline.series]


# --- Test 4: the bias recursion by hand ----------------------------------------------------------


def _drive_offset(updater: Any, rate: float, horizon: int, steps: int) -> tuple[list[Any], Any]:
    """Drive one expert whose forecast is a constant ``d`` above the target, ``horizon``-deep.

    The Matured for step ``s`` arrives at ``s + horizon`` carrying the combined
    errors the output actually took, exactly as the arena would hand them over.
    """
    d = 0.8
    forecast_row = tuple(1.0 + (h + 1) * d for h in range(horizon))
    target_row = tuple(forecast_row[h] - d for h in range(horizon))
    state = updater.init(1, horizon, {"eta": 1.0, "alpha": 0.01, "bias_rate": rate}, 0)
    outputs = []
    combined: dict[int, tuple[float, ...]] = {}
    for step in range(steps):
        feedback: tuple[Matured, ...] = ()
        if step >= horizon:
            feedback = (
                Matured(
                    step=step - horizon,
                    errors=((d,) * horizon,),
                    combined_errors=combined[step - horizon],
                ),
            )
        output, state = updater.update(
            state, feedback, step_context(1, horizon, step, (forecast_row,))
        )
        outputs.append(output.values)
        combined[step] = tuple(output.values[h] - target_row[h] for h in range(horizon))
    return outputs, state


def test_bias_recursion_by_hand() -> None:
    """One expert with a constant offset +d: the bias tracks d at rate bias_rate, exactly."""
    rate, horizon, steps = 0.25, 2, 8
    outputs, state = _drive_offset(bias_share(), rate, horizon, steps)
    expected = [0.0] * horizon
    for step, out in enumerate(outputs):
        if step >= horizon:  # the feedback for step - horizon moved the bias first
            for h in range(horizon):
                expected[h] = expected[h] + rate * (0.8 - expected[h])
        assert out == tuple(1.0 + (h + 1) * 0.8 - expected[h] for h in range(horizon)), (
            f"step {step}"
        )
    # the first maturation moved the bias toward d by exactly rate * (d - 0)
    first, first_state = _drive_offset(bias_share(), rate, horizon, horizon + 1)
    assert first_state.bias == [rate * 0.8] * horizon
    assert first[horizon] == tuple(1.0 + (h + 1) * 0.8 - rate * 0.8 for h in range(horizon))
    # applied holds one entry per not-yet-matured step, never more than horizon + 1
    assert len(first_state.applied) <= horizon + 1
    assert len(state.applied) <= horizon + 1


def test_bias_rate_one_sets_the_bias_to_the_last_residual() -> None:
    """With bias_rate = 1 the bias is the last residual: here d, forever."""
    horizon = 2
    outputs, state = _drive_offset(bias_share(), 1.0, horizon, 6)
    assert state.bias == [0.8] * horizon
    for step in range(horizon, 6):
        assert outputs[step] == tuple(1.0 + (h + 1) * 0.8 - 0.8 for h in range(horizon))


# --- Test 5: removes a level shift ---------------------------------------------------------------


def _shift_fixture(
    n_origins: int = 220, horizon: int = 2, shift_at: int = 60
) -> tuple[list[tuple[tuple[float, ...], ...]], list[float], int]:
    """Targets flat at 0, stepping up to 10 at ``shift_at``; both experts never adapt."""
    values = [0.0 if t < shift_at else 10.0 for t in range(n_origins + horizon)]
    rows = [
        (tuple(0.0 for _ in range(horizon)), tuple(0.5 for _ in range(horizon)))
        for _ in range(n_origins)
    ]
    return rows, values, shift_at


def _mae_over(trace: Any, horizon: int, first: int, last: int) -> float:
    """The mean absolute trace error over origins [first, last)."""
    total = 0.0
    count = 0
    for origin in range(first, last):
        for h in range(horizon):
            total = total + abs(trace.errors[origin][h])
            count = count + 1
    return total / count


def test_removes_a_level_shift() -> None:
    """After a level shift bias_share's mean absolute error beats Fixed-Share's."""
    rows, values, shift_at = _shift_fixture()
    n_origins, horizon = len(rows), len(rows[0][0])
    forecast = replay_forecast(bias_share(), rows, values, 2, horizon, n_origins, CONFIG)
    baseline = replay_forecast(
        FixedShare(),
        rows,
        values,
        2,
        horizon,
        n_origins,
        {"eta": 1.0, "alpha": 0.01},
        interface="weights-v1",
    )
    after = shift_at + 50  # the bias has converged long since
    assert _mae_over(forecast, horizon, after, n_origins) < _mae_over(
        baseline, horizon, after, n_origins
    )


# --- Test 6: per-series state --------------------------------------------------------------------


def test_one_instance_serves_two_series_side_by_side() -> None:
    """Two states of one instance evolve exactly like two separate instances."""
    rng_a, rng_b = random.Random(1), random.Random(2)
    n_experts, horizon, steps = 2, 2, 8
    shared = bias_share()
    state_a = shared.init(n_experts, horizon, CONFIG, 0)
    state_b = shared.init(n_experts, horizon, CONFIG, 1)
    alone_a, alone_b = bias_share(), bias_share()
    alone_state_a = alone_a.init(n_experts, horizon, CONFIG, 0)
    alone_state_b = alone_b.init(n_experts, horizon, CONFIG, 1)
    # the arena's first call carries empty feedback; step 0's bias is applied there
    _, state_a = shared.update(state_a, (), step_context(n_experts, horizon, 0))
    _, state_b = shared.update(state_b, (), step_context(n_experts, horizon, 0))
    _, alone_state_a = alone_a.update(alone_state_a, (), step_context(n_experts, horizon, 0))
    _, alone_state_b = alone_b.update(alone_state_b, (), step_context(n_experts, horizon, 0))
    for k in range(1, steps + 1):
        series_a = [[rng_a.uniform(-1.0, 1.0) for _ in range(horizon)] for _ in range(n_experts)]
        series_b = [[rng_b.uniform(-1.0, 1.0) for _ in range(horizon)] for _ in range(n_experts)]
        forecasts_a = tuple(tuple(row) for row in series_a)
        forecasts_b = tuple(tuple(row) for row in series_b)
        feedback_a = (matured(series_a, [0.0] * horizon, step=k - 1),)
        feedback_b = (matured(series_b, [0.1] * horizon, step=k - 1),)
        interleaved_a, state_a = shared.update(
            state_a, feedback_a, step_context(n_experts, horizon, k, forecasts_a)
        )
        interleaved_b, state_b = shared.update(
            state_b, feedback_b, step_context(n_experts, horizon, k, forecasts_b)
        )
        separate_a, alone_state_a = alone_a.update(
            alone_state_a, feedback_a, step_context(n_experts, horizon, k, forecasts_a)
        )
        separate_b, alone_state_b = alone_b.update(
            alone_state_b, feedback_b, step_context(n_experts, horizon, k, forecasts_b)
        )
        assert interleaved_a == separate_a
        assert interleaved_b == separate_b


# --- Test 7: missing maturation ------------------------------------------------------------------


def test_missing_maturation_raises_value_error() -> None:
    """A Matured whose step never had a bias applied makes update raise ValueError."""
    updater = bias_share()
    state = updater.init(2, 2, CONFIG, 0)
    bogus = matured([[-0.1, 0.2], [0.3, -0.4]], [0.0, 0.0], step=5)
    with pytest.raises(ValueError, match="step 5"):
        updater.update(state, (bogus,), step_context(2, 2, 0))


class Injecting:
    """Wraps bias_share, appending one Matured whose step was never forecast.

    The extra Matured rides only the non-empty feedback the arena hands over,
    so the failure lands at the first maturation, where the bogus step pops.
    """

    def __init__(self) -> None:
        self.inner = BiasShare()

    def init(self, n_experts: int, horizon: int, config: object, seed: int) -> object:
        return self.inner.init(n_experts, horizon, config, seed)

    def update(self, state: object, feedback: object, context: Context) -> object:
        if not feedback:
            return self.inner.update(state, feedback, context)
        extra = Matured(
            step=999,
            errors=tuple(tuple(0.0 for _ in range(context.horizon)) for _ in context.expert_names),
            combined_errors=tuple(0.0 for _ in range(context.horizon)),
        )
        return self.inner.update(state, (*feedback, extra), context)


def test_a_replay_that_injects_an_unknown_step_reports_failed() -> None:
    """The arena never sends an unknown step; an updater that sees one fails cleanly."""
    rows, values = hand_fixture(2, 2, 20)
    with pytest.raises(UpdaterFailure) as excinfo:
        replay_forecast(Injecting(), rows, values, 2, 2, 20, CONFIG)
    assert excinfo.value.kind == "failed"
    assert "step 999" in excinfo.value.message
    assert excinfo.value.step == 2  # the first step whose feedback is non-empty


# --- Test 8: through the arena -------------------------------------------------------------------


#: The digest of canonical_bytes(result_to_obj(result)) for bias_share at its default
#: config, seed 0, on the per_family = 1 synthetic-dev pool (the TASK_007 fixture),
#: pinned on the CI platform (Linux x86_64, CPython 3.11), like GOLDEN_DIGESTS in
#: tests/updater/test_baselines_replay.py. Changing it requires a reason in the PR:
#: bias_share's behaviour changed. The digest moves with ARENA_VERSION (the version
#: string is inside the hashed object); arena version 4 pinned this first.
GOLDEN_DIGEST = "f37faeb38f9a418dd3363141ead3f23a701d10174676b46840ce3690e1c3e20c"


def test_replays_the_synthetic_pool(synthetic_pool) -> None:
    """bias_share at its default config replays synthetic-dev to ok, digest pinned."""
    result = replay_dataset(bias_share, CONFIG, 0, synthetic_pool, interface="forecast-v2")
    assert result.status == "ok"
    assert result.updater_interface == "forecast-v2"
    digest = sha256(canonical_bytes(result_to_obj(result))).hexdigest()
    assert digest == GOLDEN_DIGEST


# --- Test 9: weights-v1 refusal ------------------------------------------------------------------


def test_weights_v1_refusal() -> None:
    """Replaying bias_share under the default weights-v1 interface is invalid_weights at 0."""
    rows, values = hand_fixture(2, 2, 20)
    with pytest.raises(UpdaterFailure) as excinfo:
        replay_forecast(bias_share(), rows, values, 2, 2, 20, CONFIG, interface="weights-v1")
    assert excinfo.value.kind == "invalid_weights"
    assert excinfo.value.step == 0


# --- Test 11: import discipline (test 10, the CLI, moved to feat/forecast-updater-cli) ------------


ALLOWED_IMPORTS = {
    "collections.abc",
    "dataclasses",
    "types",
    "evolvets.updater.interfaces",
    "evolvets.updater.baselines",
}


def _tree() -> ast.Module:
    return ast.parse((SRC / "updater" / "forecast.py").read_text(encoding="utf-8"))


def test_imports_are_exactly_the_allowed_ones() -> None:
    """forecast.py imports only the standard library, the contract and the baselines."""
    imported: set[str | None] = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    assert imported <= ALLOWED_IMPORTS


def test_no_banned_calls() -> None:
    """No builtin sum, no open, no math.fsum, and no math at all."""
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"sum", "open"}, node.func.id
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr != "fsum"
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id != "math"


def test_no_power_operator() -> None:
    """No ``**`` anywhere: products are written as explicit multiplications."""
    for node in ast.walk(_tree()):  # ``**=`` reuses ast.Pow as its AugAssign op
        assert not isinstance(node, ast.Pow)
