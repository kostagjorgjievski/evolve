"""The five hand-designed online updaters (TASK_010 PR A, roadmap entry 10).

Each baseline is an independent learner per horizon column: column ``h`` has
its own state and is updated only from column ``h`` of the matured errors, so
a baseline is ``horizon`` parallel copies of one classical algorithm. All five
share one loss: for one matured origin and one column, ``a_e = abs(error_e)``
and ``l_e = a_e / max_e a_e`` when the maximum is positive, else every
``l_e = 0`` — each loss lies in ``[0, 1]``, the worst expert at that origin
and column scores ``1``, and the loss is invariant to the series' scale.

Everything is deterministic: tie-breaking is by pool order, equality on
floats is exact, and the ``seed`` ``init`` receives is validated and ignored
(nothing here is stochastic). Float arithmetic sticks to ``+ - * /``,
comparisons, ``abs``, explicit left-to-right loops in the order stated in
docs/tasks/TASK_010_updater_baselines.md, and — in this module only —
``math.exp`` and ``math.log``, which follow the platform libm exactly as the
synthetic generator already does (TASK_004). Golden digests are therefore
pinned per platform (Linux x86_64, CPython 3.11).

AdaHedge (de Rooij, van Erven, Grünwald and Koolen, 2014, Algorithm 1) is
written for immediate feedback. With the arena's delayed feedback the weights
that enter the mixability gap at a matured origin are the *current* weights —
the ones the state produces while processing that origin — not the ones that
were played at the origin when it was forecast. This is the standard
reduction of Hedge-style algorithms to delayed feedback: the gap of a delayed
loss is measured against the current posterior, which upper-bounds nothing
and changes no regret guarantee, and it keeps every quantity a function of
the state alone.
"""

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeVar

from evolvets.updater.interfaces import ConfigValue, Context, Matured, Updater, Weights

BASELINE_IDS = ("equal", "ftl", "hedge", "fixed_share", "adahedge")

_DEFAULT_CONFIGS: dict[str, dict[str, ConfigValue]] = {
    "equal": {},
    "ftl": {},
    "hedge": {"eta": 1.0},
    "fixed_share": {"eta": 1.0, "alpha": 0.01},
    "adahedge": {},
}
#: The placeholder configuration of each baseline: runnable, not tuned (entry 11).
DEFAULT_CONFIGS: Mapping[str, Mapping[str, ConfigValue]] = MappingProxyType(
    {name: MappingProxyType(dict(values)) for name, values in _DEFAULT_CONFIGS.items()}
)

_State = TypeVar("_State")


# --- shared validation ---------------------------------------------------------------------------


def _require_int(name: str, value: object, minimum: int) -> int:
    """Return ``value`` as an int >= ``minimum`` (bools rejected), else raise ValueError."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _as_finite_float(name: str, value: object) -> float:
    """Return ``value`` as a finite float built from an exact int or float (no bools)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a float or int, got {value!r}")
    try:
        result = float(value)
    except OverflowError:  # an int too large for a float
        raise ValueError(f"{name} must be finite, got {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _checked_init(
    n_experts: int,
    horizon: int,
    config: Mapping[str, ConfigValue],
    seed: int,
    keys: tuple[str, ...],
) -> tuple[int, int]:
    """Validate the arguments every baseline's init shares; returns (n_experts, horizon)."""
    checked_experts = _require_int("n_experts", n_experts, 1)
    checked_horizon = _require_int("horizon", horizon, 1)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")  # its value is ignored
    if set(config) != set(keys):
        raise ValueError(f"config keys must be exactly {list(keys)}, got {sorted(config)}")
    return checked_experts, checked_horizon


def _checked_eta(value: object) -> float:
    """Return ``value`` as Hedge's learning rate: a finite number > 0."""
    eta = _as_finite_float("eta", value)
    if eta <= 0.0:
        raise ValueError(f"eta must be > 0, got {value!r}")
    return eta


def _checked_alpha(value: object) -> float:
    """Return ``value`` as Fixed-Share's sharing rate: a finite number in [0, 1]."""
    alpha = _as_finite_float("alpha", value)
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {value!r}")
    return alpha


def _checked_state(state: object, cls: type[_State]) -> _State:
    """Return ``state`` when it is the object this baseline's init returned."""
    if not isinstance(state, cls):
        raise ValueError(f"state must be the object init returned, got {type(state).__name__}")
    return state


def _validated_forecasts(n_experts: int, horizon: int, context: Context) -> None:
    """Check that the context's forecasts hold ``n_experts`` rows of ``horizon`` finite floats."""
    if context.horizon != horizon:
        raise ValueError(f"context.horizon must be {horizon}, got {context.horizon}")
    forecasts = context.forecasts
    if len(forecasts) != n_experts:
        raise ValueError(f"forecasts must hold {n_experts} rows, got {len(forecasts)}")
    for row in forecasts:
        if len(row) != horizon:
            raise ValueError(f"each forecast row must hold {horizon} values, got {len(row)}")
        for value in row:
            _as_finite_float("forecast value", value)


def _validated_errors(matured: Matured, index: int, n_experts: int, horizon: int) -> None:
    """Check one Matured's errors: ``n_experts`` rows of ``horizon`` finite floats."""
    errors = matured.errors
    if len(errors) != n_experts:
        raise ValueError(f"feedback[{index}].errors must hold {n_experts} rows, got {len(errors)}")
    for row in errors:
        if len(row) != horizon:
            raise ValueError(
                f"feedback[{index}] error rows must hold {horizon} values, got {len(row)}"
            )
        for value in row:
            _as_finite_float("feedback error", value)


# --- the shared loss and weight forms -------------------------------------------------------------


def _loss_column(errors: tuple[tuple[float, ...], ...], column: int, n_experts: int) -> list[float]:
    """The shared loss of one column: ``abs(error)`` over the column's maximum, else zeros."""
    top = abs(errors[0][column])
    for e in range(1, n_experts):
        scaled = abs(errors[e][column])
        top = scaled if scaled > top else top
    if top > 0.0:
        return [abs(errors[e][column]) / top for e in range(n_experts)]
    return [0.0] * n_experts


def _lowest(values: list[float]) -> float:
    """The minimum of ``values`` by a left-to-right scan (pool order breaks ties)."""
    low = values[0]
    for value in values[1:]:
        low = value if value < low else low
    return low


def _normalised(scores: list[float]) -> list[float]:
    """Normalise non-negative scores: each over their left-to-right loop total."""
    total = 0.0
    for score in scores:
        total = total + score
    return [score / total for score in scores]


def _ftl_column(losses: list[float]) -> list[float]:
    """Uniform weight on every expert whose cumulative loss equals the minimum."""
    low = _lowest(losses)
    n_leaders = 0
    for value in losses:
        if value == low:
            n_leaders += 1
    weight = 1.0 / n_leaders
    return [weight if value == low else 0.0 for value in losses]


def _hedge_column(losses: list[float], eta: float) -> list[float]:
    """Normalised exponential weights ``exp(-eta * (L_e - L_min))`` over one column."""
    low = _lowest(losses)
    scores = [math.exp(-eta * (losses[e] - low)) for e in range(len(losses))]
    return _normalised(scores)


def _rows_from_columns(
    columns: list[list[float]], n_experts: int, horizon: int
) -> tuple[tuple[float, ...], ...]:
    """Transpose per-column weight lists ``[h][e]`` into the ``[e][h]`` tuple the arena takes."""
    return tuple(tuple(columns[h][e] for h in range(horizon)) for e in range(n_experts))


def _uniform_weights(n_experts: int, horizon: int) -> tuple[tuple[float, ...], ...]:
    """Equal weight on every expert at every horizon step."""
    weight = 1.0 / n_experts
    return tuple(tuple(weight for _ in range(horizon)) for _ in range(n_experts))


# --- the five states (everything per-series lives here, never on the instance) --------------------


@dataclass
class _EqualState:
    n_experts: int
    horizon: int


@dataclass
class _FtlState:
    n_experts: int
    horizon: int
    losses: list[list[float]]  # [h][e] cumulative shared loss L_e of column h


@dataclass
class _HedgeState:
    n_experts: int
    horizon: int
    eta: float
    losses: list[list[float]]  # [h][e]


@dataclass
class _FixedShareState:
    n_experts: int
    horizon: int
    eta: float
    alpha: float
    weights: list[list[float]]  # [h][e] the column's current weights p


@dataclass
class _AdaHedgeState:
    n_experts: int
    horizon: int
    losses: list[list[float]]  # [h][e]
    gaps: list[float]  # [h] the mixability gap total Delta of column h


# --- the five baselines --------------------------------------------------------------------------


class Equal:
    """Equal weight on every expert at every step; feedback is validated, never used."""

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> _EqualState:
        experts, steps = _checked_init(n_experts, horizon, config, seed, ())
        return _EqualState(n_experts=experts, horizon=steps)

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Weights, object]:
        checked = _checked_state(state, _EqualState)
        _validated_forecasts(checked.n_experts, checked.horizon, context)
        for index, matured in enumerate(feedback):
            _validated_errors(matured, index, checked.n_experts, checked.horizon)
        return (_uniform_weights(checked.n_experts, checked.horizon), checked)


class FollowTheLeader:
    """All weight, per column, on the expert(s) with the smallest cumulative loss."""

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> _FtlState:
        experts, steps = _checked_init(n_experts, horizon, config, seed, ())
        return _FtlState(
            n_experts=experts,
            horizon=steps,
            losses=[[0.0] * experts for _ in range(steps)],
        )

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Weights, object]:
        checked = _checked_state(state, _FtlState)
        n_experts, horizon = checked.n_experts, checked.horizon
        _validated_forecasts(n_experts, horizon, context)
        for index, matured in enumerate(feedback):
            _validated_errors(matured, index, n_experts, horizon)
            for column in range(horizon):
                losses = _loss_column(matured.errors, column, n_experts)
                cumulative = checked.losses[column]
                for e in range(n_experts):
                    cumulative[e] = cumulative[e] + losses[e]
        columns = [_ftl_column(checked.losses[column]) for column in range(horizon)]
        return (_rows_from_columns(columns, n_experts, horizon), checked)


class Hedge:
    """Exponential weights on the cumulative loss, per column, at a fixed learning rate."""

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> _HedgeState:
        experts, steps = _checked_init(n_experts, horizon, config, seed, ("eta",))
        return _HedgeState(
            n_experts=experts,
            horizon=steps,
            eta=_checked_eta(config["eta"]),
            losses=[[0.0] * experts for _ in range(steps)],
        )

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Weights, object]:
        checked = _checked_state(state, _HedgeState)
        n_experts, horizon = checked.n_experts, checked.horizon
        _validated_forecasts(n_experts, horizon, context)
        for index, matured in enumerate(feedback):
            _validated_errors(matured, index, n_experts, horizon)
            for column in range(horizon):
                losses = _loss_column(matured.errors, column, n_experts)
                cumulative = checked.losses[column]
                for e in range(n_experts):
                    cumulative[e] = cumulative[e] + losses[e]
        columns = [_hedge_column(checked.losses[column], checked.eta) for column in range(horizon)]
        return (_rows_from_columns(columns, n_experts, horizon), checked)


class FixedShare:
    """Hedge with a fixed share of probability spread over every expert each step.

    Herbster and Warmuth (1998). With ``alpha = 0`` this is Hedge with
    incremental updates; with ``alpha = 1`` it is Equal. The share keeps every
    expert's weight at or above ``alpha / E``, so a baseline that follows the
    leader can still switch when the best expert changes.
    """

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> _FixedShareState:
        experts, steps = _checked_init(n_experts, horizon, config, seed, ("eta", "alpha"))
        return _FixedShareState(
            n_experts=experts,
            horizon=steps,
            eta=_checked_eta(config["eta"]),
            alpha=_checked_alpha(config["alpha"]),
            weights=[[1.0 / experts] * experts for _ in range(steps)],
        )

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Weights, object]:
        checked = _checked_state(state, _FixedShareState)
        n_experts, horizon = checked.n_experts, checked.horizon
        _validated_forecasts(n_experts, horizon, context)
        for index, matured in enumerate(feedback):
            _validated_errors(matured, index, n_experts, horizon)
            for column in range(horizon):
                losses = _loss_column(matured.errors, column, n_experts)
                _share_step(checked.weights[column], losses, checked.eta, checked.alpha, n_experts)
        rows = tuple(
            tuple(checked.weights[column][e] for column in range(horizon)) for e in range(n_experts)
        )
        return (rows, checked)


def _share_step(
    weights: list[float], losses: list[float], eta: float, alpha: float, n_experts: int
) -> None:
    """One Fixed-Share step on one column, in place (Herbster and Warmuth, 1998)."""
    support = [e for e in range(n_experts) if weights[e] > 0.0]
    low = losses[support[0]]
    for e in support[1:]:
        low = losses[e] if losses[e] < low else low
    scores = [0.0] * n_experts
    for e in support:
        scores[e] = weights[e] * math.exp(-eta * (losses[e] - low))
    total = 0.0
    for e in range(n_experts):
        total = total + scores[e]
    for e in range(n_experts):
        weights[e] = (1.0 - alpha) * (scores[e] / total) + alpha / n_experts


class AdaHedge:
    """Hedge with a learning rate adapted by the mixability gap, per column.

    de Rooij, van Erven, Grünwald and Koolen (2014), "Follow the leader if you
    can, hedge if you must", Algorithm 1. While the column's gap total
    ``Delta`` is zero the learning rate is infinite and the weights are the
    FTL weights; after that ``eta = ln(E) / Delta``.
    """

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> _AdaHedgeState:
        experts, steps = _checked_init(n_experts, horizon, config, seed, ())
        return _AdaHedgeState(
            n_experts=experts,
            horizon=steps,
            losses=[[0.0] * experts for _ in range(steps)],
            gaps=[0.0] * steps,
        )

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Weights, object]:
        checked = _checked_state(state, _AdaHedgeState)
        n_experts, horizon = checked.n_experts, checked.horizon
        _validated_forecasts(n_experts, horizon, context)
        for index, matured in enumerate(feedback):
            _validated_errors(matured, index, n_experts, horizon)
            if n_experts == 1:
                continue  # the single expert's weight is 1; there is nothing to adapt
            for column in range(horizon):
                losses = _loss_column(matured.errors, column, n_experts)
                checked.gaps[column] = _gap_step(
                    checked.losses[column], checked.gaps[column], losses, n_experts
                )
        columns = [
            _gap_column(checked.losses[c], checked.gaps[c], n_experts) for c in range(horizon)
        ]
        return (_rows_from_columns(columns, n_experts, horizon), checked)


def _gap_column(losses: list[float], gap: float, n_experts: int) -> list[float]:
    """The AdaHedge weights of one column: FTL while the gap is zero, else Hedge at ln(E)/gap."""
    if gap == 0.0:
        return _ftl_column(losses)
    return _hedge_column(losses, math.log(n_experts) / gap)


def _gap_step(cumulative: list[float], gap: float, losses: list[float], n_experts: int) -> float:
    """One AdaHedge step on one column: update ``Delta`` then ``L``, and return the new Delta.

    The weights that enter the gap are the current ones, computed from the
    state just before this origin is processed (the delayed-feedback reduction
    in the module docstring).
    """
    weights = _gap_column(cumulative, gap, n_experts)
    mixed = 0.0
    for e in range(n_experts):
        mixed = mixed + weights[e] * losses[e]
    if gap == 0.0:
        low = _lowest(cumulative)
        leaders = [e for e in range(n_experts) if cumulative[e] == low]
        mix = losses[leaders[0]]
        for e in leaders[1:]:
            mix = losses[e] if losses[e] < mix else mix
    else:
        eta = math.log(n_experts) / gap
        low = losses[0]
        for e in range(1, n_experts):
            low = losses[e] if losses[e] < low else low
        entropy = 0.0
        for e in range(n_experts):
            entropy = entropy + weights[e] * math.exp(-eta * (losses[e] - low))
        mix = low - math.log(entropy) / eta
    step = mixed - mix
    if step < 0.0:
        step = 0.0  # rounding only: the gap is never negative in exact arithmetic
    for e in range(n_experts):
        cumulative[e] = cumulative[e] + losses[e]
    return gap + step


# --- the factories and the registry ---------------------------------------------------------------


def equal() -> Updater:
    """A fresh Equal baseline."""
    return Equal()


def ftl() -> Updater:
    """A fresh FollowTheLeader baseline."""
    return FollowTheLeader()


def hedge() -> Updater:
    """A fresh Hedge baseline."""
    return Hedge()


def fixed_share() -> Updater:
    """A fresh FixedShare baseline."""
    return FixedShare()


def adahedge() -> Updater:
    """A fresh AdaHedge baseline."""
    return AdaHedge()


#: id -> zero-argument factory, in BASELINE_IDS order; "evolvets.updater.baselines:<id>"
#: is a valid sealed updater_ref for each of them.
BASELINES: Mapping[str, Callable[[], Updater]] = MappingProxyType(
    {
        "equal": equal,
        "ftl": ftl,
        "hedge": hedge,
        "fixed_share": fixed_share,
        "adahedge": adahedge,
    }
)
