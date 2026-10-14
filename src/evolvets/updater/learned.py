"""The learned updaters (TASK_014, research question 1).

Where the baselines combine the pool with a fixed rule, a learned updater
fits its combination from the matured errors themselves. This module holds
them; PR A adds ``ridge_stack`` and PR B appends ``loss_gate``. Like the
baselines, each learned updater is an independent learner per horizon column,
is deterministic (the ``seed`` ``init`` receives is validated and ignored),
and keeps everything per-series in the state object ``init`` returns, never
on the instance, so one instance can serve several series side by side.

Float arithmetic sticks to the same discipline as
:mod:`evolvets.updater.baselines`: ``+ - * /``, comparisons, ``abs`` and —
in this module only — ``math.exp`` and ``math.log``, accumulated in explicit
left-to-right loops in the order stated in
docs/tasks/TASK_014_learned_probe.md. There is no ``**``, no ``sum()`` on
floats, no ``math.fsum`` and no ``statistics``, and the module imports only
the standard library, the updater contract and the baselines' validators.

``ridge_stack`` is a constrained least-squares stacker with exponential
forgetting: an online, shrunk "optimal combination" of the pool. Each column
``h`` keeps a discounted sum ``S[h]`` of raw-error outer products ``e e^T``
and a discounted count ``m[h]``; every ``refit_every`` matured origins it
solves ``A z = 1`` with ``A = Sbar + shrink * trace * I`` (``Sbar = S / m``,
``trace`` the mean diagonal of ``Sbar``) and plays the projection of
``z / 1^T z`` onto the simplex. That is the minimiser of
``w^T Sbar w + shrink * trace * |w - u|^2`` subject to ``sum(w) = 1`` with
``u`` the uniform vector, projected to enforce ``w >= 0``; the shrinkage
towards ``u`` keeps the weights stable while ``m`` is small, and the
forgetting tracks a pool whose best member drifts.

``loss_gate`` learns to predict each expert's *next* loss instead of mixing
second moments: per column, a discounted ridge regression of the shared loss
on origin-time features — an intercept, fast and slow EWMAs of the loss, the
expert's last loss at this season index, and the expert's share of the
pool's disagreement — whose Hedge-style weights ``exp(-beta * (p_e - min p))``
over the predicted losses gate the pool. The feature rows are stored at the
origin they describe and consumed when that origin matures, so every training
row pairs the loss that materialised with what was knowable when the forecast
was made; a column with no fit yet, a degenerate refit or a non-finite
prediction falls back to uniform or to its previous coefficients rather than
failing the replay.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from evolvets.updater.baselines import (
    _as_finite_float,
    _checked_init,
    _checked_state,
    _hedge_column,
    _loss_column,
    _require_int,
    _rows_from_columns,
    _validated_errors,
    _validated_forecasts,
)
from evolvets.updater.interfaces import ConfigValue, Context, Matured, Updater, Weights

LEARNED_IDS: tuple[str, ...] = ("ridge_stack", "loss_gate")

_LEARNED_DEFAULT_CONFIGS: dict[str, dict[str, ConfigValue]] = {
    "ridge_stack": {"half_life": 336.0, "shrink": 1.0, "refit_every": 24},
    "loss_gate": {"beta": 16.0, "half_life": 336.0, "ridge": 0.01, "refit_every": 24},
}
#: The placeholder configuration of each learned updater: runnable, not tuned.
LEARNED_DEFAULT_CONFIGS: Mapping[str, Mapping[str, ConfigValue]] = MappingProxyType(
    {name: MappingProxyType(dict(values)) for name, values in _LEARNED_DEFAULT_CONFIGS.items()}
)

_INF = float("inf")
_NEG_INF = float("-inf")


def _is_finite(value: float) -> bool:
    """Return whether ``value`` is neither NaN nor an infinity, via comparisons only."""
    return value == value and value != _INF and value != _NEG_INF


# --- the numerical helpers ------------------------------------------------------------------------


def solve_linear(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float] | None:
    """Solve the ``n x n`` (``n >= 1``) system ``matrix @ x = rhs``, or return ``None``.

    Gaussian elimination with partial pivoting, generalising
    ``experts/ridge.py:_solve``: the pivot is the largest absolute value at or
    below the diagonal, ties going to the lowest row index. ``None`` means a
    pivot was ``0.0`` or a solution entry came out NaN or infinite — a
    degenerate system the caller must fall back from, not an error. The
    arguments are never mutated; a non-square matrix, an empty one, or
    ``len(rhs) != n`` raises ``ValueError``.
    """
    n = len(matrix)
    if n == 0:
        raise ValueError("matrix must hold at least one row")
    a: list[list[float]] = []
    for source_row in matrix:
        if len(source_row) != n:
            raise ValueError(f"matrix must be square, got {n} rows of length {len(source_row)}")
        a.append([float(value) for value in source_row])
    if len(rhs) != n:
        raise ValueError(f"rhs must hold {n} values, got {len(rhs)}")
    b = [float(value) for value in rhs]
    for col in range(n):
        pivot_row = col
        best = abs(a[col][col])
        for row in range(col + 1, n):
            magnitude = abs(a[row][col])
            if magnitude > best:
                best = magnitude
                pivot_row = row
        if best == 0.0:
            return None
        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]
            b[col], b[pivot_row] = b[pivot_row], b[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            for c in range(col, n):
                a[row][c] = a[row][c] - factor * a[col][c]
            b[row] = b[row] - factor * b[col]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = b[i]
        for j in range(i + 1, n):
            total = total - a[i][j] * x[j]
        x[i] = total / a[i][i]
    for value in x:
        if not _is_finite(value):
            return None
    return x


def project_to_simplex(values: Sequence[float]) -> list[float]:
    """The Euclidean projection of ``values`` onto the probability simplex.

    Duchi, Shalev-Shwartz, Singer and Chandra (2008), Algorithm 1: sort the
    entries descending, find the last position whose value exceeds the mean
    of the prefix above it, shift by that threshold and clip at zero. The
    clipped vector is finally normalised so the result sums to 1 well within
    the arena's ``1e-9``. The input must be non-empty and finite, else
    ``ValueError``.
    """
    if not values:
        raise ValueError("values must hold at least one entry")
    entries = [float(value) for value in values]
    for value in entries:
        if not _is_finite(value):
            raise ValueError(f"every value must be finite, got {value!r}")
    order = sorted(range(len(entries)), key=lambda i: (-entries[i], i))
    rho = 0
    cs = 0.0
    cs_rho = 0.0
    for k, index in enumerate(order, start=1):
        cs = cs + entries[index]
        if entries[index] - (cs - 1.0) / k > 0.0:
            rho = k
            cs_rho = cs
    if rho == 0:
        # No position qualifies: the entries are so large (beyond 2**53)
        # that (cs - 1) / k rounds back to v_(k) itself. The exact projection
        # then concentrates on the largest entry, so return that, with ties
        # going to the lowest index as everywhere in this package.
        top = entries[0]
        for value in entries[1:]:
            top = value if value > top else top
        out = [0.0] * len(entries)
        for i, value in enumerate(entries):
            if value == top:
                out[i] = 1.0
                break
        return out
    theta = (cs_rho - 1.0) / rho
    out = []
    for value in entries:
        difference = value - theta
        out.append(difference if difference > 0.0 else 0.0)
    total = 0.0
    for value in out:
        total = total + value
    return [value / total for value in out]


# --- ridge_stack ----------------------------------------------------------------------------------


def _checked_positive(name: str, value: object) -> float:
    """Return ``value`` as a finite number > 0 (bools rejected), else raise ValueError."""
    number = _as_finite_float(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return number


def _checked_half_life(value: object) -> float:
    """Return ``value`` as a forgetting half-life: a finite number > 0."""
    return _checked_positive("half_life", value)


def _checked_shrink(value: object) -> float:
    """Return ``value`` as the stacker's shrinkage: a finite number > 0."""
    return _checked_positive("shrink", value)


@dataclass
class _RidgeStackState:
    n_experts: int
    horizon: int
    gamma: float
    shrink: float
    refit_every: int
    s: list[list[list[float]]]  # [h][i][j] discounted sum of e e^T
    m: list[float]  # [h] discounted count of absorbed origins
    weights: list[list[float]]  # [h][e] the column's current weights
    absorbed: int  # matured origins seen
    fitted: bool  # whether a refit has run


class RidgeStack:
    """A constrained least-squares stacker with exponential forgetting, per column.

    Before the first matured origin, and whenever a column's discounted
    second moments degenerate (an all-zero trace, a singular system, a
    non-finite solve), the column falls back: uniform, or its previous
    weights. A refit that overflows (errors around ``1e200``) therefore
    freezes the column instead of failing the replay.
    """

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> _RidgeStackState:
        experts, steps = _checked_init(
            n_experts, horizon, config, seed, ("half_life", "shrink", "refit_every")
        )
        half_life = _checked_half_life(config["half_life"])
        return _RidgeStackState(
            n_experts=experts,
            horizon=steps,
            gamma=math.exp(-math.log(2.0) / half_life),
            shrink=_checked_shrink(config["shrink"]),
            refit_every=_require_int("refit_every", config["refit_every"], 1),
            s=[[[0.0] * experts for _ in range(experts)] for _ in range(steps)],
            m=[0.0] * steps,
            weights=[[1.0 / experts] * experts for _ in range(steps)],
            absorbed=0,
            fitted=False,
        )

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Weights, object]:
        checked = _checked_state(state, _RidgeStackState)
        n_experts, horizon = checked.n_experts, checked.horizon
        _validated_forecasts(n_experts, horizon, context)
        for index, matured in enumerate(feedback):
            _validated_errors(matured, index, n_experts, horizon)
        for matured in feedback:
            for h in range(horizon):
                square = checked.s[h]
                for i in range(n_experts):
                    e_i = matured.errors[i][h]
                    row = square[i]
                    for j in range(n_experts):
                        row[j] = checked.gamma * row[j] + e_i * matured.errors[j][h]
                checked.m[h] = checked.gamma * checked.m[h] + 1.0
            checked.absorbed = checked.absorbed + 1
        due = not checked.fitted or context.step % checked.refit_every == 0
        if checked.absorbed >= 1 and due:
            _refit_stack(checked)
        return (_rows_from_columns(checked.weights, n_experts, horizon), checked)


def _refit_stack(state: _RidgeStackState) -> None:
    """Refit every column's weights from its discounted second moments.

    A column whose system degenerates keeps its previous weights; the flag
    is set whatever each column did, so the schedule stays
    ``step % refit_every == 0`` after the first fit.
    """
    state.fitted = True
    n_experts = state.n_experts
    for h in range(state.horizon):
        count = state.m[h]
        square = state.s[h]
        sbar = [[square[i][j] / count for j in range(n_experts)] for i in range(n_experts)]
        trace = 0.0
        for i in range(n_experts):
            trace = trace + sbar[i][i] / n_experts
        if trace == 0.0:
            state.weights[h] = [1.0 / n_experts] * n_experts
            continue
        shift = state.shrink * trace
        a = [[sbar[i][j] for j in range(n_experts)] for i in range(n_experts)]
        for i in range(n_experts):
            a[i][i] = a[i][i] + shift
        z = solve_linear(a, [1.0] * n_experts)
        if z is None:
            continue
        total = 0.0
        for value in z:
            total = total + value
        if not _is_finite(total) or not total > 0.0:
            continue
        state.weights[h] = project_to_simplex([value / total for value in z])


# --- loss_gate ------------------------------------------------------------------------------------


FAST_HALF_LIFE = 4.0  # matured origins, the fast loss EWMA
SLOW_HALF_LIFE = 64.0  # matured origins, the slow loss EWMA

_N_FEATURES = 5  # x = (1.0, fast, slow, season, dev)

_FeatureRow = tuple[float, float, float, float, float]  # one expert's features at one origin
_Features = tuple[tuple[_FeatureRow, ...], ...]  # [h][e], the rows one origin stored


@dataclass
class _LossGateState:
    n_experts: int
    horizon: int
    beta: float
    gamma: float
    a_fast: float
    a_slow: float
    ridge: float
    refit_every: int
    fast: list[list[float | None]]  # [h][e] EWMA of the shared loss; None until the first
    slow: list[list[float | None]]  # [h][e]
    season_last: list[list[dict[int, float]]]  # [h][e] season index -> last matured loss
    g: list[list[list[float]]]  # [h][i][j] discounted sum of x x^T
    r: list[list[float]]  # [h][i] discounted sum of x * l
    m: list[float]  # [h] discounted count of absorbed expert-rows
    beta_hat: list[list[float] | None]  # [h] the fitted coefficients
    pending: dict[int, tuple[int, _Features]]  # step -> (season index, features)
    absorbed: int  # matured origins seen
    fitted: bool  # whether a refit has run


class LossGate:
    """An online ridge model of each expert's next loss, gated by its predictions.

    Every training row pairs the loss an origin eventually took with the
    features that were knowable when that origin was forecast: the rows are
    built at the origin, stored in ``pending`` against its step, and consumed
    when the arena hands the origin's ``Matured`` back. A column with no fit
    yet is uniform, a degenerate refit keeps the previous coefficients, and a
    non-finite prediction falls back to uniform, so the learner itself never
    fails the replay.
    """

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> _LossGateState:
        experts, steps = _checked_init(
            n_experts, horizon, config, seed, ("beta", "half_life", "ridge", "refit_every")
        )
        half_life = _checked_half_life(config["half_life"])
        return _LossGateState(
            n_experts=experts,
            horizon=steps,
            beta=_checked_positive("beta", config["beta"]),
            gamma=math.exp(-math.log(2.0) / half_life),
            a_fast=1.0 - math.exp(-math.log(2.0) / FAST_HALF_LIFE),
            a_slow=1.0 - math.exp(-math.log(2.0) / SLOW_HALF_LIFE),
            ridge=_checked_positive("ridge", config["ridge"]),
            refit_every=_require_int("refit_every", config["refit_every"], 1),
            fast=[[None] * experts for _ in range(steps)],
            slow=[[None] * experts for _ in range(steps)],
            season_last=[[{} for _ in range(experts)] for _ in range(steps)],
            g=[[[0.0] * _N_FEATURES for _ in range(_N_FEATURES)] for _ in range(steps)],
            r=[[0.0] * _N_FEATURES for _ in range(steps)],
            m=[0.0] * steps,
            beta_hat=[None] * steps,
            pending={},
            absorbed=0,
            fitted=False,
        )

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Weights, object]:
        checked = _checked_state(state, _LossGateState)
        n_experts, horizon = checked.n_experts, checked.horizon
        _validated_forecasts(n_experts, horizon, context)
        for index, matured in enumerate(feedback):
            _validated_errors(matured, index, n_experts, horizon)
        for matured in feedback:
            _absorb_origin(checked, matured)
        due = not checked.fitted or context.step % checked.refit_every == 0
        if checked.absorbed >= 1 and due:
            _refit_gate(checked)
        features = _origin_features(checked, context)
        checked.pending[context.step] = (context.season_index, features)
        return (_rows_from_columns(_gate_columns(checked, features), n_experts, horizon), checked)


def _absorb_origin(state: _LossGateState, matured: Matured) -> None:
    """Absorb one matured origin into the discounted design and the feature EWMAs.

    The design rows are the ones stored at the origin (``pending``), not
    anything recomputed now; an origin the state never forecast raises, which
    the arena records as a failed replay.
    """
    stored = state.pending.pop(matured.step, None)
    if stored is None:
        raise ValueError(f"no update call ever stored step {matured.step} on this state")
    season_index, features = stored
    n_experts = state.n_experts
    loss_columns = [_loss_column(matured.errors, h, n_experts) for h in range(state.horizon)]
    for h in range(state.horizon):
        losses = loss_columns[h]
        gram, rhs = state.g[h], state.r[h]
        for e in range(n_experts):
            row = features[h][e]
            loss = losses[e]
            for i in range(_N_FEATURES):
                value = row[i]
                gram_row = gram[i]
                for j in range(_N_FEATURES):
                    gram_row[j] = state.gamma * gram_row[j] + value * row[j]
                rhs[i] = state.gamma * rhs[i] + value * loss
        state.m[h] = state.gamma * state.m[h] + n_experts
    for h in range(state.horizon):
        losses = loss_columns[h]
        for e in range(n_experts):
            loss = losses[e]
            fast = state.fast[h][e]
            state.fast[h][e] = loss if fast is None else fast + state.a_fast * (loss - fast)
            slow = state.slow[h][e]
            state.slow[h][e] = loss if slow is None else slow + state.a_slow * (loss - slow)
            state.season_last[h][e][season_index] = loss
    state.absorbed = state.absorbed + 1


def _refit_gate(state: _LossGateState) -> None:
    """Refit every column's predicted-loss coefficients from its discounted design.

    The intercept is not shrunk; a column whose system degenerates keeps its
    previous coefficients. The flag is set whatever each column did, so the
    schedule stays ``step % refit_every == 0`` after the first fit.
    """
    state.fitted = True
    for h in range(state.horizon):
        count = state.m[h]
        gram, rhs = state.g[h], state.r[h]
        a = [[gram[i][j] / count for j in range(_N_FEATURES)] for i in range(_N_FEATURES)]
        for i in range(1, _N_FEATURES):
            a[i][i] = a[i][i] + state.ridge
        b = [rhs[i] / count for i in range(_N_FEATURES)]
        solution = solve_linear(a, b)
        if solution is not None:
            state.beta_hat[h] = solution


def _origin_features(state: _LossGateState, context: Context) -> _Features:
    """The feature rows of the current origin: ``x = (1.0, fast, slow, season, dev)``.

    ``season`` falls back to ``slow``, and every loss feature reads ``0.0``
    before the first matured loss. ``dev`` is the expert's absolute distance
    from the column's forecast mean over the column's largest distance.
    """
    n_experts, horizon = state.n_experts, state.horizon
    columns = []
    for h in range(horizon):
        mean = 0.0
        for e in range(n_experts):
            mean = mean + context.forecasts[e][h]
        mean = mean / n_experts
        deviations = []
        top = 0.0
        for e in range(n_experts):
            difference = abs(context.forecasts[e][h] - mean)
            deviations.append(difference)
            top = difference if difference > top else top
        rows = []
        for e in range(n_experts):
            fast_value = state.fast[h][e]
            fast = fast_value if fast_value is not None else 0.0
            slow_value = state.slow[h][e]
            slow = slow_value if slow_value is not None else 0.0
            season_value = state.season_last[h][e].get(context.season_index)
            season = season_value if season_value is not None else slow
            dev = deviations[e] / top if top > 0.0 else 0.0
            rows.append((1.0, fast, slow, season, dev))
        columns.append(tuple(rows))
    return tuple(columns)


def _gate_columns(state: _LossGateState, features: _Features) -> list[list[float]]:
    """The weight columns: Hedge on the predicted losses, uniform without a usable fit."""
    n_experts = state.n_experts
    uniform = [1.0 / n_experts] * n_experts
    columns = []
    for h in range(state.horizon):
        fitted = state.beta_hat[h]
        if fitted is None:
            columns.append(list(uniform))
            continue
        predictions = []
        degenerate = False
        for e in range(n_experts):
            row = features[h][e]
            total = 0.0
            for i in range(_N_FEATURES):
                total = total + fitted[i] * row[i]
            if not _is_finite(total):
                degenerate = True
                break
            predictions.append(total)
        columns.append(list(uniform) if degenerate else _hedge_column(predictions, state.beta))
    return columns


# --- the factories and the registry ---------------------------------------------------------------


def ridge_stack() -> Updater:
    """A fresh RidgeStack learned updater."""
    return RidgeStack()


def loss_gate() -> Updater:
    """A fresh LossGate learned updater."""
    return LossGate()


#: id -> zero-argument factory, in LEARNED_IDS order; "evolvets.updater.learned:<id>"
#: names each of them in records (updater_ref).
LEARNED: Mapping[str, Callable[[], Updater]] = MappingProxyType(
    {"ridge_stack": ridge_stack, "loss_gate": loss_gate}
)
