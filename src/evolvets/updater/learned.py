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
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from evolvets.updater.baselines import (
    _as_finite_float,
    _checked_init,
    _checked_state,
    _require_int,
    _rows_from_columns,
    _validated_errors,
    _validated_forecasts,
)
from evolvets.updater.interfaces import ConfigValue, Context, Matured, Updater, Weights

LEARNED_IDS: tuple[str, ...] = ("ridge_stack",)  # PR B appends "loss_gate"

_LEARNED_DEFAULT_CONFIGS: dict[str, dict[str, ConfigValue]] = {
    "ridge_stack": {"half_life": 336.0, "shrink": 1.0, "refit_every": 24},
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


def _checked_half_life(value: object) -> float:
    """Return ``value`` as a forgetting half-life: a finite number > 0."""
    half_life = _as_finite_float("half_life", value)
    if half_life <= 0.0:
        raise ValueError(f"half_life must be > 0, got {value!r}")
    return half_life


def _checked_shrink(value: object) -> float:
    """Return ``value`` as the stacker's shrinkage: a finite number > 0."""
    shrink = _as_finite_float("shrink", value)
    if shrink <= 0.0:
        raise ValueError(f"shrink must be > 0, got {value!r}")
    return shrink


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


# --- the factories and the registry ---------------------------------------------------------------


def ridge_stack() -> Updater:
    """A fresh RidgeStack learned updater."""
    return RidgeStack()


#: id -> zero-argument factory, in LEARNED_IDS order; "evolvets.updater.learned:<id>"
#: names each of them in records (updater_ref).
LEARNED: Mapping[str, Callable[[], Updater]] = MappingProxyType({"ridge_stack": ridge_stack})
