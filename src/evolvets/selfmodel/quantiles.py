"""The conformal ridge self-model and the prediction record (TASK_017 PR B).

A ridge point forecast of the dev MASE gain from the eight proposal features
plus an intercept, conformalised into five quantiles by the empirical
residual order statistics, and shrunk towards a prior while the history is
short. The fit solves the ridge normal equations by Gaussian elimination
with partial pivoting — the ``experts/ridge.py`` pattern; the equal-budget
claim would not survive a linear-algebra library, so there is none.

Monotone quantiles are a construction, not a check: the order-statistic
index is non-decreasing in ``q``, so the offsets are; every raw quantile
shares the same point forecast; ``DEFAULT_PRIOR`` is strictly increasing;
and a convex blend of two non-decreasing sequences is non-decreasing (IEEE
rounding preserves weak order). Raw quantiles cannot cross, so a sort could
only ever mask an offsets bug — there is deliberately none. The payload's
own monotone validation remains the guard on the record.

``append_prediction`` is the write path: the ``prediction`` record refers to
its proposal, and because ``LedgerWriter.append`` flushes and fsyncs before
returning, the quantiles are durable before any evaluation of that proposal
can be appended (AGENTS.md 2.3). The self-model never charges tokens and
never sees anything but features and gains — no pools, no holdout — which
is what lets PR C's allocators consume its predictions without a leak path.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Protocol

from evolvets.improver.interfaces import FEATURE_NAMES, ProposalFeatures, feature_vector
from evolvets.ledger.payloads import QUANTILE_KEYS, Payload, PayloadError, Prediction
from evolvets.ledger.schema import Budget, Record
from evolvets.selfmodel.interfaces import DEFAULT_PRIOR, METRIC_DEV_MASE_GAIN, HistoryPoint

#: The quantile level of each key, for the order-statistic index.
_QUANTILE_LEVELS: Mapping[str, float] = {
    "q05": 0.05,
    "q25": 0.25,
    "q50": 0.5,
    "q75": 0.75,
    "q95": 0.95,
}


def _require_finite_number(name: str, value: object) -> float:
    """Check a finite number (bools excluded) and return it as a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _solve(a_in: list[list[float]], b_in: list[float]) -> list[float]:
    """Solve a square system by Gaussian elimination with partial pivoting.

    The pivot is the largest absolute value in the column at or below the
    diagonal, ties going to the lowest row index — the ``experts/ridge.py``
    pattern. The ridge system is positive definite, so a zero pivot cannot
    occur.
    """
    a = [row[:] for row in a_in]
    b = b_in[:]
    n = len(b)
    for col in range(n):
        pivot_row = col
        best = abs(a[col][col])
        for row in range(col + 1, n):
            magnitude = abs(a[row][col])
            if magnitude > best:
                best = magnitude
                pivot_row = row
        if best == 0.0:
            raise ValueError("the ridge normal equations are singular")
        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]
            b[col], b[pivot_row] = b[pivot_row], b[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            for c in range(col, n):
                a[row][c] = a[row][c] - factor * a[col][c]
            b[row] = b[row] - factor * b[col]
    w = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = b[i]
        for j in range(i + 1, n):
            total = total - a[i][j] * w[j]
        w[i] = total / a[i][i]
    return w


class _RecordingRun(Protocol):
    """What append_prediction needs of the run recording it: one append.

    The runner's ``Run`` satisfies this structurally; the self-model names
    the shape rather than the class, as the improver's proposer does, so no
    runner import ever crosses into this package.
    """

    def append(
        self,
        kind: str,
        payload: Payload,
        *,
        refers_to: Sequence[str] = (),
        budget: Budget | None = None,
    ) -> Record: ...


class ConformalRidgeSelfModel:
    """Ridge point forecast of the dev MASE gain, conformalised into quantiles.

    Fitted once from the history at construction; ``predict`` is a pure
    function of the fitted weights and the residuals. No randomness
    anywhere: the same history and features always give the same dict.
    """

    def __init__(
        self,
        history: Sequence[HistoryPoint],
        *,
        prior: Mapping[str, float] = DEFAULT_PRIOR,
        prior_points: float = 8.0,
        ridge_lambda: float = 1.0,
    ) -> None:
        shrink_points = _require_finite_number("prior_points", prior_points)
        if shrink_points < 0:
            raise ValueError(f"prior_points must be >= 0, got {prior_points!r}")
        lam = _require_finite_number("ridge_lambda", ridge_lambda)
        if lam <= 0:
            raise ValueError(f"ridge_lambda must be > 0, got {ridge_lambda!r}")
        if set(prior) != set(QUANTILE_KEYS):
            raise ValueError(f"prior must have exactly the keys {list(QUANTILE_KEYS)}")
        self._prior = {
            key: _require_finite_number(f"prior[{key!r}]", prior[key]) for key in QUANTILE_KEYS
        }
        ordered = [self._prior[key] for key in QUANTILE_KEYS]
        if any(later <= earlier for earlier, later in pairwise(ordered)):
            raise ValueError(
                f"prior must be strictly increasing in {list(QUANTILE_KEYS)}, got {ordered}"
            )
        gains: list[float] = []
        for point in history:
            if not math.isfinite(point.gain):
                raise ValueError(f"every history gain must be finite, got {point.gain!r}")
            gains.append(point.gain)
        self._prior_points = shrink_points
        self._lam = lam
        self._n = len(gains)
        self._weights: list[float] | None = None
        self._residuals: list[float] = []
        if history:
            self._weights = self._fit([feature_vector(point.features) for point in history], gains)
            residuals = [
                gain - self._mu(point.features) for point, gain in zip(history, gains, strict=True)
            ]
            self._residuals = sorted(residuals)

    def _fit(self, rows: Sequence[tuple[float, ...]], gains: Sequence[float]) -> list[float]:
        """Solve (XᵀX + λI) w = Xᵀy with one intercept column ahead of the features."""
        n_columns = 1 + len(FEATURE_NAMES)
        a = [[0.0] * n_columns for _ in range(n_columns)]
        b = [0.0] * n_columns
        for row, gain in zip(rows, gains, strict=True):
            x = (1.0, *row)
            for i in range(n_columns):
                b[i] = b[i] + x[i] * gain
                for j in range(n_columns):
                    a[i][j] = a[i][j] + x[i] * x[j]
        for i in range(n_columns):
            a[i][i] = a[i][i] + self._lam
        return _solve(a, b)

    def _mu(self, features: ProposalFeatures) -> float:
        """The point forecast μ̂(x): zero with no history, else the left-to-right dot."""
        if self._weights is None:
            return 0.0
        x = (1.0, *feature_vector(features))
        total = 0.0
        for weight, value in zip(self._weights, x, strict=True):
            total = total + weight * value
        return total

    def identify(self) -> dict[str, str]:
        """This model's id and version."""
        return {"id": "conformal_ridge", "version": "1"}

    def predict(self, features: ProposalFeatures) -> dict[str, float]:
        """The five gain quantiles for ``features``, in ``QUANTILE_KEYS`` order.

        With no history this is the prior exactly. Otherwise each raw
        quantile is μ̂(x) plus the residual order statistic at index
        ``min(n - 1, max(0, ceil(q * n) - 1))``, blended with the prior at
        weight ``n / (n + prior_points)``.
        """
        if self._weights is None:
            return dict(self._prior)
        mu = self._mu(features)
        n = self._n
        weight = n / (n + self._prior_points)
        out: dict[str, float] = {}
        for key in QUANTILE_KEYS:
            index = min(n - 1, max(0, math.ceil(_QUANTILE_LEVELS[key] * n) - 1))
            raw = mu + self._residuals[index]
            out[key] = weight * raw + (1 - weight) * self._prior[key]
        return out


def append_prediction(
    run: _RecordingRun,
    *,
    predictor: Mapping[str, str],
    proposal_record_id: str,
    quantiles: Mapping[str, float],
    metric: str = METRIC_DEV_MASE_GAIN,
    horizon_generations: int = 1,
    wall_seconds: float,
) -> Record:
    """Append one ``prediction`` record for ``proposal_record_id`` and return it.

    The quantiles must carry exactly the ``QUANTILE_KEYS`` and be monotone
    (the payload re-checks both); the record charges wall time only and
    refers to its proposal. Because ``LedgerWriter.append`` flushes and
    fsyncs before returning, the prediction is durable the moment this
    returns — before any evaluation of that proposal can be appended
    (AGENTS.md 2.3).
    """
    if set(quantiles) != set(QUANTILE_KEYS):
        raise PayloadError(
            f"quantiles must have exactly the keys {list(QUANTILE_KEYS)}, got {sorted(quantiles)}"
        )
    payload = Prediction(
        predictor=dict(predictor),
        metric=metric,
        horizon_generations=horizon_generations,
        quantiles={key: quantiles[key] for key in QUANTILE_KEYS},
    )
    return run.append(
        "prediction",
        payload,
        refers_to=[proposal_record_id],
        budget=Budget(wall_seconds=wall_seconds),
    )
