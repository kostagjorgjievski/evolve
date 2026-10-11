"""Recursive one-step ridge autoregression with periodic refits.

Lags ``(1, 2, 3, m, 2m)`` plus an intercept, fitted on normalised values by
solving the 6x6 normal equations with Gaussian elimination. The refit
schedule depends only on the time index, so forecasting at some origins and
not others never changes a later forecast.
"""

import math
from collections import deque

from evolvets.experts.interfaces import (
    ExpertError,
    check_forecast,
    require_int,
    require_positive,
    require_value,
)

_N_FEATURES = 6


def _dot(w: list[float], x: list[float]) -> float:
    """Return the left-to-right dot product of ``w`` and ``x``."""
    total = 0.0
    for i in range(_N_FEATURES):
        total = total + w[i] * x[i]
    return total


def _solve(a_in: list[list[float]], b_in: list[float]) -> list[float]:
    """Solve the 6x6 system by Gaussian elimination with partial pivoting.

    The pivot is the largest absolute value in the column at or below the
    diagonal, ties going to the lowest row index. The matrix is positive
    definite, so a zero pivot cannot occur.
    """
    a = [row[:] for row in a_in]
    b = b_in[:]
    n = _N_FEATURES
    for col in range(n):
        pivot_row = col
        best = abs(a[col][col])
        for row in range(col + 1, n):
            magnitude = abs(a[row][col])
            if magnitude > best:
                best = magnitude
                pivot_row = row
        if best == 0.0:
            raise ExpertError("ar_ridge normal equations are singular")
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


class RidgeAR:
    """A ridge autoregression on lags (1, 2, 3, m, 2m), refitted on a fixed schedule."""

    name = "ar_ridge"

    def __init__(self, period: int, window: int = 240, refit_every: int = 24, ridge: float = 1.0):
        self._m = require_int("RidgeAR: period", period, 2)
        self._window = require_int("RidgeAR: window", window, 2)
        self._refit_every = require_int("RidgeAR: refit_every", refit_every, 1)
        self._ridge = require_positive("RidgeAR: ridge", ridge)
        self._fit_len = self._window + 2 * self._m
        self.min_history = self._fit_len
        self._t0 = self._fit_len - 1  # first origin on the refit schedule
        self._t = -1  # index of the last observed value
        self._buf: deque[float] = deque(maxlen=self._fit_len)
        self._mu = 0.0
        self._sd = 1.0
        self._w: list[float] | None = None

    def observe(self, y: float) -> None:
        self._buf.append(require_value("ar_ridge.observe: y", y))
        self._t += 1
        if self._t < self._t0:
            return
        if self._w is None or (self._t - self._t0) % self._refit_every == 0:
            self._refit()

    def _refit(self) -> None:
        """Fit the coefficients on the last window + 2m observed values."""
        m = self._m
        values = list(self._buf)
        count = len(values)
        mu = 0.0
        for value in values:
            mu = mu + value
        mu = mu / count
        sq = 0.0
        for value in values:
            sq = sq + (value - mu) * (value - mu)
        sd = math.sqrt(sq / count)
        if sd == 0.0:
            sd = 1.0
        z = [(value - mu) / sd for value in values]
        # z[i] is the normalised value at absolute index t - count + 1 + i.

        def z_at(index: int) -> float:
            return z[index - (self._t - count + 1)]

        a = [[0.0] * _N_FEATURES for _ in range(_N_FEATURES)]
        b = [0.0] * _N_FEATURES
        for s in range(self._t - self._window + 1, self._t + 1):
            x = [1.0, z_at(s - 1), z_at(s - 2), z_at(s - 3), z_at(s - m), z_at(s - 2 * m)]
            for i in range(_N_FEATURES):
                b[i] = b[i] + x[i] * z_at(s)
                for j in range(_N_FEATURES):
                    a[i][j] = a[i][j] + x[i] * x[j]
        for i in range(1, _N_FEATURES):
            a[i][i] = a[i][i] + self._ridge
        self._mu = mu
        self._sd = sd
        self._w = _solve(a, b)

    def forecast(self, horizon: int) -> tuple[float, ...]:
        require_int("ar_ridge.forecast: horizon", horizon, 1)
        if self._w is None:
            raise ExpertError(
                f"ar_ridge needs {self.min_history} observed values before forecasting"
            )
        w = self._w
        m = self._m
        t = self._t
        # z_hist holds the normalised last 2m observed values, pred the
        # predicted z for indices > t as the recursion produces them.
        recent = list(self._buf)
        z_hist = [(value - self._mu) / self._sd for value in recent[len(recent) - 2 * m :]]
        pred: list[float] = []

        def z_at(index: int) -> float:
            if index <= t:
                return z_hist[index - (t - 2 * m + 1)]
            return pred[index - t - 1]

        out = []
        for h in range(1, horizon + 1):
            target = t + h
            x = [
                1.0,
                z_at(target - 1),
                z_at(target - 2),
                z_at(target - 3),
                z_at(target - m),
                z_at(target - 2 * m),
            ]
            z_hat = _dot(w, x)
            pred.append(z_hat)
            out.append(self._mu + self._sd * z_hat)
        return check_forecast("ar_ridge", tuple(out))

    def params(self) -> dict[str, float | int]:
        return {
            "period": self._m,
            "window": self._window,
            "refit_every": self._refit_every,
            "ridge": self._ridge,
        }
