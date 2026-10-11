"""Four simple streaming experts: naive, seasonal naive, SES, damped Holt-Winters.

All arithmetic is left-to-right on plain floats so every forecast is
bit-reproducible; means are computed with an explicit loop, never ``sum()``.
"""

from collections import deque

from evolvets.experts.interfaces import (
    ExpertError,
    check_forecast,
    require_int,
    require_unit,
    require_value,
)


def _mean(values: list[float]) -> float:
    """Return the left-to-right loop mean of ``values``."""
    total = 0.0
    for value in values:
        total = total + value
    return total / len(values)


class Naive:
    """The last observed value, held for every horizon step."""

    name = "naive"
    min_history = 1

    def __init__(self) -> None:
        self._last: float | None = None

    def observe(self, y: float) -> None:
        self._last = require_value("naive.observe: y", y)

    def forecast(self, horizon: int) -> tuple[float, ...]:
        require_int("naive.forecast: horizon", horizon, 1)
        if self._last is None:
            raise ExpertError("naive needs 1 observed value before forecasting")
        return check_forecast("naive", (self._last,) * horizon)

    def params(self) -> dict[str, float | int]:
        return {}


class SeasonalNaive:
    """The most recent value in the same seasonal phase, cycling with the period."""

    name = "seasonal_naive"

    def __init__(self, period: int) -> None:
        self._m = require_int("SeasonalNaive: period", period, 2)
        self.min_history = self._m
        self._buf: deque[float] = deque(maxlen=self._m)

    def observe(self, y: float) -> None:
        self._buf.append(require_value("seasonal_naive.observe: y", y))

    def forecast(self, horizon: int) -> tuple[float, ...]:
        require_int("seasonal_naive.forecast: horizon", horizon, 1)
        if len(self._buf) < self._m:
            raise ExpertError(f"seasonal_naive needs {self._m} observed values before forecasting")
        out = []
        for h in range(1, horizon + 1):
            k = (h + self._m - 1) // self._m
            out.append(self._buf[h - 1 - self._m * (k - 1)])
        return check_forecast("seasonal_naive", tuple(out))

    def params(self) -> dict[str, float | int]:
        return {"period": self._m}


class SimpleExpSmoothing:
    """Simple exponential smoothing: the level is the only state."""

    name = "ses"
    min_history = 1

    def __init__(self, alpha: float = 0.3) -> None:
        self._alpha = require_unit("SimpleExpSmoothing: alpha", alpha, allow_one=False)
        self._level: float | None = None

    def observe(self, y: float) -> None:
        value = require_value("ses.observe: y", y)
        if self._level is None:
            self._level = value
        else:
            self._level = self._alpha * value + (1.0 - self._alpha) * self._level

    def forecast(self, horizon: int) -> tuple[float, ...]:
        require_int("ses.forecast: horizon", horizon, 1)
        if self._level is None:
            raise ExpertError("ses needs 1 observed value before forecasting")
        return check_forecast("ses", (self._level,) * horizon)

    def params(self) -> dict[str, float | int]:
        return {"alpha": self._alpha}


class DampedHoltWinters:
    """Additive Holt-Winters with a damped trend, fixed parameters.

    The first ``2m`` values only initialise the level, trend and seasonals;
    afterwards every value updates them. Forecasting never mutates state.
    """

    name = "holt_winters"

    def __init__(
        self,
        period: int,
        alpha: float = 0.2,
        beta: float = 0.01,
        gamma: float = 0.1,
        phi: float = 0.98,
    ) -> None:
        self._m = require_int("DampedHoltWinters: period", period, 2)
        self._alpha = require_unit("DampedHoltWinters: alpha", alpha, allow_one=False)
        self._beta = require_unit("DampedHoltWinters: beta", beta, allow_one=False)
        self._gamma = require_unit("DampedHoltWinters: gamma", gamma, allow_one=False)
        self._phi = require_unit("DampedHoltWinters: phi", phi, allow_one=True)
        self.min_history = 2 * self._m
        self._t = -1  # index of the last observed value
        self._buf: list[float] = []
        self._level = 0.0
        self._trend = 0.0
        self._seasonal: list[float] = []

    def observe(self, y: float) -> None:
        value = require_value("holt_winters.observe: y", y)
        self._t += 1
        if self._t < 2 * self._m - 1:
            self._buf.append(value)
            return
        if self._t == 2 * self._m - 1:
            self._buf.append(value)
            self._initialise()
            return
        m = self._m
        s = self._seasonal[self._t % m]
        level_prev = self._level
        self._level = self._alpha * (value - s) + (1.0 - self._alpha) * (
            level_prev + self._phi * self._trend
        )
        self._trend = (
            self._beta * (self._level - level_prev) + (1.0 - self._beta) * self._phi * self._trend
        )
        self._seasonal[self._t % m] = self._gamma * (value - self._level) + (1.0 - self._gamma) * s

    def _initialise(self) -> None:
        """Set the level, trend and seasonals from the buffered first 2m values."""
        m = self._m
        a0 = _mean(self._buf[:m])
        a1 = _mean(self._buf[m:])
        self._trend = (a1 - a0) / m
        self._level = a1 + self._trend * ((m - 1) / 2)
        self._seasonal = [((self._buf[j] - a0) + (self._buf[m + j] - a1)) / 2 for j in range(m)]
        self._buf = []

    def forecast(self, horizon: int) -> tuple[float, ...]:
        require_int("holt_winters.forecast: horizon", horizon, 1)
        if self._t < self.min_history - 1:
            raise ExpertError(
                f"holt_winters needs {self.min_history} observed values before forecasting"
            )
        m = self._m
        d = 0.0
        p = 1.0
        out = []
        for h in range(1, horizon + 1):
            p = p * self._phi
            d = d + p
            out.append(self._level + d * self._trend + self._seasonal[(self._t + h) % m])
        return check_forecast("holt_winters", tuple(out))

    def params(self) -> dict[str, float | int]:
        return {
            "period": self._m,
            "alpha": self._alpha,
            "beta": self._beta,
            "gamma": self._gamma,
            "phi": self._phi,
        }
