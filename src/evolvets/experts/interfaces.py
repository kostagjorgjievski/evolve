"""The expert interface shared by every forecaster in the pool.

Float arithmetic inside this package sticks to ``+ - * /``, comparisons,
``abs`` and ``math.sqrt`` so that forecast values are bit-identical on every
platform and Python 3.11+.
"""

from typing import Protocol

_INF = float("inf")
_NEG_INF = float("-inf")


class ExpertError(ValueError):
    """Raised when an expert is constructed, observed or asked to forecast invalidly."""


def _is_finite(value: float) -> bool:
    """Return whether ``value`` is neither NaN nor an infinity, via comparisons only."""
    return value == value and value != _INF and value != _NEG_INF


def _as_float(name: str, value: object) -> float:
    """Return ``value`` as a float (bools rejected), raising ExpertError if it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExpertError(f"{name} must be a finite float, got {value!r}")
    try:
        return float(value)
    except OverflowError:  # an int too large for a float
        raise ExpertError(f"{name} must be a finite float, got {value!r}") from None


def require_value(name: str, value: object) -> float:
    """Return ``value`` as a finite float (bools rejected), else raise ExpertError."""
    result = _as_float(name, value)
    if not _is_finite(result):
        raise ExpertError(f"{name} must be a finite float, got {value!r}")
    return result


def require_int(name: str, value: object, minimum: int) -> int:
    """Return ``value`` as an int >= ``minimum`` (bools rejected), else raise ExpertError."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExpertError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ExpertError(f"{name} must be >= {minimum}, got {value}")
    return value


def require_unit(name: str, value: object, allow_one: bool) -> float:
    """Return ``value`` as a float in ``(0, 1)`` — or ``(0, 1]`` when ``allow_one``."""
    result = _as_float(name, value)
    valid = 0.0 < result < 1.0 or (allow_one and result == 1.0)
    if not _is_finite(result) or not valid:
        bound = "(0, 1]" if allow_one else "(0, 1)"
        raise ExpertError(f"{name} must be a float in {bound}, got {value!r}")
    return result


def require_positive(name: str, value: object) -> float:
    """Return ``value`` as a finite float > 0, else raise ExpertError."""
    result = _as_float(name, value)
    if not _is_finite(result) or result <= 0.0:
        raise ExpertError(f"{name} must be a finite float > 0, got {value!r}")
    return result


def check_forecast(name: str, values: tuple[float, ...]) -> tuple[float, ...]:
    """Return ``values`` unless any entry is non-finite, in which case raise ExpertError."""
    for i, value in enumerate(values):
        if not _is_finite(value):
            raise ExpertError(f"{name} produced a non-finite forecast at step {i + 1}")
    return values


class Expert(Protocol):
    """A streaming univariate forecaster. It only knows the values it has been shown."""

    name: str
    min_history: int  # forecast() is valid once this many values have been observed

    def observe(self, y: float) -> None: ...
    def forecast(self, horizon: int) -> tuple[float, ...]: ...
    def params(self) -> dict[str, float | int]: ...  # JSON-ready, for the cache manifest
