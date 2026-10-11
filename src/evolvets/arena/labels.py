"""Horizon-aware label release: the only code that indexes the target series.

A target for origin ``s`` becomes available at ``s + H`` and not before — all
``H`` of them together (AGENTS.md 2.2). Everything outside this module that
needs a target or a history goes through :class:`LabelRelease`.
"""

from collections.abc import Sequence

from evolvets.arena.errors import ArenaError, LookAheadError

_INF = float("inf")
_NEG_INF = float("-inf")


def _require_int(name: str, value: object, minimum: int) -> int:
    """Return ``value`` as an int >= ``minimum`` (bools rejected), else raise ArenaError."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArenaError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ArenaError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_finite(name: str, value: object) -> float:
    """Return ``value`` as a finite float (bools rejected), else raise ArenaError."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArenaError(f"{name} must be a finite float, got {value!r}")
    try:
        result = float(value)
    except OverflowError:  # an int too large for a float
        raise ArenaError(f"{name} must be a finite float, got {value!r}") from None
    if not (result == result and result != _INF and result != _NEG_INF):
        raise ArenaError(f"{name} must be a finite float, got {value!r}")
    return result


class LabelRelease:
    """Holds a series' values and hands out targets only once they have matured."""

    def __init__(self, values: Sequence[float], horizon: int) -> None:
        self._values = tuple(_require_finite(f"values[{i}]", v) for i, v in enumerate(values))
        self._horizon = _require_int("horizon", horizon, 1)

    def targets(self, origin: int, now: int) -> tuple[float, ...]:
        """Return ``y_{origin+1} .. y_{origin+horizon}`` if that horizon has passed."""
        origin = _require_int("origin", origin, 0)
        now = _require_int("now", now, 0)
        if origin + self._horizon > now:
            raise LookAheadError(
                f"targets for origin {origin} (horizon {self._horizon}) are not available "
                f"until now = {origin + self._horizon}, but now = {now}"
            )
        if origin + self._horizon > len(self._values) - 1:
            raise ArenaError(
                f"origin {origin} + horizon {self._horizon} exceeds the last index "
                f"{len(self._values) - 1} of the series"
            )
        return self._values[origin + 1 : origin + self._horizon + 1]

    def history(self, end: int, now: int) -> tuple[float, ...]:
        """Return ``y_0 .. y_{end-1}`` if ``end`` is at most one past ``now``."""
        end = _require_int("end", end, 0)
        now = _require_int("now", now, 0)
        if end > now + 1:
            raise LookAheadError(
                f"history up to {end} is not available until now >= {end - 1}, but now = {now}"
            )
        if end > len(self._values):
            raise ArenaError(f"history end {end} exceeds the series length {len(self._values)}")
        return self._values[:end]

    def __len__(self) -> int:
        """Return the number of values held."""
        return len(self._values)
