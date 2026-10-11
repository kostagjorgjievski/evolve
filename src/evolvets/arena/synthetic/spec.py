"""Exact definition of a synthetic world.

A :class:`WorldSpec` fixes the mean path, the AR(1) noise parameters and the
noise bursts of a world, so that simulation, the oracle and the ledger all
agree on what the world is. Validation happens at construction time: an
invalid spec can never enter a simulation or a ledger record.
"""

import math
from dataclasses import dataclass, fields
from typing import Any, Literal

TWO_PI = 2.0 * math.pi

_TRANSITIONS = ("sudden", "gradual")


class WorldSpecError(ValueError):
    """Raised when a WorldSpec, Regime or NoiseBurst does not match its rules."""


def _require_int(name: str, value: Any) -> None:
    """Check that ``value`` is an int (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorldSpecError(f"{name} must be an int, got {value!r}")


def _require_finite_float(name: str, value: Any) -> None:
    """Check that ``value`` is a finite float."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorldSpecError(f"{name} must be a float, got {value!r}")
    if not math.isfinite(value):
        raise WorldSpecError(f"{name} must be finite, got {value!r}")


@dataclass(frozen=True)
class Regime:
    """One regime of the mean path, active from ``start`` until the next change."""

    start: int  # first index where this regime is active
    level: float  # trend value at `start`
    slope: float  # per step, measured from `start`
    season_amp: float  # >= 0
    season_phase: float  # radians, 0 <= phase < 2*pi
    ar_phi: float  # -1 < phi < 1
    sigma: float  # innovation std, > 0
    transition: Literal["sudden", "gradual"]
    ramp: int  # 0 if sudden, >= 2 if gradual
    recurs_from: int | None  # index of an earlier regime this one repeats, or None

    def __post_init__(self) -> None:
        _require_int("Regime.start", self.start)
        _require_int("Regime.ramp", self.ramp)
        if self.start < 0:
            raise WorldSpecError(f"Regime.start must be >= 0, got {self.start}")
        for name in ("level", "slope", "season_amp", "season_phase", "ar_phi", "sigma"):
            _require_finite_float(f"Regime.{name}", getattr(self, name))
        if self.season_amp < 0:
            raise WorldSpecError(f"Regime.season_amp must be >= 0, got {self.season_amp}")
        if not 0.0 <= self.season_phase < TWO_PI:
            raise WorldSpecError(
                f"Regime.season_phase must be in [0, 2*pi), got {self.season_phase}"
            )
        if not -1.0 < self.ar_phi < 1.0:
            raise WorldSpecError(f"Regime.ar_phi must be in (-1, 1), got {self.ar_phi}")
        if self.sigma <= 0:
            raise WorldSpecError(f"Regime.sigma must be > 0, got {self.sigma}")
        if self.transition not in _TRANSITIONS:
            raise WorldSpecError(
                f"Regime.transition must be one of {_TRANSITIONS}, got {self.transition!r}"
            )
        if self.transition == "sudden" and self.ramp != 0:
            raise WorldSpecError("a sudden transition must have ramp == 0")
        if self.transition == "gradual" and self.ramp < 2:
            raise WorldSpecError("a gradual transition must have ramp >= 2")
        if self.recurs_from is not None:
            _require_int("Regime.recurs_from", self.recurs_from)
            if self.recurs_from < 0:
                raise WorldSpecError(f"Regime.recurs_from must be >= 0, got {self.recurs_from}")


@dataclass(frozen=True)
class NoiseBurst:
    """A span of ``length`` steps where the innovation std is multiplied."""

    start: int
    length: int  # >= 1
    sigma_mult: float  # > 1

    def __post_init__(self) -> None:
        _require_int("NoiseBurst.start", self.start)
        _require_int("NoiseBurst.length", self.length)
        _require_finite_float("NoiseBurst.sigma_mult", self.sigma_mult)
        if self.start < 0:
            raise WorldSpecError(f"NoiseBurst.start must be >= 0, got {self.start}")
        if self.length < 1:
            raise WorldSpecError(f"NoiseBurst.length must be >= 1, got {self.length}")
        if self.sigma_mult <= 1:
            raise WorldSpecError(f"NoiseBurst.sigma_mult must be > 1, got {self.sigma_mult}")


@dataclass(frozen=True)
class WorldSpec:
    """A complete world: regimes for the mean and noise, plus noise bursts."""

    length: int  # >= 2
    period: int  # seasonal period, >= 2
    regimes: tuple[Regime, ...]
    bursts: tuple[NoiseBurst, ...]

    def __post_init__(self) -> None:
        _require_int("WorldSpec.length", self.length)
        _require_int("WorldSpec.period", self.period)
        if self.length < 2:
            raise WorldSpecError(f"WorldSpec.length must be >= 2, got {self.length}")
        if self.period < 2:
            raise WorldSpecError(f"WorldSpec.period must be >= 2, got {self.period}")
        if not self.regimes:
            raise WorldSpecError("WorldSpec.regimes must be non-empty")
        first = self.regimes[0]
        if first.start != 0:
            raise WorldSpecError(f"regimes[0].start must be 0, got {first.start}")
        if first.transition != "sudden":
            raise WorldSpecError("regimes[0].transition must be 'sudden'")
        if first.ramp != 0:
            raise WorldSpecError("regimes[0].ramp must be 0")
        if first.recurs_from is not None:
            raise WorldSpecError("regimes[0].recurs_from must be None")
        for i, regime in enumerate(self.regimes):
            if i > 0:
                if regime.start <= self.regimes[i - 1].start:
                    raise WorldSpecError(
                        f"regime starts must be strictly increasing, got "
                        f"{self.regimes[i - 1].start} then {regime.start}"
                    )
                if regime.start >= self.length:
                    raise WorldSpecError(
                        f"regime start {regime.start} must be < length {self.length}"
                    )
            if regime.transition == "gradual":
                limit = self.regimes[i + 1].start if i + 1 < len(self.regimes) else self.length
                if regime.start + regime.ramp > limit:
                    raise WorldSpecError(
                        f"ramp of regime {i} ends at {regime.start + regime.ramp}, "
                        f"beyond the limit {limit}"
                    )
            if regime.recurs_from is not None:
                self._check_recurrence(i)
        prev_burst_end = 0
        for burst in self.bursts:
            if burst.start < prev_burst_end:
                raise WorldSpecError("bursts must be sorted by start and non-overlapping")
            if burst.start + burst.length > self.length:
                raise WorldSpecError(
                    f"burst [{burst.start}, {burst.start + burst.length}) must be inside "
                    f"[0, {self.length})"
                )
            prev_burst_end = burst.start + burst.length

    def _check_recurrence(self, r: int) -> None:
        regime = self.regimes[r]
        a = regime.recurs_from
        if a is None or a >= r:
            raise WorldSpecError(f"regime {r}: recurs_from must be a smaller index, got {a}")
        ref = self.regimes[a]
        for name in ("slope", "season_amp", "season_phase", "ar_phi", "sigma"):
            if getattr(regime, name) != getattr(ref, name):
                raise WorldSpecError(
                    f"regime {r}: {name} must equal regime {a}'s for a recurring regime"
                )
        expected_level = ref.level + ref.slope * (regime.start - ref.start)
        if regime.level != expected_level:
            raise WorldSpecError(
                f"regime {r}: level must be {expected_level!r} (level_a + slope_a * "
                f"(start_r - start_a)), got {regime.level!r}"
            )

    def regime_at(self, t: int) -> int:
        """Return the index of the last regime with ``start <= t``."""
        index = 0
        for i, regime in enumerate(self.regimes):
            if regime.start <= t:
                index = i
            else:
                break
        return index

    def mean_at(self, r: int, t: int) -> float:
        """Return regime ``r``'s own mean function ``m_r(t)``."""
        regime = self.regimes[r]
        return (
            regime.level
            + regime.slope * (t - regime.start)
            + regime.season_amp * math.sin(TWO_PI * t / self.period + regime.season_phase)
        )


def spec_to_dict(spec: WorldSpec) -> dict[str, Any]:
    """Return a JSON-ready dict describing ``spec``, for hashing and the ledger."""
    return {
        "length": spec.length,
        "period": spec.period,
        "regimes": [
            {f.name: getattr(regime, f.name) for f in fields(Regime)} for regime in spec.regimes
        ],
        "bursts": [
            {f.name: getattr(burst, f.name) for f in fields(NoiseBurst)} for burst in spec.bursts
        ],
    }
