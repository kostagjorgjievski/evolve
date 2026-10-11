"""Sampling synthetic worlds from named families.

A family fixes the *shape* of a world — how many changes of which kinds,
with which transitions and bursts — while every concrete value is drawn
from ``GrammarConfig`` ranges. Only ``random.Random(seed).random()`` is
consumed, through the two uniform-draw helpers below, so sampled worlds
are reproducible across Python versions. The draw order is written down
in :func:`sample_spec` and pinned by the golden-fingerprint tests; any
change to it or to a draw formula changes fingerprints and requires
bumping ``GENERATOR_VERSION``.
"""

import hashlib
import math
import random
from dataclasses import dataclass, fields, replace
from typing import Any, Literal, cast

from evolvets.arena.synthetic.simulate import World, _require_seed, simulate
from evolvets.arena.synthetic.spec import TWO_PI, NoiseBurst, Regime, WorldSpec

FAMILIES = ("stationary", "sudden", "gradual", "recurring", "false_drift", "mixed")
CHANGE_KINDS = ("level_shift", "slope_change", "season_change", "noise_change")

_TASK_KINDS = (*CHANGE_KINDS, "recurrence")
_FLOAT_RANGES = (
    "level_range",
    "slope_range",
    "new_slope_range",
    "season_amp_range",
    "phi_range",
    "sigma_range",
    "shift_range",
    "burst_mult_range",
)
_INT_RANGES = ("ramp_range", "burst_length_range")
_SEED_MASK = 2**63 - 1


class GrammarError(ValueError):
    """Raised when a GrammarConfig, family name or SyntheticTask breaks its rules."""


def _finite_float_range(name: str, value: Any) -> tuple[float, float]:
    """Normalise a two-element float range to finite floats (bools rejected)."""
    try:
        lo, hi = value
    except (TypeError, ValueError):
        raise GrammarError(f"GrammarConfig.{name} must be a (lo, hi) pair, got {value!r}") from None
    bounds = []
    for bound in (lo, hi):
        if isinstance(bound, bool) or not isinstance(bound, (int, float)):
            raise GrammarError(f"GrammarConfig.{name} must contain floats, got {value!r}")
        try:
            as_float = float(bound)
        except OverflowError:
            raise GrammarError(f"GrammarConfig.{name} must be finite, got {value!r}") from None
        if not math.isfinite(as_float):
            raise GrammarError(f"GrammarConfig.{name} must be finite, got {value!r}")
        # -0.0 equals 0.0 but serialises differently; keep the positive zero.
        bounds.append(as_float + 0.0)
    return (bounds[0], bounds[1])


@dataclass(frozen=True)
class GrammarConfig:
    """Ranges and placement rules the sampler draws a world from."""

    length: int = 2000
    period: int = 24
    warmup: int = 300  # no change point or burst before this index
    min_segment: int = 200  # minimum gap between consecutive change points
    max_changes: int = 3  # changes drawn uniformly from 1..max_changes
    level_range: tuple[float, float] = (-5.0, 5.0)
    slope_range: tuple[float, float] = (-0.002, 0.002)
    new_slope_range: tuple[float, float] = (-0.01, 0.01)
    season_amp_range: tuple[float, float] = (0.0, 3.0)
    phi_range: tuple[float, float] = (0.0, 0.9)
    sigma_range: tuple[float, float] = (0.3, 1.0)
    shift_range: tuple[float, float] = (2.0, 5.0)  # level shift size, in stationary noise stds
    ramp_range: tuple[int, int] = (20, 150)
    max_bursts: int = 3
    burst_length_range: tuple[int, int] = (10, 50)
    burst_mult_range: tuple[float, float] = (2.0, 4.0)

    def __post_init__(self) -> None:
        for name in _FLOAT_RANGES:
            object.__setattr__(self, name, _finite_float_range(name, getattr(self, name)))
        for name in (*_FLOAT_RANGES, *_INT_RANGES):
            lo, hi = getattr(self, name)
            if lo > hi:
                raise GrammarError(f"GrammarConfig.{name} must satisfy lo <= hi, got ({lo}, {hi})")
        if self.ramp_range[0] < 2:
            raise GrammarError(
                f"GrammarConfig.ramp_range[0] must be >= 2, got {self.ramp_range[0]}"
            )
        if self.ramp_range[1] >= self.min_segment:
            raise GrammarError(
                f"GrammarConfig.ramp_range[1] must be < min_segment, got {self.ramp_range[1]} "
                f"vs {self.min_segment}"
            )
        if self.max_changes < 1:
            raise GrammarError(f"GrammarConfig.max_changes must be >= 1, got {self.max_changes}")
        slots = (self.length - self.warmup) // max(2, self.max_changes, self.max_bursts)
        if slots < self.min_segment:
            raise GrammarError(
                f"(length - warmup) // max(2, max_changes, max_bursts) = {slots} must be "
                f">= min_segment {self.min_segment}"
            )
        if self.burst_length_range[1] >= self.min_segment:
            raise GrammarError(
                f"GrammarConfig.burst_length_range[1] must be < min_segment, got "
                f"{self.burst_length_range[1]} vs {self.min_segment}"
            )
        if self.burst_mult_range[0] <= 1:
            raise GrammarError(
                f"GrammarConfig.burst_mult_range[0] must be > 1, got {self.burst_mult_range[0]}"
            )
        if self.sigma_range[0] <= 0:
            raise GrammarError(
                f"GrammarConfig.sigma_range[0] must be > 0, got {self.sigma_range[0]}"
            )
        if not -1 < self.phi_range[0] <= self.phi_range[1] < 1:
            raise GrammarError(
                f"GrammarConfig.phi_range must be inside (-1, 1), got {self.phi_range}"
            )
        if self.season_amp_range[0] < 0:
            raise GrammarError(
                f"GrammarConfig.season_amp_range[0] must be >= 0, got {self.season_amp_range[0]}"
            )


@dataclass(frozen=True)
class SyntheticTask:
    """One sampled world plus the change kinds planted in it."""

    family: str
    world: World
    change_kinds: tuple[str, ...]  # aligned with world.change_points; "recurrence" for recurrences

    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise GrammarError(f"family must be one of {FAMILIES}, got {self.family!r}")
        try:
            kinds = tuple(self.change_kinds)
        except TypeError:
            raise GrammarError(
                f"SyntheticTask.change_kinds must be a sequence of str, got {self.change_kinds!r}"
            ) from None
        for kind in kinds:
            if not isinstance(kind, str) or kind not in _TASK_KINDS:
                raise GrammarError(f"change kind must be one of {_TASK_KINDS}, got {kind!r}")
        if len(kinds) != len(self.world.change_points):
            raise GrammarError(
                f"change_kinds must align with world.change_points, got {len(kinds)} kinds "
                f"for {len(self.world.change_points)} change points"
            )
        object.__setattr__(self, "change_kinds", kinds)


def _uniform(rng: random.Random, lo: float, hi: float) -> float:
    """Draw a float uniformly from ``[lo, hi]``."""
    return lo + (hi - lo) * rng.random()


def _rand_int(rng: random.Random, lo: int, hi: int) -> int:
    """Draw an int uniformly from ``[lo, hi]``, capped at ``hi``."""
    return min(lo + int(rng.random() * (hi - lo + 1)), hi)


def _draw_base_regime(rng: random.Random, config: GrammarConfig) -> Regime:
    """Draw the anchor regime at start 0: level, slope, amp, phase, phi, sigma."""
    return Regime(
        start=0,
        level=_uniform(rng, *config.level_range),
        slope=_uniform(rng, *config.slope_range),
        season_amp=_uniform(rng, *config.season_amp_range),
        season_phase=_uniform(rng, 0.0, TWO_PI),
        ar_phi=_uniform(rng, *config.phi_range),
        sigma=_uniform(rng, *config.sigma_range),
        transition="sudden",
        ramp=0,
        recurs_from=None,
    )


def _place(rng: random.Random, config: GrammarConfig, count: int) -> tuple[int, ...]:
    """Place ``count`` events in equal slots of ``[warmup, length)``.

    Event ``i`` lands in slot ``i``; the slot width minus ``min_segment``
    is the free range inside the slot, so consecutive events are at least
    ``min_segment`` apart and every ramp or burst ends before the next event.
    """
    width = (config.length - config.warmup) // count
    return tuple(
        config.warmup + i * width + _rand_int(rng, 0, width - config.min_segment)
        for i in range(count)
    )


def _apply_kind(
    rng: random.Random, config: GrammarConfig, kind: str, prev: Regime, start: int
) -> Regime:
    """Build the regime a change of ``kind`` produces at ``start`` after ``prev``.

    The level is first re-anchored so the trend is continuous at ``start``
    (the previous trend line evaluated there); only then is the kind applied.
    Draw order per kind: ``level_shift`` draws sign then shift; ``season_change``
    draws amplitude then phase; ``noise_change`` draws phi then sigma;
    ``slope_change`` draws one slope.
    """
    if kind not in CHANGE_KINDS:
        raise GrammarError(f"change kind must be one of {CHANGE_KINDS}, got {kind!r}")
    level = prev.level + prev.slope * (start - prev.start)
    slope = prev.slope
    season_amp = prev.season_amp
    season_phase = prev.season_phase
    ar_phi = prev.ar_phi
    sigma = prev.sigma
    if kind == "level_shift":
        sign = -1.0 if rng.random() < 0.5 else 1.0
        shift = _uniform(rng, *config.shift_range)
        level += sign * shift * prev.sigma / math.sqrt(1.0 - prev.ar_phi**2)
    elif kind == "slope_change":
        slope = _uniform(rng, *config.new_slope_range)
    elif kind == "season_change":
        season_amp = _uniform(rng, *config.season_amp_range)
        season_phase = _uniform(rng, 0.0, TWO_PI)
    else:  # noise_change
        ar_phi = _uniform(rng, *config.phi_range)
        sigma = _uniform(rng, *config.sigma_range)
    return Regime(
        start=start,
        level=level,
        slope=slope,
        season_amp=season_amp,
        season_phase=season_phase,
        ar_phi=ar_phi,
        sigma=sigma,
        transition="sudden",
        ramp=0,
        recurs_from=None,
    )


def _change_chain(
    rng: random.Random,
    config: GrammarConfig,
    base: Regime,
    mode: Literal["sudden", "gradual", "mixed"],
) -> tuple[tuple[Regime, ...], tuple[str, ...]]:
    """Draw and place ``1..max_changes`` changes, then build each new regime.

    Per change, in order: the kind index, the kind's own draws (see
    :func:`_apply_kind`), then the transition — for ``gradual`` a ramp from
    ``ramp_range``; for ``mixed`` a sudden/gradual flag (probability 1/2)
    and, when gradual, the ramp after it.
    """
    count = _rand_int(rng, 1, config.max_changes)
    positions = _place(rng, config, count)
    regimes = [base]
    kinds: list[str] = []
    for start in positions:
        kind = CHANGE_KINDS[_rand_int(rng, 0, len(CHANGE_KINDS) - 1)]
        regime = _apply_kind(rng, config, kind, regimes[-1], start)
        if mode == "gradual" or (mode == "mixed" and rng.random() < 0.5):
            regime = replace(regime, transition="gradual", ramp=_rand_int(rng, *config.ramp_range))
        kinds.append(kind)
        regimes.append(regime)
    return tuple(regimes), tuple(kinds)


def _recurring_chain(
    rng: random.Random, config: GrammarConfig, base: Regime
) -> tuple[tuple[Regime, ...], tuple[str, ...]]:
    """Draw the alternating A/B regime chain of the ``recurring`` family.

    Draw order: the change count ``2..max(2, max_changes)``, the positions,
    then the single kind index and its draws for regime B. Every later
    regime copies regime A or B exactly (level on the referenced regime's
    own trend line) and is recorded as a ``recurrence`` change point.
    """
    count = _rand_int(rng, 2, max(2, config.max_changes))
    positions = _place(rng, config, count)
    kind = CHANGE_KINDS[_rand_int(rng, 0, len(CHANGE_KINDS) - 1)]
    regimes = [base, _apply_kind(rng, config, kind, base, positions[0])]
    for start in positions[1:]:
        source = 0 if len(regimes) % 2 == 0 else 1
        ref = regimes[source]
        regimes.append(
            Regime(
                start=start,
                level=ref.level + ref.slope * (start - ref.start),
                slope=ref.slope,
                season_amp=ref.season_amp,
                season_phase=ref.season_phase,
                ar_phi=ref.ar_phi,
                sigma=ref.sigma,
                transition="sudden",
                ramp=0,
                recurs_from=source,
            )
        )
    return tuple(regimes), (kind,) + ("recurrence",) * (count - 1)


def _draw_bursts(rng: random.Random, config: GrammarConfig, count: int) -> tuple[NoiseBurst, ...]:
    """Place ``count`` bursts, then draw per burst its length and multiplier.

    A count of zero places nothing and draws nothing, so the caller's draw
    order stays well-defined.
    """
    if count <= 0:
        return ()
    positions = _place(rng, config, count)
    bursts = []
    for start in positions:
        length = _rand_int(rng, *config.burst_length_range)
        sigma_mult = _uniform(rng, *config.burst_mult_range)
        bursts.append(NoiseBurst(start=start, length=length, sigma_mult=sigma_mult))
    return tuple(bursts)


def sample_spec(
    family: str, rng: random.Random, config: GrammarConfig
) -> tuple[WorldSpec, tuple[str, ...]]:
    """Sample a :class:`WorldSpec` and its change kinds for ``family``.

    Draw order, pinned by the golden-fingerprint tests: the base regime
    draws level, slope, season amplitude, phase, phi, sigma (in that
    order); then the family draws — ``sudden``/``gradual``/``mixed``: the
    change count, all change positions, then per change the kind, its own
    draws and its transition (``mixed`` draws the sudden/gradual flag
    first, and after all changes the burst count and bursts);
    ``recurring``: the change count, all positions, then regime B's kind
    and draws; ``false_drift``: the burst count, all burst positions, then
    per burst length and multiplier; ``stationary``: nothing more.
    """
    if family not in FAMILIES:
        raise GrammarError(f"family must be one of {FAMILIES}, got {family!r}")
    base = _draw_base_regime(rng, config)
    regimes: tuple[Regime, ...]
    kinds: tuple[str, ...]
    bursts: tuple[NoiseBurst, ...] = ()
    if family == "stationary":
        regimes, kinds = (base,), ()
    elif family == "sudden" or family == "gradual" or family == "mixed":
        # mypy cannot narrow a plain str to the Literal; family is one of the
        # three because FAMILIES was checked above and the other cases excluded.
        mode = cast("Literal['sudden', 'gradual', 'mixed']", family)
        regimes, kinds = _change_chain(rng, config, base, mode)
        if family == "mixed":
            bursts = _draw_bursts(rng, config, _rand_int(rng, 0, config.max_bursts))
    elif family == "recurring":
        regimes, kinds = _recurring_chain(rng, config, base)
    else:  # false_drift
        regimes, kinds = (base,), ()
        bursts = _draw_bursts(rng, config, _rand_int(rng, 1, config.max_bursts))
    spec = WorldSpec(length=config.length, period=config.period, regimes=regimes, bursts=bursts)
    return spec, kinds


def _sub_seed(label: bytes, seed: int) -> int:
    """Derive an independent sub-seed in ``[0, 2**63)`` for ``label`` from ``seed``."""
    digest = hashlib.sha256(
        b"evolvets/generator/v1/" + label + b"/" + str(seed).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & _SEED_MASK


_DEFAULT_CONFIG = GrammarConfig()


def make_task(family: str, seed: int, config: GrammarConfig = _DEFAULT_CONFIG) -> SyntheticTask:
    """Sample and simulate the task named by ``family`` and ``seed``.

    The spec and the simulation noise come from independent sub-seeds of
    ``seed``, so the same seed always yields the same task.
    """
    _require_seed(seed)
    spec, kinds = sample_spec(family, random.Random(_sub_seed(b"spec", seed)), config)
    return SyntheticTask(
        family=family, world=simulate(spec, _sub_seed(b"noise", seed)), change_kinds=kinds
    )


def config_to_dict(config: GrammarConfig) -> dict[str, Any]:
    """Return a JSON-ready dict describing ``config``, for hashing and the ledger."""
    return {f.name: getattr(config, f.name) for f in fields(GrammarConfig)}
