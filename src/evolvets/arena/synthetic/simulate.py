"""Simulation of a WorldSpec from standard-normal innovations.

The mean path and the AR(1) noise are computed step by step so that the
simulator, the oracle and the tests evaluate the same expressions on the same
floats. ``World`` records everything the oracle needs, including the actually
used ``sigma`` and ``phi`` per step.
"""

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, replace

from evolvets.arena.synthetic import GENERATOR_VERSION
from evolvets.arena.synthetic.spec import TWO_PI, WorldSpec, spec_to_dict
from evolvets.ledger.canonical import canonical_bytes

SEED_LIMIT = 2**63


@dataclass(frozen=True)
class World:
    """A simulated world: the observed series and everything derived from it."""

    spec: WorldSpec
    seed: int | None  # None when built from explicit innovations
    y: tuple[float, ...]  # observed series
    mean: tuple[float, ...]  # mu_t
    noise: tuple[float, ...]  # e_t
    sigma: tuple[float, ...]  # sigma_t actually used
    phi: tuple[float, ...]  # phi_t actually used
    change_points: tuple[int, ...]  # starts of regimes 1..R-1
    burst_spans: tuple[tuple[int, int], ...]  # (start, end exclusive)

    def __post_init__(self) -> None:
        n = self.spec.length
        for name in ("y", "mean", "noise", "sigma", "phi"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"World.{name} must have length {n}")
        if self.seed is not None:
            _require_seed(self.seed)


def _require_seed(seed: int) -> None:
    """Check that ``seed`` is an int in ``[0, 2**63)`` (bools excluded)."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an int, got {seed!r}")
    if not 0 <= seed < SEED_LIMIT:
        raise ValueError(f"seed must be in [0, 2**63), got {seed}")


def standard_normals(seed: int, n: int) -> tuple[float, ...]:
    """Draw ``n`` standard normals from ``seed`` via Box-Muller.

    Uses only ``random.Random(seed).random()`` so the output is reproducible
    across Python versions; the sine partner of each pair is discarded.
    """
    _require_seed(seed)
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"n must be an int, got {n!r}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    rng = random.Random(seed)
    values = []
    for _ in range(n):
        u1 = rng.random()
        u2 = rng.random()
        values.append(math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(TWO_PI * u2))
    return tuple(values)


def _mean_path(spec: WorldSpec) -> tuple[float, ...]:
    """Return ``mu_t`` for ``t = 0 .. length - 1`` per the spec's regimes."""
    mu = []
    for t in range(spec.length):
        r = spec.regime_at(t)
        regime = spec.regimes[r]
        if r > 0 and regime.transition == "gradual" and t < regime.start + regime.ramp:
            alpha = (t - regime.start + 1) / (regime.ramp + 1)
            m_prev = spec.mean_at(r - 1, t)
            m_curr = spec.mean_at(r, t)
            mu.append((1.0 - alpha) * m_prev + alpha * m_curr)
        else:
            mu.append(spec.mean_at(r, t))
    return tuple(mu)


def _noise_scale(spec: WorldSpec) -> tuple[tuple[float, float], ...]:
    """Return ``(sigma_t, phi_t)`` actually used at each step, bursts applied."""
    out = []
    for t in range(spec.length):
        regime = spec.regimes[spec.regime_at(t)]
        sigma = regime.sigma
        for burst in spec.bursts:
            if burst.start <= t < burst.start + burst.length:
                sigma = regime.sigma * burst.sigma_mult
                break
        out.append((sigma, regime.ar_phi))
    return tuple(out)


def simulate_from_innovations(spec: WorldSpec, z: Sequence[float]) -> World:
    """Simulate ``spec`` from the given standard-normal innovations."""
    if len(z) != spec.length:
        raise ValueError(f"z must have length {spec.length}, got {len(z)}")
    for i, value in enumerate(z):
        if not math.isfinite(value):
            raise ValueError(f"z[{i}] must be finite, got {value!r}")
    mean = _mean_path(spec)
    scale = _noise_scale(spec)
    noise: list[float] = []
    y: list[float] = []
    sigma_0, phi_0 = scale[0]
    e = sigma_0 / math.sqrt(1.0 - phi_0**2) * z[0]
    for t in range(spec.length):
        if t > 0:
            sigma_t, phi_t = scale[t]
            e = phi_t * e + sigma_t * z[t]
        noise.append(e)
        y.append(mean[t] + e)
    return World(
        spec=spec,
        seed=None,
        y=tuple(y),
        mean=mean,
        noise=tuple(noise),
        sigma=tuple(s for s, _ in scale),
        phi=tuple(p for _, p in scale),
        change_points=tuple(r.start for r in spec.regimes[1:]),
        burst_spans=tuple((b.start, b.start + b.length) for b in spec.bursts),
    )


def simulate(spec: WorldSpec, seed: int) -> World:
    """Simulate ``spec`` with standard normals drawn from ``seed``."""
    world = simulate_from_innovations(spec, standard_normals(seed, spec.length))
    return replace(world, seed=seed)


def world_fingerprint(world: World) -> str:
    """Return a stable SHA-256 fingerprint of the simulated world.

    ``y`` is formatted to 10 significant digits so the fingerprint does not
    depend on last-bit ``libm`` differences.
    """
    payload = {
        "generator_version": GENERATOR_VERSION,
        "spec": spec_to_dict(world.spec),
        "seed": world.seed,
        "y": [format(v, ".9e") for v in world.y],
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()
