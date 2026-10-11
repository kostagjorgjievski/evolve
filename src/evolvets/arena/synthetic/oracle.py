"""The true-best forecast for a simulated world.

The oracle knows the future mean path and the past noise, so its forecast is
the conditional mean of ``y_{t+h}`` given everything. It is an oracle for
scoring only: nothing outside this package may pass it to an expert or an
updater. It reads ``world.mean``, ``world.phi`` and ``world.noise[origin]``
only, never the observations or the noise after the origin.
"""

from evolvets.arena.synthetic.simulate import World


def _require_origin(origin: int, horizon: int, world: World) -> None:
    """Check origin/horizon bounds for forecasting from ``world``."""
    if isinstance(origin, bool) or not isinstance(origin, int):
        raise ValueError(f"origin must be an int, got {origin!r}")
    if isinstance(horizon, bool) or not isinstance(horizon, int):
        raise ValueError(f"horizon must be an int, got {horizon!r}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if origin < 0:
        raise ValueError(f"origin must be >= 0, got {origin}")
    if origin + horizon > len(world.mean) - 1:
        raise ValueError(f"origin + horizon must be <= length - 1, got {origin} + {horizon}")


def true_best_forecast(world: World, origin: int, horizon: int) -> tuple[float, ...]:
    """Return the conditional-mean forecast of ``y`` for steps ``origin+1..origin+H``."""
    _require_origin(origin, horizon, world)
    e_t = world.noise[origin]
    forecasts = []
    phi_prod = 1.0
    for h in range(1, horizon + 1):
        phi_prod *= world.phi[origin + h]
        forecasts.append(world.mean[origin + h] + phi_prod * e_t)
    return tuple(forecasts)


def true_best_error_variance(world: World, origin: int, horizon: int) -> tuple[float, ...]:
    """Return the forecast error variance for steps ``origin+1..origin+H``."""
    _require_origin(origin, horizon, world)
    variances = []
    # Closed form: Var_h = sum_{j=1..h} sigma_{t+j}^2 * prod_{k=j+1..h} phi_{t+k}^2,
    # which recurses as Var_h = sigma_{t+h}^2 + phi_{t+h}^2 * Var_{h-1}.
    var_h = 0.0
    for h in range(1, horizon + 1):
        var_h = world.sigma[origin + h] ** 2 + world.phi[origin + h] ** 2 * var_h
        variances.append(var_h)
    return tuple(variances)
