"""The forecast-output updaters (TASK_016 PR B, roadmap entry 14).

Where every updater before them answers the arena with weights over the pool,
a forecast-v2 updater may answer with its own point forecast. This module
holds the reference one, ``bias_share``: the later lineage's starting point.
Like the baselines it is deterministic (the ``seed`` ``init`` receives is
validated and ignored) and keeps everything per-series in the state object
``init`` returns, never on the instance, so one instance can serve several
series side by side.

Float arithmetic sticks to the same discipline as
:mod:`evolvets.updater.baselines`: ``+ - * /``, comparisons and ``abs``,
accumulated in explicit left-to-right loops in the order stated in
docs/tasks/TASK_016_forecast_updaters_and_holdout.md. There is no ``**``, no
``sum()`` on floats, no ``math.fsum``, no ``statistics`` and — unlike the
baselines — no ``math`` at all, and the module imports only the standard
library, the updater contract and the baselines' validators.

``bias_share`` plays Fixed-Share over the pool and subtracts, per horizon
column, an online estimate of that combination's own bias. Each origin's
``Matured.combined_errors`` are what the *output* (combination minus bias)
took, so adding back the bias that was applied at that origin recovers the
Fixed-Share combination's error — the quantity a bias estimate should track.
The estimate moves toward that residual at ``bias_rate`` per matured origin;
``bias_rate = 0`` keeps it at zero and reproduces Fixed-Share bit for bit.
Fixed-Share itself reads only ``matured.errors``, never
``combined_errors``, so both learners can absorb the same feedback exactly.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from evolvets.updater.baselines import (
    FixedShare,
    _as_finite_float,
    _checked_alpha,
    _checked_eta,
    _checked_init,
    _checked_state,
    _validated_errors,
    _validated_forecasts,
)
from evolvets.updater.interfaces import (
    ConfigValue,
    Context,
    ForecastOutput,
    ForecastUpdater,
    Matured,
    Output,
)

FORECAST_UPDATER_IDS: tuple[str, ...] = ("bias_share",)

_FORECAST_DEFAULT_CONFIGS: dict[str, dict[str, ConfigValue]] = {
    "bias_share": {"eta": 1.0, "alpha": 0.01, "bias_rate": 0.1},
}
#: The placeholder configuration of each forecast updater: runnable, not tuned.
FORECAST_DEFAULT_CONFIGS: Mapping[str, Mapping[str, ConfigValue]] = MappingProxyType(
    {name: MappingProxyType(dict(values)) for name, values in _FORECAST_DEFAULT_CONFIGS.items()}
)


def _checked_bias_rate(value: object) -> float:
    """Return ``value`` as the bias-correction rate: a finite number in [0, 1]."""
    rate = _as_finite_float("bias_rate", value)
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"bias_rate must be in [0, 1], got {value!r}")
    return rate


@dataclass
class _BiasShareState:
    n_experts: int
    horizon: int
    bias_rate: float
    fs_state: object  # the Fixed-Share state FixedShare().init returned
    bias: list[float]  # [h] the current per-column bias estimate
    applied: dict[int, tuple[float, ...]]  # step -> the bias subtracted at that step


class BiasShare:
    """Fixed-Share combination minus an online per-horizon estimate of its own bias."""

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> _BiasShareState:
        experts, steps = _checked_init(
            n_experts, horizon, config, seed, ("eta", "alpha", "bias_rate")
        )
        eta = _checked_eta(config["eta"])
        alpha = _checked_alpha(config["alpha"])
        bias_rate = _checked_bias_rate(config["bias_rate"])
        return _BiasShareState(
            n_experts=experts,
            horizon=steps,
            bias_rate=bias_rate,
            fs_state=FixedShare().init(experts, steps, {"eta": eta, "alpha": alpha}, seed),
            bias=[0.0] * steps,
            applied={},
        )

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Output, object]:
        checked = _checked_state(state, _BiasShareState)
        n_experts, horizon = checked.n_experts, checked.horizon
        _validated_forecasts(n_experts, horizon, context)
        for index, matured in enumerate(feedback):
            _validated_errors(matured, index, n_experts, horizon)
        for matured in feedback:
            step = matured.step
            if step not in checked.applied:
                raise ValueError(f"no update call ever applied a bias at step {step}")
            applied_row = checked.applied.pop(step)
            for h in range(horizon):
                residual = matured.combined_errors[h] + applied_row[h]
                checked.bias[h] = checked.bias[h] + checked.bias_rate * (residual - checked.bias[h])
        weights, checked.fs_state = FixedShare().update(checked.fs_state, feedback, context)
        out = []
        for h in range(horizon):
            base = 0.0
            for e in range(n_experts):
                base = base + weights[e][h] * context.forecasts[e][h]
            out.append(base - checked.bias[h])
        checked.applied[context.step] = tuple(checked.bias)
        return (ForecastOutput(tuple(out)), checked)


# --- the factories and the registry ---------------------------------------------------------------


def bias_share() -> ForecastUpdater:
    """A fresh BiasShare forecast updater."""
    return BiasShare()


#: id -> zero-argument factory, in FORECAST_UPDATER_IDS order. Each entry is
#: replayed under interface "forecast-v2" (runner/cli.py, ``arena replay``).
FORECAST_UPDATERS: Mapping[str, Callable[[], ForecastUpdater]] = MappingProxyType(
    {"bias_share": bias_share}
)
