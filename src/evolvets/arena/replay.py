"""Replay of one series: the loop that drives an updater and feeds it errors.

The updater sees expert forecasts and matured errors only, through
:class:`~evolvets.updater.interfaces.Context` and
:class:`~evolvets.updater.interfaces.Matured`; every target goes through
:class:`~evolvets.arena.labels.LabelRelease`. Nothing in this module indexes
the target series directly. Float arithmetic sticks to ``+ - * /``,
comparisons and ``abs``, accumulated left-to-right in the order stated in
docs/tasks/TASK_007_arena_replay.md, so traces are bit-identical everywhere.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol, cast

from evolvets.arena.errors import ArenaError, UpdaterFailure
from evolvets.arena.labels import LabelRelease
from evolvets.experts.pool import EXPERT_NAMES
from evolvets.updater.interfaces import ConfigValue, Context, Matured, Updater

SEED_LIMIT = 2**63

_INF = float("inf")
_NEG_INF = float("-inf")
_SUM_TOLERANCE = 1e-9


class Forecasts(Protocol):
    """One cached series: every pooled expert's forecasts for every origin.

    :class:`~evolvets.cache.store.SeriesForecasts` (the simple five) and
    :class:`~evolvets.arena.pool.PoolForecasts` (the five plus foundation
    point rows) both satisfy it, so the replay loop cannot tell a pool spec
    from the forecasts object it was handed.
    """

    series_id: str
    first_origin: int
    last_origin: int

    def at(self, origin: int) -> tuple[tuple[float, ...], ...]: ...  # [expert][h - 1]


def _require_int(name: str, value: object, minimum: int) -> int:
    """Return ``value`` as an int >= ``minimum`` (bools rejected), else raise ArenaError."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArenaError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ArenaError(f"{name} must be >= {minimum}, got {value}")
    return value


def _checked_expert_names(expert_names: Sequence[str]) -> tuple[str, ...]:
    """Return ``expert_names`` as a non-empty tuple of unique strings, else raise."""
    names = tuple(expert_names)
    if not names:
        raise ArenaError("expert_names must name at least one expert")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ArenaError(f"every expert name must be a non-empty str, got {name!r}")
    if len(set(names)) != len(names):
        raise ArenaError(f"expert_names must be unique, got {names}")
    return names


def _require_seed(seed: object) -> None:
    """Raise ArenaError unless ``seed`` is an int (not bool) in ``[0, SEED_LIMIT)``."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ArenaError(f"seed must be an int, got {seed!r}")
    if seed < 0 or seed >= SEED_LIMIT:
        raise ArenaError(f"seed must be in [0, {SEED_LIMIT}), got {seed}")


def series_seed(seed: int, series_id: str) -> int:
    """Derive one reproducible seed per series from the run seed and the series id."""
    _require_seed(seed)
    digest = sha256(
        b"evolvets/replay/v1\x00" + str(seed).encode() + b"\x00" + series_id.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & (SEED_LIMIT - 1)


@dataclass(frozen=True)
class SeriesTrace:
    """The combined-forecast errors of one replayed series, one row per origin."""

    series_id: str
    first_origin: int  # origin of step 0
    end_origin: int  # exclusive; last replayed origin is end_origin - 1
    horizon: int
    errors: tuple[tuple[float, ...], ...]  # [origin - first_origin][h - 1], combined - target


def _failure_message(exc: BaseException) -> str:
    """Format an updater exception, truncated to 1000 characters."""
    return f"{type(exc).__name__}: {exc}"[:1000]


def _is_finite(value: float) -> bool:
    """Return whether ``value`` is neither NaN nor an infinity, via comparisons only."""
    return value == value and value != _INF and value != _NEG_INF


def _validated_weights(
    weights: object, horizon: int, series_id: str, step: int, n_experts: int
) -> tuple[tuple[float, ...], ...]:
    """Type-check and copy what an updater returned, then validate the copy.

    Only exact ``list``/``tuple`` containers and exact ``int``/``float``
    elements are accepted, so a subclass whose ``__iter__`` or ``__mul__``
    lies cannot serve one contents to validation and another to the
    combination. The returned copy is what was validated and what is used.
    """
    try:
        if type(weights) is not list and type(weights) is not tuple:
            raise TypeError(f"weights must be a list or a tuple, got {type(weights).__name__}")
        rows: list[tuple[float, ...]] = []
        for row in cast("Sequence[object]", weights):
            if type(row) is not list and type(row) is not tuple:
                raise TypeError(
                    f"each weights row must be a list or a tuple, got {type(row).__name__}"
                )
            entries: list[float] = []
            for element in cast("Sequence[object]", row):
                if type(element) is not int and type(element) is not float:
                    raise TypeError(
                        f"weights entries must be int or float, got {type(element).__name__}"
                    )
                entries.append(float(element))
            rows.append(tuple(entries))
        stored = tuple(rows)
    except Exception as exc:
        raise UpdaterFailure("invalid_weights", _failure_message(exc), series_id, step) from None
    if len(stored) != n_experts:
        raise UpdaterFailure(
            "invalid_weights",
            f"weights must have {n_experts} rows, got {len(stored)}",
            series_id,
            step,
        )
    for row in stored:
        if len(row) != horizon:
            raise UpdaterFailure(
                "invalid_weights",
                f"each weights row must have {horizon} entries, got {len(row)}",
                series_id,
                step,
            )
        for weight in row:
            if not _is_finite(weight) or weight < 0.0:
                raise UpdaterFailure(
                    "invalid_weights",
                    f"weights entries must be finite and >= 0, got {weight!r}",
                    series_id,
                    step,
                )
    for h in range(horizon):
        total = 0.0
        for e in range(n_experts):
            total = total + stored[e][h]
        if abs(total - 1.0) > _SUM_TOLERANCE:
            raise UpdaterFailure(
                "invalid_weights",
                f"weights column {h + 1} sums to {total!r}, which is not 1 within 1e-9",
                series_id,
                step,
            )
    return stored


def replay_series(
    updater: Updater,
    config: Mapping[str, ConfigValue],
    seed: int,
    series_id: str,
    forecasts: Forecasts,
    values: Sequence[float],
    period: int,
    horizon: int,
    end_origin: int,
    expert_names: Sequence[str] = EXPERT_NAMES,
) -> SeriesTrace:
    """Replay one series: run the updater origin by origin and score its combinations.

    ``expert_names`` names the pool the forecasts hold, in pool order; the
    default is the simple five. Feedback for origin ``s`` arrives at exactly
    ``s + horizon``, once, as :class:`~evolvets.updater.interfaces.Matured`;
    the last ``min(horizon, end_origin - first_origin)`` origins are matured
    in a final drain with no further ``update`` calls. Raises
    :class:`UpdaterFailure` if the updater fails or returns invalid weights,
    :class:`ArenaError` for bad arguments.
    """
    horizon = _require_int("horizon", horizon, 1)
    names = _checked_expert_names(expert_names)
    n_experts = len(names)
    for forecast_row in forecasts.at(forecasts.first_origin):
        if len(forecast_row) != horizon:
            raise ArenaError(
                f"horizon {horizon} does not match the cached forecasts' row length "
                f"{len(forecast_row)}"
            )
    period = _require_int("period", period, 2)
    end_origin = _require_int("end_origin", end_origin, 1)
    _require_seed(seed)
    t0 = forecasts.first_origin
    if t0 >= end_origin:
        raise ArenaError(f"end_origin {end_origin} must be greater than the first origin {t0}")
    if end_origin - 1 > forecasts.last_origin:
        raise ArenaError(
            f"end_origin {end_origin} needs forecasts up to origin {end_origin - 1}, "
            f"but the cache stops at {forecasts.last_origin}"
        )
    if len(values) < end_origin + horizon:
        raise ArenaError(
            f"scoring every origin up to end_origin {end_origin} with horizon {horizon} "
            f"requires a value for every target, through index {end_origin + horizon - 1}"
        )
    release = LabelRelease(values, horizon)

    try:
        state: object = updater.init(n_experts, horizon, dict(config), series_seed(seed, series_id))
    except Exception as exc:
        raise UpdaterFailure("failed", _failure_message(exc), series_id, None) from None

    combined: list[tuple[float, ...]] = []
    errors: list[tuple[float, ...]] = []

    def mature(origin: int, now: int) -> Matured:
        """Read origin's targets (matured), record its trace row and build its feedback."""
        targets = release.targets(origin, now)
        origin_forecasts = forecasts.at(origin)
        combined_row = combined[origin - t0]
        combined_errors = tuple(combined_row[h] - targets[h] for h in range(horizon))
        errors.append(combined_errors)
        expert_errors = tuple(
            tuple(origin_forecasts[e][h] - targets[h] for h in range(horizon))
            for e in range(n_experts)
        )
        return Matured(step=origin - t0, errors=expert_errors, combined_errors=combined_errors)

    for t in range(t0, end_origin):
        k = t - t0
        feedback: tuple[Matured, ...] = ()
        if t - horizon >= t0:
            feedback = (mature(t - horizon, t),)
        current = forecasts.at(t)
        context = Context(
            step=k,
            season_index=t % period,
            period=period,
            horizon=horizon,
            expert_names=names,
            forecasts=current,
        )
        try:
            result: object = updater.update(state, feedback, context)
        except Exception as exc:
            raise UpdaterFailure("failed", _failure_message(exc), series_id, k) from None
        if type(result) is not tuple:
            raise UpdaterFailure("failed", "update must return (weights, state)", series_id, k)
        pair = cast("tuple[object, object]", result)
        if len(pair) != 2:
            raise UpdaterFailure("failed", "update must return (weights, state)", series_id, k)
        stored = _validated_weights(pair[0], horizon, series_id, k, n_experts)
        state = pair[1]
        totals: list[float] = []
        for h in range(horizon):
            total = 0.0
            for e in range(n_experts):
                total = total + stored[e][h] * current[e][h]
            totals.append(total)
        combined.append(tuple(totals))

    for t in range(end_origin, end_origin + horizon):
        if t - horizon >= t0:
            mature(t - horizon, t)

    return SeriesTrace(
        series_id=series_id,
        first_origin=t0,
        end_origin=end_origin,
        horizon=horizon,
        errors=tuple(errors),
    )
