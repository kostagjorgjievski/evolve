"""Hindsight oracles over the expert pool: what selection could have achieved.

The oracles are arena computations over targets, like the synthetic true-best
forecast: for every series they report, per pool expert, the MASE of having
played that expert alone (``expert_mase``); the best MASE of a sequence that
switches experts at most ``m`` times (``switch_mase``); and the MASE of the
best expert at every origin, the switching ceiling (``select_mase``). They are
computed in hindsight, after every target has matured, and are never given to
an updater.

The oracles bound single-expert selection only: they are the floor for any
rule that commits to one expert (fixed, switched or per-origin), and they are
not a floor for weighted combinations, whose averaged forecast can beat every
single expert at an origin. The Gate 1 report must not present them as one.

Float arithmetic sticks to ``+ - * /`` and comparisons, accumulated
left-to-right in the order docs/tasks/TASK_010_updater_baselines.md states, so
oracle values are bit-identical everywhere. Targets are read only through
:class:`~evolvets.arena.labels.LabelRelease`.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from evolvets.arena import ARENA_VERSION
from evolvets.arena.dev import DevPool, DevSeries
from evolvets.arena.errors import ArenaError
from evolvets.arena.labels import LabelRelease
from evolvets.arena.metrics import mase_scale, origin_losses
from evolvets.arena.pool import load_pool_forecasts, pool_expert_names, pool_leakage_risk

#: The switch budgets ``switch_mase`` reports, per series and pooled.
ORACLE_SWITCHES = (0, 1, 2, 4, 8, 16, 32)

_INF = float("inf")
_NEG_INF = float("-inf")


@dataclass(frozen=True)
class SeriesOracle:
    """One series' hindsight oracles, over the pool's experts."""

    series_id: str
    expert_mase: tuple[float, ...]  # per pool expert, in pool order
    switch_mase: tuple[float, ...]  # per ORACLE_SWITCHES entry, same order
    select_mase: float  # best expert at every origin (unlimited switches)


@dataclass(frozen=True)
class OracleResult:
    """The oracles of one whole pool, in pool order.

    ``pool_spec``, ``foundation_cache_ids`` and ``leakage_risk`` pin the pool
    the oracles were computed over, exactly as :class:`ReplayResult` does.
    """

    dataset: str
    cache_id: str
    arena_version: str
    pool_spec: str
    foundation_cache_ids: tuple[str, ...]
    leakage_risk: bool
    expert_names: tuple[str, ...]
    switches: tuple[int, ...]  # == ORACLE_SWITCHES
    global_expert: str  # the single expert best over the whole pool
    metrics: dict[str, float]
    series: tuple[SeriesOracle, ...]


def _require_int(name: str, value: object, minimum: int) -> int:
    """Return ``value`` as an int >= ``minimum`` (bools rejected), else raise ArenaError."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ArenaError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise ArenaError(f"{name} must be >= {minimum}, got {value}")
    return value


def _is_finite(value: float) -> bool:
    """Return whether ``value`` is neither NaN nor an infinity, via comparisons only."""
    return value == value and value != _INF and value != _NEG_INF


def _lowest(values: Sequence[float]) -> float:
    """The minimum of ``values`` by a left-to-right scan (pool order breaks ties)."""
    low = values[0]
    for value in values[1:]:
        low = value if value < low else low
    return low


def expert_origin_losses(pool: DevPool, series: DevSeries) -> tuple[tuple[float, ...], ...]:
    """Per expert, in pool order, the mean absolute error of every scored origin.

    Each scored origin's targets are read once, exactly at maturity, through
    :meth:`LabelRelease.targets`; expert ``e``'s row uses the same arithmetic
    as :func:`~evolvets.arena.metrics.origin_losses`, so it equals the
    per-origin losses of a one-hot replay on ``e``.
    """
    horizon = pool.manifest.horizon
    release = LabelRelease(series.values, horizon)
    forecasts = load_pool_forecasts(pool, series.series_id)
    n_experts = len(pool_expert_names(pool))
    rows: list[list[tuple[float, ...]]] = [[] for _ in range(n_experts)]
    for origin in range(series.score_start, series.score_end):
        targets = release.targets(origin, now=origin + horizon)
        at = forecasts.at(origin)
        if len(at) != n_experts:
            raise ArenaError(
                f"{series.series_id}: the pool names {n_experts} experts, but its forecasts "
                f"hold {len(at)} rows at origin {origin}"
            )
        for e in range(n_experts):
            rows[e].append(tuple(at[e][h] - targets[h] for h in range(horizon)))
    return tuple(origin_losses(rows[e]) for e in range(n_experts))


def switching_total(losses: Sequence[Sequence[float]], max_switches: int) -> float:
    """The minimum total loss over expert sequences with at most ``max_switches`` switches.

    A sequence picks one expert per origin; a switch is an index ``k`` where
    the pick differs from the previous origin's. The dynamic programme keeps
    ``D[j][e]``, the best total over sequences up to the current origin that
    end at expert ``e`` with at most ``j`` switches, updating every ``j`` from
    the previous origin's values; additions run left to right in origin order,
    so for ``max_switches = 0`` the result is the best row's loop sum exactly.
    """
    m = _require_int("max_switches", max_switches, 0)
    rows = tuple(tuple(row) for row in losses)
    if not rows:
        raise ArenaError("losses must hold at least one expert's row")
    n = len(rows[0])
    if n == 0:
        raise ArenaError("every losses row must hold at least one origin")
    for row in rows:
        if len(row) != n:
            raise ArenaError(f"losses rows must share one length, got {len(row)} and {n}")
        for value in row:
            if not _is_finite(value):
                raise ArenaError(f"losses values must be finite, got {value!r}")
    n_experts = len(rows)
    if m >= n - 1:  # every origin can pick its own expert
        total = 0.0
        for k in range(n):
            column = [rows[e][k] for e in range(n_experts)]
            total = total + _lowest(column)
        return total
    previous = [[rows[e][0] for e in range(n_experts)] for _ in range(m + 1)]
    for k in range(1, n):
        best_previous = [_lowest(previous[j]) for j in range(m + 1)]
        current: list[list[float]] = []
        for j in range(m + 1):
            updated: list[float] = []
            for e in range(n_experts):
                best = previous[j][e]
                if j > 0:
                    switched = best_previous[j - 1]
                    best = switched if switched < best else best
                updated.append(best + rows[e][k])
            current.append(updated)
        previous = current
    return _lowest(previous[m])


def oracle_dataset(pool: DevPool) -> OracleResult:
    """Compute the hindsight oracles of every series of ``pool``, in pool order.

    A pool or cache problem raises :class:`ArenaError`; nothing untrusted
    runs, so there is no failed status.
    """
    names = pool_expert_names(pool)
    n_experts = len(names)
    if not pool.series:
        raise ArenaError("cannot compute the oracles of an empty pool")
    per_series: list[SeriesOracle] = []
    for series in pool.series:
        release = LabelRelease(series.values, pool.manifest.horizon)
        scale = mase_scale(release, series.period, series.scale_end, series.score_start)
        n = series.score_end - series.score_start
        losses = expert_origin_losses(pool, series)
        expert_mase: list[float] = []
        for row in losses:
            total = 0.0
            for value in row:
                total = total + value
            expert_mase.append(total / n / scale)
        switch_mase: list[float] = []
        for m in ORACLE_SWITCHES:
            switch_mase.append(switching_total(losses, m) / n / scale)
        select_total = 0.0
        for k in range(n):
            column = [losses[e][k] for e in range(n_experts)]
            select_total = select_total + _lowest(column)
        per_series.append(
            SeriesOracle(
                series_id=series.series_id,
                expert_mase=tuple(expert_mase),
                switch_mase=tuple(switch_mase),
                select_mase=select_total / n / scale,
            )
        )

    n_series = len(per_series)
    metrics: dict[str, float] = {"n_series": float(n_series)}
    best_total = 0.0
    for oracle in per_series:
        best_total = best_total + _lowest(oracle.expert_mase)
    metrics["best_expert_mase"] = best_total / n_series
    expert_totals = [0.0] * n_experts
    for oracle in per_series:
        for e in range(n_experts):
            expert_totals[e] = expert_totals[e] + oracle.expert_mase[e]
    expert_means = [expert_totals[e] / n_series for e in range(n_experts)]
    global_expert = 0
    for e in range(1, n_experts):
        if expert_means[e] < expert_means[global_expert]:
            global_expert = e
    metrics["global_expert_mase"] = expert_means[global_expert]
    for index, m in enumerate(ORACLE_SWITCHES):
        switch_total = 0.0
        for oracle in per_series:
            switch_total = switch_total + oracle.switch_mase[index]
        metrics[f"switch_mase_m{m}"] = switch_total / n_series
    select_mean_total = 0.0
    for oracle in per_series:
        select_mean_total = select_mean_total + oracle.select_mase
    metrics["select_mase"] = select_mean_total / n_series
    for e in range(n_experts):
        metrics[f"expert_mase/{names[e]}"] = expert_means[e]

    return OracleResult(
        dataset=pool.dataset,
        cache_id=pool.cache_id,
        arena_version=ARENA_VERSION,
        pool_spec=pool.pool_spec,
        foundation_cache_ids=tuple(cache.cache_id for cache in pool.foundation),
        leakage_risk=pool_leakage_risk(pool.dataset, pool.foundation),
        expert_names=names,
        switches=ORACLE_SWITCHES,
        global_expert=names[global_expert],
        metrics=metrics,
        series=tuple(per_series),
    )


def oracle_to_obj(result: OracleResult) -> dict[str, object]:
    """Return the JSON-ready object of ``result``; ``canonical_bytes`` accepts it."""
    return {
        "dataset": result.dataset,
        "cache_id": result.cache_id,
        "arena_version": result.arena_version,
        "pool_spec": result.pool_spec,
        "foundation_cache_ids": list(result.foundation_cache_ids),
        "leakage_risk": result.leakage_risk,
        "expert_names": list(result.expert_names),
        "switches": list(result.switches),
        "global_expert": result.global_expert,
        "metrics": dict(result.metrics),
        "series": [
            {
                "series_id": oracle.series_id,
                "expert_mase": list(oracle.expert_mase),
                "switch_mase": list(oracle.switch_mase),
                "select_mase": oracle.select_mase,
            }
            for oracle in result.series
        ],
    }
