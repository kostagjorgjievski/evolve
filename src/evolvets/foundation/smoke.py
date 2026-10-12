"""The foundation-model smoke check: one fixed series, batch invariance, usefulness.

``evolvets foundation smoke`` (TASK_009 PR C) runs a pretrained adapter on a
fixed sine-plus-trend series of length 512 at 20 origins and checks that
(1) the adapter produces well-formed finite forecasts for every origin,
(2) batching does not change the 0.5 quantile beyond 1e-3 relative, and
(3) the forecast's MAE on that series beats the naive (repeat-last) MAE.
It is a diagnostic: it reports a :class:`SmokeReport` and never raises on a
failed check — structural adapter faults are caught as problems too, so the
one-line JSON summary is always printable. Nothing here imports torch; the
adapter is passed in, so the fake adapter can be smoked in plain CI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from evolvets.cache.store import DEFAULT_HORIZON
from evolvets.foundation.build import _checked_row, _fix_crossing
from evolvets.foundation.errors import FoundationError
from evolvets.foundation.interfaces import QUANTILES, Adapter
from evolvets.foundation.registry import ModelSpec

LENGTH = 512
PERIOD = 24
N_ORIGINS = 20
BATCH_SIZES = (1, 16)
#: Maximum tolerated relative change of the 0.5 quantile between batch sizes.
MAX_REL_BATCH_DIFF = 1e-3


@dataclass(frozen=True)
class SmokeReport:
    """What the smoke check found, in the fields its JSON summary prints."""

    model: str
    device: str
    horizon: int
    max_context: int
    n_origins: int
    batch_sizes: tuple[int, ...]
    max_rel_batch_diff: float
    mae: float
    naive_mae: float
    ok: bool
    problems: tuple[str, ...]


def smoke_series() -> tuple[float, ...]:
    """Return the fixed series every smoke run forecasts: sine plus trend, length 512.

    The trend (0.15 per step against a ±1.0 sine) is strong enough that a
    forecaster tracking it — as any working model and the fake adapter do —
    clearly beats the naive repeat-last forecast on this series, while the
    sine keeps the extrapolation non-trivial.
    """
    return tuple(1.0 * math.sin(2.0 * math.pi * t / PERIOD) + 0.15 * t for t in range(LENGTH))


def _origins() -> list[int]:
    """Return the 20 smoke origins, evenly spread, each with its full horizon inside the series."""
    last = LENGTH - 1 - DEFAULT_HORIZON
    return [round(PERIOD + i * (last - PERIOD) / (N_ORIGINS - 1)) for i in range(N_ORIGINS)]


def _forecast_origins(
    adapter: Adapter,
    values: tuple[float, ...],
    origins: list[int],
    max_context: int,
    batch_size: int,
) -> list[list[list[float]]]:
    """Forecast exactly ``origins`` through ``adapter`` in batches, validated like the builder.

    The context at origin ``t`` is ``values[max(0, t + 1 - L) : t + 1]`` — the
    builder's exact look-ahead rule — and every returned row goes through the
    builder's own validation and monotone rearrangement.
    """
    horizon = DEFAULT_HORIZON
    contexts = [values[max(0, t + 1 - max_context) : t + 1] for t in origins]
    rows: list[list[list[float]]] = []
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start : start + batch_size]
        output = adapter.forecast(batch, horizon)
        if len(output) != len(batch):
            raise FoundationError(
                f"{adapter.name}: {len(batch)} contexts sent, {len(output)} forecast rows returned"
            )
        for i, entry in enumerate(output):
            rows.append(_checked_row(adapter.name, origins[start + i], entry, horizon))
    _fix_crossing(rows, horizon)
    return rows


def run_smoke(spec: ModelSpec, adapter: Adapter) -> SmokeReport:
    """Smoke-test ``adapter`` against ``spec`` on the fixed series; report, never raise."""
    origins = _origins()
    values = smoke_series()
    device = str(getattr(adapter, "device", "unknown"))

    try:
        if adapter.name != spec.name:
            raise FoundationError(
                f"adapter name {adapter.name!r} does not match the model spec name {spec.name!r}"
            )
        single = _forecast_origins(adapter, values, origins, spec.max_context, batch_size=1)
        batched = _forecast_origins(adapter, values, origins, spec.max_context, batch_size=16)
    except FoundationError as exc:
        return _failed(spec, device, f"malformed adapter output: {exc}")

    point = QUANTILES.index(0.5)
    max_rel = 0.0
    mae_sum = 0.0
    naive_sum = 0.0
    compared = 0
    for i, origin in enumerate(origins):
        for h in range(DEFAULT_HORIZON):
            actual = values[origin + 1 + h]
            one = single[i][point][h]
            many = batched[i][point][h]
            diff = abs(one - many)
            max_rel = max(max_rel, diff / max(abs(one), abs(many), 1e-12))
            mae_sum += abs(actual - one)
            naive_sum += abs(actual - values[origin])
            compared += 1

    problems: list[str] = []
    if max_rel > MAX_REL_BATCH_DIFF:
        problems.append(
            f"batch size changes the 0.5 quantile by {max_rel:.3e} relative "
            f"(> {MAX_REL_BATCH_DIFF:.0e})"
        )
    mae = mae_sum / compared
    naive_mae = naive_sum / compared
    if not mae < naive_mae:
        problems.append(f"forecast MAE {mae:.6f} does not beat the naive MAE {naive_mae:.6f}")

    return SmokeReport(
        model=spec.name,
        device=device,
        horizon=DEFAULT_HORIZON,
        max_context=spec.max_context,
        n_origins=len(origins),
        batch_sizes=BATCH_SIZES,
        max_rel_batch_diff=max_rel,
        mae=mae,
        naive_mae=naive_mae,
        ok=not problems,
        problems=tuple(problems),
    )


def _failed(spec: ModelSpec, device: str, problem: str) -> SmokeReport:
    """Return the report of a smoke run whose adapter output could not be used at all."""
    return SmokeReport(
        model=spec.name,
        device=device,
        horizon=DEFAULT_HORIZON,
        max_context=spec.max_context,
        n_origins=0,
        batch_sizes=BATCH_SIZES,
        max_rel_batch_diff=math.inf,
        mae=math.inf,
        naive_mae=math.inf,
        ok=False,
        problems=(problem,),
    )
