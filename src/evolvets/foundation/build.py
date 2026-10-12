"""Turn one series into per-origin contexts and batch them through an adapter.

Origin convention (as in TASK_004/006/007): origin ``t`` means ``y_0 .. y_t``
are observed and the forecast for step ``h = 1 .. H`` targets ``y_{t+h}``.
The context handed to the adapter at origin ``t`` is exactly
``values[max(0, t + 1 - L) : t + 1]`` — no other value of the series ever
reaches the model.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

from evolvets.foundation.errors import FoundationError
from evolvets.foundation.interfaces import QUANTILES, Adapter, SeriesForecast


def _finite_values(name: str, values: Sequence[float]) -> tuple[float, ...]:
    """Return ``values`` as a tuple of finite floats (bools rejected), else raise."""
    if len(values) == 0:
        raise FoundationError(f"{name} must be non-empty")
    out: list[float] = []
    for i, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise FoundationError(f"{name}[{i}] must be a finite float, got {value!r}")
        result = float(value)
        if not math.isfinite(result):
            raise FoundationError(f"{name}[{i}] must be a finite float, got {value!r}")
        out.append(result)
    return tuple(out)


def _require_int(name: str, value: int, minimum: int) -> int:
    """Return ``value`` as an int >= ``minimum`` (bools rejected), else raise FoundationError."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise FoundationError(f"{name} must be an int, got {value!r}")
    if value < minimum:
        raise FoundationError(f"{name} must be >= {minimum}, got {value}")
    return value


def origin_contexts(
    values: Sequence[float], first: int, max_context: int
) -> Iterator[tuple[int, tuple[float, ...]]]:
    """Yield ``(t, values[max(0, t + 1 - max_context) : t + 1])`` for every origin ``t``.

    ``t`` runs from ``first`` to ``len(values) - 1`` inclusive.
    """
    data = _finite_values("values", values)
    first_ok = _require_int("first", first, 0)
    if first_ok >= len(data):
        raise FoundationError(f"first must be < len(values) = {len(data)}, got {first_ok}")
    context_ok = _require_int("max_context", max_context, 1)
    return _yield_contexts(data, first_ok, context_ok)


def _yield_contexts(
    data: tuple[float, ...], first: int, max_context: int
) -> Iterator[tuple[int, tuple[float, ...]]]:
    for t in range(first, len(data)):
        yield t, data[max(0, t + 1 - max_context) : t + 1]


def _checked_row(
    name: str, origin: int, entry: Sequence[Sequence[float]], horizon: int
) -> list[list[float]]:
    """Return ``entry`` as ``[quantile][h]`` finite floats, raising FoundationError if malformed."""
    row: list[list[float]] = []
    if not isinstance(entry, Sequence) or isinstance(entry, (str, bytes)):
        raise FoundationError(
            f"{name}: origin {origin}: quantile rows are not a sequence: {entry!r}"
        )
    if len(entry) != len(QUANTILES):
        raise FoundationError(
            f"{name}: origin {origin}: {len(entry)} quantile rows, expected {len(QUANTILES)}"
        )
    for q, quantile in enumerate(entry):
        if not isinstance(quantile, Sequence) or isinstance(quantile, (str, bytes)):
            raise FoundationError(
                f"{name}: origin {origin}: quantile {q} is not a sequence: {quantile!r}"
            )
        if len(quantile) != horizon:
            raise FoundationError(
                f"{name}: origin {origin}: quantile {q} has {len(quantile)} steps,"
                f" expected horizon {horizon}"
            )
        step_values: list[float] = []
        for h, value in enumerate(quantile):
            if isinstance(value, bool) or not isinstance(value, float):
                raise FoundationError(
                    f"{name}: origin {origin}: quantile {q} step {h + 1} is not a float: {value!r}"
                )
            if not math.isfinite(value):
                raise FoundationError(
                    f"{name}: origin {origin}: quantile {q} step {h + 1} is not finite: {value!r}"
                )
            step_values.append(value)
        row.append(step_values)
    return row


def _fix_crossing(rows: list[list[list[float]]], horizon: int) -> int:
    """Sort non-monotone quantile columns ascending in place; return how many steps were sorted."""
    sorted_steps = 0
    n_quantiles = len(QUANTILES)
    for row in rows:
        for h in range(horizon):
            column = [row[q][h] for q in range(n_quantiles)]
            if any(column[q] > column[q + 1] for q in range(n_quantiles - 1)):
                column.sort()
                for q in range(n_quantiles):
                    row[q][h] = column[q]
                sorted_steps += 1
    return sorted_steps


def forecast_series(
    adapter: Adapter,
    values: Sequence[float],
    first: int,
    max_context: int,
    horizon: int,
    batch_size: int,
) -> SeriesForecast:
    """Forecast every origin of ``values`` through ``adapter``, in origin order.

    Contexts are sent in batches of ``batch_size`` (the last batch may be
    shorter). The adapter's output is validated — context count, quantile
    count, horizon, finite floats — and quantiles that are not non-decreasing
    at a step are sorted ascending (monotone rearrangement), with
    :attr:`SeriesForecast.sorted_steps` counting the (origin, step) pairs
    sorted. Bad output is never patched beyond that or skipped.
    """
    horizon_ok = _require_int("horizon", horizon, 1)
    batch_ok = _require_int("batch_size", batch_size, 1)
    pairs = list(origin_contexts(values, first, max_context))
    contexts = [context for _, context in pairs]
    origins = [t for t, _ in pairs]

    rows: list[list[list[float]]] = []
    for start in range(0, len(contexts), batch_ok):
        batch = contexts[start : start + batch_ok]
        batch_origins = origins[start : start + batch_ok]
        output = adapter.forecast(batch, horizon_ok)
        if not isinstance(output, Sequence) or isinstance(output, (str, bytes)):
            raise FoundationError(
                f"{adapter.name}: origins {batch_origins[0]}..{batch_origins[-1]}:"
                f" forecast output is not a sequence of per-context rows: {type(output).__name__}"
            )
        if len(output) != len(batch):
            raise FoundationError(
                f"{adapter.name}: origins {batch_origins[0]}..{batch_origins[-1]}:"
                f" {len(batch)} contexts sent, {len(output)} forecast rows returned"
            )
        for i, entry in enumerate(output):
            rows.append(_checked_row(adapter.name, batch_origins[i], entry, horizon_ok))

    sorted_steps = _fix_crossing(rows, horizon_ok)
    frozen = tuple(tuple(tuple(step) for step in row) for row in rows)
    return SeriesForecast(values=frozen, sorted_steps=sorted_steps)
