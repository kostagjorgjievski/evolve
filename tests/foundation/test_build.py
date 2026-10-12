"""Build tests: contexts, look-ahead, batching, validation, crossing (PR A tests 1-7, 9)."""

import math
import subprocess
import sys

import pytest

from evolvets.foundation.build import forecast_series, origin_contexts
from evolvets.foundation.errors import FoundationError
from evolvets.foundation.fake import FakeAdapter
from evolvets.foundation.interfaces import QUANTILES


def _series(n):
    """A deterministic, strictly increasing series where each value encodes its index."""
    return [float(i) + 0.25 * math.sin(i) for i in range(n)]


# --- Test 1: origin_contexts yields exactly the documented slices ----------------------


@pytest.mark.parametrize(
    ("first", "max_context"),
    [(4, 3), (0, 3), (4, 12), (4, 50)],  # L smaller than, equal to and larger than the series
)
def test_origin_contexts_slices(first, max_context):
    """Every origin t gets values[max(0, t + 1 - L) : t + 1], for t = first .. len - 1."""
    values = _series(12)
    out = list(origin_contexts(values, first, max_context))
    assert [t for t, _ in out] == list(range(first, 12))
    for t, context in out:
        assert context == tuple(values[max(0, t + 1 - max_context) : t + 1])


# --- Test 2: no look-ahead -------------------------------------------------------------


def test_no_look_ahead():
    """Values after a cut cannot change forecasts at or before it, and never reach the adapter."""
    values = _series(40)
    cut = 20
    changed = values[: cut + 1] + [v * 0.5 - 3.0 for v in values[cut + 1 :]]
    before = forecast_series(FakeAdapter(), values, first=5, max_context=7, horizon=4, batch_size=8)
    after = forecast_series(FakeAdapter(), changed, first=5, max_context=7, horizon=4, batch_size=8)
    for k in range(cut - 5 + 1):
        assert before.values[k] == after.values[k], f"origin {5 + k}"
    # The test is meaningful: forecasts after the cut do change.
    assert before.values[cut + 1 - 5] != after.values[cut + 1 - 5]


def test_contexts_never_reach_beyond_their_origin():
    """Each recorded context is exactly values[max(0, t + 1 - L) : t + 1] for its origin."""
    values = _series(40)
    first, max_context = 5, 7
    adapter = FakeAdapter()
    forecast_series(adapter, values, first, max_context, 4, 8)
    assert len(adapter.seen) == 40 - first
    for k, context in enumerate(adapter.seen):
        t = first + k
        assert context == tuple(values[max(0, t + 1 - max_context) : t + 1])


# --- Test 3: batch-size invariance -----------------------------------------------------


def test_batch_size_invariance():
    """Batch sizes 1, 7 and larger than the origin count give identical outputs."""
    values = _series(20)
    reference = None
    for batch_size in (1, 7, 1000):
        out = forecast_series(
            FakeAdapter(), values, first=3, max_context=5, horizon=6, batch_size=batch_size
        )
        if reference is None:
            reference = out
        else:
            assert out.values == reference.values
            assert out.sorted_steps == reference.sorted_steps


# --- Test 4: output shape --------------------------------------------------------------


def test_output_shape():
    """[n_origins][len(QUANTILES)][horizon] with n_origins = len(values) - first."""
    values = _series(15)
    first, horizon = 4, 5
    out = forecast_series(FakeAdapter(), values, first, 3, horizon, 4)
    assert len(out.values) == 15 - first
    for row in out.values:
        assert len(row) == len(QUANTILES)
        for quantile in row:
            assert len(quantile) == horizon
    assert out.sorted_steps == 0  # the fake output is monotone in the quantile level


# --- Test 5: malformed adapter output --------------------------------------------------


class _StubAdapter:
    """Always returns the same prepared structure, ignoring the contexts it is sent.

    Used with a batch size that swallows every origin in one call, so the
    caller sees exactly ``len(output)`` rows for ``n_origins`` contexts.
    """

    def __init__(self, output):
        self.name = "stub"
        self.output = output

    def forecast(self, contexts, horizon):
        return self.output


def _good_output(n_origins, horizon):
    return [
        [[10.0 + q + 0.1 * h for h in range(horizon)] for q in range(len(QUANTILES))]
        for _ in range(n_origins)
    ]


def _wrong_context_count(output):
    output.pop()


def _too_many_contexts(output):
    output.append(output[0])


def _wrong_quantile_count(output):
    output[1].pop()


def _wrong_horizon(output):
    output[2][4].append(99.0)


def _nan(output):
    output[0][0][2] = float("nan")


def _inf(output):
    output[0][4][1] = float("inf")


def _non_float(output):
    output[1][2][0] = "x"


def _non_float_int(output):
    output[1][2][0] = 3


def _row_not_a_sequence(output):
    output[0] = 1.0


@pytest.mark.parametrize(
    "mutate",
    [
        _wrong_context_count,
        _too_many_contexts,
        _wrong_quantile_count,
        _wrong_horizon,
        _nan,
        _inf,
        _non_float,
        _non_float_int,
        _row_not_a_sequence,
    ],
    ids=[
        "wrong-context-count",
        "too-many-contexts",
        "wrong-quantile-count",
        "wrong-horizon",
        "nan",
        "inf",
        "non-float",
        "non-float-int",
        "row-not-a-sequence",
    ],
)
def test_malformed_adapter_output(mutate):
    """Every kind of malformed output raises FoundationError naming the adapter."""
    output = _good_output(3, 4)
    mutate(output)
    with pytest.raises(FoundationError, match="stub"):
        forecast_series(_StubAdapter(output), [1.0, 2.0, 3.0, 4.0], 1, 2, 4, 10)


# --- Test 6: crossed quantiles are sorted and counted ----------------------------------


def _row_from_columns(columns):
    """Build a [quantile][h] row from columns[h][q]."""
    return [[columns[h][q] for h in range(len(columns))] for q in range(len(columns[0]))]


def test_crossed_quantiles_sorted_and_counted():
    """Crossed columns are sorted ascending; sorted_steps counts the (origin, step) pairs."""
    monotone = [float(q) for q in range(len(QUANTILES))]
    crossed = monotone[::-1]
    rows = [
        _row_from_columns([monotone, crossed, monotone]),  # crosses at step 2
        _row_from_columns([crossed, monotone, crossed]),  # crosses at steps 1 and 3
        _row_from_columns([monotone, monotone, monotone]),  # never crosses
    ]
    out = forecast_series(_StubAdapter(rows), [1.0, 2.0, 3.0, 4.0], 1, 2, 3, 3)
    assert out.sorted_steps == 3
    expected = [
        _row_from_columns([monotone, sorted(crossed), monotone]),
        _row_from_columns([sorted(crossed), monotone, sorted(crossed)]),
        _row_from_columns([monotone, monotone, monotone]),
    ]
    frozen = tuple(tuple(tuple(step) for step in row) for row in expected)
    assert out.values == frozen
    # Every column is now non-decreasing in the quantile level.
    for row in out.values:
        for h in range(3):
            column = [row[q][h] for q in range(len(QUANTILES))]
            assert column == sorted(column)


# --- Test 7: invalid arguments ---------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"values": []},
        {"values": [1.0, float("nan"), 3.0]},
        {"values": [1.0, float("inf"), 3.0]},
        {"values": [1.0, True, 3.0]},
        {"first": -1},
        {"first": 4},
        {"first": True},
        {"max_context": 0},
        {"max_context": True},
        {"horizon": 0},
        {"horizon": True},
        {"batch_size": 0},
        {"batch_size": True},
    ],
    ids=[
        "empty-values",
        "nan-value",
        "inf-value",
        "bool-value",
        "first-negative",
        "first-too-large",
        "first-bool",
        "max-context-zero",
        "max-context-bool",
        "horizon-zero",
        "horizon-bool",
        "batch-size-zero",
        "batch-size-bool",
    ],
)
def test_forecast_series_invalid_arguments(overrides):
    """Each invalid argument raises FoundationError, bools included."""
    args = {
        "values": [1.0, 2.0, 3.0, 4.0],
        "first": 1,
        "max_context": 2,
        "horizon": 3,
        "batch_size": 2,
    }
    args.update(overrides)
    with pytest.raises(FoundationError):
        forecast_series(FakeAdapter(), **args)


@pytest.mark.parametrize(
    "overrides",
    [
        {"values": []},
        {"values": [1.0, float("nan")]},
        {"first": -1},
        {"first": 2},
        {"first": True},
        {"max_context": 0},
        {"max_context": True},
    ],
    ids=["empty", "nan", "negative", "too-large", "bool", "zero", "bool-context"],
)
def test_origin_contexts_invalid_arguments(overrides):
    """origin_contexts validates its own arguments the same way."""
    args = {"values": [1.0, 2.0], "first": 0, "max_context": 1}
    args.update(overrides)
    with pytest.raises(FoundationError):
        list(origin_contexts(**args))


# --- Test 9: importing the package needs no torch --------------------------------------


def test_import_without_torch():
    """import evolvets.foundation and .registry succeed with torch absent from sys.modules."""
    code = (
        "import sys; import evolvets.foundation, evolvets.foundation.registry; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
