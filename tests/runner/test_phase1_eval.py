"""The Phase 1 subprocess evaluator (TASK_018 PR A1: tests 1-7).

The evaluator is the wall the artifact grammar is not: every outcome — a
clean replay, a timed-out burn, a memory burn, a crashing artifact, a dead
worker, a garbage reply — must arrive as an EvalOutcome, never as an
exception in the parent. Test 1 anchors the numbers: the subprocess replay
of the reference artifact is bit-equal to an in-process replay_dataset call
on the same pool, which is what lets the engine's ledger records and the
Gate 1 world speak the same floats. The artifacts below are grammatical
(they validate) and pathological (they burn wall or memory or crash) — the
grammar's own blind spots, which is exactly what the deadline and the cap
are for.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from evolvets.arena.dev import load_dev_pool
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.improver.artifacts import (
    REFERENCE_ARTIFACT_CONFIG,
    REFERENCE_ARTIFACT_SOURCE,
    artifact_updater,
    validate_artifact_source,
)
from evolvets.runner import phase1_eval
from evolvets.runner.phase1_eval import EvalOutcome, evaluate_source

#: A grammatical artifact whose update loops a trillion times: burns wall,
#: not memory, so only the deadline can stop it.
TIMEOUT_ARTIFACT = '''"""A grammatical artifact that loops forever."""

def init(n_experts, horizon, config, seed):
    return {"horizon": horizon}


def update(state, feedback, context):
    total = 0.0
    for i in range(1000000000000):
        total = total + 1.0
    return ((0.0,) * state["horizon"], state)
'''

#: A grammatical artifact allocating a giant list: burns memory, not wall.
MEMORY_ARTIFACT = '''"""A grammatical artifact that allocates a giant list."""

def init(n_experts, horizon, config, seed):
    return {"horizon": horizon}


def update(state, feedback, context):
    big = list(range(500000000))
    return ((0.0,) * state["horizon"], state)
'''

#: A grammatical artifact that raises: the replay's error must surface.
CRASH_ARTIFACT = '''"""A grammatical artifact that raises."""

def init(n_experts, horizon, config, seed):
    return {}


def update(state, feedback, context):
    raise ValueError("boom: artifact crashed")
'''


@pytest.fixture(scope="module")
def tiny_pool(tmp_path_factory: pytest.TempPathFactory):
    """The synthetic-dev fixture pool (one series per family) and its request's path fields."""
    base = tmp_path_factory.mktemp("phase1-eval")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-dev", base, base, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    pool = load_dev_pool("synthetic-dev", cache_root, base, base)
    fields = {
        "dataset": "synthetic-dev",
        "pool_spec": "simple",
        "foundation_models": [],
        "cache_root": str(cache_root),
        "data_root": str(base),
        "manifest_dir": str(base),
        "foundation_root": str(base / "foundation_cache"),
    }
    return pool, fields


def reference_request(fields: dict) -> dict:
    """One evaluate_source request: the pinned reference artifact at its placeholder config."""
    return {
        "source": REFERENCE_ARTIFACT_SOURCE,
        "config": dict(REFERENCE_ARTIFACT_CONFIG),
        "seed": 0,
        "interface": "forecast-v2",
        **fields,
    }


# --- Test 1: the ok path is bit-equal to an in-process replay -----------------------


def test_ok_path_bit_equal_to_in_process_replay(tiny_pool) -> None:
    """The reference artifact replays ok in the subprocess, bit-equal to an in-process call."""
    pool, fields = tiny_pool
    outcome = evaluate_source(reference_request(fields), timeout_seconds=240.0, memory_gib=2.0)
    assert isinstance(outcome, EvalOutcome)
    assert outcome.status == "ok"
    assert outcome.error is None
    expected = replay_dataset(
        artifact_updater(REFERENCE_ARTIFACT_SOURCE),
        REFERENCE_ARTIFACT_CONFIG,
        0,
        pool,
        interface="forecast-v2",
    )
    assert outcome.result_obj == result_to_obj(expected)  # metrics and per-series detail
    assert outcome.wall_seconds > 0.0
    assert len(outcome.series_wall_seconds) == len(pool.series)
    assert all(value >= 0.0 for value in outcome.series_wall_seconds)


# --- Test 2: the wall deadline ------------------------------------------------------


def test_timeout_kills_the_worker_and_leaves_no_zombie(tiny_pool) -> None:
    """A trillion-iteration loop hits the deadline: timeout, bounded wall, no zombie."""
    _, fields = tiny_pool
    validate_artifact_source(TIMEOUT_ARTIFACT)  # grammatical: the grammar cannot stop it
    request = {**reference_request(fields), "source": TIMEOUT_ARTIFACT}
    outcome = evaluate_source(request, timeout_seconds=3.0, memory_gib=2.0)
    assert outcome.status == "timeout"
    assert outcome.result_obj is None
    assert outcome.series_wall_seconds == ()
    assert outcome.error is not None
    assert "3 s wall deadline" in outcome.error
    assert 3.0 <= outcome.wall_seconds < 30.0  # bounded: the kill lands at the deadline
    # the kill leaves no zombie: a later evaluation still works
    later = evaluate_source(reference_request(fields), timeout_seconds=240.0, memory_gib=2.0)
    assert later.status == "ok"


# --- Test 3: the memory cap ---------------------------------------------------------


def test_memory_cap_burns_out_inside_the_deadline(tiny_pool) -> None:
    """A giant list trips RLIMIT_AS inside the deadline; the parent is never the victim."""
    _, fields = tiny_pool
    validate_artifact_source(MEMORY_ARTIFACT)
    request = {**reference_request(fields), "source": MEMORY_ARTIFACT}
    started = time.monotonic()
    outcome = evaluate_source(request, timeout_seconds=120.0, memory_gib=0.5)
    elapsed = time.monotonic() - started
    assert outcome.status == "failed"  # MemoryError inside the worker, or its hard exit
    assert outcome.result_obj is None
    assert outcome.error is not None
    assert elapsed < 120.0 and outcome.wall_seconds < 120.0  # failed inside, not at, the deadline
    follow_up = evaluate_source(reference_request(fields), timeout_seconds=240.0, memory_gib=2.0)
    assert follow_up.status == "ok"


# --- Test 4: a crashing artifact ----------------------------------------------------


def test_artifact_crash_surfaces_the_replays_error(tiny_pool) -> None:
    """A ValueError inside update becomes a failed outcome with the replay's error text."""
    _, fields = tiny_pool
    validate_artifact_source(CRASH_ARTIFACT)
    request = {**reference_request(fields), "source": CRASH_ARTIFACT}
    outcome = evaluate_source(request, timeout_seconds=240.0, memory_gib=2.0)
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert "ValueError: boom: artifact crashed" in outcome.error
    assert outcome.result_obj is None


# --- Test 5: worker death -----------------------------------------------------------


def test_worker_death_is_a_failed_outcome(tiny_pool) -> None:
    """A nonexistent cache path kills the worker; the parent reports the stderr tail."""
    _, fields = tiny_pool
    request = reference_request(
        {**fields, "cache_root": str(Path(fields["cache_root"]) / "missing")}
    )
    outcome = evaluate_source(request, timeout_seconds=240.0, memory_gib=2.0)
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert outcome.error.startswith("worker: ")
    assert "ArenaError" in outcome.error  # the tail names the child's actual failure
    assert outcome.result_obj is None


# --- Test 6: a garbage reply --------------------------------------------------------


def test_garbage_reply_is_a_failed_outcome(monkeypatch: pytest.MonkeyPatch) -> None:
    """A stubbed worker printing noise gives a failed outcome; no exception escapes."""
    monkeypatch.setattr(
        phase1_eval, "_CHILD_COMMAND", (sys.executable, "-c", "print('this is not json')")
    )
    outcome = evaluate_source({"anything": 1}, timeout_seconds=30.0, memory_gib=1.0)
    assert outcome.status == "failed"
    assert outcome.error is not None
    assert outcome.error.startswith("worker: ")
    assert outcome.result_obj is None


def test_unstamped_and_oversized_replies_are_failed_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A JSON reply without the worker stamp, or one over the byte cap, is a failure."""
    monkeypatch.setattr(phase1_eval, "_CHILD_COMMAND", (sys.executable, "-c", "print('{}')"))
    unstamped = evaluate_source({}, timeout_seconds=30.0, memory_gib=1.0)
    assert unstamped.status == "failed"
    assert unstamped.error is not None
    assert unstamped.error.startswith("worker: ")
    monkeypatch.setattr(
        phase1_eval,
        "_CHILD_COMMAND",
        (sys.executable, "-c", "import sys; sys.stdout.write('x' * 5000)"),
    )
    monkeypatch.setattr(phase1_eval, "MAX_REPLY_BYTES", 1024)
    oversized = evaluate_source({}, timeout_seconds=30.0, memory_gib=1.0)
    assert oversized.status == "failed"
    assert oversized.error is not None
    assert oversized.error.startswith("worker: ")
    assert "exceeds" in oversized.error


# --- Test 7: interface pass-through -------------------------------------------------


def test_interface_pass_through_weights_v1_refuses_a_flat_forecast(tiny_pool) -> None:
    """The reference artifact under weights-v1 evaluates invalid_weights: the arena's rule."""
    _, fields = tiny_pool
    request = {**reference_request(fields), "interface": "weights-v1"}
    outcome = evaluate_source(request, timeout_seconds=240.0, memory_gib=2.0)
    assert outcome.status == "invalid_weights"
    assert outcome.result_obj is None
    assert outcome.error is not None
