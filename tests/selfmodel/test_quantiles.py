"""Tests for the conformal ridge self-model and the prediction record (TASK_017 PR B).

Tests 3, 5 and 6 pin exact floats: the expected values are derived in the
same operation order as the implementation (``2 / 3``, ``1 - 2 / 3``, the
blend as ``w * raw + (1 - w) * prior``), because 2/3 is not exactly
representable and decimal literals would drift. The property test (4)
checks monotone quantiles over seeded draws — it has teeth because the
implementation runs no sort that could mask a crossing. The record tests
(9-11) run against a real run directory whose proposal comes from the
non-LLM proposer, so the flush-before-evaluation rule (AGENTS.md 2.3) is
checked exactly the way (f)'s lineage loop will exercise it.
"""

from __future__ import annotations

import json
import math
import random
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from evolvets.improver.artifacts import REFERENCE_ARTIFACT_SOURCE
from evolvets.improver.interfaces import ProposalFeatures
from evolvets.improver.propose import propose_and_record
from evolvets.ledger.payloads import (
    QUANTILE_KEYS,
    Evaluation,
    PayloadError,
    Prediction,
    payload_from_dict,
    payload_to_dict,
)
from evolvets.ledger.schema import Budget, Record
from evolvets.ledger.store import LedgerReader
from evolvets.ledger.validate import validate_run
from evolvets.runner.runs import Run
from evolvets.selfmodel.interfaces import DEFAULT_PRIOR, HistoryPoint
from evolvets.selfmodel.quantiles import ConformalRidgeSelfModel, append_prediction

COMMIT = "a" * 40
LIMITS = Budget(wall_seconds=1e9, evaluations=1e9)
RUN_CONFIG: dict[str, Any] = {"selfmodel": "conformal_ridge", "metric": "dev_mase_gain"}


def features(**overrides: Any) -> ProposalFeatures:
    """ProposalFeatures with all-zero defaults; each test overrides what it varies."""
    values: dict[str, Any] = {
        "family": "params",
        "changed_lines": 0,
        "const_changes": 0,
        "max_rel_const_change": 0.0,
        "generation": 0,
        "parent_gain": 0.0,
        "is_llm": False,
    }
    values.update(overrides)
    return ProposalFeatures(**values)


def drawn_features(rng: random.Random) -> ProposalFeatures:
    """Features drawn uniformly over their ranges, the clips included."""
    return ProposalFeatures(
        family=rng.choice(("params", "logic")),
        changed_lines=rng.randint(0, 30),
        const_changes=rng.randint(0, 8),
        max_rel_const_change=rng.uniform(0.0, 2.0),
        generation=rng.randint(0, 80),
        parent_gain=rng.uniform(-2.0, 2.0),
        is_llm=rng.random() < 0.5,
    )


def start_run(root: Path, run_id: str) -> Run:
    """A run that declares requires_predictions: the rule tests 10 and 11 exercise."""
    return Run.start(
        root,
        run_id,
        condition="selfmodel",
        seed=11,
        config=RUN_CONFIG,
        budget_limits=LIMITS,
        git_commit=COMMIT,
        requires_predictions=True,
    )


def proposal_in(run: Run) -> Record:
    """One proposal through the non-LLM proposer: no client, no key, no network."""
    return propose_and_record(
        run,
        proposer_id="random_param",
        parent_source=REFERENCE_ARTIFACT_SOURCE,
        parent_record_id=None,
        generation=0,
        parent_gain=0.0,
        history_records=[],
        rng=random.Random(0),
    )


def dev_evaluation() -> Evaluation:
    """One ok dev evaluation, the record whose prediction must precede it."""
    return Evaluation(
        pool="dev", status="ok", metrics={"mase": 0.8}, per_task_blob=None, error=None
    )


def records_of(root: Path, run_id: str) -> list[Record]:
    return list(LedgerReader(root, run_id).records())


# --- Tests 1-2: identity, prior, empty history --------------------------------------


def test_identity_and_prior() -> None:
    """identify() is exact; the default prior is finite and strictly increasing."""
    model = ConformalRidgeSelfModel([])
    assert model.identify() == {"id": "conformal_ridge", "version": "1"}
    ordered = [DEFAULT_PRIOR[key] for key in QUANTILE_KEYS]
    assert all(math.isfinite(value) for value in ordered)
    assert all(earlier < later for earlier, later in pairwise(ordered))


def test_empty_history_returns_the_prior_exactly() -> None:
    """No history: every prediction is the prior, whatever the features."""
    model = ConformalRidgeSelfModel([])
    assert model.predict(features()) == dict(DEFAULT_PRIOR)
    assert model.predict(drawn_features(random.Random(5))) == dict(DEFAULT_PRIOR)


# --- Test 3: the orthogonal hand computation ----------------------------------------


def test_orthogonal_hand_computation() -> None:
    """Two points differing only in parent_gain: every quantile equals the hand value.

    The intercept and family columns get zero weight (their targets cancel),
    so w_parent_gain solves (2 + 1) w = 2 and μ̂ = ±2/3. The residuals are
    ±(1 - 2/3); with n = 2 the order statistics sit at indexes 0, 0, 0, 1, 1
    for q05..q95. 2/3 is not exactly representable, so the expected floats
    are derived in the implementation's operation order, never as decimals.
    """
    history = [
        HistoryPoint(features(parent_gain=1.0), 1.0),
        HistoryPoint(features(parent_gain=-1.0), -1.0),
    ]
    model = ConformalRidgeSelfModel(history, ridge_lambda=1.0)
    mu = 2 / 3  # w_parent_gain: (2 + 1) w = 2, from the hand-built normal equations
    res = 1 - 2 / 3  # the residual magnitude; the same float at either sign
    weight = 2 / (2 + 8.0)  # n / (n + prior_points), the default prior_points
    raw_low = mu - res  # μ̂ + the first order statistic, which is -res
    raw_high = mu + res
    plus = model.predict(features(parent_gain=1.0))
    assert set(plus) == set(QUANTILE_KEYS)
    assert plus == {
        key: weight * raw + (1 - weight) * DEFAULT_PRIOR[key]
        for key, raw in (
            ("q05", raw_low),
            ("q25", raw_low),
            ("q50", raw_low),
            ("q75", raw_high),
            ("q95", raw_high),
        )
    }
    assert model.predict(features(parent_gain=-1.0)) == {
        key: weight * raw + (1 - weight) * DEFAULT_PRIOR[key]
        for key, raw in (
            ("q05", -(mu + res)),
            ("q25", -(mu + res)),
            ("q50", -(mu + res)),
            ("q75", -(mu - res)),
            ("q95", -(mu - res)),
        )
    }


# --- Test 4: monotone by construction (property) ------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n", [1, 2, 5, 20])
def test_quantiles_are_monotone_by_construction(seed: int, n: int) -> None:
    """Seeded histories and queries always give five non-decreasing quantiles.

    Monotonicity holds by construction — the shrinkage paragraph of the task
    — so this test has teeth exactly because no sort runs that could mask a
    crossing.
    """
    rng = random.Random(seed * 101 + n)
    history = [HistoryPoint(drawn_features(rng), rng.uniform(-0.05, 0.05)) for _ in range(n)]
    model = ConformalRidgeSelfModel(history)
    for _ in range(5):
        values = model.predict(drawn_features(rng))
        ordered = [values[key] for key in QUANTILE_KEYS]
        assert all(earlier <= later for earlier, later in pairwise(ordered))


# --- Test 5: the offset order statistics --------------------------------------------


def test_offset_order_statistics() -> None:
    """n = 4: the offsets are the residual order statistics at indexes 0, 0, 1, 2, 3.

    The parent-gain pairs (+1, +1, -1, -1) cancel in every column's target
    and cross-term, so every weight is exactly zero and μ̂ ≡ 0; the residuals
    are the gains themselves, and prior_points = 0 leaves them raw.
    """
    gains = [0.03, -0.03, 0.011, -0.011]  # telescopes to exactly 0.0 left-to-right
    parent_gains = [1.0, 1.0, -1.0, -1.0]
    history = [
        HistoryPoint(features(parent_gain=pg), gain)
        for pg, gain in zip(parent_gains, gains, strict=True)
    ]
    model = ConformalRidgeSelfModel(history, prior_points=0.0)
    assert model.predict(features(parent_gain=0.3, family="logic")) == {
        "q05": -0.03,  # index 0: ceil(0.05 * 4) - 1 = 0
        "q25": -0.03,  # index 0: ceil(0.25 * 4) - 1 = 0
        "q50": -0.011,  # index 1: ceil(0.50 * 4) - 1 = 1
        "q75": 0.011,  # index 2: ceil(0.75 * 4) - 1 = 2
        "q95": 0.03,  # index 3: ceil(0.95 * 4) - 1 = 3
    }


# --- Test 6: shrinkage --------------------------------------------------------------


def test_shrinkage_between_raw_and_prior() -> None:
    """prior_points = 0 gives the raw quantiles; a large one pulls strictly between."""
    history = [
        HistoryPoint(features(parent_gain=1.0), 0.03),
        HistoryPoint(features(parent_gain=1.0), -0.03),
        HistoryPoint(features(parent_gain=-1.0), 0.011),
        HistoryPoint(features(parent_gain=-1.0), -0.011),
    ]
    raw = ConformalRidgeSelfModel(history, prior_points=0.0)
    shrunk = ConformalRidgeSelfModel(history, prior_points=1e6)
    query = features(parent_gain=0.3)
    raw_values = raw.predict(query)
    shrunk_values = shrunk.predict(query)
    for key in QUANTILE_KEYS:
        low, high = sorted((raw_values[key], DEFAULT_PRIOR[key]))
        assert low < shrunk_values[key] < high


# --- Tests 7-8: determinism and finiteness ------------------------------------------


def test_determinism() -> None:
    """The same history gives the same dict, call after call, model after model."""
    rng = random.Random(9)
    history = [HistoryPoint(drawn_features(rng), rng.uniform(-0.05, 0.05)) for _ in range(12)]
    first = ConformalRidgeSelfModel(history)
    second = ConformalRidgeSelfModel(history)
    query = drawn_features(rng)
    assert first.predict(query) == first.predict(query)
    assert first.predict(query) == second.predict(query)


def test_non_finite_gain_raises() -> None:
    """A non-finite gain in the history is rejected at construction."""
    with pytest.raises(ValueError, match="finite"):
        ConformalRidgeSelfModel([HistoryPoint(features(), float("nan"))])
    with pytest.raises(ValueError, match="finite"):
        ConformalRidgeSelfModel([HistoryPoint(features(), float("inf"))])


# --- Tests 9-11: the prediction record ----------------------------------------------


def test_append_prediction_record(tmp_path: Path) -> None:
    """Kind, refers_to, payload fields, budget, the payload round-trip, run validity."""
    root = tmp_path / "ledgers"
    run = start_run(root, "selfmodel-record")
    proposal = proposal_in(run)
    model = ConformalRidgeSelfModel([])
    with pytest.raises(PayloadError, match="monotone"):
        append_prediction(
            run,
            predictor=model.identify(),
            proposal_record_id=proposal.record_id,
            quantiles={"q05": 0.5, "q25": 0.4, "q50": 0.3, "q75": 0.2, "q95": 0.1},
            wall_seconds=0.5,
        )
    record = append_prediction(
        run,
        predictor=model.identify(),
        proposal_record_id=proposal.record_id,
        quantiles=model.predict(features()),
        wall_seconds=0.5,
    )
    run.end("completed", {})

    assert record.kind == "prediction"
    assert record.refers_to == [proposal.record_id]
    assert record.budget == Budget(wall_seconds=0.5)
    payload = record.payload
    assert isinstance(payload, Prediction)
    assert payload.predictor == {"id": "conformal_ridge", "version": "1"}
    assert payload.metric == "dev_mase_gain"
    assert payload.horizon_generations == 1
    assert payload.quantiles == dict(DEFAULT_PRIOR)
    assert payload_from_dict("prediction", payload_to_dict(payload)) == payload
    assert validate_run(records_of(root, "selfmodel-record")) == []


def test_prediction_is_flushed_before_evaluation(tmp_path: Path) -> None:
    """After append_prediction, the last line of main.jsonl on disk is that prediction."""
    root = tmp_path / "ledgers"
    run = start_run(root, "selfmodel-flush")
    proposal = proposal_in(run)
    record = append_prediction(
        run,
        predictor={"id": "conformal_ridge", "version": "1"},
        proposal_record_id=proposal.record_id,
        quantiles=ConformalRidgeSelfModel([]).predict(features()),
        wall_seconds=0.25,
    )
    with (root / "selfmodel-flush" / "main.jsonl").open(encoding="utf-8") as stream:
        last_line = stream.read().splitlines()[-1]
    assert json.loads(last_line)["record_id"] == record.record_id
    assert json.loads(last_line)["kind"] == "prediction"
    run.append(
        "evaluation",
        dev_evaluation(),
        refers_to=[proposal.record_id],
        budget=Budget(evaluations=1),
    )
    run.end("completed", {})
    assert validate_run(records_of(root, "selfmodel-flush")) == []


def test_evaluation_without_prediction_is_reported_exactly(tmp_path: Path) -> None:
    """A run that evaluates with no prediction logged first: the one violation, named."""
    root = tmp_path / "ledgers"
    run = start_run(root, "selfmodel-ordering")
    proposal = proposal_in(run)
    run.append(
        "evaluation",
        dev_evaluation(),
        refers_to=[proposal.record_id],
        budget=Budget(evaluations=1),
    )
    run.end("completed", {})
    violations = validate_run(records_of(root, "selfmodel-ordering"))
    assert violations == [
        "evaluation at seq 2 (main) has no prediction logged before it "
        f"for proposal {proposal.record_id}"
    ]
