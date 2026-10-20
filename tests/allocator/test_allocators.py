"""Tests for the allocator slot's core five and the allocation record (TASK_017 PR C1).

These are tests 1-6 and 9-12 and 14 of the task's PR C list; tests 7, 8 and 13
(bandit_ucb and bo_gp) arrive with the rivals in PR C2, which also extends the
parametrisations to all seven ids. The rng-dependent tests pin pre-verified
seeds — for ``random``, seed 1 samples {0, 1} and seed 3 samples {1, 4} of
five candidates; for ``shuffled`` over the values [0.9, 0.5, 0.7], seed 1
permutes them to [0.5, 0.7, 0.9], moving the top score from the first
candidate to the third. The record test (12) runs against a real run directory
whose proposals come from the non-LLM proposer and carry a prediction each,
exactly the record shape (f)'s lineage loop will allocate over.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import pytest

from evolvets.allocator.allocators import ALLOCATOR_IDS, append_allocation, make_allocator
from evolvets.allocator.interfaces import AllocatorError, Candidate
from evolvets.improver.artifacts import REFERENCE_ARTIFACT_SOURCE
from evolvets.improver.interfaces import ProposalFeatures
from evolvets.improver.propose import propose_and_record
from evolvets.ledger.payloads import (
    Allocation,
    PayloadError,
    payload_from_dict,
    payload_to_dict,
)
from evolvets.ledger.schema import Budget, Record
from evolvets.ledger.store import LedgerReader
from evolvets.ledger.validate import validate_run
from evolvets.runner.runs import Run
from evolvets.selfmodel.interfaces import DEFAULT_PRIOR, HistoryPoint
from evolvets.selfmodel.quantiles import append_prediction

COMMIT = "a" * 40
LIMITS = Budget(wall_seconds=1e9, evaluations=1e9)
RUN_CONFIG: dict[str, Any] = {"allocator": "predictive", "k": 1}

#: The ids make_allocator constructs; PR C2 (feat/allocator-rivals) adds
#: bandit_ucb and bo_gp here and to the dispatch, extending tests 9, 10 and 14.
CONSTRUCTIBLE_IDS = ("evaluate_all", "fixed", "random", "predictive", "shuffled")

#: The ids that decide without predictions; PR C2 adds the two rivals.
NO_PREDICTION_IDS = ("evaluate_all", "fixed", "random")


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


def quantiles(q50: float, q75: float) -> dict[str, float]:
    """A monotone quantile dict pinning q50 and q75 (q75 must be >= q50)."""
    return {
        "q05": q50 - 0.02,
        "q25": q50 - 0.01,
        "q50": q50,
        "q75": q75,
        "q95": q75 + 0.01,
    }


def candidate(name: str, q50: float, q75: float | None = None) -> Candidate:
    """One candidate named ``name`` with monotone quantiles around ``q50``."""
    return Candidate(
        record_id=name,
        features=features(),
        quantiles=quantiles(q50, q75 if q75 is not None else q50 + 0.01),
    )


def history_of(*gains: float) -> list[HistoryPoint]:
    """A history of one evaluated proposal per gain, all params-family."""
    return [HistoryPoint(features(), gain) for gain in gains]


def records_of(root: Path, run_id: str) -> list[Record]:
    return list(LedgerReader(root, run_id).records())


# --- Test 1: registry and constructors ----------------------------------------------


def test_registry_and_constructor_validation() -> None:
    """The registry order is exact; every parameter is validated before dispatch."""
    assert ALLOCATOR_IDS == (
        "evaluate_all",
        "fixed",
        "random",
        "predictive",
        "shuffled",
        "bandit_ucb",
        "bo_gp",
    )
    with pytest.raises(AllocatorError, match="unknown allocator"):
        make_allocator("no_such_allocator")
    for rival in ("bandit_ucb", "bo_gp"):
        with pytest.raises(AllocatorError, match="feat/allocator-rivals"):
            make_allocator(rival)
    for allocator_id in CONSTRUCTIBLE_IDS:
        assert make_allocator(allocator_id).identify() == {"id": allocator_id, "version": "1"}
    with pytest.raises(ValueError, match="k must be >= 1"):
        make_allocator("fixed", k=0)
    with pytest.raises(ValueError, match="k must be an int"):
        make_allocator("fixed", k=True)
    with pytest.raises(ValueError, match="quantile must be one of"):
        make_allocator("predictive", quantile="q99")
    for name, value in (
        ("bandit_c", 0.0),
        ("bandit_c", -1.0),
        ("gp_gamma", 0.0),
        ("gp_noise", 0.0),
        ("gp_beta", -2.0),
        ("gp_prior_std", 0.0),
    ):
        with pytest.raises(ValueError, match=f"{name} must be > 0"):
            make_allocator("fixed", **{name: value})
    with pytest.raises(ValueError, match="bandit_c must be a number"):
        make_allocator("fixed", bandit_c=True)
    with pytest.raises(ValueError, match="history_cap must be >= 1"):
        make_allocator("fixed", history_cap=0)


# --- Test 2: evaluate_all -----------------------------------------------------------


def test_evaluate_all() -> None:
    """The first evaluations_left candidates, k ignored; 1.0/0.0 scores."""
    candidates = [candidate("c0", 0.2), candidate("c1", 0.9), candidate("c2", 0.5)]
    allocator = make_allocator("evaluate_all")  # k defaults to 1 and is ignored
    two = allocator.choose(candidates, [], 2, random.Random(0))
    assert two.chosen == ("c0", "c1")
    assert two.scores == {"c0": 1.0, "c1": 1.0, "c2": 0.0}
    for left in (3, 5):
        assert allocator.choose(candidates, [], left, random.Random(0)).chosen == (
            "c0",
            "c1",
            "c2",
        )


# --- Test 3: fixed ------------------------------------------------------------------


def test_fixed() -> None:
    """The first k candidates in order, whatever the quantiles say."""
    candidates = [candidate("c0", 0.1), candidate("c1", 0.9), candidate("c2", 0.5)]
    history = history_of(0.1, -0.1)
    first = make_allocator("fixed", k=1).choose(candidates, history, 5, random.Random(0))
    assert first.chosen == ("c0",)
    assert first.scores == {"c0": 1.0, "c1": 0.0, "c2": 0.0}
    assert make_allocator("fixed", k=2).choose(candidates, [], 5, random.Random(0)).chosen == (
        "c0",
        "c1",
    )
    assert make_allocator("fixed", k=5).choose(candidates, [], 5, random.Random(0)).chosen == (
        "c0",
        "c1",
        "c2",
    )
    # quantiles and history are invisible: the decision is the same without them
    blind = [
        Candidate(record_id=c.record_id, features=c.features, quantiles={}) for c in candidates
    ]
    assert make_allocator("fixed", k=2).choose(blind, [], 5, random.Random(0)) == make_allocator(
        "fixed", k=2
    ).choose(candidates, history, 5, random.Random(3))


# --- Test 4: random -----------------------------------------------------------------


def test_random() -> None:
    """A seeded sample: same seed, same set in candidate order; different seeds differ."""
    candidates = [candidate(f"c{i}", 0.1 * i) for i in range(5)]
    allocator = make_allocator("random", k=2)
    first = allocator.choose(candidates, [], 9, random.Random(1))
    assert first == allocator.choose(candidates, [], 9, random.Random(1))
    assert first.chosen == ("c0", "c1")  # seed 1 samples {0, 1}
    assert first.scores == {"c0": 1.0, "c1": 1.0, "c2": 0.0, "c3": 0.0, "c4": 0.0}
    assert allocator.choose(candidates, [], 9, random.Random(3)).chosen == ("c1", "c4")
    # the cap: k = 2 but only one evaluation left
    assert allocator.choose(candidates, [], 1, random.Random(1)).chosen == ("c1",)


# --- Test 5: predictive -------------------------------------------------------------


def test_predictive() -> None:
    """Ranking by the quantile, ties keeping candidate order; no quantiles, no run."""
    candidates = [
        candidate("c0", 0.2),
        candidate("c1", 0.9),
        candidate("c2", 0.9),
        candidate("c3", 0.5),
    ]
    allocator = make_allocator("predictive", k=1)
    top = allocator.choose(candidates, [], 9, random.Random(0))
    assert top.chosen == ("c1",)  # the tie with c2 keeps candidate order
    assert top.scores == {"c0": 0.2, "c1": 0.9, "c2": 0.9, "c3": 0.5}
    assert make_allocator("predictive", k=2).choose(candidates, [], 9, random.Random(0)).chosen == (
        "c1",
        "c2",
    )
    # quantile="q75" ranks by q75: c0's spread beats c2's level
    spread = [
        candidate("c0", 0.1, 0.8),
        candidate("c1", 0.4, 0.5),
        candidate("c2", 0.6, 0.7),
    ]
    assert allocator.choose(spread, [], 9, random.Random(0)).chosen == ("c2",)
    assert make_allocator("predictive", quantile="q75").choose(
        spread, [], 9, random.Random(0)
    ).chosen == ("c0",)
    blind = [Candidate(record_id="c0", features=features(), quantiles={})]
    with pytest.raises(AllocatorError, match="does not carry the quantiles"):
        allocator.choose(blind, [], 9, random.Random(0))


# --- Test 6: shuffled ---------------------------------------------------------------


def test_shuffled() -> None:
    """Predictive's rule over permuted values: same rule, destroyed information."""
    candidates = [candidate("c0", 0.9), candidate("c1", 0.5), candidate("c2", 0.7)]
    allocator = make_allocator("shuffled", k=1)
    decision = allocator.choose(candidates, [], 9, random.Random(1))
    # seed 1 permutes [0.9, 0.5, 0.7] to [0.5, 0.7, 0.9]: the top moves to c2
    assert decision.chosen == ("c2",)
    assert decision.scores == {"c0": 0.5, "c1": 0.7, "c2": 0.9}
    assert sorted(decision.scores.values()) == [0.5, 0.7, 0.9]
    assert decision != make_allocator("predictive", k=1).choose(candidates, [], 9, random.Random(1))
    assert decision == allocator.choose(candidates, [], 9, random.Random(1))
    with pytest.raises(AllocatorError, match="does not carry the quantiles"):
        allocator.choose(
            [Candidate(record_id="c0", features=features(), quantiles={})], [], 9, random.Random(1)
        )


# --- Tests 9-10: evaluations_left and empty candidates ------------------------------


@pytest.mark.parametrize("allocator_id", CONSTRUCTIBLE_IDS)
def test_evaluations_left_caps_and_zero(allocator_id: str) -> None:
    """Zero left chooses nothing but still scores; the cap is never exceeded."""
    candidates = [candidate(f"c{i}", 0.1 * ((i * 3) % 4)) for i in range(4)]
    allocator = make_allocator(allocator_id, k=2)
    none_left = allocator.choose(candidates, [], 0, random.Random(0))
    assert none_left.chosen == ()
    assert set(none_left.scores) == {"c0", "c1", "c2", "c3"}
    if allocator_id in NO_PREDICTION_IDS:
        assert set(none_left.scores.values()) == {0.0}
    else:
        assert sorted(none_left.scores.values()) == sorted(c.quantiles["q50"] for c in candidates)
    with pytest.raises(AllocatorError, match="evaluations_left"):
        allocator.choose(candidates, [], -1, random.Random(0))
    for left in (1, 3, 9):
        decision = allocator.choose(candidates, [], left, random.Random(left))
        expected = min(left, 4) if allocator_id == "evaluate_all" else min(2, left, 4)
        assert len(decision.chosen) == expected


@pytest.mark.parametrize("allocator_id", CONSTRUCTIBLE_IDS)
def test_empty_candidates_raise(allocator_id: str) -> None:
    """No allocator chooses over an empty candidate list."""
    with pytest.raises(AllocatorError, match="at least one candidate"):
        make_allocator(allocator_id).choose([], [], 3, random.Random(0))


# --- Test 11: determinism -----------------------------------------------------------


def test_determinism() -> None:
    """Equal arguments give equal decisions; the seeded ones need equal seeds."""
    candidates = [candidate(f"c{i}", 0.1 * i) for i in range(5)]
    history = history_of(0.01, -0.02, 0.03)
    predictive = make_allocator("predictive", k=2)
    assert predictive.choose(candidates, history, 9, random.Random(0)) == predictive.choose(
        candidates, history, 9, random.Random(5)
    )
    random_allocator = make_allocator("random", k=2)
    assert random_allocator.choose(candidates, [], 9, random.Random(4)) == (
        random_allocator.choose(candidates, [], 9, random.Random(4))
    )
    assert random_allocator.choose(candidates, [], 9, random.Random(1)).chosen != (
        random_allocator.choose(candidates, [], 9, random.Random(3)).chosen
    )
    shuffled = make_allocator("shuffled", k=1)
    assert shuffled.choose(candidates, [], 9, random.Random(1)) == shuffled.choose(
        candidates, [], 9, random.Random(1)
    )
    assert shuffled.choose(candidates, [], 9, random.Random(1)) != shuffled.choose(
        candidates, [], 9, random.Random(5)
    )


# --- Test 12: the allocation record -------------------------------------------------


def test_append_allocation(tmp_path: Path) -> None:
    """The payload round-trips, refers_to is the candidates, validate_run is clean."""
    root = tmp_path / "ledgers"
    run = Run.start(
        root,
        "allocator-record",
        condition="allocator",
        seed=13,
        config=RUN_CONFIG,
        budget_limits=LIMITS,
        git_commit=COMMIT,
        requires_predictions=True,
    )
    proposal_ids = []
    for _ in range(2):
        proposal = propose_and_record(
            run,
            proposer_id="random_param",
            parent_source=REFERENCE_ARTIFACT_SOURCE,
            parent_record_id=None,
            generation=0,
            parent_gain=0.0,
            history_records=[],
            rng=random.Random(0),
        )
        proposal_ids.append(proposal.record_id)
        append_prediction(
            run,
            predictor={"id": "conformal_ridge", "version": "1"},
            proposal_record_id=proposal.record_id,
            quantiles=DEFAULT_PRIOR,
            wall_seconds=0.1,
        )
    allocator = make_allocator("predictive", k=1)
    candidates = [
        Candidate(record_id=record_id, features=features(), quantiles=DEFAULT_PRIOR)
        for record_id in proposal_ids
    ]
    decision = allocator.choose(candidates, [], 9, random.Random(0))
    record = append_allocation(
        run,
        allocator=allocator.identify(),
        candidate_ids=proposal_ids,
        chosen_ids=list(decision.chosen),
        scores=dict(decision.scores),
        wall_seconds=0.125,
    )
    run.end("completed", {})

    assert record.kind == "allocation"
    assert record.refers_to == proposal_ids
    assert record.budget == Budget(wall_seconds=0.125)
    payload = record.payload
    assert isinstance(payload, Allocation)
    assert payload.allocator == {"id": "predictive", "version": "1"}
    assert payload.candidates == proposal_ids
    assert payload.chosen == list(decision.chosen)
    assert payload.scores == dict(decision.scores)
    assert payload_from_dict("allocation", payload_to_dict(payload)) == payload
    with pytest.raises(PayloadError, match="scores keys must be candidate"):
        Allocation(
            allocator={"id": "predictive", "version": "1"},
            candidates=proposal_ids,
            chosen=[],
            scores={"not-a-candidate": 1.0},
        )
    with pytest.raises(PayloadError, match="chosen must not contain duplicates"):
        Allocation(
            allocator={"id": "predictive", "version": "1"},
            candidates=proposal_ids,
            chosen=[proposal_ids[0], proposal_ids[0]],
            scores={},
        )
    assert validate_run(records_of(root, "allocator-record")) == []


# --- Test 14: no predictions needed -------------------------------------------------


@pytest.mark.parametrize("allocator_id", NO_PREDICTION_IDS)
def test_no_predictions_needed(allocator_id: str) -> None:
    """The non-predictive core allocators decide over candidates with empty quantiles."""
    candidates = [Candidate(record_id=f"c{i}", features=features(), quantiles={}) for i in range(3)]
    decision = make_allocator(allocator_id, k=2).choose(candidates, [], 9, random.Random(0))
    # evaluate_all ignores k and takes everything the budget allows
    expected = 3 if allocator_id == "evaluate_all" else 2
    assert len(decision.chosen) == expected
