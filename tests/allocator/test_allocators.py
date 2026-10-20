"""Tests for the allocator slot and its seven implementations (TASK_017 PR C).

Tests 7, 8 and 13 pin exact floats: every expected score is recomputed in
the test with the same loops and the same operation order as the
implementation (the left-to-right mean, ``2 * math.log(N + 1) / n_arm``, the
kernel-and-solve scalar chain), because decimal literals would drift. Tests
4, 6 and 11 pin rng outcomes to pre-verified seeds. The record test (12)
runs against a real run directory whose proposals come from the non-LLM
proposer and whose predictions precede the allocation, so ``validate_run``
sees the record exactly the way (f)'s lineage loop will write it.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from evolvets.allocator.allocators import (
    ALLOCATOR_IDS,
    ARM_UNSEEN_SCORE,
    append_allocation,
    make_allocator,
)
from evolvets.allocator.interfaces import AllocatorError, Candidate
from evolvets.improver.artifacts import REFERENCE_ARTIFACT_SOURCE
from evolvets.improver.interfaces import ProposalFeatures, feature_vector
from evolvets.improver.propose import propose_and_record
from evolvets.ledger.payloads import (
    Allocation,
    PayloadError,
    payload_from_dict,
    payload_to_dict,
)
from evolvets.ledger.schema import Budget
from evolvets.ledger.store import LedgerReader
from evolvets.ledger.validate import validate_run
from evolvets.runner.runs import Run
from evolvets.selfmodel.interfaces import HistoryPoint
from evolvets.selfmodel.quantiles import append_prediction

COMMIT = "a" * 40
LIMITS = Budget(wall_seconds=1e9, evaluations=1e9)
RUN_CONFIG: dict[str, Any] = {"allocator": "predictive", "k": 2}

#: The quantiles every prediction in the record test carries (the prior's shape).
PREDICTION_QUANTILES: dict[str, float] = {
    "q05": -0.02,
    "q25": -0.005,
    "q50": 0.002,
    "q75": 0.01,
    "q95": 0.04,
}


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


def quantiles_around(q50: float, spread: float = 0.01) -> dict[str, float]:
    """Five monotone quantiles centred on ``q50``."""
    return {
        "q05": q50 - 2 * spread,
        "q25": q50 - spread,
        "q50": q50,
        "q75": q50 + spread,
        "q95": q50 + 2 * spread,
    }


def candidate(record_id: str, family: str = "params", **overrides: Any) -> Candidate:
    """One candidate with default features and a centred quantile block."""
    return Candidate(record_id, features(family=family, **overrides), quantiles_around(0.0))


def candidates_with(values: Sequence[float]) -> list[Candidate]:
    """Candidates c0..cn whose q50 is ``values[i]``, in candidate order."""
    return [
        Candidate(f"c{i}", features(), quantiles_around(value)) for i, value in enumerate(values)
    ]


def squared_distance(x: Sequence[float], y: Sequence[float]) -> float:
    """The squared distance exactly as allocators.py accumulates it."""
    total = 0.0
    for a, b in zip(x, y, strict=True):
        diff = a - b
        total = total + diff * diff
    return total


# --- Test 1: the registry and the constructors ---------------------------------------


def test_registry_and_constructors() -> None:
    """ALLOCATOR_IDS is exact; identify() is the id/version pair; bad knobs raise."""
    assert ALLOCATOR_IDS == (
        "evaluate_all",
        "fixed",
        "random",
        "predictive",
        "shuffled",
        "bandit_ucb",
        "bo_gp",
    )
    for allocator_id in ALLOCATOR_IDS:
        assert make_allocator(allocator_id).identify() == {"id": allocator_id, "version": "1"}
    with pytest.raises(AllocatorError, match="unknown allocator"):
        make_allocator("greedy")
    with pytest.raises(ValueError, match="k must be an int >= 1"):
        make_allocator("fixed", k=0)
    with pytest.raises(ValueError, match="k must be an int >= 1"):
        make_allocator("fixed", k=True)
    with pytest.raises(ValueError, match="quantile must be one of"):
        make_allocator("predictive", quantile="q99")
    for name in ("bandit_c", "gp_gamma", "gp_noise", "gp_beta", "gp_prior_std"):
        with pytest.raises(ValueError, match=f"{name} must be"):
            make_allocator("bo_gp", **{name: 0.0})
    with pytest.raises(ValueError, match="history_cap must be an int >= 1"):
        make_allocator("bandit_ucb", history_cap=0)


# --- Test 2: evaluate_all ------------------------------------------------------------


def test_evaluate_all_takes_the_first_evaluations_left() -> None:
    """No selection at all: the first evaluations_left candidates, k ignored."""
    allocator = make_allocator("evaluate_all")  # k defaults to 1 and is ignored
    cands = candidates_with([0.1, 0.2, 0.3])
    first_two = allocator.choose(cands, [], 2, random.Random(0))
    assert first_two.chosen == ("c0", "c1")
    assert first_two.scores == {"c0": 1.0, "c1": 1.0, "c2": 0.0}
    assert allocator.choose(cands, [], 3, random.Random(0)).chosen == ("c0", "c1", "c2")
    assert allocator.choose(cands, [], 10, random.Random(0)).chosen == ("c0", "c1", "c2")


# --- Test 3: fixed -------------------------------------------------------------------


def test_fixed_takes_the_first_k_regardless_of_everything() -> None:
    """The first k candidates; the decision ignores quantiles and history entirely."""
    cands = candidates_with([0.1, 0.2, 0.3])
    blind = [Candidate(f"c{i}", features(), {}) for i in range(3)]
    history = [HistoryPoint(features(family="logic"), 0.5)]
    for k, expected in ((1, ("c0",)), (2, ("c0", "c1")), (5, ("c0", "c1", "c2"))):
        allocator = make_allocator("fixed", k=k)
        decision = allocator.choose(cands, [], 10, random.Random(0))
        assert decision.chosen == expected
        assert decision == allocator.choose(blind, history, 10, random.Random(9))
    scores = make_allocator("fixed", k=1).choose(cands, [], 10, random.Random(0)).scores
    assert scores == {"c0": 1.0, "c1": 0.0, "c2": 0.0}


# --- Test 4: random ------------------------------------------------------------------


def test_random_is_seeded_and_lists_in_candidate_order() -> None:
    """Same seed, same set; the ids appear in candidate order; the cap holds."""
    cands = candidates_with([0.1, 0.2, 0.3, 0.4, 0.5])
    allocator = make_allocator("random", k=3)
    # pre-verified: Random(0).sample(range(5), 3) is {0, 3, 4}; Random(1) is {0, 1, 4}
    assert allocator.choose(cands, [], 10, random.Random(0)).chosen == ("c0", "c3", "c4")
    assert allocator.choose(cands, [], 10, random.Random(0)) == allocator.choose(
        cands, [], 10, random.Random(0)
    )
    assert allocator.choose(cands, [], 10, random.Random(1)).chosen == ("c0", "c1", "c4")
    assert len(allocator.choose(cands, [], 2, random.Random(0)).chosen) == 2


# --- Test 5: predictive --------------------------------------------------------------


def test_predictive_ranks_by_the_quantile() -> None:
    """Ranking by the configured quantile, ties keeping candidate order."""
    cands = candidates_with([0.3, 0.1, 0.2])
    allocator = make_allocator("predictive", k=2)
    decision = allocator.choose(cands, [], 10, random.Random(0))
    assert decision.chosen == ("c0", "c2")
    assert decision.scores == {"c0": 0.3, "c1": 0.1, "c2": 0.2}
    assert allocator.choose(candidates_with([0.2, 0.2, 0.1]), [], 10, random.Random(0)).chosen == (
        "c0",
        "c1",
    )
    wide = [
        Candidate("c0", features(), {"q05": 0.0, "q25": 0.1, "q50": 0.2, "q75": 0.21, "q95": 0.3}),
        Candidate(
            "c1", features(), {"q05": -0.1, "q25": 0.05, "q50": 0.19, "q75": 0.24, "q95": 0.4}
        ),
    ]
    assert make_allocator("predictive", k=1).choose(wide, [], 10, random.Random(0)).chosen == (
        "c0",
    )
    assert make_allocator("predictive", k=1, quantile="q75").choose(
        wide, [], 10, random.Random(0)
    ).chosen == ("c1",)
    with pytest.raises(AllocatorError, match="must carry all of"):
        allocator.choose([Candidate("c0", features(), {})], [], 10, random.Random(0))


# --- Test 6: shuffled ----------------------------------------------------------------


def test_shuffled_permutes_the_scores_across_candidates() -> None:
    """Same rule as predictive over permuted scores; the permuted values are recorded."""
    cands = candidates_with([0.9, 0.1, 0.5])
    allocator = make_allocator("shuffled", k=1)
    # pre-verified: Random(4).shuffle over [0.9, 0.1, 0.5] gives [0.5, 0.1, 0.9]
    decision = allocator.choose(cands, [], 10, random.Random(4))
    assert decision.scores == {"c0": 0.5, "c1": 0.1, "c2": 0.9}
    assert decision.chosen == ("c2",)
    assert decision == allocator.choose(cands, [], 10, random.Random(4))
    assert (
        decision.chosen
        != make_allocator("predictive", k=1).choose(cands, [], 10, random.Random(0)).chosen
    )


# --- Test 7: bandit_ucb --------------------------------------------------------------


def test_bandit_ucb_explores_then_takes_the_higher_ucb_arm() -> None:
    """Unseen arms first (params breaks ties); then the hand-formula UCB winner."""
    cands = [
        candidate("c0", family="logic"),
        candidate("c1", family="params"),
        candidate("c2", family="params"),
    ]
    allocator = make_allocator("bandit_ucb", k=2)
    unseen = allocator.choose(cands, [], 10, random.Random(0))
    assert unseen.chosen == ("c1", "c2")  # both arms unseen: FAMILY_ORDER puts params first
    assert unseen.scores == {
        "c0": ARM_UNSEEN_SCORE,
        "c1": ARM_UNSEEN_SCORE,
        "c2": ARM_UNSEEN_SCORE,
    }
    strong_params = [HistoryPoint(features(family="params"), 0.8) for _ in range(6)]
    explored = allocator.choose(cands, strong_params, 10, random.Random(0))
    assert explored.chosen == ("c0", "c1")  # the unseen logic arm still goes first
    assert explored.scores["c0"] == ARM_UNSEEN_SCORE
    # both arms have history: every score is the hand formula, in the same
    # operation order as the implementation (left-to-right totals, then the mean)
    params_total = 0.0
    for _ in range(6):
        params_total = params_total + 0.8
    logic_total = 0.0
    for _ in range(3):
        logic_total = logic_total + -0.1
    n_params, n_logic = 6, 3
    n_total = n_params + n_logic
    params_score = params_total / n_params + 1.0 * math.sqrt(2 * math.log(n_total + 1) / n_params)
    logic_score = logic_total / n_logic + 1.0 * math.sqrt(2 * math.log(n_total + 1) / n_logic)
    history = strong_params + [HistoryPoint(features(family="logic"), -0.1) for _ in range(3)]
    both = allocator.choose(cands, history, 10, random.Random(0))
    assert both.scores == {"c0": logic_score, "c1": params_score, "c2": params_score}
    winner = make_allocator("bandit_ucb", k=1).choose(cands, history, 10, random.Random(0))
    assert params_score > logic_score  # the hand formula puts params on top here
    assert winner.chosen == ("c1",)
    reversed_history = [HistoryPoint(features(family="logic"), 0.5) for _ in range(6)] + [
        HistoryPoint(features(family="params"), -0.1) for _ in range(3)
    ]
    flipped = make_allocator("bandit_ucb", k=1).choose(
        cands, reversed_history, 10, random.Random(0)
    )
    assert flipped.chosen == ("c0",)  # the stronger logic arm wins instead


# --- Test 8: bo_gp -------------------------------------------------------------------


def test_bo_gp_without_history_scores_the_prior() -> None:
    """No history: mean = 0.0, std = gp_prior_std, so every score is gp_beta * gp_prior_std."""
    cands = candidates_with([0.1, 0.2, 0.3])
    allocator = make_allocator("bo_gp", k=2, gp_beta=2.0, gp_prior_std=1.0)
    decision = allocator.choose(cands, [], 10, random.Random(0))
    assert decision.scores == {"c0": 2.0 * 1.0, "c1": 2.0 * 1.0, "c2": 2.0 * 1.0}
    assert decision.chosen == ("c0", "c1")  # equal scores: candidate order breaks the tie


def test_bo_gp_with_one_history_point_matches_the_hand_loops() -> None:
    """The acquisitions equal an in-test recomputation of the kernel-and-solve loops."""
    gamma, noise, beta = 1.0, 0.01, 2.0
    history = [HistoryPoint(features(family="params", generation=10, parent_gain=0.25), 0.4)]
    cands = [
        candidate("c0", family="params"),
        candidate("c1", family="logic", max_rel_const_change=0.5),
    ]
    allocator = make_allocator("bo_gp", k=1, gp_gamma=gamma, gp_noise=noise, gp_beta=beta)
    decision = allocator.choose(cands, history, 10, random.Random(0))
    x_history = feature_vector(history[0].features)
    kernel = math.exp(-gamma * squared_distance(x_history, x_history))
    matrix = [[kernel]]
    matrix[0][0] = matrix[0][0] + noise
    alpha = [history[0].gain / matrix[0][0]]
    expected: dict[str, float] = {}
    for cand in cands:
        x = feature_vector(cand.features)
        k_star = math.exp(-gamma * squared_distance(x, x_history))
        v = k_star / matrix[0][0]
        mean = 0.0 + k_star * alpha[0]
        variance = 1.0 - (0.0 + k_star * v)
        std = math.sqrt(variance) if variance > 0.0 else 0.0
        expected[cand.record_id] = mean + beta * std
    assert decision.scores == expected
    best = max(expected, key=lambda name: expected[name])
    assert decision.chosen == (best,)


def test_bo_gp_keeps_a_real_std_at_a_near_duplicate() -> None:
    """A candidate on top of the history point: variance is positive, no NaN, no negatives."""
    history = [HistoryPoint(features(family="params"), 0.3)]
    twin = candidate("twin", family="params")  # the same feature vector as the history point
    allocator = make_allocator("bo_gp", k=1, gp_noise=0.01)
    decision = allocator.choose([twin], history, 10, random.Random(0))
    (score,) = decision.scores.values()
    assert math.isfinite(score)
    # variance = 1 - 1 / (1 + noise) > 0; the floor is what keeps the sqrt real
    # when rounding would drive it to <= 0, and the recomputation applies it too
    variance = 1.0 - (0.0 + 1.0 / (1.0 + 0.01))
    assert variance > 0.0
    assert score == 0.3 / (1.0 + 0.01) + 2.0 * math.sqrt(variance)


# --- Test 9: evaluations_left (parametrised across all seven) -------------------------


@pytest.mark.parametrize("allocator_id", ALLOCATOR_IDS)
def test_evaluations_left_caps_and_guards(allocator_id: str) -> None:
    """Zero empties the choice, negative raises, and the cap is never exceeded."""
    cands = candidates_with([0.5 - 0.1 * i for i in range(7)])
    # evaluate_all caps at len(candidates) alone (it ignores k), so it runs with k >= el
    k = 10 if allocator_id == "evaluate_all" else 4
    allocator = make_allocator(allocator_id, k=k)
    empty = allocator.choose(cands, [], 0, random.Random(0))
    assert empty.chosen == ()
    assert set(empty.scores) == {f"c{i}" for i in range(7)}
    with pytest.raises(AllocatorError, match="evaluations_left"):
        allocator.choose(cands, [], -1, random.Random(0))
    assert len(allocator.choose(cands, [], 3, random.Random(0)).chosen) <= min(k, 3, 7)
    assert len(allocator.choose(cands[:5], [], 10, random.Random(0)).chosen) <= min(k, 10, 5)


# --- Test 10: empty candidates (parametrised across all seven) ------------------------


@pytest.mark.parametrize("allocator_id", ALLOCATOR_IDS)
def test_empty_candidates_raise(allocator_id: str) -> None:
    """Choosing from nothing is an allocator error, not an empty decision."""
    allocator = make_allocator(allocator_id)
    with pytest.raises(AllocatorError, match="at least one candidate"):
        allocator.choose([], [], 5, random.Random(0))


# --- Test 11: determinism ------------------------------------------------------------


def test_determinism() -> None:
    """The rankers repeat whatever the rng; the stochastic ones follow their seed."""
    cands = candidates_with([0.1, 0.3, 0.2])
    history = [
        HistoryPoint(features(family="params"), 0.1),
        HistoryPoint(features(family="logic"), -0.2),
    ]
    for allocator_id in ("predictive", "bandit_ucb", "bo_gp"):
        allocator = make_allocator(allocator_id, k=2)
        assert allocator.choose(cands, history, 10, random.Random(0)) == allocator.choose(
            cands, history, 10, random.Random(5)
        )
    for allocator_id in ("random", "shuffled"):
        allocator = make_allocator(allocator_id, k=2)
        same = allocator.choose(cands, [], 10, random.Random(3))
        assert same == allocator.choose(cands, [], 10, random.Random(3))
        assert same != allocator.choose(cands, [], 10, random.Random(4))


# --- Test 12: the allocation record --------------------------------------------------


def test_append_allocation_record(tmp_path: Path) -> None:
    """Payload round-trip, refers_to, rejections, budget and run validity."""
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
    proposals = [
        propose_and_record(
            run,
            proposer_id="random_param",
            parent_source=REFERENCE_ARTIFACT_SOURCE,
            parent_record_id=None,
            generation=0,
            parent_gain=0.0,
            history_records=[],
            rng=random.Random(seed),
        )
        for seed in range(3)
    ]
    for proposal in proposals:
        append_prediction(
            run,
            predictor={"id": "conformal_ridge", "version": "1"},
            proposal_record_id=proposal.record_id,
            quantiles=PREDICTION_QUANTILES,
            wall_seconds=0.1,
        )
    ids = [proposal.record_id for proposal in proposals]
    allocator = make_allocator("predictive", k=2)
    ranked = [
        Candidate(proposal.record_id, features(), quantiles_around(0.002 * (index + 1)))
        for index, proposal in enumerate(proposals)
    ]
    decision = allocator.choose(ranked, [], 10, random.Random(0))
    with pytest.raises(PayloadError, match="chosen must not contain duplicates"):
        append_allocation(
            run,
            allocator=allocator.identify(),
            candidate_ids=ids,
            chosen_ids=[ids[0], ids[0]],
            scores={},
            wall_seconds=0.0,
        )
    with pytest.raises(PayloadError, match="scores keys must be candidate record ids"):
        append_allocation(
            run,
            allocator=allocator.identify(),
            candidate_ids=ids,
            chosen_ids=[ids[0]],
            scores={"not-a-candidate": 1.0},
            wall_seconds=0.0,
        )
    record = append_allocation(
        run,
        allocator=allocator.identify(),
        candidate_ids=ids,
        chosen_ids=list(decision.chosen),
        scores=dict(decision.scores),
        wall_seconds=0.02,
    )
    run.end("completed", {})

    assert record.kind == "allocation"
    assert record.refers_to == ids
    assert record.budget == Budget(wall_seconds=0.02)
    payload = record.payload
    assert isinstance(payload, Allocation)
    assert payload.allocator == {"id": "predictive", "version": "1"}
    assert payload.candidates == ids
    assert payload.chosen == list(decision.chosen)
    assert payload.scores == dict(decision.scores)
    assert payload_from_dict("allocation", payload_to_dict(payload)) == payload
    assert validate_run(list(LedgerReader(root, "allocator-record").records())) == []


# --- Test 13: the history cap --------------------------------------------------------


def test_bandit_ucb_fits_at_most_the_last_history_cap_points() -> None:
    """A 150-point history decides as its last 100 points alone do."""
    early = [HistoryPoint(features(family="logic"), 0.8) for _ in range(50)]
    late = [HistoryPoint(features(family="params"), 0.015) for _ in range(70)] + [
        HistoryPoint(features(family="logic"), -0.5) for _ in range(30)
    ]
    cands = [
        candidate("c0", family="params"),
        candidate("c1", family="logic"),
        candidate("c2", family="params"),
    ]
    allocator = make_allocator("bandit_ucb", k=2)
    capped = allocator.choose(cands, early + late, 10, random.Random(0))
    assert capped == allocator.choose(cands, late, 10, random.Random(0))
    assert capped.chosen == ("c0", "c2")  # params has the higher UCB on the last 100
    uncapped = make_allocator("bandit_ucb", k=2, history_cap=150)
    assert uncapped.choose(cands, early + late, 10, random.Random(0)) != capped


def test_bo_gp_fits_at_most_the_last_history_cap_points() -> None:
    """A 150-point history decides as its last 100 points alone do."""
    early = [HistoryPoint(features(family="params", generation=i), 0.9) for i in range(25)] + [
        HistoryPoint(features(family="logic", generation=i), -0.3) for i in range(25)
    ]
    late = [HistoryPoint(features(family="params", generation=i), 0.05) for i in range(50)] + [
        HistoryPoint(features(family="logic", generation=i), 0.5) for i in range(50)
    ]
    cands = [candidate("c0", family="params"), candidate("c1", family="logic")]
    allocator = make_allocator("bo_gp", k=1)
    capped = allocator.choose(cands, early + late, 10, random.Random(0))
    assert capped == allocator.choose(cands, late, 10, random.Random(0))
    uncapped = make_allocator("bo_gp", k=1, history_cap=150)
    assert uncapped.choose(cands, early + late, 10, random.Random(0)) != capped


# --- Test 14: the allocators that never see predictions ------------------------------


@pytest.mark.parametrize("allocator_id", ["evaluate_all", "fixed", "random", "bandit_ucb", "bo_gp"])
def test_allocators_decide_without_predictions(allocator_id: str) -> None:
    """Every non-predictive allocator decides over candidates whose quantiles are {}."""
    cands = [Candidate(f"c{i}", features(family=("params", "logic")[i % 2]), {}) for i in range(4)]
    allocator = make_allocator(allocator_id, k=2)
    decision = allocator.choose(cands, [], 10, random.Random(0))
    assert set(decision.scores) == {f"c{i}" for i in range(4)}
    # evaluate_all caps at len(candidates) alone; the others respect k
    expected_size = 4 if allocator_id == "evaluate_all" else 2
    assert len(decision.chosen) == expected_size
    assert all(math.isfinite(score) for score in decision.scores.values())
