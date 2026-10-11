"""Kind-specific validation rules for the eight later payload kinds."""

import pytest

from evolvets.ledger.payloads import (
    Acceptance,
    Allocation,
    Baseline,
    Commitment,
    Correction,
    Evaluation,
    PayloadError,
    Prediction,
    Proposal,
)
from evolvets.ledger.schema import ZERO_HASH, RecordError, make_record

SEED_KWARGS = dict(
    seq=0,
    prev_hash=ZERO_HASH,
    run_id="run-1",
    created_at="2026-10-11T10:00:00.000000Z",
)


def _record(stream, kind, payload):
    return make_record(stream=stream, kind=kind, payload=payload, **SEED_KWARGS)


def test_non_monotone_quantiles_rejected():
    with pytest.raises(PayloadError, match="monotone"):
        Prediction(
            predictor={"id": "selfmodel", "version": "1"},
            metric="mase",
            horizon_generations=1,
            quantiles={"q05": 0.5, "q25": 0.2, "q50": 0.3, "q75": 0.4, "q95": 0.6},
        )


def test_quantiles_exact_key_set(prediction):
    data = dict(prediction.quantiles)
    del data["q50"]
    with pytest.raises(PayloadError, match="quantiles"):
        Prediction(
            predictor=prediction.predictor,
            metric="mase",
            horizon_generations=1,
            quantiles=data,
        )


def test_prediction_horizon_must_be_positive():
    with pytest.raises(PayloadError, match="horizon_generations"):
        Prediction(
            predictor={"id": "s", "version": "1"},
            metric="mase",
            horizon_generations=0,
            quantiles={"q05": 0.1, "q25": 0.2, "q50": 0.3, "q75": 0.4, "q95": 0.6},
        )


def test_proposal_bad_target_rejected(proposal):
    with pytest.raises(PayloadError, match="target"):
        Proposal(**{**vars(proposal), "target": "arena"})


def test_proposal_planted_defect_key_set(proposal):
    with pytest.raises(PayloadError, match="planted_defect"):
        Proposal(**{**vars(proposal), "planted_defect": {"defect_id": "d1"}})


def test_proposal_null_prompt_response_allowed(proposal):
    assert proposal.prompt_blob is None
    assert proposal.response_blob is None


def test_proposal_bad_temperature_rejected(proposal):
    bad = dict(proposal.proposer)
    bad["temperature"] = float("nan")
    with pytest.raises(PayloadError, match="temperature"):
        Proposal(**{**vars(proposal), "proposer": bad})


def test_allocation_chosen_must_be_subset(allocation):
    other = str(allocation.candidates[0])
    with pytest.raises(PayloadError, match="subset"):
        Allocation(
            allocator=allocation.allocator,
            candidates=[other],
            chosen=[other, str(allocation.candidates[1])],
            scores={},
        )


def test_allocation_candidates_must_be_uuids(allocation):
    with pytest.raises(PayloadError, match="UUID"):
        Allocation(
            allocator=allocation.allocator,
            candidates=["not-a-uuid"],
            chosen=[],
            scores={},
        )


def test_allocation_empty_candidates_rejected(allocation):
    with pytest.raises(PayloadError, match="candidates"):
        Allocation(allocator=allocation.allocator, candidates=[], chosen=[], scores={})


def test_evaluation_bad_status_rejected(evaluation):
    with pytest.raises(PayloadError, match="status"):
        Evaluation(**{**vars(evaluation), "status": "crashed"})


def test_evaluation_bad_pool_rejected(evaluation):
    with pytest.raises(PayloadError, match="pool"):
        Evaluation(**{**vars(evaluation), "pool": "probe"})


def test_evaluation_metrics_must_be_floats(evaluation):
    with pytest.raises(PayloadError, match="metrics"):
        Evaluation(**{**vars(evaluation), "metrics": {"mase": "low"}})


def test_sealed_pool_in_main_stream_rejected(evaluation, baseline):
    for kind, payload in (("evaluation", evaluation), ("baseline", baseline)):
        sealed = type(payload)(**{**vars(payload), "pool": "sealed"})
        with pytest.raises(RecordError, match="sealed stream"):
            _record("main", kind, sealed)


def test_sealed_pool_in_sealed_stream_allowed(evaluation):
    sealed = Evaluation(**{**vars(evaluation), "pool": "sealed"})
    record = _record("sealed", "evaluation", sealed)
    assert record.payload.pool == "sealed"


def test_baseline_bad_pool_rejected(baseline):
    with pytest.raises(PayloadError, match="pool"):
        Baseline(**{**vars(baseline), "pool": "test"})


def test_acceptance_requires_bool():
    with pytest.raises(PayloadError, match="accepted"):
        Acceptance(accepted="yes", rule="r", new_artifact=None)


def test_commitment_requires_blob_ref():
    with pytest.raises(PayloadError, match="content_hash"):
        Commitment(subject="s", content_hash="nope", external_timestamp=None)


def test_correction_allows_null_replacement():
    correction = Correction(reason="r", replacement=None)
    assert correction.replacement is None
