"""Kind-specific validation rules for the eight later payload kinds."""

import uuid

import pytest

from evolvets.ledger.payloads import (
    EVALUATION_STATUSES,
    Acceptance,
    Allocation,
    Baseline,
    Commitment,
    Correction,
    Evaluation,
    PayloadError,
    Prediction,
    Proposal,
    payload_from_dict,
    payload_to_dict,
)
from evolvets.ledger.schema import ZERO_HASH, RecordError, make_record

SEED_KWARGS = dict(
    seq=0,
    prev_hash=ZERO_HASH,
    run_id="run-1",
    created_at="2026-10-11T10:00:00.000000Z",
)


def _record(stream, kind, payload, refers_to=()):
    return make_record(
        stream=stream, kind=kind, payload=payload, refers_to=refers_to, **SEED_KWARGS
    )


@pytest.mark.parametrize(
    ("kind", "count"), [("prediction", 1), ("evaluation", 1), ("acceptance", 2), ("correction", 1)]
)
def test_linked_kinds_accept_exact_refers_to(kind, count, request):
    payload = request.getfixturevalue(kind)
    refs = [str(uuid.uuid4()) for _ in range(count)]
    record = _record("main", kind, payload, refers_to=refs)
    assert record.refers_to == refs


@pytest.mark.parametrize(
    ("kind", "count"), [("prediction", 1), ("evaluation", 1), ("acceptance", 2), ("correction", 1)]
)
def test_linked_kinds_reject_wrong_refers_to_length(kind, count, request):
    payload = request.getfixturevalue(kind)
    wrong = [str(uuid.uuid4()) for _ in range(count + 1)]
    with pytest.raises(RecordError, match="refer to exactly"):
        _record("main", kind, payload, refers_to=wrong)
    with pytest.raises(RecordError, match="refer to exactly"):
        _record("main", kind, payload, refers_to=[])


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
    ref = [str(uuid.uuid4())]
    for kind, payload in (("evaluation", evaluation), ("baseline", baseline)):
        sealed = type(payload)(**{**vars(payload), "pool": "sealed"})
        with pytest.raises(RecordError, match="sealed stream"):
            _record("main", kind, sealed, refers_to=ref)


def test_sealed_pool_in_sealed_stream_allowed(evaluation):
    sealed = Evaluation(**{**vars(evaluation), "pool": "sealed"})
    record = _record("sealed", "evaluation", sealed, refers_to=[str(uuid.uuid4())])
    assert record.payload.pool == "sealed"


def test_baseline_bad_pool_rejected(baseline):
    with pytest.raises(PayloadError, match="pool"):
        Baseline(**{**vars(baseline), "pool": "test"})


def test_baseline_ok_round_trips_with_and_without_blob(baseline, blob_ref):
    with_blob = Baseline(**{**vars(baseline), "per_task_blob": blob_ref("per-task")})
    assert payload_from_dict("baseline", payload_to_dict(with_blob)) == with_blob
    assert payload_from_dict("baseline", payload_to_dict(baseline)) == baseline


@pytest.mark.parametrize("status", sorted(EVALUATION_STATUSES))
def test_baseline_accepts_every_status(baseline, status):
    payload = Baseline(
        **{
            **vars(baseline),
            "status": status,
            "metrics": {} if status != "ok" else {"mase": 0.81},
            "error": None if status == "ok" else "updater raised ValueError: boom",
        }
    )
    assert payload.status == status


@pytest.mark.parametrize("bad", ["done", "", None, "OK"])
def test_baseline_bad_status_rejected(baseline, bad):
    with pytest.raises(PayloadError, match="status"):
        Baseline(**{**vars(baseline), "status": bad})


def test_baseline_ok_with_error_rejected(baseline):
    with pytest.raises(PayloadError, match="error must be None"):
        Baseline(**{**vars(baseline), "error": "should not be here"})


@pytest.mark.parametrize("bad", [None, "", 3])
def test_baseline_non_ok_without_error_rejected(baseline, bad):
    with pytest.raises(PayloadError, match="non-empty string"):
        Baseline(**{**vars(baseline), "status": "failed", "metrics": {}, "error": bad})


def test_baseline_non_ok_with_error_accepted(baseline):
    failed = Baseline(
        **{**vars(baseline), "status": "failed", "metrics": {}, "error": "ValueError: boom"}
    )
    assert failed.error == "ValueError: boom"


def test_baseline_non_ok_with_metrics_rejected(baseline):
    with pytest.raises(PayloadError, match="metrics must be empty"):
        Baseline(**{**vars(baseline), "status": "failed", "error": "updater crashed"})


def test_baseline_timeout_with_empty_metrics_accepted(baseline):
    timeout = Baseline(
        **{**vars(baseline), "status": "timeout", "metrics": {}, "error": "series too slow"}
    )
    assert timeout.metrics == {}


def test_baseline_ok_with_empty_metrics_accepted(baseline):
    assert Baseline(**{**vars(baseline), "metrics": {}}).metrics == {}


@pytest.mark.parametrize("bad", ["sha256:xyz", "abc", 5])
def test_baseline_bad_per_task_blob_rejected(baseline, bad):
    with pytest.raises(PayloadError, match="per_task_blob"):
        Baseline(**{**vars(baseline), "per_task_blob": bad})


def test_baseline_failed_with_blob_accepted(baseline, blob_ref):
    failed = Baseline(
        **{
            **vars(baseline),
            "status": "failed",
            "metrics": {},
            "error": "worker died",
            "per_task_blob": blob_ref("failure-detail"),
        }
    )
    assert failed.per_task_blob is not None


@pytest.mark.parametrize("field", ["status", "per_task_blob", "error"])
def test_baseline_missing_field_rejected(baseline, field):
    data = payload_to_dict(baseline)
    del data[field]
    with pytest.raises(PayloadError, match="missing fields"):
        payload_from_dict("baseline", data)


def test_baseline_unknown_field_rejected(baseline):
    data = payload_to_dict(baseline)
    data["mystery"] = 1
    with pytest.raises(PayloadError, match="unknown fields"):
        payload_from_dict("baseline", data)


def test_baseline_nan_in_hyperparams_rejected(baseline):
    with pytest.raises(PayloadError, match="non-finite"):
        Baseline(**{**vars(baseline), "hyperparams": {"learning_rate": float("nan")}})


def test_baseline_nan_in_metrics_rejected(baseline):
    with pytest.raises(PayloadError, match="metrics"):
        Baseline(**{**vars(baseline), "metrics": {"mase": float("nan")}})


def test_sealed_baseline_timeout_in_sealed_stream_allowed(baseline):
    sealed = Baseline(
        **{
            **vars(baseline),
            "pool": "sealed",
            "status": "timeout",
            "metrics": {},
            "error": "series etth1/HUFL exceeded 600.0s at step 3",
        }
    )
    record = _record("sealed", "baseline", sealed)
    assert record.payload.pool == "sealed"
    assert record.payload.status == "timeout"


def test_acceptance_requires_bool():
    with pytest.raises(PayloadError, match="accepted"):
        Acceptance(accepted="yes", rule="r", new_artifact=None)


def test_commitment_requires_blob_ref():
    with pytest.raises(PayloadError, match="content_hash"):
        Commitment(subject="s", content_hash="nope", external_timestamp=None)


def test_correction_allows_null_replacement():
    correction = Correction(reason="r", replacement=None)
    assert correction.replacement is None


def test_non_llm_proposer_accepted(proposal):
    assert proposal.proposer["model"] is None
    assert proposal.proposer["temperature"] is None
    assert proposal.prompt_blob is None and proposal.response_blob is None


def test_proposer_bad_model_type_rejected(proposal):
    bad = dict(proposal.proposer)
    bad["model"] = 7
    with pytest.raises(PayloadError, match=r"proposer\.model"):
        Proposal(**{**vars(proposal), "proposer": bad})


def test_proposer_bad_temperature_type_rejected(proposal):
    bad = dict(proposal.proposer)
    bad["temperature"] = "hot"
    with pytest.raises(PayloadError, match=r"proposer\.temperature"):
        Proposal(**{**vars(proposal), "proposer": bad})


def test_evaluation_ok_with_error_rejected(evaluation):
    with pytest.raises(PayloadError, match="error must be None"):
        Evaluation(**{**vars(evaluation), "error": "should not be here"})


def test_evaluation_failed_without_error_rejected(evaluation):
    with pytest.raises(PayloadError, match="non-empty string"):
        Evaluation(**{**vars(evaluation), "status": "failed", "error": None})


def test_evaluation_failed_with_empty_error_rejected(evaluation):
    with pytest.raises(PayloadError, match="non-empty string"):
        Evaluation(**{**vars(evaluation), "status": "timeout", "error": ""})


def test_evaluation_failed_with_error_accepted(evaluation):
    failed = Evaluation(**{**vars(evaluation), "status": "failed", "error": "sandbox OOM"})
    assert failed.error == "sandbox OOM"


def test_allocation_duplicate_candidates_rejected(allocation):
    duplicated = [*allocation.candidates, allocation.candidates[0]]
    with pytest.raises(PayloadError, match="duplicates"):
        Allocation(
            allocator=allocation.allocator,
            candidates=duplicated,
            chosen=[],
            scores={},
        )


def test_allocation_duplicate_chosen_rejected(allocation):
    with pytest.raises(PayloadError, match="duplicates"):
        Allocation(
            allocator=allocation.allocator,
            candidates=allocation.candidates,
            chosen=[allocation.candidates[0], allocation.candidates[0]],
            scores={},
        )


def test_allocation_scores_must_reference_candidates(allocation):
    with pytest.raises(PayloadError, match="scores keys"):
        Allocation(
            allocator=allocation.allocator,
            candidates=allocation.candidates,
            chosen=[],
            scores={str(uuid.uuid4()): 0.5},
        )
