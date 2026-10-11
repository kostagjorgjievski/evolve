"""Ordering validator: one passing and one failing fixture per rule."""

import uuid

from evolvets.ledger.payloads import (
    Acceptance,
    Evaluation,
    Prediction,
    Proposal,
    RunEnd,
    RunStart,
)
from evolvets.ledger.schema import ZERO_HASH, Budget, make_record
from evolvets.ledger.validate import validate_run

BLOB = f"sha256:{'c' * 64}"


def mk(
    kind, payload, *, stream="main", seq=0, created_at, refers_to=(), budget=None, record_id=None
):
    return make_record(
        stream=stream,
        seq=seq,
        prev_hash=ZERO_HASH if seq == 0 else "ab" * 32,
        run_id="run-1",
        kind=kind,
        payload=payload,
        refers_to=refers_to,
        budget=budget,
        created_at=created_at,
        record_id=record_id,
    )


def run_start(at, limits=None, record_id=None, stream="main", seq=0):
    payload = RunStart(
        condition="c",
        seed=1,
        config_blob=BLOB,
        git_commit="abc",
        protocol_versions={"ledger": "1.0", "arena": "0", "generator": "0", "cache_manifest": "0"},
        budget_limits=limits if limits is not None else Budget(),
    )
    return mk("run_start", payload, seq=seq, created_at=at, record_id=record_id, stream=stream)


def proposal(at, seq, record_id=None, budget=None):
    payload = Proposal(
        target="updater",
        parent_artifact=BLOB,
        child_artifact=BLOB,
        diff_blob=BLOB,
        observation_blob=BLOB,
        prompt_blob=None,
        response_blob=None,
        diagnosis="d",
        proposer={"id": "i", "version": "1", "model": None, "temperature": None},
        improver_version="1",
        planted_defect=None,
    )
    return mk("proposal", payload, seq=seq, created_at=at, record_id=record_id, budget=budget)


def prediction(at, seq, target_id):
    payload = Prediction(
        predictor={"id": "s", "version": "1"},
        metric="mase",
        horizon_generations=1,
        quantiles={"q05": 0.1, "q25": 0.2, "q50": 0.3, "q75": 0.4, "q95": 0.6},
    )
    return mk("prediction", payload, seq=seq, created_at=at, refers_to=[target_id])


def evaluation(at, seq, target_id, stream="main"):
    payload = Evaluation(
        pool="dev" if stream == "main" else "sealed",
        status="ok",
        metrics={"mase": 0.5},
        per_task_blob=None,
        error=None,
    )
    return mk("evaluation", payload, stream=stream, seq=seq, created_at=at, refers_to=[target_id])


def acceptance(at, seq, proposal_id, evaluation_id):
    payload = Acceptance(accepted=True, rule="beats-parent", new_artifact=None)
    return mk("acceptance", payload, seq=seq, created_at=at, refers_to=[proposal_id, evaluation_id])


def run_end(at, seq, stream="main"):
    return mk(
        "run_end", RunEnd(status="completed", summary={}), seq=seq, created_at=at, stream=stream
    )


def test_valid_run_has_no_violations():
    p = proposal("2026-10-11T11:00:01.000000Z", 1)
    e = evaluation("2026-10-11T11:00:03.000000Z", 3, p.record_id)
    records = [
        run_start("2026-10-11T11:00:00.000000Z"),
        p,
        prediction("2026-10-11T11:00:02.000000Z", 2, p.record_id),
        e,
        acceptance("2026-10-11T11:00:04.000000Z", 4, p.record_id, e.record_id),
        run_end("2026-10-11T11:00:05.000000Z", 5),
    ]
    assert validate_run(records) == []


def test_run_start_must_be_first():
    records = [proposal("2026-10-11T11:00:01.000000Z", 0), run_start("2026-10-11T11:00:00.000000Z")]
    violations = validate_run(records)
    assert any("first record of main must be run_start" in v for v in violations)


def test_run_end_must_be_last():
    records = [
        run_start("2026-10-11T11:00:00.000000Z"),
        run_end("2026-10-11T11:00:01.000000Z", 1),
        proposal("2026-10-11T11:00:02.000000Z", 2),
    ]
    violations = validate_run(records)
    assert any("run_end must be the last record" in v for v in violations)


def test_unknown_reference_rejected():
    records = [
        run_start("2026-10-11T11:00:00.000000Z"),
        prediction("2026-10-11T11:00:01.000000Z", 1, str(uuid.uuid4())),
    ]
    assert any("refers to unknown record" in v for v in validate_run(records))


def test_forward_in_time_reference_rejected():
    late = proposal("2026-10-11T11:00:05.000000Z", 1)
    early_reference = mk(
        "prediction",
        Prediction(
            predictor={"id": "s", "version": "1"},
            metric="mase",
            horizon_generations=1,
            quantiles={"q05": 0.1, "q25": 0.2, "q50": 0.3, "q75": 0.4, "q95": 0.6},
        ),
        seq=2,
        created_at="2026-10-11T11:00:01.000000Z",
        refers_to=[late.record_id],
    )
    records = [run_start("2026-10-11T11:00:00.000000Z"), late, early_reference]
    assert any("created later" in v for v in validate_run(records))


def test_acceptance_kinds_enforced():
    p = proposal("2026-10-11T11:00:01.000000Z", 1)
    pred = prediction("2026-10-11T11:00:02.000000Z", 2, p.record_id)
    wrong_order = acceptance("2026-10-11T11:00:04.000000Z", 3, pred.record_id, p.record_id)
    records = [run_start("2026-10-11T11:00:00.000000Z"), p, pred, wrong_order]
    violations = validate_run(records)
    assert any("does not refer to a proposal first" in v for v in violations)
    assert any("does not refer to an evaluation second" in v for v in violations)


def test_evaluation_needs_prediction_logged_before():
    p = proposal("2026-10-11T11:00:01.000000Z", 1)
    e = evaluation("2026-10-11T11:00:02.000000Z", 2, p.record_id)
    late_prediction = prediction("2026-10-11T11:00:03.000000Z", 3, p.record_id)
    records = [run_start("2026-10-11T11:00:00.000000Z"), p, e, late_prediction]
    assert any("no prediction logged before it" in v for v in validate_run(records))


def test_cross_stream_prediction_ordering():
    p = proposal("2026-10-11T11:00:01.000000Z", 1)
    late = prediction("2026-10-11T11:00:05.000000Z", 2, p.record_id)
    early_evaluation = evaluation("2026-10-11T11:00:02.000000Z", 0, p.record_id, stream="sealed")
    records = [run_start("2026-10-11T11:00:00.000000Z"), p, late, early_evaluation]
    assert any("no prediction logged before it" in v for v in validate_run(records))


def test_cross_stream_prediction_after_evaluation_passes():
    p = proposal("2026-10-11T11:00:01.000000Z", 1)
    pred = prediction("2026-10-11T11:00:02.000000Z", 2, p.record_id)
    e = evaluation("2026-10-11T11:00:03.000000Z", 0, p.record_id, stream="sealed")
    records = [run_start("2026-10-11T11:00:00.000000Z"), p, pred, e]
    assert validate_run(records) == []


def test_budget_limit_enforced():
    limits = Budget(evaluations=1)
    spendy = proposal("2026-10-11T11:00:01.000000Z", 1, budget=Budget(evaluations=2))
    records = [run_start("2026-10-11T11:00:00.000000Z", limits=limits), spendy]
    assert any("budget evaluations exceeded" in v for v in validate_run(records))


def test_budget_within_limit_passes():
    limits = Budget(evaluations=2)
    spendy = proposal("2026-10-11T11:00:01.000000Z", 1, budget=Budget(evaluations=1))
    records = [run_start("2026-10-11T11:00:00.000000Z", limits=limits), spendy]
    assert validate_run(records) == []


def test_duplicate_record_id_rejected():
    shared = str(uuid.uuid4())
    records = [
        run_start("2026-10-11T11:00:00.000000Z", record_id=shared),
        proposal("2026-10-11T11:00:01.000000Z", 1, record_id=shared),
    ]
    assert any("duplicate record_id" in v for v in validate_run(records))


def test_prediction_must_refer_to_a_proposal():
    start = run_start("2026-10-11T11:00:00.000000Z")
    records = [start, prediction("2026-10-11T11:00:02.000000Z", 1, start.record_id)]
    assert any(
        "prediction at seq 1 does not refer to a proposal" in v for v in validate_run(records)
    )


def test_evaluation_must_refer_to_a_proposal():
    start = run_start("2026-10-11T11:00:00.000000Z")
    records = [
        start,
        proposal("2026-10-11T11:00:01.000000Z", 1),
        evaluation("2026-10-11T11:00:03.000000Z", 2, start.record_id),
    ]
    assert any(
        "evaluation at seq 2 does not refer to a proposal" in v for v in validate_run(records)
    )


def test_cross_stream_prediction_with_equal_timestamp_rejected():
    p = proposal("2026-10-11T11:00:01.000000Z", 1)
    tie = prediction("2026-10-11T11:00:02.000000Z", 2, p.record_id)
    e = evaluation("2026-10-11T11:00:02.000000Z", 0, p.record_id, stream="sealed")
    records = [run_start("2026-10-11T11:00:00.000000Z"), p, tie, e]
    assert any("no prediction logged before it" in v for v in validate_run(records))


def test_second_run_start_rejected():
    records = [
        run_start("2026-10-11T11:00:00.000000Z"),
        proposal("2026-10-11T11:00:01.000000Z", 1),
        run_start("2026-10-11T11:00:02.000000Z", seq=2),
    ]
    violations = validate_run(records)
    assert any("2 run_start records; exactly one is allowed" in v for v in violations)


def test_run_start_in_sealed_stream_rejected():
    records = [
        run_start("2026-10-11T11:00:00.000000Z"),
        proposal("2026-10-11T11:00:01.000000Z", 1),
        run_start("2026-10-11T11:00:02.000000Z", stream="sealed"),
    ]
    assert any(
        "run_start at seq 0 is only allowed in the main stream" in v for v in validate_run(records)
    )


def test_run_end_in_sealed_stream_rejected():
    records = [
        run_start("2026-10-11T11:00:00.000000Z"),
        proposal("2026-10-11T11:00:01.000000Z", 1),
        run_end("2026-10-11T11:00:02.000000Z", 0, stream="sealed"),
    ]
    assert any(
        "run_end at seq 0 is only allowed in the main stream" in v for v in validate_run(records)
    )
