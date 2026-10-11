"""Payload construction, validation and round-trips."""

import pytest

from evolvets.ledger.payloads import (
    PayloadError,
    RunEnd,
    RunStart,
    payload_from_dict,
    payload_to_dict,
    validate_kind_payload,
)
from evolvets.ledger.schema import Budget, RecordError

PROTOCOL_VERSIONS = {
    "ledger": "1.0",
    "arena": "0",
    "generator": "0",
    "cache_manifest": "0",
}


@pytest.mark.parametrize("kind", ["run_start", "run_end"])
def test_round_trip(kind, request):
    payload = request.getfixturevalue(kind)
    assert payload_from_dict(kind, payload_to_dict(payload)) == payload


def test_validate_kind_payload_accepts_matching(run_start):
    validate_kind_payload("run_start", run_start)


def test_validate_kind_payload_rejects_mismatch(run_end):
    with pytest.raises(PayloadError):
        validate_kind_payload("run_start", run_end)


def test_unknown_kind_rejected():
    with pytest.raises(PayloadError, match="unknown kind"):
        payload_from_dict("mystery", {})


def test_unknown_field_rejected(run_start):
    data = payload_to_dict(run_start)
    data["extra"] = 1
    with pytest.raises(PayloadError, match="unknown fields"):
        payload_from_dict("run_start", data)


def test_missing_field_rejected(run_start):
    data = payload_to_dict(run_start)
    del data["git_commit"]
    with pytest.raises(PayloadError, match="missing fields"):
        payload_from_dict("run_start", data)


def test_nan_rejected_at_construction():
    with pytest.raises(PayloadError, match="non-finite"):
        RunEnd(status="completed", summary={"score": float("nan")})


def test_infinity_rejected_at_construction():
    with pytest.raises(PayloadError, match="non-finite"):
        RunEnd(status="completed", summary={"score": float("inf")})


def test_budget_rejects_non_finite():
    with pytest.raises(RecordError, match="finite"):
        Budget(llm_input_tokens=float("nan"))


def test_budget_rejects_negative():
    with pytest.raises(RecordError, match="non-negative"):
        Budget(wall_seconds=-1)


def test_bad_blob_ref_rejected():
    with pytest.raises(PayloadError, match="config_blob"):
        RunStart(
            condition="c",
            seed=1,
            config_blob="sha256:tooshort",
            git_commit="abc",
            protocol_versions=dict(PROTOCOL_VERSIONS),
            budget_limits=Budget(),
        )


def test_protocol_versions_key_set_enforced():
    with pytest.raises(PayloadError, match="protocol_versions"):
        RunStart(
            condition="c",
            seed=1,
            config_blob="sha256:" + "0" * 64,
            git_commit="abc",
            protocol_versions={"ledger": "1.0"},
            budget_limits=Budget(),
        )


def test_run_end_status_literal_enforced():
    with pytest.raises(PayloadError, match="status"):
        RunEnd(status="finished", summary={})


def test_seed_must_be_int():
    with pytest.raises(PayloadError, match="seed"):
        RunStart(
            condition="c",
            seed="seven",
            config_blob="sha256:" + "0" * 64,
            git_commit="abc",
            protocol_versions=dict(PROTOCOL_VERSIONS),
            budget_limits=Budget(),
        )
