"""Canonical serialization determinism and record hashing."""

import uuid

import pytest

from evolvets.ledger.canonical import canonical_bytes
from evolvets.ledger.schema import ZERO_HASH, Budget, RecordError, make_record


def _sample_kwargs(payload):
    return dict(
        stream="main",
        seq=3,
        prev_hash="ab" * 32,
        run_id="run-1",
        kind="run_end",
        payload=payload,
        refers_to=[str(uuid.uuid4())],
        budget=Budget(evaluations=1),
        created_at="2026-10-11T09:00:00.000000Z",
        record_id=str(uuid.uuid4()),
    )


def test_same_record_same_bytes(run_end):
    kwargs = _sample_kwargs(run_end)
    first = make_record(**kwargs)
    second = make_record(**kwargs)
    assert first.record_hash == second.record_hash
    assert canonical_bytes(first.to_dict()) == canonical_bytes(second.to_dict())


def test_different_content_different_hash(run_end):
    base = _sample_kwargs(run_end)
    first = make_record(**base)
    tweaked = dict(base)
    tweaked["created_at"] = "2026-10-11T09:00:00.000001Z"
    second = make_record(**tweaked)
    assert first.record_hash != second.record_hash


def test_first_record_requires_zero_prev_hash(run_end):
    kwargs = _sample_kwargs(run_end)
    kwargs.update(seq=0, prev_hash="cd" * 32)
    with pytest.raises(RecordError, match="prev_hash"):
        make_record(**kwargs)


def test_unknown_stream_rejected(run_end):
    kwargs = _sample_kwargs(run_end)
    kwargs.update(stream="other")
    with pytest.raises(RecordError, match="stream"):
        make_record(**kwargs)


def test_zero_hash_is_64_zeros():
    assert ZERO_HASH == "0" * 64


def test_unknown_kind_rejected_at_construction(run_end):
    kwargs = _sample_kwargs(run_end)
    kwargs.update(kind="bogus")
    with pytest.raises(ValueError, match="unknown kind"):
        make_record(**kwargs)


def test_kind_payload_mismatch_rejected_at_construction(run_start):
    kwargs = _sample_kwargs(run_start)
    kwargs.update(kind="run_end", payload=run_start)
    with pytest.raises(ValueError, match="must be"):
        make_record(**kwargs)
