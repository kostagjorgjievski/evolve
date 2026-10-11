"""Blob store: round-trip, dedup, corruption detection, reference walking."""

import hashlib

import pytest

from evolvets.ledger.blobs import BlobError, BlobStore, find_blob_refs


@pytest.fixture()
def store(tmp_path):
    return BlobStore(tmp_path / "blobs")


def test_round_trip(store):
    ref = store.put(b"weights = [0.2, 0.8]")
    assert ref == f"sha256:{hashlib.sha256(b'weights = [0.2, 0.8]').hexdigest()}"
    assert store.get(ref) == b"weights = [0.2, 0.8]"


def test_same_content_dedups_to_same_ref(store):
    first = store.put(b"identical")
    second = store.put(b"identical")
    assert first == second
    assert store.get(first) == b"identical"


def test_different_content_gets_different_refs(store):
    assert store.put(b"a") != store.put(b"b")


def test_blob_is_sharded_by_first_two_hex(store):
    digest = hashlib.sha256(b"shard me").hexdigest()
    ref = store.put(b"shard me")
    assert (store._dir / digest[:2] / digest).exists()
    assert ref == f"sha256:{digest}"


def test_corrupted_blob_detected(store):
    ref = store.put(b"original content")
    digest = ref[len("sha256:") :]
    path = store._dir / digest[:2] / digest
    path.write_bytes(b"tampered content")
    with pytest.raises(BlobError, match="does not hash"):
        store.get(ref)


def test_missing_blob_rejected(store):
    ref = f"sha256:{'0' * 64}"
    with pytest.raises(BlobError, match="missing blob"):
        store.get(ref)


def test_malformed_ref_rejected(store):
    with pytest.raises(BlobError, match="malformed"):
        store.get("not-a-ref")


def test_find_blob_refs_walks_nested_structures():
    nested = {
        "config_blob": f"sha256:{'a' * 64}",
        "metrics": {"mase": 0.7, "per_task_blob": None},
        "children": [{"diff_blob": f"sha256:{'b' * 64}"}, "plain string"],
    }
    refs = set(find_blob_refs(nested))
    assert refs == {f"sha256:{'a' * 64}", f"sha256:{'b' * 64}"}
