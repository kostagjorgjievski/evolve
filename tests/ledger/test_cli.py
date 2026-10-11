"""CLI: `evolvets ledger verify` and `evolvets ledger unlock`."""

import os
import time

import pytest

from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.canonical import canonical_json
from evolvets.ledger.payloads import Prediction, Proposal, RunStart
from evolvets.ledger.schema import ZERO_HASH, Budget, make_record
from evolvets.ledger.store import STALE_EMPTY_LOCK_SECONDS, LedgerWriter
from evolvets.runner.cli import main

RUN_ID = "run-cli"


def make_proposal(store):
    return Proposal(
        target="updater",
        parent_artifact=store.put(b"parent code"),
        child_artifact=store.put(b"child code"),
        diff_blob=store.put(b"diff text"),
        observation_blob=store.put(b"observations"),
        prompt_blob=None,
        response_blob=None,
        diagnosis="slow adaptation",
        proposer={"id": "improver", "version": "1", "model": None, "temperature": None},
        improver_version="1",
        planted_defect=None,
    )


def write_chained(root, specs):
    """Write a hand-built but hash-valid main stream from (kind, payload, refs)."""
    path = root / RUN_ID / "main.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    prev_hash = ZERO_HASH
    with path.open("w", encoding="utf-8") as handle:
        for seq, (kind, payload, refs) in enumerate(specs):
            record = make_record(
                stream="main",
                seq=seq,
                prev_hash=prev_hash,
                run_id=RUN_ID,
                kind=kind,
                payload=payload,
                refers_to=refs,
                created_at=f"2026-10-11T11:00:{seq:02d}.000000Z",
            )
            handle.write(canonical_json(record.to_dict()) + "\n")
            prev_hash = record.record_hash


@pytest.fixture()
def populated_run(root, run_start, run_end):
    """A clean run: blobs stored, run_start, proposal, prediction, run_end."""
    store = BlobStore(root / RUN_ID / "blobs")
    store.put(b"config")
    with LedgerWriter(root, RUN_ID, "main") as writer:
        writer.append("run_start", run_start)
        proposal = writer.append("proposal", make_proposal(store))
        writer.append(
            "prediction",
            Prediction(
                predictor={"id": "selfmodel", "version": "1"},
                metric="mase",
                horizon_generations=1,
                quantiles={"q05": 0.1, "q25": 0.2, "q50": 0.3, "q75": 0.4, "q95": 0.6},
            ),
            refers_to=[proposal.record_id],
        )
        writer.append("run_end", run_end)


def test_verify_clean_run_exits_zero(root, populated_run, capsys):
    assert main(["ledger", "verify", str(root), RUN_ID]) == 0
    assert "OK" in capsys.readouterr().out


def test_verify_detects_chain_break(root, populated_run, capsys):
    path = root / RUN_ID / "main.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:2] + lines[3:]) + "\n", encoding="utf-8")
    assert main(["ledger", "verify", str(root), RUN_ID]) == 1
    assert "chain break" in capsys.readouterr().out


def test_verify_detects_ordering_violation(root, run_start, run_end, capsys):
    store = BlobStore(root / RUN_ID / "blobs")
    store.put(b"config")
    write_chained(
        root,
        [
            ("run_start", run_start, ()),
            ("run_end", run_end, ()),
            ("proposal", make_proposal(store), ()),
        ],
    )
    assert main(["ledger", "verify", str(root), RUN_ID]) == 1
    assert "run_end must be the last record" in capsys.readouterr().out


def test_verify_detects_missing_blob(root, populated_run, capsys):
    victim = next(p for p in (root / RUN_ID / "blobs").rglob("*") if p.is_file())
    victim.unlink()
    assert main(["ledger", "verify", str(root), RUN_ID]) == 1
    assert "blob" in capsys.readouterr().out


def test_verify_detects_corrupted_blob(root, populated_run, capsys):
    victim = next(p for p in (root / RUN_ID / "blobs").rglob("*") if p.is_file())
    victim.write_bytes(b"corrupted")
    assert main(["ledger", "verify", str(root), RUN_ID]) == 1
    assert "blob" in capsys.readouterr().out


def test_verify_detects_budget_exceeded(root, run_start, run_end, capsys):
    store = BlobStore(root / RUN_ID / "blobs")
    store.put(b"config")
    tight = RunStart(
        condition=run_start.condition,
        seed=run_start.seed,
        config_blob=run_start.config_blob,
        git_commit=run_start.git_commit,
        protocol_versions=run_start.protocol_versions,
        budget_limits=Budget(evaluations=0),
    )
    with LedgerWriter(root, RUN_ID, "main") as writer:
        writer.append("run_start", tight, budget=Budget(evaluations=1))
        writer.append("run_end", run_end)
    assert main(["ledger", "verify", str(root), RUN_ID]) == 1
    assert "budget" in capsys.readouterr().out


def test_verify_missing_run_fails(root, capsys):
    assert main(["ledger", "verify", str(root), "no-such-run"]) == 1
    assert "missing" in capsys.readouterr().out


def _plant_empty_lock(root, run_id, age_seconds):
    lock_path = root / run_id / "main.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("")
    old = time.time() - age_seconds
    os.utime(lock_path, (old, old))
    return lock_path


def test_unlock_breaks_aged_empty_lock(root, populated_run, capsys):
    lock_path = _plant_empty_lock(root, RUN_ID, STALE_EMPTY_LOCK_SECONDS + 60)
    assert main(["ledger", "unlock", str(root), RUN_ID, "main"]) == 0
    assert not lock_path.exists()
    assert "OK" in capsys.readouterr().out


def test_unlock_refuses_fresh_empty_lock(root, populated_run, capsys):
    lock_path = _plant_empty_lock(root, RUN_ID, 0)
    assert main(["ledger", "unlock", str(root), RUN_ID, "main"]) == 1
    assert lock_path.exists()
    assert "refusing" in capsys.readouterr().out


def test_unlock_refuses_live_lock(root, populated_run, capsys):
    writer = LedgerWriter(root, RUN_ID, "main")
    try:
        assert main(["ledger", "unlock", str(root), RUN_ID, "main"]) == 1
    finally:
        writer.close()
    assert "refusing" in capsys.readouterr().out


def test_unlock_then_write_works(root, capsys):
    run_id = "run-unlocked"
    store = BlobStore(root / run_id / "blobs")
    config = store.put(b"cfg")
    _plant_empty_lock(root, run_id, STALE_EMPTY_LOCK_SECONDS + 60)
    assert main(["ledger", "unlock", str(root), run_id, "main"]) == 0
    payload = RunStart(
        condition="c",
        seed=2,
        config_blob=config,
        git_commit="def456",
        protocol_versions={
            "ledger": "1.0",
            "arena": "0",
            "generator": "0",
            "cache_manifest": "0",
        },
        budget_limits=Budget(),
    )
    with LedgerWriter(root, run_id, "main") as writer:
        writer.append("run_start", payload)
    assert main(["ledger", "verify", str(root), run_id]) == 0
    assert "OK" in capsys.readouterr().out


def test_no_command_prints_help_and_exits_zero(capsys):
    assert main([]) == 0
    assert "evolvets" in capsys.readouterr().out
