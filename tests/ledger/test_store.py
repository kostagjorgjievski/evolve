"""Writer and reader behaviour: append, resume, locking, tamper detection."""

import json
import socket
import subprocess
import sys
import textwrap

import pytest

import evolvets.ledger.store as store_module
from evolvets.ledger.canonical import canonical_json
from evolvets.ledger.schema import make_record
from evolvets.ledger.store import (
    LedgerCorruptError,
    LedgerLockedError,
    LedgerReader,
    LedgerWriter,
    VerifyResult,
)

RUN_ID = "run-001"


def write_two(root, run_start, run_end):
    """Write one run_start and one run_end record; return them."""
    with LedgerWriter(root, RUN_ID, "main") as writer:
        first = writer.append("run_start", run_start)
        second = writer.append("run_end", run_end)
    return first, second


def rewrite_lines(path, lines):
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def test_write_and_read_back(root, run_start, run_end):
    first, second = write_two(root, run_start, run_end)
    reader = LedgerReader(root, RUN_ID)
    assert list(reader.records()) == [first, second]
    assert reader.verify() == VerifyResult(True, None, None, "ok", 2)


def test_reader_single_stream_selection(root, run_start, run_end):
    write_two(root, run_start, run_end)
    reader = LedgerReader(root, RUN_ID)
    assert len(list(reader.records(streams=["main"]))) == 2
    assert list(reader.records(streams=["sealed"])) == []
    with pytest.raises(ValueError, match="stream"):
        list(reader.records(streams=["other"]))


def test_reopen_resumes_seq_and_chain(root, run_start, run_end):
    with LedgerWriter(root, RUN_ID, "main") as writer:
        first = writer.append("run_start", run_start)
    with LedgerWriter(root, RUN_ID, "main") as writer:
        second = writer.append("run_end", run_end)
    assert second.seq == 1
    assert second.prev_hash == first.record_hash
    assert LedgerReader(root, RUN_ID).verify().ok


def test_second_writer_on_same_stream_raises(root, run_start):
    writer = LedgerWriter(root, RUN_ID, "main")
    try:
        with pytest.raises(LedgerLockedError):
            LedgerWriter(root, RUN_ID, "main")
    finally:
        writer.close()
    with LedgerWriter(root, RUN_ID, "main") as writer:
        writer.append("run_start", run_start)
    assert LedgerReader(root, RUN_ID).verify().ok


def test_corrupt_stream_refuses_to_resume(root, run_start):
    with LedgerWriter(root, RUN_ID, "main") as writer:
        writer.append("run_start", run_start)
    path = root / RUN_ID / "main.jsonl"
    rewrite_lines(
        path, [path.read_text(encoding="utf-8").rstrip("\n").replace('"seed":7', '"seed":8')]
    )
    with pytest.raises(LedgerCorruptError):
        LedgerWriter(root, RUN_ID, "main")


def test_record_survives_subprocess_crash(root):
    code = textwrap.dedent(
        """
        import sys
        from evolvets.ledger.payloads import RunStart
        from evolvets.ledger.schema import Budget
        from evolvets.ledger.store import LedgerWriter

        root, run_id = sys.argv[1], sys.argv[2]
        payload = RunStart(
            condition="c",
            seed=1,
            config_blob="sha256:" + "0" * 64,
            git_commit="abc123",
            protocol_versions={
                "ledger": "1.0", "arena": "0", "generator": "0", "cache_manifest": "0"
            },
            budget_limits=Budget(),
        )
        writer = LedgerWriter(root, run_id, "main")
        writer.append("run_start", payload)
        raise RuntimeError("crash right after the acknowledged append")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(root), RUN_ID], capture_output=True, text=True
    )
    assert result.returncode != 0
    reader = LedgerReader(root, RUN_ID)
    verified = reader.verify()
    assert verified.ok and verified.records == 1


def test_tamper_edited_byte_detected(root, run_start, run_end):
    write_two(root, run_start, run_end)
    path = root / RUN_ID / "main.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"seed":7', '"seed":8')
    rewrite_lines(path, lines)
    result = LedgerReader(root, RUN_ID).verify()
    assert not result.ok
    assert result.seq == 0
    assert "record_hash mismatch" in result.reason


def test_tamper_deleted_line_detected(root, run_start, run_end):
    write_two(root, run_start, run_end)
    path = root / RUN_ID / "main.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    rewrite_lines(path, lines[1:])
    result = LedgerReader(root, RUN_ID).verify()
    assert not result.ok
    assert result.seq == 0
    assert "chain break" in result.reason


def test_tamper_swapped_lines_detected(root, run_start, run_end):
    write_two(root, run_start, run_end)
    path = root / RUN_ID / "main.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    rewrite_lines(path, lines)
    result = LedgerReader(root, RUN_ID).verify()
    assert not result.ok
    assert result.seq == 0
    assert "chain break" in result.reason


def test_tamper_forged_append_detected(root, run_start, run_end):
    write_two(root, run_start, run_end)
    path = root / RUN_ID / "main.jsonl"
    forged = make_record(
        stream="main",
        seq=2,
        prev_hash="ff" * 32,
        run_id=RUN_ID,
        kind="run_end",
        payload=run_end,
    )
    lines = path.read_text(encoding="utf-8").splitlines()
    lines.append(canonical_json(forged.to_dict()))
    rewrite_lines(path, lines)
    result = LedgerReader(root, RUN_ID).verify()
    assert not result.ok
    assert result.seq == 2
    assert "chain break" in result.reason


def test_store_exposes_no_mutation_api():
    forbidden = ("delete", "update", "remove", "truncate", "rewrite")
    for name, obj in vars(store_module).items():
        for word in forbidden:
            assert word not in name.lower(), f"store exposes {name!r}"
        if isinstance(obj, type) and obj.__module__ == store_module.__name__:
            for member in vars(obj):
                for word in forbidden:
                    assert word not in member.lower(), f"{obj.__name__} exposes {member!r}"


def test_torn_tail_detected_and_writer_refuses(root, run_start, run_end):
    write_two(root, run_start, run_end)
    path = root / RUN_ID / "main.jsonl"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.rstrip("\n"), encoding="utf-8")
    result = LedgerReader(root, RUN_ID).verify()
    assert not result.ok
    assert result.seq == 1
    assert "torn tail" in result.reason
    with pytest.raises(LedgerCorruptError):
        LedgerWriter(root, RUN_ID, "main")


def test_verify_fails_for_missing_run(root):
    result = LedgerReader(root, "no-such-run").verify()
    assert not result.ok
    assert "missing" in result.reason
    assert result.records == 0


def _make_dead_pid():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    return child.pid


def _plant_lock(root, run_id, pid, host):
    lock_path = root / run_id / "main.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"pid": pid, "host": host, "created_at": "now"}))
    return lock_path


def test_stale_lock_with_live_pid_is_refused(root, run_start):
    writer = LedgerWriter(root, RUN_ID, "main")
    try:
        with pytest.raises(LedgerLockedError) as excinfo:
            LedgerWriter(root, RUN_ID, "main")
        assert "stale lock:" not in str(excinfo.value)
        with pytest.raises(LedgerLockedError, match="refusing to break"):
            LedgerWriter.break_stale_lock(root, RUN_ID, "main")
    finally:
        writer.close()


def test_stale_lock_with_dead_pid_can_be_broken(root, run_start):
    lock_path = _plant_lock(root, RUN_ID, _make_dead_pid(), socket.gethostname())
    with pytest.raises(LedgerLockedError, match="stale"):
        LedgerWriter(root, RUN_ID, "main")
    LedgerWriter.break_stale_lock(root, RUN_ID, "main")
    assert not lock_path.exists()
    with LedgerWriter(root, RUN_ID, "main") as writer:
        writer.append("run_start", run_start)
    assert LedgerReader(root, RUN_ID).verify().ok


def test_foreign_host_lock_is_not_stale(root):
    _plant_lock(root, RUN_ID, _make_dead_pid(), "some-other-host")
    with pytest.raises(LedgerLockedError) as excinfo:
        LedgerWriter(root, RUN_ID, "main")
    assert "stale lock:" not in str(excinfo.value)
    with pytest.raises(LedgerLockedError, match="refusing to break"):
        LedgerWriter.break_stale_lock(root, RUN_ID, "main")
