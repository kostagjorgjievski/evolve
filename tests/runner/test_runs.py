"""Run lifecycle: run_start shape, reuse refusal, abort paths, budget sums."""

import re
import subprocess
from pathlib import Path

import pytest

from evolvets.arena.versions import protocol_versions
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.canonical import canonical_bytes
from evolvets.ledger.payloads import Baseline, RunStart
from evolvets.ledger.schema import Budget
from evolvets.ledger.store import LedgerLockedError, LedgerReader, LedgerWriter
from evolvets.ledger.validate import validate_run
from evolvets.runner.cli import main
from evolvets.runner.runs import Run, RunError, current_git_commit, new_run_id

COMMIT = "a" * 40
RUN_ID_RE = re.compile(r"^gate1-tune-\d{8}T\d{6}Z-[0-9a-f]{8}$")
_BUDGET_FIELDS = (
    "llm_input_tokens",
    "llm_output_tokens",
    "wall_seconds",
    "gpu_seconds",
    "evaluations",
)


@pytest.fixture()
def root(tmp_path):
    return tmp_path / "ledgers"


def start_run(root, run_id="run-a", config=None, **kwargs):
    """Start a run with test defaults for the keyword-only arguments."""
    defaults: dict = {
        "condition": "gate1-dev-tune",
        "seed": 7,
        "config": config if config is not None else {"experiment": "gate1"},
        "budget_limits": Budget(wall_seconds=1000.0, evaluations=1000),
        "git_commit": COMMIT,
    }
    defaults.update(kwargs)
    return Run.start(Path(root), run_id, **defaults)


def dev_baseline(mase=0.81):
    """One dev-pool baseline payload, the kind of record a run appends."""
    return Baseline(
        updater_id="hedge",
        hyperparams={"eta": 0.5},
        pool="dev",
        status="ok",
        metrics={"mase": mase},
        per_task_blob=None,
        error=None,
    )


def read_records(root, run_id):
    return list(LedgerReader(root, run_id).records())


def snapshot(root, run_id):
    """Map every file under the run directory to its bytes."""
    directory = Path(root) / run_id
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_run_start_is_the_first_and_only_record(root):
    with start_run(root) as run:
        records = read_records(root, "run-a")
        assert len(records) == 1
        record = records[0]
        assert record.seq == 0
        assert record.kind == "run_start"
        payload = record.payload
        assert isinstance(payload, RunStart)
        assert payload.condition == "gate1-dev-tune"
        assert payload.seed == 7
        assert payload.requires_predictions is False
        assert payload.git_commit == COMMIT
        assert payload.budget_limits == Budget(wall_seconds=1000.0, evaluations=1000)
        assert payload.protocol_versions == protocol_versions()
        assert BlobStore(root / "run-a" / "blobs").get(payload.config_blob) == canonical_bytes(
            {"experiment": "gate1"}
        )
        assert run.run_start_id == record.record_id
        assert run.run_id == "run-a"
        assert run.limits == Budget(wall_seconds=1000.0, evaluations=1000)
        run.end("completed", {})


def test_started_appended_ended_run_verifies(root, capsys):
    with start_run(root) as run:
        run.append("baseline", dev_baseline(), budget=Budget(wall_seconds=0.5, evaluations=1))
        run.end("completed", {"replays": 1})
    assert main(["ledger", "verify", str(root), "run-a"]) == 0
    assert "OK" in capsys.readouterr().out
    assert validate_run(read_records(root, "run-a")) == []


def test_existing_main_stream_refuses_reuse(root):
    with start_run(root, "run-a") as run:
        run.end("completed", {})
    before = snapshot(root, "run-a")
    with pytest.raises(RunError, match="refusing to reuse"):
        start_run(root, "run-a")
    assert snapshot(root, "run-a") == before


def test_existing_sealed_stream_refuses_reuse(root):
    with LedgerWriter(root, "run-b", "sealed") as writer:
        writer.append(
            "baseline",
            Baseline(
                updater_id="oracle",
                hyperparams={},
                pool="sealed",
                status="ok",
                metrics={"mase": 1.0},
                per_task_blob=None,
                error=None,
            ),
        )
    before = snapshot(root, "run-b")
    with pytest.raises(RunError, match="refusing to reuse"):
        start_run(root, "run-b")
    assert snapshot(root, "run-b") == before
    assert not (root / "run-b" / "main.jsonl").exists()


@pytest.mark.parametrize("config", [{"x": float("nan")}, {"s": {1}}])
def test_bad_config_creates_nothing(root, config):
    with pytest.raises(RunError, match="serializable"):
        start_run(root, "run-a", config=config)
    assert not (root / "run-a").exists()


def test_config_with_non_string_key_is_refused(root):
    with pytest.raises(RunError, match="string keys"):
        start_run(root, "run-a", config={1: "a"})
    assert not (root / "run-a").exists()


@pytest.mark.parametrize("run_id", ["run/a", "..", "run/../a", "run a"])
def test_bad_run_id_creates_nothing(root, run_id):
    with pytest.raises(RunError, match="run id must match"):
        start_run(root, run_id)
    assert list(Path(root).glob("*")) == [] or not (root / run_id).exists()


def test_exception_inside_with_aborts_and_propagates(root):
    with pytest.raises(RuntimeError, match="boom"), start_run(root):
        raise RuntimeError("boom")
    records = read_records(root, "run-a")
    assert [record.kind for record in records] == ["run_start", "run_end"]
    assert records[-1].payload.status == "aborted"
    assert records[-1].payload.summary["error"] == "RuntimeError: boom"


def test_keyboard_interrupt_aborts(root):
    with pytest.raises(KeyboardInterrupt), start_run(root):
        raise KeyboardInterrupt()
    records = read_records(root, "run-a")
    assert records[-1].payload.status == "aborted"
    assert records[-1].payload.summary["error"].startswith("KeyboardInterrupt")


def test_normal_exit_without_end_aborts(root):
    with start_run(root):
        pass
    records = read_records(root, "run-a")
    assert records[-1].payload.status == "aborted"
    assert records[-1].payload.summary["error"] == "run exited without run_end"


def test_exit_after_end_adds_nothing_and_refuses_more(root):
    with start_run(root) as run:
        run.end("completed", {"n": 1})
    records = read_records(root, "run-a")
    assert [record.kind for record in records] == ["run_start", "run_end"]
    assert records[-1].payload.status == "completed"
    with pytest.raises(RunError, match="already ended"):
        run.end("completed", {})
    with pytest.raises(RunError, match="ended"):
        run.append("baseline", dev_baseline())
    assert len(read_records(root, "run-a")) == 2


def test_spent_sums_record_budgets_field_by_field(root):
    with start_run(root) as run:
        run.append("baseline", dev_baseline(), budget=Budget(wall_seconds=1.5, evaluations=1))
        run.append("baseline", dev_baseline(0.9), budget=Budget(wall_seconds=2.0, evaluations=0))
        assert run.spent == Budget(wall_seconds=3.5, evaluations=1)
        run.end("completed", {})
    summed = {field: 0.0 for field in _BUDGET_FIELDS}
    for record in read_records(root, "run-a"):
        for field in _BUDGET_FIELDS:
            summed[field] += getattr(record.budget, field)
    assert run.spent == Budget(**summed)


def test_append_refuses_run_lifecycle_kinds(root):
    with start_run(root) as run:
        for kind in ("run_start", "run_end"):
            with pytest.raises(RunError, match="refuses"):
                run.append(kind, dev_baseline())
        assert len(read_records(root, "run-a")) == 1
        run.end("completed", {})


def test_open_run_holds_the_main_lock(root):
    with start_run(root) as run:
        with pytest.raises(LedgerLockedError):
            LedgerWriter(root, "run-a", "main")
        run.end("completed", {})
    with LedgerWriter(root, "run-a", "main") as writer:
        assert writer.stream == "main"


def run_git(repo, *arguments):
    """Run one git command in ``repo``, failing the test on a non-zero exit."""
    subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True)


@pytest.fixture()
def git_repo(tmp_path):
    """A clean git repository with one committed tracked file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    run_git(repo, "init")
    run_git(repo, "config", "user.email", "test@example.com")
    run_git(repo, "config", "user.name", "Test")
    (repo / "tracked.txt").write_text("one\n", encoding="utf-8")
    run_git(repo, "add", "tracked.txt")
    run_git(repo, "commit", "-m", "init")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    return repo, repo / "tracked.txt", head


def test_current_git_commit_accepts_untracked_files(git_repo):
    repo, _tracked, head = git_repo
    (repo / "untracked.txt").write_text("ignored\n", encoding="utf-8")
    assert current_git_commit(repo) == head
    assert re.fullmatch(r"[0-9a-f]{40}", head)


def test_current_git_commit_refuses_modified_file(git_repo):
    repo, tracked, _head = git_repo
    tracked.write_text("two\n", encoding="utf-8")
    with pytest.raises(RunError, match="uncommitted changes"):
        current_git_commit(repo)


def test_current_git_commit_refuses_staged_file(git_repo):
    repo, tracked, _head = git_repo
    tracked.write_text("three\n", encoding="utf-8")
    run_git(repo, "add", "tracked.txt")
    with pytest.raises(RunError, match="uncommitted changes"):
        current_git_commit(repo)


def test_current_git_commit_refuses_plain_directory(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(RunError):
        current_git_commit(plain)


def test_new_run_id_shape_and_uniqueness():
    first = new_run_id("gate1-tune")
    assert RUN_ID_RE.match(first)
    assert new_run_id("gate1-tune") != first
    with pytest.raises(RunError, match="prefix"):
        new_run_id("Bad/Prefix")
