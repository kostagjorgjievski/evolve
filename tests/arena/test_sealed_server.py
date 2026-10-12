"""Sealed server tests: the spawned server process and its caller-side client.

PR C part 2 (tests 1-4 and 10-14 of the TASK_008 PR C list): ``client.py``
spawns the server, which alone opens the sealed data, replays the updater
through one fresh worker per series and writes the baseline to the run's
sealed stream; the caller gets back only a receipt. The worker and its proxy
are tested in ``tests/arena/test_sealed_worker.py``; toy updaters live in
:mod:`tests.arena.sealed_toys`.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import multiprocessing
import os
import subprocess
import sys
import time
from datetime import date
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.arena.realdata.fetch import fetch
from evolvets.arena.realdata.prepare import prepare
from evolvets.arena.sealed.client import (
    SealedError,
    SealedPaths,
    SealedReceipt,
    SealedRequest,
    evaluate_sealed,
)
from evolvets.arena.sealed.pools import build_sealed_cache, init_salt, load_sealed_pool
from evolvets.arena.versions import protocol_versions
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import Baseline, Commitment, RunEnd, RunStart
from evolvets.ledger.schema import Budget
from evolvets.ledger.store import LedgerReader, LedgerWriter
from evolvets.ledger.validate import validate_run
from tests.arena.sealed_toys import EqualWeights, OneHotNaive
from tests.arena.test_dev import ROWS
from tests.arena.test_realdata_prepare import (
    _build_etth1_fixture,
    _install,
    _processed,
    _record_opens,
)

TOYS = "tests.arena.sealed_toys"
SRC = Path(__file__).parents[2] / "src" / "evolvets"
IN_PROCESS: dict[str, type] = {
    f"{TOYS}:EqualWeights": EqualWeights,
    f"{TOYS}:OneHotNaive": OneHotNaive,
}
TOY_REFS = list(IN_PROCESS)


@pytest.fixture(scope="module")
def real_env(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """The 1200-row fixture ETTh1, prepared, with its sealed cache built."""
    base = tmp_path_factory.mktemp("server-real-env")
    etth1 = _build_etth1_fixture(base / "fixtures", rows=ROWS)
    monkeypatch = pytest.MonkeyPatch()
    try:
        _install(monkeypatch, etth1=etth1)
        data_root, manifest_dir = base / "data", base / "manifests"
        fetch(etth1, data_root, manifest_dir, today=date(2026, 10, 11))
        prepare("etth1", data_root, manifest_dir)
        sealed_root = base / "sealed_cache"
        build_sealed_cache("etth1", data_root, manifest_dir, sealed_root)
        return data_root, manifest_dir, sealed_root
    finally:
        monkeypatch.undo()


@pytest.fixture(scope="module")
def synthetic_env(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """A salted synthetic sealed pool with one world per family, its cache built."""
    base = tmp_path_factory.mktemp("server-synthetic-env")
    data_root, manifest_dir = base / "data", base / "manifests"
    init_salt(data_root, manifest_dir, per_family=1)
    sealed_root = base / "sealed_cache"
    build_sealed_cache("synthetic-sealed", data_root, manifest_dir, sealed_root)
    return data_root, manifest_dir, sealed_root


def _paths(env: tuple[Path, Path, Path]) -> SealedPaths:
    """The caller-side paths for one sealed environment."""
    data_root, manifest_dir, sealed_root = env
    return SealedPaths(
        data_root=data_root, manifest_dir=manifest_dir, sealed_cache_root=sealed_root
    )


def _request(updater_ref: str, config: dict[str, Any] | None = None) -> SealedRequest:
    """A valid request for ``updater_ref`` over the real ETTh1 sealed pool."""
    return SealedRequest(
        dataset="etth1",
        updater_id="toy",
        updater_ref=updater_ref,
        config=dict(config or {}),
        seed=3,
    )


def _begin_run(
    ledger_root: Path,
    run_id: str,
    *,
    first: str = "run_start",
    arena: str | None = None,
    run_end: bool = False,
) -> None:
    """Write a main stream that satisfies, or deliberately breaks, one precondition."""
    blobs = BlobStore(ledger_root / run_id / "blobs")
    versions = protocol_versions() if arena is None else {**protocol_versions(), "arena": arena}
    with LedgerWriter(ledger_root, run_id, "main") as writer:
        if first == "run_start":
            writer.append(
                "run_start",
                RunStart(
                    condition="sealed-test",
                    seed=1,
                    requires_predictions=True,
                    config_blob=blobs.put(b"run config"),
                    git_commit="abc123",
                    protocol_versions=versions,
                    budget_limits=Budget(
                        llm_input_tokens=1e6, wall_seconds=3600.0, evaluations=100
                    ),
                ),
            )
        else:
            writer.append(
                "commitment",
                Commitment(
                    subject="not a run_start",
                    content_hash=blobs.put(b"note"),
                    external_timestamp=None,
                ),
            )
        if run_end:
            writer.append("run_end", RunEnd(status="completed", summary={}))


def _baseline(ledger_root: Path, run_id: str) -> Baseline:
    """The last baseline of the run's sealed stream."""
    records = list(LedgerReader(ledger_root, run_id).records(["sealed"]))
    assert records, "expected at least one sealed-stream record"
    assert isinstance(records[-1].payload, Baseline)
    return records[-1].payload


def _blob(ledger_root: Path, run_id: str, baseline: Baseline) -> dict[str, Any]:
    """The parsed per-task blob of ``baseline``."""
    assert baseline.per_task_blob is not None
    raw = BlobStore(ledger_root / run_id / "blobs").get(baseline.per_task_blob)
    obj: Any = json.loads(raw.decode("utf-8"))
    return obj


def _evaluate(
    env: tuple[Path, Path, Path],
    updater_ref: str,
    *,
    ledger_root: Path,
    run_id: str,
    config: dict[str, Any] | None = None,
    timeout: float = 600.0,
) -> SealedReceipt:
    """One sealed evaluation of a toy over ``env``, with a valid run begun."""
    _begin_run(ledger_root, run_id)
    return evaluate_sealed(
        _request(updater_ref, config),
        ledger_root=ledger_root,
        run_id=run_id,
        paths=_paths(env),
        series_timeout_seconds=timeout,
    )


# --- Test 1: the same numbers as an in-process replay ---------------------------------------------


@pytest.mark.parametrize("ref", TOY_REFS)
def test_same_numbers_as_in_process_real(
    ref: str, real_env: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """A sealed run's metrics and blob equal an in-process replay's, exactly."""
    ledger_root = tmp_path / "ledgers"
    receipt = _evaluate(real_env, ref, ledger_root=ledger_root, run_id="run-1")
    assert receipt.status == "ok"
    baseline = _baseline(ledger_root, "run-1")
    data_root, manifest_dir, sealed_root = real_env
    pool = load_sealed_pool("etth1", data_root, manifest_dir, sealed_root)
    expected = replay_dataset(IN_PROCESS[ref], {}, 3, pool)
    assert baseline.metrics == expected.metrics
    assert _blob(ledger_root, "run-1", baseline)["result"] == result_to_obj(expected)


@pytest.mark.parametrize("ref", TOY_REFS)
def test_same_numbers_as_in_process_synthetic(
    ref: str, synthetic_env: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """The salted synthetic pool scores identically through the server."""
    ledger_root = tmp_path / "ledgers"
    _begin_run(ledger_root, "run-1")
    receipt = evaluate_sealed(
        dataclasses.replace(_request(ref), dataset="synthetic-sealed"),
        ledger_root=ledger_root,
        run_id="run-1",
        paths=_paths(synthetic_env),
    )
    assert receipt.status == "ok"
    baseline = _baseline(ledger_root, "run-1")
    data_root, manifest_dir, sealed_root = synthetic_env
    pool = load_sealed_pool("synthetic-sealed", data_root, manifest_dir, sealed_root)
    expected = replay_dataset(IN_PROCESS[ref], {}, 3, pool)
    assert baseline.metrics == expected.metrics
    assert _blob(ledger_root, "run-1", baseline)["result"] == result_to_obj(expected)


# --- Test 2: the record ---------------------------------------------------------------------------


def test_the_record(real_env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Exactly one well-formed baseline per request, chained by seq."""
    ledger_root = tmp_path / "ledgers"
    config: dict[str, Any] = {"alpha": 0.5, "rounds": 3, "note": "x", "strict": True}
    receipt = _evaluate(
        real_env, f"{TOYS}:EqualWeights", ledger_root=ledger_root, run_id="run-1", config=config
    )
    assert receipt.status == "ok"
    reader = LedgerReader(ledger_root, "run-1")
    records = list(reader.records(["sealed"]))
    assert len(records) == 1
    record = records[0]
    assert record.kind == "baseline"
    assert isinstance(record.payload, Baseline)
    assert record.payload.pool == "sealed"
    assert record.payload.updater_id == "toy"
    assert record.payload.hyperparams == config
    assert record.budget.evaluations == 1
    assert record.budget.wall_seconds > 0
    assert reader.verify(check_blobs=True).ok
    assert validate_run(list(reader.records())) == []
    second = evaluate_sealed(
        _request(f"{TOYS}:EqualWeights", config),
        ledger_root=ledger_root,
        run_id="run-1",
        paths=_paths(real_env),
    )
    assert second.status == "ok"
    records = list(LedgerReader(ledger_root, "run-1").records(["sealed"]))
    assert [entry.seq for entry in records] == [0, 1]


# --- Test 3: the receipt only ---------------------------------------------------------------------


def test_the_reply_is_a_receipt_only(
    real_env: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SealedReceipt has exactly two fields; the raw reply exactly those two keys."""
    assert [field.name for field in dataclasses.fields(SealedReceipt)] == [
        "record_id",
        "status",
    ]
    raw_replies: list[bytes] = []
    real_recv_bytes = Connection.recv_bytes

    def _spy(conn: Connection, *args: Any) -> bytes:
        raw = real_recv_bytes(conn, *args)
        raw_replies.append(raw)
        return raw

    monkeypatch.setattr(Connection, "recv_bytes", _spy)
    ledger_root = tmp_path / "ledgers"
    receipt = _evaluate(real_env, f"{TOYS}:EqualWeights", ledger_root=ledger_root, run_id="run-1")
    assert len(raw_replies) == 1
    reply: Any = json.loads(raw_replies[0].decode("utf-8"))
    assert set(reply) == {"record_id", "status"}
    assert reply == {"record_id": receipt.record_id, "status": receipt.status}


# --- Test 4: the caller never touches sealed data -------------------------------------------------


def test_the_caller_never_touches_sealed_data(
    real_env: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """During evaluate_sealed the caller opens no sealed file, salt or sealed cache."""
    ledger_root = tmp_path / "ledgers"
    data_root, _manifest_dir, sealed_root = real_env
    opened: list[str] = []
    _record_opens(monkeypatch, opened)
    receipt = _evaluate(real_env, f"{TOYS}:EqualWeights", ledger_root=ledger_root, run_id="run-1")
    assert receipt.status == "ok"
    assert str(_processed(data_root, "etth1", "sealed")) not in opened
    assert not any(path.startswith(str(data_root / "sealed")) for path in opened)
    assert not any(path.startswith(str(sealed_root)) for path in opened)


# --- Timeouts end to end --------------------------------------------------------------------------


def test_timeout_end_to_end(real_env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """A series past its deadline becomes a timeout baseline, not a crash."""
    ledger_root = tmp_path / "ledgers"
    spy = tmp_path / "sleep.jsonl"
    started = time.monotonic()
    receipt = _evaluate(
        real_env,
        f"{TOYS}:sleeping_at_step3",
        ledger_root=ledger_root,
        run_id="run-1",
        config={"spy_path": str(spy)},
        timeout=2.0,
    )
    assert time.monotonic() - started < 60.0
    assert receipt.status == "timeout"
    baseline = _baseline(ledger_root, "run-1")
    assert baseline.error is not None
    assert baseline.error.startswith("series etth1/HUFL exceeded")
    assert baseline.metrics == {}
    assert _blob(ledger_root, "run-1", baseline)["result"]["failed_step"] == 3


# --- Test 10: refusals ----------------------------------------------------------------------------


REFUSALS = [
    ("no-main", "run_start"),
    ("not-run-start", "not a run_start"),
    ("arena-0", "protocol versions"),
    ("run-end", "run_end"),
]


def _begin_broken_run(ledger_root: Path, run_id: str, mode: str) -> None:
    """A main stream missing exactly one of the server's preconditions."""
    if mode == "no-main":
        return  # no ledger at all for this run
    if mode == "not-run-start":
        _begin_run(ledger_root, run_id, first="other")
    elif mode == "arena-0":
        _begin_run(ledger_root, run_id, arena="0")
    elif mode == "run-end":
        _begin_run(ledger_root, run_id, run_end=True)
    else:  # pragma: no cover - the parametrize list is exhaustive
        raise AssertionError(mode)


@pytest.mark.parametrize(("mode", "needle"), REFUSALS)
def test_refusals(
    mode: str,
    needle: str,
    real_env: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """Each broken precondition is refused: SealedError, no record, no lock."""
    ledger_root = tmp_path / "ledgers"
    run_id = f"run-{mode}"
    _begin_broken_run(ledger_root, run_id, mode)
    with pytest.raises(SealedError) as excinfo:
        evaluate_sealed(
            _request(f"{TOYS}:EqualWeights"),
            ledger_root=ledger_root,
            run_id=run_id,
            paths=_paths(real_env),
        )
    assert needle in str(excinfo.value)
    run_dir = ledger_root / run_id
    assert not (run_dir / "sealed.jsonl").exists()
    assert not (run_dir / "sealed.lock").exists()


def test_refusal_when_the_sealed_stream_is_locked(
    real_env: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """A second concurrent request on the same run fails fast and writes nothing."""
    ledger_root = tmp_path / "ledgers"
    _begin_run(ledger_root, "run-1")
    with LedgerWriter(ledger_root, "run-1", "sealed"):  # held open by the test
        with pytest.raises(SealedError) as excinfo:
            evaluate_sealed(
                _request(f"{TOYS}:EqualWeights"),
                ledger_root=ledger_root,
                run_id="run-1",
                paths=_paths(real_env),
            )
        assert "locked" in str(excinfo.value)
    assert not (ledger_root / "run-1" / "sealed.lock").exists()
    assert list(LedgerReader(ledger_root, "run-1").records(["sealed"])) == []


def test_caller_side_validation_spawns_nothing(
    real_env: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed requests are refused in the caller, before any process is spawned."""
    ledger_root = tmp_path / "ledgers"
    _begin_run(ledger_root, "run-1")
    spawns: list[tuple[Any, ...]] = []
    real_get_context = multiprocessing.get_context

    def _spy(*args: Any, **kwargs: Any) -> Any:
        spawns.append(args)
        return real_get_context(*args, **kwargs)

    monkeypatch.setattr(multiprocessing, "get_context", _spy)
    good = _request(f"{TOYS}:EqualWeights")
    bad: list[tuple[SealedRequest, float, str]] = [
        (dataclasses.replace(good, dataset="m4"), 1.0, "dataset"),
        (dataclasses.replace(good, updater_ref="no_colon_here"), 1.0, "updater_ref"),
        (dataclasses.replace(good, config={"x": float("nan")}), 1.0, "finite"),
        (dataclasses.replace(good, seed=-1), 1.0, "seed"),
        (good, 0, "series_timeout_seconds"),
    ]
    for request, timeout, needle in bad:
        with pytest.raises(SealedError) as excinfo:
            evaluate_sealed(
                request,
                ledger_root=ledger_root,
                run_id="run-1",
                paths=_paths(real_env),
                series_timeout_seconds=timeout,
            )
        assert needle in str(excinfo.value), (needle, str(excinfo.value))
    assert spawns == []


# --- Test 11: server failures ---------------------------------------------------------------------


def test_server_failure_before_the_pool_loads(tmp_path: Path) -> None:
    """A missing sealed cache: exit 2, a log entry, no record, no caller path leak."""
    ledger_root = tmp_path / "ledgers"
    _begin_run(ledger_root, "run-1")
    paths = SealedPaths(
        data_root=tmp_path / "data",
        manifest_dir=tmp_path / "manifests",
        sealed_cache_root=tmp_path / "missing_sealed_cache",
    )
    with pytest.raises(SealedError) as excinfo:
        evaluate_sealed(
            _request(f"{TOYS}:EqualWeights"),
            ledger_root=ledger_root,
            run_id="run-1",
            paths=paths,
        )
    message = str(excinfo.value)
    assert "server.log" in message
    assert str(paths.sealed_cache_root) not in message
    assert list(LedgerReader(ledger_root, "run-1").records(["sealed"])) == []
    log = (tmp_path / "data" / "sealed" / "server.log").read_text(encoding="utf-8")
    assert "ArenaError" in log
    assert "run-1" in log
    assert "toy" in log
    assert not (ledger_root / "run-1" / "sealed.lock").exists()


def test_server_killed_mid_run(real_env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """A killed server leaves a verifiable stream, a breakable lock and no live worker."""
    ledger_root = tmp_path / "ledgers"
    spy = tmp_path / "spy.jsonl"
    with pytest.raises(SealedError):
        _evaluate(
            real_env,
            f"{TOYS}:KillServer",
            ledger_root=ledger_root,
            run_id="run-1",
            config={"spy_path": str(spy)},
        )
    assert LedgerReader(ledger_root, "run-1").verify().ok
    LedgerWriter.break_stale_lock(ledger_root, "run-1", "sealed")
    pid = json.loads(spy.read_text(encoding="utf-8").splitlines()[0])["pid"]
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return  # the orphaned worker saw EOF and exited
        time.sleep(0.05)
    pytest.fail(f"the orphaned worker pid {pid} is still alive after 10 s")


# --- Test 12: import isolation --------------------------------------------------------------------


def _evolvets_modules(import_line: str) -> list[str]:
    """The evolvets modules a fresh interpreter loads for ``import_line``."""
    code = (
        "import sys, json; "
        + import_line
        + "; print(json.dumps(sorted(m for m in sys.modules if m.startswith('evolvets'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    modules: Any = json.loads(out.stdout)
    return modules


def test_client_imports_no_data_path() -> None:
    """Importing the client loads no module that can read sealed data."""
    modules = _evolvets_modules("import evolvets.arena.sealed.client")
    assert "evolvets.arena.sealed.pools" not in modules
    assert "evolvets.arena.realdata.prepare" not in modules
    assert "evolvets.cache.store" not in modules


def test_worker_imports_only_updater_interfaces() -> None:
    """Importing the worker loads nothing from evolvets but the updater contract."""
    modules = _evolvets_modules("import evolvets.arena.sealed.worker")
    assert modules == [
        "evolvets",
        "evolvets.arena",
        "evolvets.arena.sealed",
        "evolvets.arena.sealed.worker",
        "evolvets.updater",
        "evolvets.updater.interfaces",
    ]


# --- Test 13: AST guards --------------------------------------------------------------------------


def _callee_name(func: ast.expr) -> str | None:
    """The bare name a call target is invoked by, if it is one."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def test_ast_guards() -> None:
    """No pickle, no recv, spawn only, and the worker imports only updater.interfaces."""
    sealed_dir = SRC / "arena" / "sealed"
    modules = sorted(sealed_dir.glob("*.py"))
    assert {path.name for path in modules} >= {
        "__init__.py",
        "client.py",
        "names.py",
        "pools.py",
        "remote.py",
        "server.py",
        "worker.py",
    }
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert not any(
                    alias.name == "pickle" or alias.name.startswith("pickle.")
                    for alias in node.names
                ), f"{path.name} imports pickle"
            if isinstance(node, ast.ImportFrom):
                target = node.module or ""
                assert not (target == "pickle" or target.startswith("pickle.")), (
                    f"{path.name} imports pickle"
                )
            if isinstance(node, ast.Call):
                name = _callee_name(node.func)
                assert name != "recv", f"{path.name} calls .recv(); only recv_bytes is allowed"
                if name == "get_context":
                    assert node.args and isinstance(node.args[0], ast.Constant), (
                        f"{path.name} calls get_context without a literal method"
                    )
                    assert node.args[0].value == "spawn", f"{path.name} spawns a non-spawn context"
    worker = ast.parse((sealed_dir / "worker.py").read_text(encoding="utf-8"), filename="worker.py")
    for node in ast.walk(worker):
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and (node.module == "evolvets" or node.module.startswith("evolvets."))
        ):
            assert node.module == "evolvets.updater.interfaces", (
                f"worker.py imports {node.module}; only evolvets.updater.interfaces is allowed"
            )
        if isinstance(node, ast.Import):
            assert not any(
                alias.name == "evolvets" or alias.name.startswith("evolvets.")
                for alias in node.names
            ), "worker.py must import no whole evolvets module"


# --- Test 14: determinism -------------------------------------------------------------------------


def test_determinism(real_env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Two runs of the same request give equal metrics and equal blob bytes."""
    ledger_root = tmp_path / "ledgers"
    metrics: list[dict[str, float]] = []
    blobs: list[bytes] = []
    for run_id in ("run-a", "run-b"):
        receipt = _evaluate(real_env, f"{TOYS}:OneHotNaive", ledger_root=ledger_root, run_id=run_id)
        assert receipt.status == "ok"
        baseline = _baseline(ledger_root, run_id)
        metrics.append(baseline.metrics)
        assert baseline.per_task_blob is not None
        blobs.append(BlobStore(ledger_root / run_id / "blobs").get(baseline.per_task_blob))
    assert metrics[0] == metrics[1]
    assert blobs[0] == blobs[1]
