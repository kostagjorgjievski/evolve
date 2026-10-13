"""The sealed oracle: the server computes the oracles of a sealed pool and records them.

TASK_010 PR C, tests 1 to 7 and 9: ``evaluate_sealed_oracle`` spawns the same
sealed server as ``evaluate_sealed``, which loads the sealed pool, runs
``oracle_dataset`` (no worker, no untrusted code) and appends a ``baseline``
with ``updater_id = "oracle"`` that charges wall-clock and no evaluation. The
numbers equal an in-process ``oracle_dataset`` exactly; the caller still
learns only a receipt and touches no sealed file.
"""

from __future__ import annotations

import json
import multiprocessing
import subprocess
import sys
from datetime import date, datetime, timedelta
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.oracles import oracle_dataset, oracle_to_obj
from evolvets.arena.realdata.fetch import fetch
from evolvets.arena.realdata.prepare import prepare
from evolvets.arena.realdata.sources import Source
from evolvets.arena.sealed import server
from evolvets.arena.sealed.client import (
    SealedError,
    SealedOracleRequest,
    SealedPaths,
    SealedReceipt,
    SealedRequest,
    evaluate_sealed,
    evaluate_sealed_oracle,
)
from evolvets.arena.sealed.pools import (
    build_sealed_cache,
    build_sealed_foundation_cache,
    init_salt,
    load_sealed_pool,
)
from evolvets.foundation.fake import FAKE_SPEC, FakeAdapter
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import Baseline
from evolvets.ledger.store import LedgerReader, LedgerWriter
from evolvets.ledger.validate import validate_run
from tests.arena.test_dev import ROWS
from tests.arena.test_realdata_prepare import (
    _ETTH1_HEADER,
    _TS_FORMAT,
    _build_etth1_fixture,
    _install,
    _processed,
    _record_opens,
)
from tests.arena.test_sealed_server import (
    REFUSALS,
    _begin_broken_run,
    _begin_run,
    _blob,
    _paths,
)

TOYS = "tests.arena.sealed_toys"


@pytest.fixture(scope="module")
def real_env(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """The 1200-row fixture ETTh1, prepared, with its sealed cache built."""
    base = tmp_path_factory.mktemp("oracle-real-env")
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
    base = tmp_path_factory.mktemp("oracle-synthetic")
    data_root, manifest_dir = base / "data", base / "manifests"
    init_salt(data_root, manifest_dir, per_family=1)
    sealed_root = base / "sealed_cache"
    build_sealed_cache("synthetic-sealed", data_root, manifest_dir, sealed_root)
    return data_root, manifest_dir, sealed_root


@pytest.fixture(scope="module")
def flat_warmup_env(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """Fixture ETTh1 whose values are constant over the whole warmup segment.

    The pool loads (pool loading never computes the MASE scale) but the warmup
    scale is zero, so ``oracle_dataset`` raises ``ArenaError`` inside the
    server. Values drift again from index 600, well before the sealed segment,
    so the experts and the cache build see varying data where it is scored.
    """
    base = tmp_path_factory.mktemp("oracle-flat-warmup")
    fixtures = base / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    moment = datetime(2016, 7, 1)
    lines = [_ETTH1_HEADER]
    for index in range(ROWS):
        drift = 0.0 if index < 600 else index / 100
        lines.append(
            moment.strftime(_TS_FORMAT)
            + ","
            + ",".join(str(column + drift) for column in range(1, 8))
        )
        moment += timedelta(hours=1)
    etth1_file = fixtures / "ETTh1.csv"
    etth1_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    etth1 = Source("etth1", etth1_file.as_uri(), etth1_file.name, "MIT")
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


def _oracle_request(dataset: str = "etth1", **kwargs: Any) -> SealedOracleRequest:
    """A valid oracle request for ``dataset``."""
    return SealedOracleRequest(dataset=dataset, **kwargs)


def _baseline(ledger_root: Path, run_id: str) -> Baseline:
    """The last baseline of the run's sealed stream."""
    records = list(LedgerReader(ledger_root, run_id).records(["sealed"]))
    assert records, "expected at least one sealed-stream record"
    assert isinstance(records[-1].payload, Baseline)
    return records[-1].payload


def _evaluate_oracle(
    env: tuple[Path, Path, Path],
    *,
    ledger_root: Path,
    run_id: str,
    request: SealedOracleRequest | None = None,
) -> SealedReceipt:
    """One oracle request over ``env``, with a valid run begun."""
    _begin_run(ledger_root, run_id)
    return evaluate_sealed_oracle(
        request if request is not None else _oracle_request(dataset=_dataset_of(env)),
        ledger_root=ledger_root,
        run_id=run_id,
        paths=_paths(env),
    )


def _dataset_of(env: tuple[Path, Path, Path]) -> str:
    """The dataset name of one sealed environment."""
    data_root, _manifest_dir, _sealed_root = env
    if (data_root / "sealed" / "synthetic.salt").exists():
        return "synthetic-sealed"
    return "etth1"


# --- Test 1: the same numbers as in-process -------------------------------------------------------


@pytest.mark.parametrize("env_name", ["real", "synthetic"])
def test_same_numbers_as_in_process(
    env_name: str,
    real_env: tuple[Path, Path, Path],
    synthetic_env: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """A sealed oracle run's metrics and blob equal an in-process oracle_dataset's."""
    env = real_env if env_name == "real" else synthetic_env
    dataset = _dataset_of(env)
    ledger_root = tmp_path / "ledgers"
    receipt = _evaluate_oracle(env, ledger_root=ledger_root, run_id="run-1")
    assert receipt.status == "ok"
    baseline = _baseline(ledger_root, "run-1")
    data_root, manifest_dir, sealed_root = env
    pool = load_sealed_pool(dataset, data_root, manifest_dir, sealed_root)
    expected = oracle_dataset(pool)
    assert baseline.metrics == expected.metrics
    assert _blob(ledger_root, "run-1", baseline)["result"] == oracle_to_obj(expected)


# --- Test 2: the record ---------------------------------------------------------------------------


def test_the_record(real_env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """One baseline per oracle request: updater_id 'oracle', no evaluation charged."""
    ledger_root = tmp_path / "ledgers"
    receipt = _evaluate_oracle(real_env, ledger_root=ledger_root, run_id="run-1")
    assert receipt.status == "ok"
    reader = LedgerReader(ledger_root, "run-1")
    records = list(reader.records(["sealed"]))
    assert len(records) == 1
    record = records[0]
    assert record.kind == "baseline"
    assert isinstance(record.payload, Baseline)
    assert record.payload.updater_id == "oracle"
    assert record.payload.hyperparams == {}
    assert record.payload.pool == "sealed"
    assert record.payload.error is None
    assert record.budget.evaluations == 0
    assert record.budget.wall_seconds > 0
    assert reader.verify(check_blobs=True).ok
    assert validate_run(list(reader.records())) == []


def test_updater_then_oracle_chain_by_seq(
    real_env: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """An updater request and an oracle request in one run give seq 0 and 1."""
    ledger_root = tmp_path / "ledgers"
    _begin_run(ledger_root, "run-1")
    updater = evaluate_sealed(
        SealedRequest(
            dataset="etth1",
            updater_id="toy",
            updater_ref=f"{TOYS}:EqualWeights",
            config={},
            seed=3,
        ),
        ledger_root=ledger_root,
        run_id="run-1",
        paths=_paths(real_env),
    )
    oracle = evaluate_sealed_oracle(
        _oracle_request("etth1"), ledger_root=ledger_root, run_id="run-1", paths=_paths(real_env)
    )
    assert updater.status == "ok"
    assert oracle.status == "ok"
    records = list(LedgerReader(ledger_root, "run-1").records(["sealed"]))
    assert [entry.seq for entry in records] == [0, 1]
    assert records[0].payload.updater_id == "toy"
    assert records[1].payload.updater_id == "oracle"


# --- Test 3: refusals -----------------------------------------------------------------------------


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
        evaluate_sealed_oracle(
            _oracle_request(), ledger_root=ledger_root, run_id=run_id, paths=_paths(real_env)
        )
    assert needle in str(excinfo.value)
    run_dir = ledger_root / run_id
    assert not (run_dir / "sealed.jsonl").exists()
    assert not (run_dir / "sealed.lock").exists()


def test_refusal_when_the_sealed_stream_is_locked(
    real_env: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """A concurrent oracle request on a locked sealed run fails fast, writing nothing."""
    ledger_root = tmp_path / "ledgers"
    _begin_run(ledger_root, "run-1")
    with LedgerWriter(ledger_root, "run-1", "sealed"):  # held open by the test
        with pytest.raises(SealedError) as excinfo:
            evaluate_sealed_oracle(
                _oracle_request(),
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
    """Malformed oracle requests are refused in the caller, before any spawn."""
    ledger_root = tmp_path / "ledgers"
    _begin_run(ledger_root, "run-1")
    spawns: list[tuple[Any, ...]] = []
    real_get_context = multiprocessing.get_context

    def _spy(*args: Any, **kwargs: Any) -> Any:
        spawns.append(args)
        return real_get_context(*args, **kwargs)

    monkeypatch.setattr(multiprocessing, "get_context", _spy)
    bad = [
        _oracle_request(dataset="m4"),
        _oracle_request(pool_spec=""),
        _oracle_request(foundation_models=()),
        _oracle_request(foundation_models=("", "fake")),
    ]
    for request in bad:
        with pytest.raises(SealedError):
            evaluate_sealed_oracle(
                request, ledger_root=ledger_root, run_id="run-1", paths=_paths(real_env)
            )
    assert spawns == []


# --- Test 4: server failures ----------------------------------------------------------------------


def test_server_failure_before_the_pool_loads(tmp_path: Path) -> None:
    """A missing sealed cache: exit 2, a log entry, no record."""
    ledger_root = tmp_path / "ledgers"
    _begin_run(ledger_root, "run-1")
    paths = SealedPaths(
        data_root=tmp_path / "data",
        manifest_dir=tmp_path / "manifests",
        sealed_cache_root=tmp_path / "missing_sealed_cache",
    )
    with pytest.raises(SealedError) as excinfo:
        evaluate_sealed_oracle(
            _oracle_request(), ledger_root=ledger_root, run_id="run-1", paths=paths
        )
    assert "server.log" in str(excinfo.value)
    assert list(LedgerReader(ledger_root, "run-1").records(["sealed"])) == []
    log = (tmp_path / "data" / "sealed" / "server.log").read_text(encoding="utf-8")
    assert "ArenaError" in log
    assert "run-1" in log
    assert "oracle" in log


def test_degenerate_scale_writes_a_failed_record(
    flat_warmup_env: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """A pool that loads but has a zero MASE scale: one failed record, a normal reply."""
    data_root, manifest_dir, sealed_root = flat_warmup_env
    pool = load_sealed_pool("etth1", data_root, manifest_dir, sealed_root)  # loads; no scale yet
    assert len(pool.series) > 0
    ledger_root = tmp_path / "ledgers"
    receipt = _evaluate_oracle(flat_warmup_env, ledger_root=ledger_root, run_id="run-1")
    assert receipt.status == "failed"
    baseline = _baseline(ledger_root, "run-1")
    assert baseline.updater_id == "oracle"
    assert baseline.error is not None
    assert baseline.error.startswith("sealed server error: ArenaError")
    assert baseline.per_task_blob is None
    assert baseline.metrics == {}
    reader = LedgerReader(ledger_root, "run-1")
    assert len(list(reader.records(["sealed"]))) == 1
    assert reader.verify(check_blobs=True).ok
    assert validate_run(list(reader.records())) == []
    log = (data_root / "sealed" / "server.log").read_text(encoding="utf-8")
    assert "ArenaError" in log


# --- Test 5: the caller never touches sealed data -------------------------------------------------


def test_the_caller_never_touches_sealed_data(
    real_env: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """During evaluate_sealed_oracle the caller opens no sealed file, salt or sealed cache."""
    ledger_root = tmp_path / "ledgers"
    data_root, _manifest_dir, sealed_root = real_env
    opened: list[str] = []
    _record_opens(monkeypatch, opened)
    receipt = _evaluate_oracle(real_env, ledger_root=ledger_root, run_id="run-1")
    assert receipt.status == "ok"
    assert str(_processed(data_root, "etth1", "sealed")) not in opened
    assert not any(path.startswith(str(data_root / "sealed")) for path in opened)
    assert not any(path.startswith(str(sealed_root)) for path in opened)


def test_the_reply_is_a_receipt_only(
    real_env: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller's raw reply has exactly the keys record_id and status."""
    raw_replies: list[bytes] = []
    real_recv_bytes = Connection.recv_bytes

    def _spy(conn: Connection, *args: Any) -> bytes:
        raw = real_recv_bytes(conn, *args)
        raw_replies.append(raw)
        return raw

    monkeypatch.setattr(Connection, "recv_bytes", _spy)
    ledger_root = tmp_path / "ledgers"
    receipt = _evaluate_oracle(real_env, ledger_root=ledger_root, run_id="run-1")
    assert len(raw_replies) == 1
    reply: Any = json.loads(raw_replies[0].decode("utf-8"))
    assert set(reply) == {"record_id", "status"}
    assert reply == {"record_id": receipt.record_id, "status": receipt.status}


# --- Test 6: the updater request is unchanged -----------------------------------------------------


@pytest.mark.parametrize(
    "request_obj",
    [
        {"dataset": "etth1"},  # no kind at all
        {"kind": "nope", "dataset": "etth1"},  # an unknown kind
        # kind "updater" with the oracle key set: the wrong shape for its kind
        {"kind": "updater", "dataset": "etth1", "pool_spec": "simple", "foundation_models": None},
    ],
)
def test_a_malformed_request_kind_exits_two(request_obj: dict[str, Any], tmp_path: Path) -> None:
    """A request with a missing, unknown or mismatched kind makes the server exit 2."""
    parent_conn, child_conn = multiprocessing.Pipe(duplex=True)
    request_json = json.dumps(request_obj).encode("utf-8")
    paths_json = json.dumps(
        {
            "data_root": str(tmp_path / "data"),
            "manifest_dir": str(tmp_path / "manifests"),
            "sealed_cache_root": str(tmp_path / "sealed_cache"),
        }
    ).encode("utf-8")
    try:
        with pytest.raises(SystemExit) as excinfo:
            server._serve(
                child_conn, request_json, tmp_path / "ledgers", "run-1", paths_json, 600.0
            )
        assert excinfo.value.code == 2
    finally:
        parent_conn.close()
        child_conn.close()


# --- Test 7: the pooled spec ----------------------------------------------------------------------


def test_pooled_oracle_names_the_foundation_expert(
    synthetic_env: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """simple+foundation with the fake model: expert_mase/fake is in the record."""
    data_root, manifest_dir, sealed_root = synthetic_env
    build_sealed_foundation_cache(
        "synthetic-sealed", FakeAdapter(), FAKE_SPEC, data_root, manifest_dir, sealed_root, 32
    )
    ledger_root = tmp_path / "ledgers"
    receipt = _evaluate_oracle(
        synthetic_env,
        ledger_root=ledger_root,
        run_id="run-1",
        request=_oracle_request(
            "synthetic-sealed", pool_spec="simple+foundation", foundation_models=("fake",)
        ),
    )
    assert receipt.status == "ok"
    baseline = _baseline(ledger_root, "run-1")
    assert "expert_mase/fake" in baseline.metrics
    pool = load_sealed_pool(
        "synthetic-sealed",
        data_root,
        manifest_dir,
        sealed_root,
        pool_spec="simple+foundation",
        foundation_models=("fake",),
    )
    assert baseline.metrics == oracle_dataset(pool).metrics


# --- Test 9: import isolation ---------------------------------------------------------------------


def test_client_imports_no_data_path_or_oracles() -> None:
    """Importing the client loads no module that can read sealed data, nor the oracles."""
    code = (
        "import sys, json; import evolvets.arena.sealed.client; "
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('evolvets'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    modules: Any = json.loads(out.stdout)
    for forbidden in (
        "evolvets.arena.sealed.pools",
        "evolvets.arena.oracles",
        "evolvets.arena.realdata.prepare",
        "evolvets.cache.store",
    ):
        assert forbidden not in modules


# --- Determinism ----------------------------------------------------------------------------------


def test_determinism(real_env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Two oracle runs give equal metrics and equal blob bytes."""
    ledger_root = tmp_path / "ledgers"
    metrics: list[dict[str, float]] = []
    blobs: list[bytes] = []
    for run_id in ("run-a", "run-b"):
        receipt = _evaluate_oracle(real_env, ledger_root=ledger_root, run_id=run_id)
        assert receipt.status == "ok"
        baseline = _baseline(ledger_root, run_id)
        metrics.append(baseline.metrics)
        assert baseline.per_task_blob is not None
        blobs.append(BlobStore(ledger_root / run_id / "blobs").get(baseline.per_task_blob))
    assert metrics[0] == metrics[1]
    assert blobs[0] == blobs[1]
