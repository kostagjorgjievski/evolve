"""Sealed worker tests: the per-series worker process and its server-side proxy.

PR C part 1 (tests 5 to 9 of the TASK_008 PR C list): :mod:`remote.py` and
:mod:`worker.py`, driven in-process by handing :func:`replay_dataset` the
:class:`RemoteUpdaterFactory` over the real sealed pool. The sealed server
process and its client come in the follow-up PR ``feat/arena-sealed-server``.
Toy updaters live in :mod:`tests.arena.sealed_toys`.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path

import pytest

from evolvets.arena.evaluate import ReplayResult, replay_dataset
from evolvets.arena.realdata.fetch import fetch
from evolvets.arena.realdata.prepare import prepare
from evolvets.arena.sealed.pools import build_sealed_cache, load_sealed_pool
from evolvets.arena.sealed.remote import RemoteUpdaterFactory, SeriesTimeout
from tests.arena.sealed_toys import Spy
from tests.arena.test_dev import ETTH1_IDS, ROWS
from tests.arena.test_realdata_prepare import _build_etth1_fixture, _install

TOYS = "tests.arena.sealed_toys"

FAILURE_CASES = [
    ("failing_init_series0", "failed", 0, None, ("ValueError",)),
    ("failing_update_series2_step7", "failed", 2, 7, ("RuntimeError",)),
    ("invalid_weights_series0_step1", "invalid_weights", 0, 1, ()),
    ("nan_weights_series0_step1", "invalid_weights", 0, 1, ()),
    ("set_weights_series0_step1", "failed", 0, 1, ()),
    ("exit3_series0_step5", "failed", 0, 5, ("exit code 3",)),
    ("big_weights_series0_step1", "failed", 0, 1, ()),
]


@pytest.fixture(scope="module")
def real_env(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """The 1200-row fixture ETTh1, prepared, with its sealed cache built."""
    base = tmp_path_factory.mktemp("real-env")
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


def _drive(
    env: tuple[Path, Path, Path],
    updater_ref: str,
    config: dict[str, object] | None = None,
    timeout_seconds: float = 600.0,
) -> ReplayResult:
    """Replay one toy updater over the sealed pool through the remote factory.

    The factory's shutdown runs in a ``finally``, so the last series' worker is
    reaped however the replay ends — the sealed server does the same.
    """
    data_root, manifest_dir, sealed_root = env
    pool = load_sealed_pool("etth1", data_root, manifest_dir, sealed_root)
    factory = RemoteUpdaterFactory(
        [series.series_id for series in pool.series], updater_ref, timeout_seconds
    )
    try:
        result = replay_dataset(factory, dict(config or {}), 0, pool)
    finally:
        factory.shutdown()
    return result


# --- Test 5: the worker sees exactly the replay stream --------------------------------------------


def test_the_worker_sees_exactly_the_replay_stream(
    real_env: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """The worker's recording equals an in-process replay's, series by series."""
    remote_spy, local_spy = tmp_path / "remote.jsonl", tmp_path / "local.jsonl"
    _drive(real_env, f"{TOYS}:Spy", config={"spy_path": str(remote_spy)})
    data_root, manifest_dir, sealed_root = real_env
    pool = load_sealed_pool("etth1", data_root, manifest_dir, sealed_root)
    replay_dataset(Spy, {"spy_path": str(local_spy)}, 0, pool)
    remote = remote_spy.read_text(encoding="utf-8")
    assert len(remote.splitlines()) > 1000  # every origin of every series
    assert remote == local_spy.read_text(encoding="utf-8")


# --- Test 6: a fresh process per series -----------------------------------------------------------


def test_a_fresh_worker_process_per_series(
    real_env: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """One distinct pid per series, and the module-level init counter is 1 each time."""
    spy = tmp_path / "pids.jsonl"
    _drive(real_env, f"{TOYS}:PidCounter", config={"spy_path": str(spy)})
    lines = [json.loads(line) for line in spy.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == len(ETTH1_IDS)
    assert len({line["pid"] for line in lines}) == len(ETTH1_IDS)
    assert all(line["inits"] == 1 for line in lines)


# --- Test 7: timeouts -----------------------------------------------------------------------------


def test_timeout_during_update(real_env: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """A series that sleeps past its deadline: SeriesTimeout naming it, dead worker."""
    spy = tmp_path / "sleep.jsonl"
    started = time.monotonic()
    with pytest.raises(SeriesTimeout) as excinfo:
        _drive(
            real_env,
            f"{TOYS}:sleeping_at_step3",
            config={"spy_path": str(spy)},
            timeout_seconds=2.0,
        )
    assert time.monotonic() - started < 15.0
    assert excinfo.value.series_id == "etth1/HUFL"
    assert excinfo.value.step == 3
    pid = json.loads(spy.read_text(encoding="utf-8").splitlines()[0])["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_timeout_during_factory(real_env: tuple[Path, Path, Path]) -> None:
    """A toy that never finishes constructing: SeriesTimeout with step None."""
    with pytest.raises(SeriesTimeout) as excinfo:
        _drive(real_env, f"{TOYS}:sleeping_factory", timeout_seconds=2.0)
    assert excinfo.value.series_id == "etth1/HUFL"
    assert excinfo.value.step is None


# --- Test 8: updater faults become results --------------------------------------------------------


@pytest.mark.parametrize(("factory", "status", "series", "step", "needle"), FAILURE_CASES)
def test_updater_faults_become_results(
    real_env: tuple[Path, Path, Path],
    tmp_path: Path,
    factory: str,
    status: str,
    series: int,
    step: int | None,
    needle: tuple[str, ...],
) -> None:
    """Every updater fault becomes a replay result of the right status, series and step."""
    result = _drive(
        real_env, f"{TOYS}:{factory}", config={"spy_path": str(tmp_path / "notes.jsonl")}
    )
    assert result.status == status
    assert result.failed_series == ETTH1_IDS[series]
    assert result.failed_step == step
    if status == "failed":
        assert result.metrics == {}
    for text in needle:
        assert result.error is not None and text in result.error


def test_missing_module_fails_at_init(real_env: tuple[Path, Path, Path]) -> None:
    """An updater_ref naming a missing module: failed, step None, type name in the error."""
    result = _drive(real_env, "no.such.module:Thing")
    assert result.status == "failed"
    assert result.error is not None and "ModuleNotFoundError" in result.error
    assert result.failed_step is None
    assert result.failed_series == "etth1/HUFL"


# --- Test 9: no stray output ----------------------------------------------------------------------


def test_no_stray_output(
    real_env: tuple[Path, Path, Path], capfd: pytest.CaptureFixture[str]
) -> None:
    """A printing toy leaves no marker in the test process's captured output."""
    _drive(real_env, f"{TOYS}:PrintNoise")
    captured = capfd.readouterr()
    assert "sealed-toy-marker" not in captured.out
    assert "sealed-toy-marker" not in captured.err
