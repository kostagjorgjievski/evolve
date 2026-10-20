"""CLI tests for ``evolvets arena``: dev replays and oracles (TASK_010 PR C, test 8).

Both commands are dev only: the dataset choices are the dev datasets (the
cache datasets, the scan presets of TASK_015 and the holdout replicates of
TASK_016 PR C), so a sealed name never reaches the pool loader, and neither
command takes a sealed path or a ledger path. One canonical-JSON line per
run, with the per-series detail stripped; an ``ArenaError`` is a FAIL line
and exit 1, while a failed updater is a result like any other and exits 0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.dev import load_dev_pool
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.arena.oracles import oracle_dataset
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.runner import cli
from evolvets.updater.baselines import DEFAULT_CONFIGS, hedge


@pytest.fixture(scope="module")
def synthetic_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A synthetic-dev cache with one series per family, and the paths to load it."""
    base = tmp_path_factory.mktemp("arena-cli")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-dev", base, base, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    return {
        "base": base,
        "cache_root": cache_root,
        "data_root": base,
        "manifest_dir": base,
    }


def _argv(env: dict[str, Path], *args: str) -> list[str]:
    """An ``arena`` command line pointed at the fixture cache."""
    return [
        "arena",
        *args,
        "--cache-root",
        str(env["cache_root"]),
        "--data-root",
        str(env["data_root"]),
        "--manifest-dir",
        str(env["manifest_dir"]),
    ]


def _run(env: dict[str, Path], capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, Any]:
    """Run one ``arena`` command; return its exit code and the parsed JSON line."""
    code = cli.main(_argv(env, *args))
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1, out
    return code, json.loads(out[0])


def _pool(env: dict[str, Path]) -> Any:
    """The dev pool the CLI loads for the fixture cache."""
    return load_dev_pool("synthetic-dev", env["cache_root"], env["data_root"], env["manifest_dir"])


def test_replay_prints_the_in_process_numbers(
    synthetic_env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """One JSON line: the replay's result without series, plus its wall-clock."""
    code, payload = _run(synthetic_env, capsys, "replay", "synthetic-dev", "--updater", "hedge")
    assert code == 0
    assert set(payload) == {"result", "wall_seconds"}
    assert isinstance(payload["wall_seconds"], float)
    result = payload["result"]
    assert "series" not in result
    expected = replay_dataset(hedge, DEFAULT_CONFIGS["hedge"], 0, _pool(synthetic_env))
    assert expected.status == "ok"
    expected_obj = result_to_obj(expected)
    del expected_obj["series"]
    assert result == expected_obj


def test_a_failed_updater_is_a_result_not_an_error(
    synthetic_env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """A config the updater refuses is printed as a failed result, exit 0."""
    code, payload = _run(
        synthetic_env,
        capsys,
        "replay",
        "synthetic-dev",
        "--updater",
        "hedge",
        "--config",
        '{"eta": 0}',
    )
    assert code == 0
    assert payload["result"]["status"] == "failed"
    assert payload["result"]["metrics"] == {}


def test_replay_rejects_bad_arguments(synthetic_env: dict[str, Path]) -> None:
    """An unknown updater, a non-object config and a sealed dataset name all exit non-zero."""
    for argv in (
        ["replay", "synthetic-dev", "--updater", "nope"],
        ["replay", "synthetic-dev", "--updater", "hedge", "--config", "[1]"],
        ["replay", "sealed", "--updater", "hedge"],
        ["replay", "synthetic-sealed", "--updater", "hedge"],
        ["oracle", "sealed"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(_argv(synthetic_env, *argv))
        assert excinfo.value.code != 0, argv


def test_oracle_prints_the_in_process_numbers(
    synthetic_env: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """One JSON line: the oracles without series, matching an in-process oracle_dataset."""
    code, payload = _run(synthetic_env, capsys, "oracle", "synthetic-dev")
    assert code == 0
    assert set(payload) == {"result", "wall_seconds"}
    assert "series" not in payload["result"]
    assert payload["result"]["metrics"] == oracle_dataset(_pool(synthetic_env)).metrics


def test_a_missing_cache_fails_cleanly(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An ArenaError from the pool loader is one FAIL line and exit 1."""
    code = cli.main(
        [
            "arena",
            "oracle",
            "synthetic-dev",
            "--cache-root",
            str(tmp_path / "missing"),
            "--data-root",
            str(tmp_path),
            "--manifest-dir",
            str(tmp_path),
        ]
    )
    assert code == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].startswith("FAIL synthetic-dev: ")


def test_oracle_over_a_scan_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``arena oracle`` over a scan preset succeeds (TASK_015 PR A, test 12)."""
    cache_root = tmp_path / "cache_store"
    source, series = public_inputs("synthetic-shift3", tmp_path, tmp_path, per_family=1)
    build_cache("synthetic-shift3", source, series, cache_root)
    code = cli.main(
        [
            "arena",
            "oracle",
            "synthetic-shift3",
            "--cache-root",
            str(cache_root),
            "--data-root",
            str(tmp_path),
            "--manifest-dir",
            str(tmp_path),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1, out
    payload = json.loads(out[0])
    assert set(payload) == {"result", "wall_seconds"}
    assert payload["result"]["dataset"] == "synthetic-shift3"
    assert "series" not in payload["result"]


def test_oracle_and_replay_over_a_holdout_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``arena oracle`` and ``arena replay`` accept a holdout (TASK_016 PR C, test 11)."""
    cache_root = tmp_path / "cache_store"
    source, series = public_inputs("synthetic-shift3-holdout", tmp_path, tmp_path, per_family=1)
    build_cache("synthetic-shift3-holdout", source, series, cache_root)
    argv = [
        "--cache-root",
        str(cache_root),
        "--data-root",
        str(tmp_path),
        "--manifest-dir",
        str(tmp_path),
    ]
    code = cli.main(["arena", "oracle", "synthetic-shift3-holdout", *argv])
    assert code == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1, out
    payload = json.loads(out[0])
    assert set(payload) == {"result", "wall_seconds"}
    assert payload["result"]["dataset"] == "synthetic-shift3-holdout"
    assert "series" not in payload["result"]

    code = cli.main(
        ["arena", "replay", "synthetic-shift3-holdout", "--updater", "fixed_share", *argv]
    )
    assert code == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1, out
    result = json.loads(out[0])["result"]
    assert result["status"] == "ok"
    assert result["updater_interface"] == "weights-v1"
