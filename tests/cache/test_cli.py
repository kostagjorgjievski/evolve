"""CLI tests for ``evolvets cache``: build and verify over fixture data."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from evolvets.arena.realdata.fetch import fetch
from evolvets.arena.realdata.prepare import prepare
from evolvets.cache.store import cache_dir
from evolvets.runner import cli
from tests.arena.test_realdata_prepare import _build_etth1_fixture, _install

_PUBLIC_ROWS = 344  # the fixture's sealed segment start
_START = 287  # first_origin(24)
_OK_LINE = re.compile(r"^OK (\S+): (\d+) series, (\d+) origins, cache [0-9a-f]{64}$")


@pytest.fixture()
def prepared_etth1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """Fetched and prepared fixture ETTh1 plus an empty cache root."""
    etth1 = _build_etth1_fixture(tmp_path / "fixtures")
    _install(monkeypatch, etth1=etth1)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    fetch(etth1, data_root, manifest_dir, today=date(2026, 10, 11))
    prepare("etth1", data_root, manifest_dir)
    return data_root, manifest_dir, tmp_path / "cache_store"


def _args(data_root: Path, manifest_dir: Path, cache_root: Path) -> list[str]:
    return [
        "--cache-root",
        str(cache_root),
        "--data-root",
        str(data_root),
        "--manifest-dir",
        str(manifest_dir),
    ]


def test_build_then_verify_etth1(
    prepared_etth1: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    data_root, manifest_dir, cache_root = prepared_etth1
    args = _args(data_root, manifest_dir, cache_root)

    assert cli.main(["cache", "build", "etth1", *args]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1
    match = _OK_LINE.match(out[0])
    assert match is not None, out[0]
    assert match.group(1, 2, 3) == ("etth1", "7", str(7 * (_PUBLIC_ROWS - _START)))

    assert cli.main(["cache", "verify", "etth1", *args]) == 0
    assert capsys.readouterr().out.splitlines() == out  # same line again

    # idempotent rebuild is an OK line, not a re-write
    assert cli.main(["cache", "build", "etth1", *args]) == 0
    assert capsys.readouterr().out.splitlines() == out


def test_verify_a_corrupt_cache_fails(
    prepared_etth1: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    data_root, manifest_dir, cache_root = prepared_etth1
    args = _args(data_root, manifest_dir, cache_root)
    assert cli.main(["cache", "build", "etth1", *args]) == 0
    capsys.readouterr()

    series_file = cache_dir(cache_root, "etth1") / "series" / "0000.f64"
    raw = series_file.read_bytes()
    series_file.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))
    assert cli.main(["cache", "verify", "etth1", *args]) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].startswith("FAIL etth1: ")

    series_file.write_bytes(raw)  # restored
    assert cli.main(["cache", "verify", "etth1", *args]) == 0


def test_build_synthetic_dev_needs_per_family(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_root = tmp_path / "cache_store"
    missing = ["cache", "build", "synthetic-dev", "--cache-root", str(cache_root)]
    assert cli.main(missing) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].startswith("FAIL synthetic-dev: per_family")
    assert not cache_root.exists()

    assert cli.main([*missing, "--per-family", "1"]) == 0
    out = capsys.readouterr().out.splitlines()
    match = _OK_LINE.match(out[0])
    assert match is not None, out[0]
    assert match.group(1, 2, 3) == ("synthetic-dev", "6", str(6 * (2000 - _START)))

    assert cli.main(["cache", "verify", "synthetic-dev", "--cache-root", str(cache_root)]) == 0
    assert capsys.readouterr().out.splitlines() == out


def test_build_then_verify_synthetic_shift1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A scan preset builds and verifies like synthetic-dev: one OK line, same again."""
    cache_root = tmp_path / "cache_store"
    assert (
        cli.main(
            [
                "cache",
                "build",
                "synthetic-shift1",
                "--cache-root",
                str(cache_root),
                "--per-family",
                "1",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out.splitlines()
    match = _OK_LINE.match(out[0])
    assert match is not None, out[0]
    assert match.group(1, 2, 3) == ("synthetic-shift1", "5", str(5 * (2000 - _START)))  # 5 families

    assert cli.main(["cache", "verify", "synthetic-shift1", "--cache-root", str(cache_root)]) == 0
    assert capsys.readouterr().out.splitlines() == out


# --- TASK_016 PR C: the holdout builds like any synthetic dataset (test 11) -----------------------


def test_build_then_verify_synthetic_shift2_holdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A holdout replicate builds and verifies: one OK line, five families, same again."""
    cache_root = tmp_path / "cache_store"
    assert (
        cli.main(
            [
                "cache",
                "build",
                "synthetic-shift2-holdout",
                "--cache-root",
                str(cache_root),
                "--per-family",
                "1",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out.splitlines()
    match = _OK_LINE.match(out[0])
    assert match is not None, out[0]
    assert match.group(1, 2, 3) == (
        "synthetic-shift2-holdout",
        "5",
        str(5 * (2000 - _START)),
    )  # the base's 5 families, indexes from 1000

    assert (
        cli.main(["cache", "verify", "synthetic-shift2-holdout", "--cache-root", str(cache_root)])
        == 0
    )
    assert capsys.readouterr().out.splitlines() == out


def test_unknown_dataset_is_rejected_by_argparse() -> None:
    for argv in (
        ["cache", "build", "nope"],
        ["cache", "verify", "nope"],
        ["cache", "build", "synthetic-shift9"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(argv)
        assert excinfo.value.code == 2
