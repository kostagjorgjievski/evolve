"""CLI tests for ``evolvets sealed``: init-salt and build-cache over fixture data."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from evolvets.arena.realdata.fetch import fetch
from evolvets.arena.realdata.prepare import prepare
from evolvets.arena.sealed.pools import salt_path
from evolvets.ledger.canonical import sha256_hex
from evolvets.runner import cli
from tests.arena.test_realdata_prepare import _build_etth1_fixture, _install, _processed

_ROWS = 1200  # the TASK_007 fixture: sealed [718, 926), reserve from 1094
_OK_HASH = re.compile(r"^OK init-salt: salt sha256 ([0-9a-f]{64}); manifest (\S+)$")
_OK_CACHE = re.compile(r"^OK etth1: (\d+) series, cache ([0-9a-f]{64})$")


@pytest.fixture()
def prepared_etth1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """Fetched and prepared 1200-row fixture ETTh1 plus an empty sealed cache root."""
    etth1 = _build_etth1_fixture(tmp_path / "fixtures", rows=_ROWS)
    _install(monkeypatch, etth1=etth1)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    fetch(etth1, data_root, manifest_dir, today=date(2026, 10, 11))
    prepare("etth1", data_root, manifest_dir)
    return data_root, manifest_dir, tmp_path / "sealed_cache"


def _build_args(data_root: Path, manifest_dir: Path, sealed_cache_root: Path) -> list[str]:
    return [
        "--data-root",
        str(data_root),
        "--manifest-dir",
        str(manifest_dir),
        "--sealed-cache-root",
        str(sealed_cache_root),
    ]


def test_init_salt_requires_per_family(tmp_path: Path) -> None:
    """Without --per-family argparse exits non-zero and nothing is written."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["sealed", "init-salt", "--data-root", str(tmp_path / "data")])
    assert excinfo.value.code != 0
    assert not (tmp_path / "data").exists()


def test_init_salt_prints_the_hash_not_the_salt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The OK line carries the salt's hash and the manifest path, never the salt bytes."""
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    argv = ["sealed", "init-salt", "--per-family", "3", "--data-root", str(data_root)]
    assert cli.main([*argv, "--manifest-dir", str(manifest_dir)]) == 0
    out = capsys.readouterr().out.splitlines()
    match = _OK_HASH.match(out[0]) if out else None
    assert match is not None, out
    salt = salt_path(data_root).read_bytes()
    assert len(salt) == 32
    assert match.group(1) == sha256_hex(salt)
    assert salt.hex() not in capsys.readouterr().out
    assert match.group(2) == str(manifest_dir / "synthetic-sealed.yaml")
    assert json.loads((manifest_dir / "synthetic-sealed.yaml").read_text())["per_family"] == 3


def test_build_cache_etth1(
    prepared_etth1: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """build-cache prints the series count and the sealed cache id."""
    data_root, manifest_dir, sealed_cache_root = prepared_etth1
    argv = [
        "sealed",
        "build-cache",
        "etth1",
        *_build_args(data_root, manifest_dir, sealed_cache_root),
    ]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out.splitlines()
    match = _OK_CACHE.match(out[0]) if out else None
    assert match is not None, out
    assert match.group(1) == "7"
    manifest_file = sealed_cache_root / "etth1" / "manifest.json"
    assert match.group(2) == sha256_hex(manifest_file.read_bytes())
    assert not (sealed_cache_root / "synthetic-sealed").exists()


def test_build_cache_on_a_tampered_sealed_file(
    prepared_etth1: tuple[Path, Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """A tampered sealed file fails with the ArenaError message and exit code 1."""
    data_root, manifest_dir, sealed_cache_root = prepared_etth1
    sealed_file = _processed(data_root, "etth1", "sealed")
    obj = json.loads(sealed_file.read_text(encoding="utf-8"))
    obj["series"][0]["values"][800] = 5.0
    sealed_file.write_text(json.dumps(obj), encoding="utf-8")
    argv = [
        "sealed",
        "build-cache",
        "etth1",
        *_build_args(data_root, manifest_dir, sealed_cache_root),
    ]
    assert cli.main(argv) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].startswith("FAIL etth1: ")
    assert "does not match the split manifest" in out[0]
    assert not sealed_cache_root.exists()
