"""Fetch and data-CLI tests: pinning, restore, tamper and failure cases."""

from __future__ import annotations

import hashlib
import urllib.request
from datetime import date
from pathlib import Path

import pytest

from evolvets.arena.realdata import sources
from evolvets.arena.realdata.fetch import (
    FetchError,
    fetch,
    file_sha256,
    raw_path,
    verify_raw,
)
from evolvets.arena.realdata.manifest import raw_manifest_path, read_raw_manifest
from evolvets.arena.realdata.sources import Source
from evolvets.runner import cli

_FIXTURE_BYTES = b"date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n2016-07-01 00:00:00,1,2,3,4,5,6,7\n"
_OTHER_BYTES = b"completely different content\n"


def _fixture_source(tmp_path: Path, content: bytes = _FIXTURE_BYTES) -> Source:
    """Return a source whose url points at a fixture file under ``tmp_path``."""
    fixture = tmp_path / "fixtures" / "ETTh1.csv"
    fixture.parent.mkdir(parents=True, exist_ok=True)
    fixture.write_bytes(content)
    return Source(name="etth1", url=fixture.as_uri(), filename="ETTh1.csv", license="MIT")


def _target(tmp_path: Path, source: Source) -> Path:
    return raw_path(tmp_path / "data", source)


def _manifest_file(tmp_path: Path, source: Source) -> Path:
    return raw_manifest_path(tmp_path / "manifests", source.name)


def _part(target: Path) -> Path:
    return target.parent / (target.name + ".part")


def test_fresh_fetch_writes_target_and_manifest(tmp_path: Path) -> None:
    source = _fixture_source(tmp_path)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    manifest = fetch(source, data_root, manifest_dir, today=date(2026, 10, 11))
    target = _target(tmp_path, source)
    assert target.read_bytes() == _FIXTURE_BYTES
    assert not _part(target).exists()
    assert manifest == read_raw_manifest(_manifest_file(tmp_path, source), source)
    assert manifest.dataset == "etth1"
    assert manifest.url == source.url
    assert manifest.filename == "ETTh1.csv"
    assert manifest.license == "MIT"
    assert manifest.sha256 == hashlib.sha256(_FIXTURE_BYTES).hexdigest()
    assert manifest.size_bytes == len(_FIXTURE_BYTES)
    assert manifest.downloaded_on == "2026-10-11"
    assert manifest.manifest_version == "1"


def test_second_fetch_never_touches_the_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fixture_source(tmp_path)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    first = fetch(source, data_root, manifest_dir, today=date(2026, 10, 11))

    def _no_network(url: object, timeout: object = None) -> object:
        raise AssertionError("network access during a pinned fetch")

    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    assert fetch(source, data_root, manifest_dir, today=date(2027, 1, 1)) == first


def test_missing_target_is_restored_manifest_unchanged(tmp_path: Path) -> None:
    source = _fixture_source(tmp_path)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    fetch(source, data_root, manifest_dir, today=date(2026, 10, 11))
    target, manifest_file = _target(tmp_path, source), _manifest_file(tmp_path, source)
    manifest_before = manifest_file.read_bytes()
    target.unlink()

    fetch(source, data_root, manifest_dir, today=date(2027, 1, 1))
    assert target.read_bytes() == _FIXTURE_BYTES
    assert manifest_file.read_bytes() == manifest_before


def test_changed_source_content_is_refused(tmp_path: Path) -> None:
    source = _fixture_source(tmp_path)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    fetch(source, data_root, manifest_dir, today=date(2026, 10, 11))
    target, manifest_file = _target(tmp_path, source), _manifest_file(tmp_path, source)
    manifest_before = manifest_file.read_bytes()
    target.unlink()
    _fixture_source(tmp_path, content=_FIXTURE_BYTES + b"extra row\n")

    with pytest.raises(FetchError, match="content changed"):
        fetch(source, data_root, manifest_dir)
    assert not target.exists()
    assert not _part(target).exists()
    assert manifest_file.read_bytes() == manifest_before


def test_tampered_target_is_refused_and_left_alone(tmp_path: Path) -> None:
    source = _fixture_source(tmp_path)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    fetch(source, data_root, manifest_dir, today=date(2026, 10, 11))
    target = _target(tmp_path, source)
    target.write_bytes(_OTHER_BYTES)

    with pytest.raises(FetchError, match="does not match"):
        fetch(source, data_root, manifest_dir)
    assert target.read_bytes() == _OTHER_BYTES


def test_unpinned_target_is_refused(tmp_path: Path) -> None:
    source = _fixture_source(tmp_path)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    target, manifest_file = _target(tmp_path, source), _manifest_file(tmp_path, source)
    target.parent.mkdir(parents=True)
    target.write_bytes(_OTHER_BYTES)

    with pytest.raises(FetchError, match="no manifest pins it"):
        fetch(source, data_root, manifest_dir)
    assert not manifest_file.exists()
    assert target.read_bytes() == _OTHER_BYTES


class _FailingStream:
    """A urlopen stand-in whose read() yields one chunk, then fails."""

    def __init__(self, first: bytes) -> None:
        self._first = first

    def read(self, size: int = -1) -> bytes:
        chunk, self._first = self._first, b""
        if chunk:
            return chunk
        raise OSError("connection reset mid-stream")

    def __enter__(self) -> _FailingStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


def test_mid_stream_failure_cleans_up_and_stale_part_is_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _fixture_source(tmp_path)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    target, manifest_file = _target(tmp_path, source), _manifest_file(tmp_path, source)
    target.parent.mkdir(parents=True)
    stale = _part(target)
    stale.write_bytes(b"junk left by a hard kill")

    real_urlopen = urllib.request.urlopen
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda url, timeout=None: _FailingStream(b"partial data")
    )
    with pytest.raises(FetchError, match="downloading"):
        fetch(source, data_root, manifest_dir)
    assert not target.exists()
    assert not stale.exists()
    assert not manifest_file.exists()

    # A hard kill between the cleanup above and the next download is the one
    # case that leaves a stale .part behind. Recreate it by hand (as the task
    # spec puts it) and require the next fetch to truncate, not append: the
    # target must equal the fixture and hash to the manifest just written.
    stale.write_bytes(b"junk left by a hard kill")
    monkeypatch.setattr(urllib.request, "urlopen", real_urlopen)
    manifest = fetch(source, data_root, manifest_dir, today=date(2026, 10, 11))
    assert target.read_bytes() == _FIXTURE_BYTES
    assert file_sha256(target) == (manifest.sha256, manifest.size_bytes)
    assert manifest.sha256 == hashlib.sha256(_FIXTURE_BYTES).hexdigest()
    assert not stale.exists()


def test_empty_source_is_refused(tmp_path: Path) -> None:
    source = _fixture_source(tmp_path, content=b"")
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    target, manifest_file = _target(tmp_path, source), _manifest_file(tmp_path, source)

    with pytest.raises(FetchError, match="zero bytes"):
        fetch(source, data_root, manifest_dir)
    assert not target.exists()
    assert not _part(target).exists()
    assert not manifest_file.exists()


def test_verify_raw_passes_and_detects_missing_or_changed(tmp_path: Path) -> None:
    source = _fixture_source(tmp_path)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    manifest = fetch(source, data_root, manifest_dir, today=date(2026, 10, 11))
    target = _target(tmp_path, source)

    assert verify_raw(source, data_root, manifest_dir) == manifest
    assert file_sha256(target) == (manifest.sha256, manifest.size_bytes)

    target.unlink()
    with pytest.raises(FetchError, match="missing"):
        verify_raw(source, data_root, manifest_dir)

    target.write_bytes(_FIXTURE_BYTES[:-1] + b"X")
    with pytest.raises(FetchError, match="does not match"):
        verify_raw(source, data_root, manifest_dir)


@pytest.fixture()
def cli_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Source]:
    """Point the registry's etth1/ecl entries at local file:// fixtures."""
    registry: dict[str, Source] = {}
    for name, filename in (
        ("etth1", "ETTh1.csv"),
        ("ecl", "electricityloaddiagrams20112014.zip"),
    ):
        fixture = tmp_path / "fixtures" / filename
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_bytes(f"{name} fixture bytes".encode())
        registry[name] = Source(name=name, url=fixture.as_uri(), filename=filename, license="MIT")
    monkeypatch.setattr(sources, "SOURCES", registry)
    return registry


def _data_args(data_root: Path, manifest_dir: Path) -> list[str]:
    return ["--data-root", str(data_root), "--manifest-dir", str(manifest_dir)]


def test_cli_fetch_and_verify_all(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cli_sources: dict[str, Source]
) -> None:
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"

    assert cli.main(["data", "fetch", "all", *_data_args(data_root, manifest_dir)]) == 0
    expected = [
        f"OK {name}: {hashlib.sha256(f'{name} fixture bytes'.encode()).hexdigest()} "
        f"{len(f'{name} fixture bytes')} bytes"
        for name in ("etth1", "ecl")
    ]
    assert capsys.readouterr().out.splitlines() == expected

    assert cli.main(["data", "verify", "all", *_data_args(data_root, manifest_dir)]) == 0
    assert capsys.readouterr().out.splitlines() == expected

    tampered = raw_path(data_root, cli_sources["ecl"])
    tampered.write_bytes(b"tampered")
    assert cli.main(["data", "verify", "all", *_data_args(data_root, manifest_dir)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0] == expected[0]
    assert out[1].startswith("FAIL ecl: ")


def test_cli_reports_pathological_manifest_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cli_sources: dict[str, Source]
) -> None:
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    manifest_dir.mkdir(parents=True)
    raw_manifest_path(manifest_dir, "etth1").write_text("[" * 100_000, encoding="utf-8")

    assert cli.main(["data", "verify", "etth1", *_data_args(data_root, manifest_dir)]) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    out_lines = captured.out.splitlines()
    assert len(out_lines) == 1
    assert out_lines[0].startswith("FAIL etth1: ")


def test_cli_rejects_unknown_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["data", "fetch", "nope"])
    assert excinfo.value.code == 2
    capsys.readouterr()
