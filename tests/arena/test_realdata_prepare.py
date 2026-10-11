"""Prepare/load_public tests: fixtures, idempotence, crash safety, no look-ahead.

The golden hashes pinned at the bottom of this file fix the exact bytes
produced from the deterministic fixtures below. If a change moves them,
the processed series changed and REALDATA_VERSION must be bumped.
"""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import pathlib
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from evolvets.arena.realdata import prepare as prepare_module
from evolvets.arena.realdata import sources
from evolvets.arena.realdata.fetch import FetchError, fetch, raw_path
from evolvets.arena.realdata.manifest import raw_manifest_path, read_raw_manifest
from evolvets.arena.realdata.prepare import (
    PrepareError,
    load_public,
    prepare,
    read_split_manifest,
    splits_manifest_path,
)
from evolvets.arena.realdata.sources import Source
from evolvets.arena.realdata.splits import (
    MAX_HORIZON,
    Segment,
    Split,
    chronological_split,
)
from evolvets.runner import cli

_TS_FORMAT = "%Y-%m-%d %H:%M:%S"
_ETTH1_HEADER = "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT"
_ROWS = 520  # > 3 * MAX_HORIZON + 5; the sealed segment starts at index 344
_ECL_WINDOW_START = "2012-01-01 00:00:00"
_ECL_WINDOW_HOURS = _ROWS
_ECL_SUBSET = 2
_MANIFEST_KEYS = {
    "manifest_version",
    "realdata_version",
    "dataset",
    "raw_sha256",
    "public_sha256",
    "sealed_sha256",
    "max_horizon",
    "split_percent",
    "series",
}


def _kw(value: float) -> str:
    """Format a float with a decimal comma, as the ECL file does."""
    return f"{value}".replace(".", ",")


def _build_etth1_fixture(fixtures: Path, rows: int = _ROWS, variant: int = 0) -> Source:
    """Build a deterministic ETTh1 fixture; ``variant`` shifts values at index >= 344."""
    fixtures.mkdir(parents=True, exist_ok=True)
    moment = datetime(2016, 7, 1)
    lines = [_ETTH1_HEADER]
    for index in range(rows):
        values = ",".join(
            str(column + index / 100 + (variant if index >= 344 else 0)) for column in range(1, 8)
        )
        lines.append(moment.strftime(_TS_FORMAT) + "," + values)
        moment += timedelta(hours=1)
    fixture = fixtures / ("ETTh1.csv" if variant == 0 else f"ETTh1.v{variant}.csv")
    fixture.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Source(name="etth1", url=fixture.as_uri(), filename=fixture.name, license="MIT")


def _build_ecl_fixture(fixtures: Path) -> Source:
    """Build a deterministic ECL zip: 2 clients, 15-minute readings around the window."""
    fixtures.mkdir(parents=True, exist_ok=True)
    window = datetime.strptime(_ECL_WINDOW_START, _TS_FORMAT)
    lines = ['"";"MT_A";"MT_B"']
    for quarter in range(-2, 4 * _ECL_WINDOW_HOURS + 3):
        stamp = (window + timedelta(minutes=15 * quarter)).strftime(_TS_FORMAT)
        lines.append(f'"{stamp}";{_kw((quarter % 7) * 0.25)};{quarter % 3}')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("__MACOSX/LD2011_2014.txt", "junk")
        archive.writestr("LD2011_2014.txt", "\n".join(lines) + "\n")
    fixture = fixtures / "electricityloaddiagrams20112014.zip"
    fixture.write_bytes(buffer.getvalue())
    return Source(name="ecl", url=fixture.as_uri(), filename=fixture.name, license="MIT")


def _install(
    monkeypatch: pytest.MonkeyPatch,
    etth1: Source | None = None,
    ecl: Source | None = None,
) -> None:
    """Point the registry and the prepare constants at small fixtures."""
    registry = dict(sources.SOURCES)
    if etth1 is not None:
        registry["etth1"] = etth1
    if ecl is not None:
        registry["ecl"] = ecl
    monkeypatch.setattr(sources, "SOURCES", registry)
    monkeypatch.setattr(prepare_module, "ECL_WINDOW_START", _ECL_WINDOW_START)
    monkeypatch.setattr(prepare_module, "ECL_WINDOW_HOURS", _ECL_WINDOW_HOURS)
    monkeypatch.setattr(prepare_module, "ECL_SUBSET_SIZE", _ECL_SUBSET)


@pytest.fixture()
def fetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Fetched (but not prepared) fixture datasets; returns (data_root, manifest_dir)."""
    etth1 = _build_etth1_fixture(tmp_path / "fixtures")
    ecl = _build_ecl_fixture(tmp_path / "fixtures")
    _install(monkeypatch, etth1, ecl)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    for source in (etth1, ecl):
        fetch(source, data_root, manifest_dir, today=date(2026, 10, 11))
    return data_root, manifest_dir


def _processed(data_root: Path, dataset: str, label: str) -> Path:
    return data_root / "processed" / f"{dataset}.{label}.json"


def test_prepare_writes_files_and_manifest_idempotently(
    fetched: tuple[Path, Path],
) -> None:
    data_root, manifest_dir = fetched
    prepare("etth1", data_root, manifest_dir)
    prepare("ecl", data_root, manifest_dir)
    files = [
        _processed(data_root, "etth1", "public"),
        _processed(data_root, "etth1", "sealed"),
        _processed(data_root, "ecl", "public"),
        _processed(data_root, "ecl", "sealed"),
        splits_manifest_path(manifest_dir, "etth1"),
        splits_manifest_path(manifest_dir, "ecl"),
    ]
    before = {file: file.read_bytes() for file in files}
    for dataset in ("etth1", "ecl"):
        prepare(dataset, data_root, manifest_dir)
    assert {file: file.read_bytes() for file in files} == before

    for dataset, count in (("etth1", 7), ("ecl", _ECL_SUBSET)):
        manifest = json.loads(splits_manifest_path(manifest_dir, dataset).read_text())
        assert set(manifest) == _MANIFEST_KEYS
        assert len(manifest["series"]) == count
        public_series = json.loads(_processed(data_root, dataset, "public").read_text())["series"]
        for manifest_entry, series in zip(manifest["series"], public_series, strict=True):
            assert len(series["values"]) == 344  # values[:sealed.start] only
            assert manifest_entry["length"] == _ROWS
        sealed = json.loads(_processed(data_root, dataset, "sealed").read_text())
        assert all(len(series["values"]) == _ROWS for series in sealed["series"])


def test_prepare_rejects_unknown_dataset(fetched: tuple[Path, Path]) -> None:
    data_root, manifest_dir = fetched
    with pytest.raises(PrepareError, match="unknown dataset"):
        prepare("nope", data_root, manifest_dir)


def test_prepare_refuses_a_differing_existing_manifest(fetched: tuple[Path, Path]) -> None:
    data_root, manifest_dir = fetched
    prepare("etth1", data_root, manifest_dir)
    manifest_file = splits_manifest_path(manifest_dir, "etth1")
    manifest = json.loads(manifest_file.read_text())
    manifest["series"][0]["segments"]["dev"]["start"] += 1
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    files = [
        _processed(data_root, "etth1", "public"),
        _processed(data_root, "etth1", "sealed"),
        manifest_file,
    ]
    before = {file: file.read_bytes() for file in files}
    with pytest.raises(PrepareError, match="different content"):
        prepare("etth1", data_root, manifest_dir)
    assert {file: file.read_bytes() for file in files} == before


def test_prepare_requires_a_verified_raw_file(
    fetched: tuple[Path, Path],
) -> None:
    data_root, manifest_dir = fetched
    raw_path(data_root, sources.SOURCES["etth1"]).write_bytes(b"tampered raw file")
    with pytest.raises(FetchError):
        prepare("etth1", data_root, manifest_dir)
    assert not (data_root / "processed").exists()
    assert not splits_manifest_path(manifest_dir, "etth1").exists()


def test_prepare_crash_before_manifest_is_recoverable(fetched: tuple[Path, Path]) -> None:
    data_root, manifest_dir = fetched
    manifest_file = splits_manifest_path(manifest_dir, "etth1")
    real_replace = os.replace

    def replace(src: object, dst: object, *args: object, **kwargs: object) -> object:
        if str(dst) == str(manifest_file):
            raise OSError("simulated crash while moving the split manifest")
        return real_replace(src, dst, *args, **kwargs)  # type: ignore[arg-type]

    patcher = pytest.MonkeyPatch()
    patcher.setattr(os, "replace", replace)
    try:
        with pytest.raises(OSError, match="simulated crash"):
            prepare("etth1", data_root, manifest_dir)
    finally:
        patcher.undo()
    assert not manifest_file.exists()
    assert not manifest_file.with_name(manifest_file.name + ".tmp").exists()
    assert _processed(data_root, "etth1", "public").exists()  # written before the manifest

    prepare("etth1", data_root, manifest_dir)  # rerun succeeds
    load_public("etth1", data_root, manifest_dir)


def test_no_look_ahead_into_sealed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plain = _build_etth1_fixture(tmp_path / "fixtures", variant=0)
    shifted = _build_etth1_fixture(tmp_path / "fixtures", variant=1)  # differs at index >= 344
    roots = {}
    for tag, source in (("a", plain), ("b", shifted)):
        _install(monkeypatch, etth1=source)
        data_root, manifest_dir = tmp_path / tag / "data", tmp_path / tag / "manifests"
        fetch(source, data_root, manifest_dir, today=date(2026, 10, 11))
        prepare("etth1", data_root, manifest_dir)
        roots[tag] = (data_root, manifest_dir)
    (root_a, manifests_a), (root_b, manifests_b) = roots["a"], roots["b"]
    assert (
        _processed(root_a, "etth1", "public").read_bytes()
        == _processed(root_b, "etth1", "public").read_bytes()
    )
    assert (
        _processed(root_a, "etth1", "sealed").read_bytes()
        != _processed(root_b, "etth1", "sealed").read_bytes()
    )
    assert load_public("etth1", root_a, manifests_a) == load_public("etth1", root_b, manifests_b)


def _record_opens(monkeypatch: pytest.MonkeyPatch, opened: list[str]) -> None:
    """Record every file open through builtins.open, io.open, os.open and Path.open."""
    real_builtins_open = builtins.open
    real_io_open = io.open
    real_os_open = os.open
    real_path_open = pathlib.Path.open

    def record(file: object) -> None:
        opened.append(str(os.fspath(file)) if not isinstance(file, int) else f"<fd {file}>")

    def open_rec(file: object, *args: object, **kwargs: object) -> object:
        record(file)
        return real_builtins_open(file, *args, **kwargs)  # type: ignore[arg-type]

    def io_open_rec(file: object, *args: object, **kwargs: object) -> object:
        record(file)
        return real_io_open(file, *args, **kwargs)  # type: ignore[arg-type]

    def os_open_rec(path: object, flags: int, mode: int = 0o777, **kwargs: object) -> int:
        record(path)
        return real_os_open(path, flags, mode, **kwargs)  # type: ignore[arg-type]

    def path_open_rec(self: pathlib.Path, *args: object, **kwargs: object) -> object:
        record(self)
        return real_path_open(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "open", open_rec)
    monkeypatch.setattr(io, "open", io_open_rec)
    monkeypatch.setattr(os, "open", os_open_rec)
    monkeypatch.setattr(pathlib.Path, "open", path_open_rec)


def test_load_public_never_opens_sealed_or_raw(
    fetched: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, manifest_dir = fetched
    prepare("etth1", data_root, manifest_dir)
    _processed(data_root, "etth1", "sealed").unlink()
    raw_path(data_root, sources.SOURCES["etth1"]).unlink()

    opened: list[str] = []
    _record_opens(monkeypatch, opened)
    series = load_public("etth1", data_root, manifest_dir)
    expected_split = chronological_split(_ROWS, MAX_HORIZON)
    assert len(series) == 7
    assert all(entry.split == expected_split for entry in series)
    hufl = series[0]
    assert hufl.series_id == "etth1/HUFL"
    assert hufl.start == "2016-07-01 00:00:00"
    assert hufl.values == tuple(1 + index / 100 for index in range(344))

    allowed = {
        str(splits_manifest_path(manifest_dir, "etth1")),
        str(_processed(data_root, "etth1", "public")),
    }
    assert opened, "load_public should have opened its manifest and public file"
    assert set(opened) <= allowed, opened


def test_load_public_rejects_a_tampered_public_file(fetched: tuple[Path, Path]) -> None:
    data_root, manifest_dir = fetched
    prepare("etth1", data_root, manifest_dir)
    public = _processed(data_root, "etth1", "public")
    public.write_bytes(public.read_bytes()[:-1] + b" ")
    with pytest.raises(PrepareError, match="does not match"):
        load_public("etth1", data_root, manifest_dir)


def test_load_public_rejects_missing_pieces(fetched: tuple[Path, Path]) -> None:
    data_root, manifest_dir = fetched
    with pytest.raises(PrepareError, match="is missing"):
        load_public("etth1", data_root, manifest_dir)
    prepare("etth1", data_root, manifest_dir)
    _processed(data_root, "etth1", "public").unlink()
    with pytest.raises(PrepareError, match="is missing"):
        load_public("etth1", data_root, manifest_dir)


def test_cli_prepare_and_verify_all(
    fetched: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    data_root, manifest_dir = fetched
    args = ["--data-root", str(data_root), "--manifest-dir", str(manifest_dir)]
    assert cli.main(["data", "prepare", "all", *args]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("OK etth1: 7 series, processed ")
    assert out[1].startswith(f"OK ecl: {_ECL_SUBSET} series, processed ")
    assert cli.main(["data", "verify", "all", *args]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 2

    _processed(data_root, "ecl", "public").unlink()
    assert cli.main(["data", "verify", "all", *args]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("OK etth1: ")
    assert out[1].startswith("FAIL ecl: ")


def test_golden_processed_hashes(fetched: tuple[Path, Path]) -> None:
    """Pin the processed bytes for the fixtures above (bump REALDATA_VERSION to move)."""
    data_root, manifest_dir = fetched
    prepare("etth1", data_root, manifest_dir)
    prepare("ecl", data_root, manifest_dir)
    digests = {
        f"{dataset}.{label}": hashlib.sha256(
            _processed(data_root, dataset, label).read_bytes()
        ).hexdigest()
        for dataset in ("etth1", "ecl")
        for label in ("public", "sealed")
    }
    assert digests == {
        "etth1.public": "1c95588120684ed5ed1d767161f51de5f95eb4035c63b67a7fcf314672cf58a7",
        "etth1.sealed": "f4bfd75c102d61c51490b52c1ec04a659c98c70bed95e43c1902076fedae685c",
        "ecl.public": "81de97493d5263fc55bfe9987dbd52d662f3d10fac96fcb32a9650e5fd3286ff",
        "ecl.sealed": "3f51f5309d1329ff7899f0fe9423312bdfae0f4a311e11deedb6bcff748eb3ac",
    }


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ETTH1_SEGMENTS = (
    Segment("warmup", 0, 3383),
    Segment("dev", 3551, 9471),
    Segment("sealed", 9639, 14713),
    Segment("reserve", 14881, 17420),
)
_ECL_SEGMENTS = (
    Segment("warmup", 0, 5160),
    Segment("dev", 5328, 14358),
    Segment("sealed", 14526, 22266),
    Segment("reserve", 22434, 26304),
)
_ECL_CLIENTS = [
    "MT_001",
    "MT_010",
    "MT_011",
    "MT_018",
    "MT_052",
    "MT_067",
    "MT_090",
    "MT_119",
    "MT_138",
    "MT_157",
    "MT_172",
    "MT_193",
    "MT_194",
    "MT_208",
    "MT_227",
    "MT_238",
    "MT_251",
    "MT_284",
    "MT_297",
    "MT_306",
]


def test_committed_split_manifests() -> None:
    manifest_dir = _REPO_ROOT / "data" / "manifests"
    for dataset, expected in (
        ("etth1", Split(length=17_420, gap=MAX_HORIZON, segments=_ETTH1_SEGMENTS)),
        ("ecl", Split(length=26_304, gap=MAX_HORIZON, segments=_ECL_SEGMENTS)),
    ):
        spec = read_split_manifest(manifest_dir, dataset)
        raw = read_raw_manifest(raw_manifest_path(manifest_dir, dataset), sources.SOURCES[dataset])
        assert spec.raw_sha256 == raw.sha256
        for _series_id, start, split in spec.series:
            assert split == expected
            assert start == ("2016-07-01 00:00:00" if dataset == "etth1" else "2012-01-01 00:00:00")
        obj = json.loads(splits_manifest_path(manifest_dir, dataset).read_text())
        assert obj["max_horizon"] == 168
        assert obj["realdata_version"] == "1"
        assert obj["manifest_version"] == "1"
        assert obj["split_percent"] == [20, 35, 30]
    etth1 = read_split_manifest(manifest_dir, "etth1")
    assert [entry[0] for entry in etth1.series] == [
        "etth1/HUFL",
        "etth1/HULL",
        "etth1/MUFL",
        "etth1/MULL",
        "etth1/LUFL",
        "etth1/LULL",
        "etth1/OT",
    ]
    ecl = read_split_manifest(manifest_dir, "ecl")
    assert [entry[0] for entry in ecl.series] == [f"ecl/{client}" for client in _ECL_CLIENTS]
