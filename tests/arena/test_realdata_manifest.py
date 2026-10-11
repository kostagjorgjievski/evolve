"""Raw-manifest tests: round trip, validation, registry, committed manifests."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from evolvets.arena.realdata.manifest import (
    ManifestError,
    RawManifest,
    raw_manifest_path,
    read_raw_manifest,
    write_raw_manifest,
)
from evolvets.arena.realdata.sources import SOURCES, Source

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ETTH1_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
    "1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv"
)
_PINNED = {
    "etth1": ("f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066", 2_589_657),
    "ecl": ("f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1", 261_335_609),
}


def _source() -> Source:
    return Source(
        name="etth1", url="file:///fixtures/ETTh1.csv", filename="ETTh1.csv", license="MIT"
    )


def _manifest() -> RawManifest:
    return RawManifest(
        dataset="etth1",
        url="file:///fixtures/ETTh1.csv",
        filename="ETTh1.csv",
        license="MIT",
        sha256="a" * 64,
        size_bytes=42,
        downloaded_on="2026-10-11",
    )


def test_round_trip_and_exact_file_text(tmp_path: Path) -> None:
    manifest = _manifest()
    path = raw_manifest_path(tmp_path, "etth1")
    write_raw_manifest(path, manifest)
    assert read_raw_manifest(path, _source()) == manifest
    expected = (
        json.dumps(asdict(manifest), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )
    assert path.read_text(encoding="utf-8") == expected


def test_constructor_rejects_invalid_values() -> None:
    good = asdict(_manifest())
    cases = {
        "sha256": ["A" * 64, "a" * 63, ""],
        "size_bytes": [0, -1, True],
        "downloaded_on": ["2026-1-1", "2026-02-30", "2026/10/11"],
        "manifest_version": ["2", ""],
        "dataset": ["", 7],
        "url": [""],
        "filename": [""],
        "license": [""],
    }
    for field, values in cases.items():
        for value in values:
            with pytest.raises(ManifestError, match=field):
                RawManifest(**{**good, field: value})


def test_read_rejects_malformed_files(tmp_path: Path) -> None:
    good = asdict(_manifest())
    source = _source()
    tamperings: list[str] = []

    def write(obj_text: str) -> Path:
        path = tmp_path / f"m{len(tamperings)}.yaml"
        tamperings.append(path.name)
        path.write_text(obj_text, encoding="utf-8")
        return path

    paths = [
        write(json.dumps({k: v for k, v in good.items() if k != "license"})),
        write(json.dumps({**good, "extra": 1})),
        write("[1, 2]"),
        write("{not json"),
        write(json.dumps({**good, "sha256": good["sha256"].upper()})),
        write(json.dumps({**good, "sha256": good["sha256"][:63]})),
        write(json.dumps({**good, "size_bytes": 0})),
        write(json.dumps({**good, "size_bytes": -1})),
        write(json.dumps({**good, "size_bytes": True})),
        write(json.dumps({**good, "downloaded_on": "2026-1-1"})),
        write(json.dumps({**good, "downloaded_on": "2026-02-30"})),
        write(json.dumps({**good, "manifest_version": "2"})),
        write(json.dumps({**good, "dataset": "ecl"})),
        write(json.dumps({**good, "url": "file:///other.csv"})),
        write(json.dumps({**good, "filename": "other.csv"})),
        write(json.dumps({**good, "license": "other"})),
    ]
    for path in paths:
        with pytest.raises(ManifestError):
            read_raw_manifest(path, source)


def test_sources_registry_is_exactly_the_two_pinned_sources() -> None:
    assert set(SOURCES) == {"etth1", "ecl"}
    etth1 = SOURCES["etth1"]
    assert etth1.name == "etth1"
    assert etth1.url == _ETTH1_URL
    assert "1d16c8f4f943005d613b5bc962e9eeb06058cf07" in etth1.url
    assert etth1.filename == "ETTh1.csv"
    assert etth1.license == "CC-BY-ND-4.0"
    ecl = SOURCES["ecl"]
    assert ecl.name == "ecl"
    assert (
        ecl.url
        == "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip"
    )
    assert ecl.filename == "electricityloaddiagrams20112014.zip"
    assert ecl.license == "CC-BY-4.0"


def test_committed_raw_manifests_match_the_pinned_sources() -> None:
    manifest_dir = _REPO_ROOT / "data" / "manifests"
    for name, (sha256, size_bytes) in _PINNED.items():
        manifest = read_raw_manifest(raw_manifest_path(manifest_dir, name), SOURCES[name])
        assert manifest.sha256 == sha256
        assert manifest.size_bytes == size_bytes
