"""Store tests: round trip, layout, idempotence, crash safety, no look-ahead.

The golden cache id pinned at the bottom of this file fixes the exact bytes
the deterministic fixtures above produce through the expert pool. If it
moves, the cache layout or manifest changed (bump CACHE_VERSION) or a
forecast value changed (bump EXPERTS_VERSION).
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from evolvets.cache import CACHE_VERSION
from evolvets.cache.interfaces import CachedSeries, CacheManifest, SeriesInput
from evolvets.cache.store import (
    DEFAULT_HORIZON,
    CacheError,
    build_cache,
    cache_dir,
    cache_id,
    load_series,
    read_manifest,
    verify_cache,
)
from evolvets.experts import EXPERTS_VERSION
from evolvets.experts.pool import EXPERT_NAMES, first_origin, make_pool, run_pool
from evolvets.ledger.canonical import canonical_bytes, sha256_hex

_PERIOD = 24
_HORIZON = 5
_START = first_origin(_PERIOD)  # 287 with the default pool
_SOURCE = {"kind": "test", "note": "fixture"}


def _values(seed: int, length: int) -> tuple[float, ...]:
    """Return deterministic values: integer arithmetic and exact divisions only."""
    return tuple(
        ((i * 53 + seed * 29) % 211) / 8.0 + (i % _PERIOD) * 0.25 - 13.0 for i in range(length)
    )


def _fixture_series() -> tuple[SeriesInput, ...]:
    """Return the two fixture series (lengths 300 and 350, period 24)."""
    return (
        SeriesInput("fixture/a", _values(1, 300), _PERIOD),
        SeriesInput("fixture/b", _values(2, 350), _PERIOD),
    )


def _build(root: Path, series: tuple[SeriesInput, ...] | None = None) -> CacheManifest:
    """Build the fixture cache (or the given series) under ``root``."""
    return build_cache(
        "fixture", _SOURCE, _fixture_series() if series is None else series, root, horizon=_HORIZON
    )


@pytest.fixture()
def built(tmp_path: Path) -> tuple[Path, CacheManifest]:
    """A built two-series fixture cache; returns (cache_root, manifest)."""
    return tmp_path, _build(tmp_path)


def _cache_files(root: Path) -> list[Path]:
    """Return the manifest and every series file of the fixture cache."""
    base = cache_dir(root, "fixture")
    return [base / "manifest.json", *(base / "series" / f"{i:04d}.f64" for i in range(2))]


def test_round_trip_matches_run_pool(built: tuple[Path, CacheManifest]) -> None:
    root, manifest = built
    for sid, seed, length in (("fixture/a", 1, 300), ("fixture/b", 2, 350)):
        forecasts = run_pool(_values(seed, length), _PERIOD, _HORIZON)
        loaded = load_series(root, manifest, sid)
        assert loaded.series_id == sid
        assert loaded.first_origin == _START
        assert loaded.last_origin == length - 1
        for offset, entry in enumerate(forecasts):
            assert loaded.at(_START + offset) == entry
        middle = len(forecasts) // 2
        for name, expert_entry in zip(EXPERT_NAMES, forecasts[middle], strict=True):
            assert loaded.expert_forecast(_START + middle, name) == expert_entry


def test_file_layout_and_offset_formula(built: tuple[Path, CacheManifest]) -> None:
    root, _ = built
    forecasts = run_pool(_values(2, 350), _PERIOD, _HORIZON)
    path = cache_dir(root, "fixture") / "series" / "0001.f64"
    blob = path.read_bytes()
    n_experts = len(EXPERT_NAMES)
    assert len(blob) == (350 - _START) * n_experts * _HORIZON * 8
    for origin, expert, h in ((_START, 0, 1), (300, 2, 3), (349, n_experts - 1, _HORIZON)):
        element = ((origin - _START) * n_experts + expert) * _HORIZON + (h - 1)
        (value,) = struct.unpack_from("<d", blob, element * 8)
        assert value == forecasts[origin - _START][expert][h - 1]


def test_manifest_content_and_ids(built: tuple[Path, CacheManifest]) -> None:
    root, manifest = built
    path = cache_dir(root, "fixture") / "manifest.json"
    raw = path.read_bytes()
    obj = json.loads(raw)
    assert set(obj) == {
        "cache_version",
        "experts_version",
        "dataset",
        "source",
        "horizon",
        "experts",
        "series",
    }
    assert obj["source"] == _SOURCE  # copied verbatim
    assert DEFAULT_HORIZON == 24
    assert cache_id(root, "fixture") == sha256_hex(raw)
    assert read_manifest(root, "fixture") == manifest
    series_meta = manifest.series
    assert [s.series_id for s in series_meta] == ["fixture/a", "fixture/b"]
    assert [s.file for s in series_meta] == ["series/0000.f64", "series/0001.f64"]
    assert [s.n_origins for s in series_meta] == [300 - _START, 350 - _START]
    assert all(s.first_origin == _START and s.period == _PERIOD for s in series_meta)
    for entry in series_meta:
        assert entry.sha256 == sha256_hex((cache_dir(root, "fixture") / entry.file).read_bytes())
    assert manifest == CacheManifest(
        cache_version=CACHE_VERSION,
        experts_version=EXPERTS_VERSION,
        dataset="fixture",
        source=dict(_SOURCE),
        horizon=_HORIZON,
        experts=tuple((e.name, e.params()) for e in make_pool(_PERIOD)),
        series=(
            CachedSeries(
                series_id="fixture/a",
                length=300,
                period=_PERIOD,
                first_origin=_START,
                n_origins=300 - _START,
                file="series/0000.f64",
                sha256=series_meta[0].sha256,
                values_sha256=sha256_hex(canonical_bytes(list(_values(1, 300)))),
            ),
            CachedSeries(
                series_id="fixture/b",
                length=350,
                period=_PERIOD,
                first_origin=_START,
                n_origins=350 - _START,
                file="series/0001.f64",
                sha256=series_meta[1].sha256,
                values_sha256=sha256_hex(canonical_bytes(list(_values(2, 350)))),
            ),
        ),
    )


def test_rebuild_is_idempotent(built: tuple[Path, CacheManifest]) -> None:
    root, manifest = built
    files = _cache_files(root)
    before = {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in files}
    rebuilt = _build(root)
    assert rebuilt == manifest
    assert {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in files} == before
    assert list(root.glob(".fixture.tmp-*")) == []


def test_conflicting_rebuild_is_refused(built: tuple[Path, CacheManifest]) -> None:
    root, _ = built
    files = _cache_files(root)
    before = {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in files}
    original = _fixture_series()
    changed_value = (
        SeriesInput("fixture/a", (*_values(1, 300)[:-1], 99.0), _PERIOD),
        original[1],
    )
    changed_horizon = _HORIZON + 1
    changed_source = {"kind": "test", "note": "other"}
    with pytest.raises(CacheError, match="cache id"):
        _build(root, changed_value)
    with pytest.raises(CacheError, match="cache id"):
        build_cache("fixture", _SOURCE, original, root, horizon=changed_horizon)
    with pytest.raises(CacheError, match="cache id"):
        build_cache("fixture", changed_source, original, root, horizon=_HORIZON)
    assert {file: (file.read_bytes(), file.stat().st_mtime_ns) for file in files} == before
    assert list(root.glob(".fixture.tmp-*")) == []


def test_crashed_rename_leaves_no_trace_and_rerun_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final = tmp_path / "fixture"
    real_rename = os.rename

    def rename(src: object, dst: object, *args: object, **kwargs: object) -> str:
        if Path(dst) == final:  # type: ignore[arg-type]
            raise OSError("simulated crash on the final rename")
        return real_rename(src, dst, *args, **kwargs)  # type: ignore[arg-type,return-value]

    monkeypatch.setattr(os, "rename", rename)
    with pytest.raises(OSError, match="simulated crash"):
        _build(tmp_path)
    monkeypatch.undo()
    assert not final.exists()
    assert list(tmp_path.glob(".fixture.tmp-*")) == []
    _build(tmp_path)  # rerun succeeds

    leftover = tmp_path / ".fixture.tmp-deadbeef"  # a hard kill leaves only this
    (leftover / "series").mkdir(parents=True)
    (leftover / "series" / "0000.f64").write_bytes(b"junk")
    assert read_manifest(tmp_path, "fixture").dataset == "fixture"
    verify_cache(tmp_path, "fixture")
    _build(tmp_path)  # still idempotent
    assert list(tmp_path.glob(".fixture.tmp-*")) == [leftover]


def test_corrupt_series_files_are_refused(built: tuple[Path, CacheManifest]) -> None:
    root, manifest = built
    path = cache_dir(root, "fixture") / "series" / "0000.f64"
    raw = path.read_bytes()

    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))  # one byte flipped
    with pytest.raises(CacheError, match="does not match"):
        load_series(root, manifest, "fixture/a")
    with pytest.raises(CacheError, match="does not match"):
        verify_cache(root, "fixture")

    path.write_bytes(raw[:-8])  # truncated
    with pytest.raises(CacheError, match="expected"):
        load_series(root, manifest, "fixture/a")
    with pytest.raises(CacheError, match="expected"):
        verify_cache(root, "fixture")

    path.unlink()  # missing
    with pytest.raises(CacheError, match="cannot be read"):
        load_series(root, manifest, "fixture/a")
    with pytest.raises(CacheError, match="cannot be read"):
        verify_cache(root, "fixture")

    path.write_bytes(raw)  # restored
    assert verify_cache(root, "fixture") == manifest


def test_tampered_manifests_are_refused(built: tuple[Path, CacheManifest]) -> None:
    root, _ = built
    path = cache_dir(root, "fixture") / "manifest.json"
    original = json.loads(path.read_bytes())

    def tamper(mutate: dict[str, object], label: str) -> None:
        path.write_text(json.dumps(mutate), encoding="utf-8")
        with pytest.raises(CacheError, match=label):
            read_manifest(root, "fixture")

    tamper({**original, "extra": 1}, "keys must be exactly")
    tamper({k: v for k, v in original.items() if k != "horizon"}, "keys must be exactly")
    tamper({**original, "cache_version": "2"}, "refused")
    tamper({**original, "experts": list(reversed(original["experts"]))}, "pool order")
    series = [
        {**original["series"][0], "n_origins": original["series"][0]["n_origins"] + 1},
        original["series"][1],
    ]
    tamper({**original, "series": series}, "length - first_origin")


_INVALID_INPUT_CASES = (
    (
        "duplicate-ids",
        "fixture",
        _HORIZON,
        (_fixture_series()[0], SeriesInput("fixture/a", _values(2, 300), _PERIOD)),
    ),
    ("empty-id", "fixture", _HORIZON, (SeriesInput("", _values(2, 300), _PERIOD),)),
    ("slash-in-dataset", "Bad/Name", _HORIZON, _fixture_series()[:1]),
    ("uppercase-dataset", "ECL", _HORIZON, _fixture_series()[:1]),
    ("horizon-zero", "fixture", 0, _fixture_series()),
    ("horizon-true", "fixture", True, _fixture_series()),
    (
        "series-exactly-first-origin-long",
        "fixture",
        _HORIZON,
        (SeriesInput("fixture/s", _values(1, _START), _PERIOD),),
    ),
    (
        "nan-value",
        "fixture",
        _HORIZON,
        (
            SeriesInput(
                "fixture/nan",
                (*_values(1, 300)[:100], float("nan"), *_values(1, 300)[101:]),
                _PERIOD,
            ),
        ),
    ),
    (
        "mixed-periods",
        "fixture",
        _HORIZON,
        (_fixture_series()[0], SeriesInput("fixture/c", _values(3, 300), 12)),
    ),
)


@pytest.mark.parametrize(
    ("dataset", "horizon", "series"),
    [(case[1], case[2], case[3]) for case in _INVALID_INPUT_CASES],
    ids=[case[0] for case in _INVALID_INPUT_CASES],
)
def test_invalid_inputs_raise_and_create_nothing(
    dataset: str, horizon: int, series: tuple[SeriesInput, ...], tmp_path: Path
) -> None:
    with pytest.raises(CacheError):
        build_cache(dataset, _SOURCE, series, tmp_path, horizon=horizon)
    assert list(tmp_path.iterdir()) == []


def test_origin_and_expert_bounds(built: tuple[Path, CacheManifest]) -> None:
    root, manifest = built
    loaded = load_series(root, manifest, "fixture/b")
    assert loaded.last_origin == 349
    assert len(loaded.at(_START)) == len(EXPERT_NAMES)
    for bad in (_START - 1, loaded.last_origin + 1):
        with pytest.raises(CacheError, match="outside"):
            loaded.at(bad)
    with pytest.raises(CacheError, match="must be an int"):
        loaded.at(True)
    with pytest.raises(CacheError, match="unknown expert"):
        loaded.expert_forecast(_START, "lightgbm")


def test_no_look_ahead_through_the_cache(tmp_path: Path) -> None:
    cut = 320
    shared = _values(7, 350)
    altered = shared[: cut + 1] + tuple(v * 1.5 + 3.0 for v in shared[cut + 1 :])
    _build(tmp_path, (SeriesInput("la/x", shared, _PERIOD), SeriesInput("la/y", altered, _PERIOD)))
    file_x = (tmp_path / "fixture" / "series" / "0000.f64").read_bytes()
    file_y = (tmp_path / "fixture" / "series" / "0001.f64").read_bytes()
    prefix = (cut - _START + 1) * len(EXPERT_NAMES) * _HORIZON * 8
    assert file_x[:prefix] == file_y[:prefix]  # origins <= cut cannot see later values
    assert file_x != file_y


def test_golden_cache_id(built: tuple[Path, CacheManifest]) -> None:
    """Pin the fixture's cache id.

    If this moves, either the cache layout or manifest changed (bump
    CACHE_VERSION) or an expert's forecast values changed (bump
    EXPERTS_VERSION).
    """
    root, _ = built
    golden = "4a0f7ddaeb9fd816020b2b55ee1d2427334b6a2e10ad30d9c61ed08851a75469"
    assert cache_id(root, "fixture") == golden
