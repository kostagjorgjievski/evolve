"""Dev-pool tests: boundaries, oracle, isolation from sealed/raw, mismatches (tests 1-3)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from evolvets.arena.dev import DevPool, load_dev_pool
from evolvets.arena.errors import ArenaError
from evolvets.arena.realdata import sources
from evolvets.arena.realdata.fetch import fetch, raw_path
from evolvets.arena.realdata.prepare import prepare, splits_manifest_path
from evolvets.arena.synthetic.grammar import FAMILIES
from evolvets.arena.synthetic.oracle import true_best_forecast
from evolvets.arena.synthetic.seeds import dev_tasks
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache, cache_dir
from tests.arena.test_realdata_prepare import (
    _build_etth1_fixture,
    _install,
    _processed,
    _record_opens,
)

ROWS = 1200  # > 3 * MAX_HORIZON + 5; warmup [0, 139), dev [307, 550), sealed from 718
H = 24
SCALE_END, SCORE_START, SCORE_END = 139, 307, 550
SEALED_START = 718
ETTH1_IDS = [
    "etth1/HUFL",
    "etth1/HULL",
    "etth1/MUFL",
    "etth1/MULL",
    "etth1/LUFL",
    "etth1/LULL",
    "etth1/OT",
]


def build_real_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows: int = ROWS, variant: int = 0
) -> tuple[Path, Path, Path]:
    """Fetch, prepare and cache fixture ETTh1; returns (data_root, manifest_dir, cache_root)."""
    etth1 = _build_etth1_fixture(tmp_path / "fixtures", rows=rows, variant=variant)
    _install(monkeypatch, etth1=etth1)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    fetch(etth1, data_root, manifest_dir, today=date(2026, 10, 11))
    prepare("etth1", data_root, manifest_dir)
    cache_root = tmp_path / "cache_store"
    source, series = public_inputs("etth1", data_root, manifest_dir)
    build_cache("etth1", source, series, cache_root)
    return data_root, manifest_dir, cache_root


@pytest.fixture()
def real_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """The 1200-row fixture ETTh1, prepared, with its cache built."""
    return build_real_env(tmp_path, monkeypatch)


@pytest.fixture()
def synthetic_cache(tmp_path: Path) -> Path:
    """A synthetic-dev cache with one series per family."""
    cache_root = tmp_path / "cache_store"
    source, series = public_inputs("synthetic-dev", tmp_path, tmp_path, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    return cache_root


# --- Test 1: the real pool ----------------------------------------------------------------


def test_real_pool_boundaries_and_values(real_env: tuple[Path, Path, Path]) -> None:
    """Manifest order, split boundaries, and values cut at the last scored target."""
    data_root, manifest_dir, cache_root = real_env
    pool = load_dev_pool("etth1", cache_root, data_root, manifest_dir)
    assert isinstance(pool, DevPool)
    assert pool.dataset == "etth1"
    assert pool.cache_root == cache_root
    assert pool.cache_id
    assert [s.series_id for s in pool.series] == ETTH1_IDS
    for entry in pool.series:
        assert entry.period == 24
        assert entry.scale_end == SCALE_END
        assert entry.score_start == SCORE_START
        assert entry.score_end == SCORE_END
        assert entry.change_points == ()
        assert entry.oracle is None
        assert len(entry.values) == SCORE_END + H  # nothing past the last scored target
    hufl = pool.series[0]
    assert hufl.values == tuple(1 + i / 100 for i in range(SCORE_END + H))
    assert len(hufl.values) < SEALED_START  # the public file holds more; the pool keeps less


def test_real_pool_never_opens_sealed_or_raw(
    real_env: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pool loads from the split manifest, public file and cache manifest only."""
    data_root, manifest_dir, cache_root = real_env
    sealed_file = _processed(data_root, "etth1", "sealed")
    raw_file = raw_path(data_root, sources.SOURCES["etth1"])
    sealed_bytes, raw_bytes = sealed_file.read_bytes(), raw_file.read_bytes()
    sealed_file.unlink()
    raw_file.unlink()

    opened: list[str] = []
    _record_opens(monkeypatch, opened)
    pool = load_dev_pool("etth1", cache_root, data_root, manifest_dir)
    assert len(pool.series) == len(ETTH1_IDS)
    allowed = {
        str(splits_manifest_path(manifest_dir, "etth1")),
        str(_processed(data_root, "etth1", "public")),
        str(cache_dir(cache_root, "etth1") / "manifest.json"),
    }
    assert opened, "load_dev_pool should have opened its manifest, public file and cache"
    assert set(opened) <= allowed, opened

    sealed_file.write_bytes(sealed_bytes)  # leave the environment as found
    raw_file.write_bytes(raw_bytes)


# --- Test 2: the synthetic pool -------------------------------------------------------------


def test_synthetic_pool(synthetic_cache: Path) -> None:
    """Six family series, warmup scoring start, world change points and oracle."""
    pool = load_dev_pool("synthetic-dev", synthetic_cache, Path("data"), Path("data/manifests"))
    assert [s.series_id for s in pool.series] == [f"{family}/0" for family in FAMILIES]
    worlds = {f"{family}/0": dev_tasks(family, 1)[0].world for family in FAMILIES}
    for entry in pool.series:
        world = worlds[entry.series_id]
        assert entry.period == world.spec.period
        assert entry.scale_end == 300
        assert entry.score_start == 300  # GrammarConfig().warmup
        assert entry.score_end == 2000 - H
        assert entry.change_points == world.change_points
        assert entry.values == world.y[: 2000 - H + H]
        assert entry.oracle is not None
        assert len(entry.oracle) == 2000 - H - 300
        assert entry.oracle[0] == true_best_forecast(world, 300, H)
        assert entry.oracle[-1] == true_best_forecast(world, entry.score_end - 1, H)


# --- Test 3: pool mismatches -----------------------------------------------------------------


def test_pool_from_a_different_fixture_variant(
    real_env: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache built from variant 0 cannot be paired with variant 1's public values."""
    _data_root, _manifest_dir, cache_root = real_env
    shifted = _build_etth1_fixture(tmp_path / "fixtures", rows=ROWS, variant=1)
    _install(monkeypatch, etth1=shifted)
    other_data, other_manifests = tmp_path / "v1" / "data", tmp_path / "v1" / "manifests"
    fetch(shifted, other_data, other_manifests, today=date(2026, 10, 11))
    prepare("etth1", other_data, other_manifests)
    with pytest.raises(ArenaError, match="public values hash"):
        load_dev_pool("etth1", cache_root, other_data, other_manifests)


def test_pool_refuses_an_edited_generator_version(synthetic_cache: Path) -> None:
    """A manifest whose source.generator_version was rewritten is refused."""
    manifest_file = cache_dir(synthetic_cache, "synthetic-dev") / "manifest.json"
    obj = json.loads(manifest_file.read_text(encoding="utf-8"))
    obj["source"]["generator_version"] = "999"
    manifest_file.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ArenaError, match="generator_version"):
        load_dev_pool("synthetic-dev", synthetic_cache, Path("data"), Path("data/manifests"))


def test_missing_cache_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ArenaError, match=r"manifest\.json"):
        load_dev_pool("etth1", tmp_path / "cache_store", tmp_path, tmp_path)


def test_520_row_fixture_scores_before_the_first_origin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The TASK_005 520-row fixture's dev segment starts at 171, before origin 287."""
    data_root, manifest_dir, cache_root = build_real_env(tmp_path, monkeypatch, rows=520)
    with pytest.raises(ArenaError, match="first origin"):
        load_dev_pool("etth1", cache_root, data_root, manifest_dir)


def test_unknown_dataset_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ArenaError, match="unknown dataset"):
        load_dev_pool("nope", tmp_path, tmp_path, tmp_path)


def test_missing_public_file(real_env: tuple[Path, Path, Path]) -> None:
    """PrepareError inside load_public escapes as ArenaError."""
    data_root, manifest_dir, cache_root = real_env
    _processed(data_root, "etth1", "public").unlink()
    with pytest.raises(ArenaError, match="is missing"):
        load_dev_pool("etth1", cache_root, data_root, manifest_dir)
