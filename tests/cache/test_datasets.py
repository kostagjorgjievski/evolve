"""Datasets tests: public inputs come from public data only.

The real-data tests reuse the TASK_005 fixture helpers (small deterministic
ETTh1/ECL stand-ins) so the cache's view of the real data is checked against
the same fixtures that pin the processed files themselves.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from evolvets.arena.realdata import sources
from evolvets.arena.realdata.fetch import fetch, raw_path
from evolvets.arena.realdata.prepare import (
    PrepareError,
    prepare,
    read_split_manifest,
    splits_manifest_path,
)
from evolvets.arena.synthetic import GENERATOR_VERSION
from evolvets.arena.synthetic.grammar import FAMILIES
from evolvets.arena.synthetic.seeds import dev_tasks
from evolvets.cache import datasets as datasets_module
from evolvets.cache.datasets import CACHE_DATASETS, public_inputs
from evolvets.cache.store import CacheError
from tests.arena.test_realdata_prepare import (
    _build_etth1_fixture,
    _install,
    _processed,
    _record_opens,
)


@pytest.fixture()
def prepared_etth1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """Fetched and prepared fixture ETTh1; returns (data_root, manifest_dir)."""
    etth1 = _build_etth1_fixture(tmp_path / "fixtures")
    _install(monkeypatch, etth1=etth1)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    fetch(etth1, data_root, manifest_dir, today=date(2026, 10, 11))
    prepare("etth1", data_root, manifest_dir)
    return data_root, manifest_dir


def test_real_inputs_are_public_values_only(
    prepared_etth1: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """public_inputs reads the public file only: sealed and raw are never opened."""
    data_root, manifest_dir = prepared_etth1
    _processed(data_root, "etth1", "sealed").unlink()
    raw_path(data_root, sources.SOURCES["etth1"]).unlink()
    spec = read_split_manifest(manifest_dir, "etth1")

    opened: list[str] = []
    _record_opens(monkeypatch, opened)
    source, series = public_inputs("etth1", data_root, manifest_dir)
    assert source == {
        "kind": "realdata",
        "dataset": "etth1",
        "realdata_version": spec.realdata_version,
        "public_sha256": spec.public_sha256,
    }
    assert len(series) == 7  # manifest order, values[:sealed.start] only
    hufl = series[0]
    assert hufl.series_id == "etth1/HUFL"
    assert hufl.period == 24
    assert hufl.values == tuple(1 + index / 100 for index in range(344))
    assert all(entry.period == 24 and len(entry.values) == 344 for entry in series)
    allowed = {
        str(splits_manifest_path(manifest_dir, "etth1")),
        str(_processed(data_root, "etth1", "public")),
    }
    assert opened, "public_inputs should have opened its manifest and public file"
    assert set(opened) <= allowed, opened


def test_real_inputs_need_the_prepared_data(prepared_etth1: tuple[Path, Path]) -> None:
    """Without a prepared public file there is nothing public to cache."""
    data_root, manifest_dir = prepared_etth1
    _processed(data_root, "etth1", "public").unlink()
    with pytest.raises(PrepareError, match="is missing"):
        public_inputs("etth1", data_root, manifest_dir)


def test_synthetic_inputs() -> None:
    source, series = public_inputs(
        "synthetic-dev", Path("data"), Path("data/manifests"), per_family=2
    )
    assert source == {
        "kind": "synthetic",
        "pool": "dev",
        "generator_version": GENERATOR_VERSION,
        "per_family": "2",
    }
    expected_ids = [f"{family}/{i}" for family in FAMILIES for i in range(2)]
    assert [entry.series_id for entry in series] == expected_ids  # FAMILIES order
    by_id = {entry.series_id: entry for entry in series}
    for family in FAMILIES:
        for i, task in enumerate(dev_tasks(family, 2)):
            entry = by_id[f"{family}/{i}"]
            assert entry.values == task.world.y
            assert entry.period == task.world.spec.period


@pytest.mark.parametrize("per_family", [None, 0, -1, True])
def test_synthetic_inputs_reject_a_missing_or_small_per_family(per_family: object) -> None:
    with pytest.raises(CacheError, match="per_family"):
        public_inputs("synthetic-dev", Path("data"), Path("data/manifests"), per_family=per_family)  # type: ignore[arg-type]


class _RaisesOnUse:
    """Passes World's length check but raises on any real access."""

    def __init__(self, length: int) -> None:
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"public_inputs touched a latent World field ({name})")


def test_synthetic_inputs_never_touch_latent_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only world.y and world.spec.period are read; mean/noise/sigma/phi are not."""

    def fake_dev_tasks(family: str, n: int) -> list[object]:
        out = []
        for task in dev_tasks(family, n):
            length = len(task.world.y)
            world = replace(
                task.world,
                mean=_RaisesOnUse(length),
                noise=_RaisesOnUse(length),
                sigma=_RaisesOnUse(length),
                phi=_RaisesOnUse(length),
            )
            out.append(replace(task, world=world))
        return out

    monkeypatch.setattr(datasets_module, "dev_tasks", fake_dev_tasks)
    source, series = public_inputs("synthetic-dev", tmp_path, tmp_path, per_family=1)
    assert source["per_family"] == "1"
    assert len(series) == len(FAMILIES)
    for entry, family in zip(series, FAMILIES, strict=True):
        assert entry.values == dev_tasks(family, 1)[0].world.y


def test_unknown_dataset_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CacheError, match="unknown dataset"):
        public_inputs("nope", tmp_path, tmp_path)
    assert CACHE_DATASETS == ("etth1", "ecl", "synthetic-dev")
