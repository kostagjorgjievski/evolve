"""Holdout tests: registry, worlds, disjointness from dev, scan unchanged (TASK_016 PR C).

The holdout datasets are fresh-seed replicates of the scan presets at indexes
from 1000 (``docs/tasks/TASK_016_forecast_updaters_and_holdout.md`` PR C): the
same preset, the same seed rule, never used for selection.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolvets.arena.dev import load_dev_pool
from evolvets.arena.errors import ArenaError
from evolvets.arena.sealed.names import SEALED_DATASETS
from evolvets.arena.synthetic.grammar import make_task
from evolvets.arena.synthetic.holdout import (
    HOLDOUT_DATASETS,
    HOLDOUT_INDEX_START,
    HOLDOUT_INDEX_STOP,
    HOLDOUT_PRESETS,
    HoldoutPreset,
    holdout_series_id,
    holdout_tasks,
)
from evolvets.arena.synthetic.presets import (
    SCAN_DATASETS,
    SCAN_PRESETS,
    scan_world_seed,
)
from evolvets.cache.datasets import CACHE_DATASETS, public_inputs
from evolvets.cache.store import CacheError, build_cache, cache_dir
from tests.arena.test_synthetic_presets import GOLDEN_SHIFT1_SUDDEN_0

#: The holdout range as one number, pinned: [1000, 2000).
_HOLDOUT_WIDTH = HOLDOUT_INDEX_STOP - HOLDOUT_INDEX_START


# --- Test 1: the registry ------------------------------------------------------------------------


def test_registry_covers_holdout_datasets_in_order() -> None:
    """The mapping iterates in HOLDOUT_DATASETS order and every name is its own key."""
    assert tuple(HOLDOUT_PRESETS) == HOLDOUT_DATASETS
    for name in HOLDOUT_DATASETS:
        assert HOLDOUT_PRESETS[name].name == name
        assert cache_dir(Path("cache_store"), name) == Path("cache_store") / name


def test_each_holdout_replicates_its_shift_preset() -> None:
    assert HOLDOUT_PRESETS["synthetic-shift2-holdout"] == HoldoutPreset(
        name="synthetic-shift2-holdout",
        base="synthetic-shift2",
        index_start=HOLDOUT_INDEX_START,
    )
    assert HOLDOUT_PRESETS["synthetic-shift3-holdout"].base == "synthetic-shift3"
    for preset in HOLDOUT_PRESETS.values():
        assert preset.base in SCAN_PRESETS
        assert preset.index_start == HOLDOUT_INDEX_START


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        HOLDOUT_PRESETS["synthetic-shift2-holdout"] = HOLDOUT_PRESETS[  # type: ignore[index]
            "synthetic-shift3-holdout"
        ]


def test_holdout_datasets_are_disjoint_from_every_other_pool() -> None:
    assert not set(HOLDOUT_DATASETS) & set(CACHE_DATASETS)
    assert not set(HOLDOUT_DATASETS) & set(SCAN_DATASETS)
    assert not set(HOLDOUT_DATASETS) & set(SEALED_DATASETS)


# --- Test 2: the worlds --------------------------------------------------------------------------


@pytest.mark.parametrize("name", HOLDOUT_DATASETS)
@pytest.mark.parametrize("family", SCAN_PRESETS["synthetic-shift2"].families)
def test_holdout_tasks_match_hand_built_tasks(name: str, family: str) -> None:
    """Each task is exactly make_task at its own scan seed, under the base's config."""
    base = HOLDOUT_PRESETS[name].base
    for index, task in enumerate(holdout_tasks(name, family, 3)):
        hand = make_task(
            family,
            scan_world_seed(base, family, HOLDOUT_INDEX_START + index),
            SCAN_PRESETS[base].config,
        )
        assert task.world.y == hand.world.y
        assert task.change_kinds == hand.change_kinds


@pytest.mark.parametrize("name", HOLDOUT_DATASETS)
def test_holdout_tasks_of_zero_is_empty(name: str) -> None:
    for family in SCAN_PRESETS[HOLDOUT_PRESETS[name].base].families:
        assert holdout_tasks(name, family, 0) == []


@pytest.mark.parametrize("bad_n", [1001, -1, True])
def test_holdout_tasks_reject_a_bad_n(bad_n: int) -> None:
    with pytest.raises(ValueError, match="n must be"):
        holdout_tasks("synthetic-shift2-holdout", "sudden", bad_n)


def test_holdout_tasks_reject_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="dataset must be"):
        holdout_tasks("synthetic-shift1-holdout", "sudden", 1)  # type: ignore[arg-type]


def test_holdout_tasks_reject_a_family_the_base_does_not_have() -> None:
    with pytest.raises(ValueError, match="family must be"):
        holdout_tasks("synthetic-shift2-holdout", "stationary", 1)


# --- Test 3: disjoint from the dev scan range ----------------------------------------------------


@pytest.mark.parametrize("name", HOLDOUT_DATASETS)
def test_holdout_seeds_are_disjoint_from_the_dev_scan_range(name: str) -> None:
    """All 1000 dev scan seeds per family, computed; the holdout range shares none."""
    base = HOLDOUT_PRESETS[name].base
    for family in SCAN_PRESETS[base].families:
        dev = {scan_world_seed(base, family, index) for index in range(HOLDOUT_INDEX_START)}
        assert len(dev) == HOLDOUT_INDEX_START
        holdout = {
            scan_world_seed(base, family, HOLDOUT_INDEX_START + index) for index in range(50)
        }
        assert not holdout & dev


@pytest.mark.parametrize("name", HOLDOUT_DATASETS)
def test_the_first_holdout_world_differs_from_dev_scan_index_0(name: str) -> None:
    base = HOLDOUT_PRESETS[name].base
    for family in SCAN_PRESETS[base].families:
        first = holdout_tasks(name, family, 1)[0]
        dev_first = make_task(family, scan_world_seed(base, family, 0), SCAN_PRESETS[base].config)
        assert first.world.y != dev_first.world.y


# --- Test 4: same distribution, by construction ---------------------------------------------------


@pytest.mark.parametrize("name", HOLDOUT_DATASETS)
def test_each_holdout_uses_its_base_config_and_families(name: str) -> None:
    """The base's config is used as-is (identity) and its families are the shift ones."""
    base = SCAN_PRESETS[HOLDOUT_PRESETS[name].base]
    assert base.families == ("sudden", "gradual", "recurring", "false_drift", "mixed")
    task = holdout_tasks(name, "sudden", 1)[0]
    hand = make_task(
        "sudden", scan_world_seed(base.name, "sudden", HOLDOUT_INDEX_START), base.config
    )
    assert task.world.spec == hand.world.spec


def test_holdout_series_id_carries_the_index_start() -> None:
    assert holdout_series_id("sudden", 0) == "sudden/1000"
    assert holdout_series_id("mixed", 49) == "mixed/1049"
    assert _HOLDOUT_WIDTH == 1000


# --- Test 5: the scan datasets are unchanged ------------------------------------------------------


def test_the_scan_presets_and_their_seed_rule_are_unchanged() -> None:
    """The TASK_015 golden seed still holds, so the scan seed domain is untouched."""
    assert tuple(SCAN_PRESETS) == SCAN_DATASETS
    assert scan_world_seed("synthetic-shift1", "sudden", 0) == GOLDEN_SHIFT1_SUDDEN_0


def test_a_scan_per_family_reaching_the_holdout_range_is_refused(tmp_path: Path) -> None:
    """public_inputs refuses a scan per_family above 1000 with a CacheError."""
    with pytest.raises(CacheError, match="holdout range"):
        public_inputs("synthetic-shift1", tmp_path, tmp_path, per_family=HOLDOUT_INDEX_START + 1)


def test_a_scan_manifest_claiming_a_holdout_sized_per_family_is_refused(tmp_path: Path) -> None:
    """load_dev_pool refuses a hand-made scan manifest whose per_family reaches 1000."""
    cache_root = tmp_path / "cache_store"
    source, series = public_inputs("synthetic-shift1", tmp_path, tmp_path, per_family=1)
    build_cache("synthetic-shift1", source, series, cache_root)
    manifest_file = cache_dir(cache_root, "synthetic-shift1") / "manifest.json"
    obj = json.loads(manifest_file.read_text(encoding="utf-8"))
    obj["source"]["per_family"] = str(HOLDOUT_INDEX_START + 1)
    manifest_file.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ArenaError, match="holdout range"):
        load_dev_pool("synthetic-shift1", cache_root, tmp_path, tmp_path)
