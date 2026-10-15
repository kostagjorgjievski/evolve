"""Preset tests: registry, rules, severity ladder, seeds, tasks, hashing (TASK_015 PR A)."""

from __future__ import annotations

from dataclasses import fields, replace
from itertools import pairwise
from pathlib import Path

import pytest

from evolvets.arena.sealed.names import SEALED_DATASETS
from evolvets.arena.synthetic import GENERATOR_VERSION
from evolvets.arena.synthetic.grammar import FAMILIES, GrammarConfig, make_task
from evolvets.arena.synthetic.presets import (
    SCAN_DATASETS,
    SCAN_PRESETS,
    preset_sha256,
    preset_to_dict,
    scan_tasks,
    scan_world_seed,
)
from evolvets.arena.synthetic.seeds import dev_tasks, world_seed
from evolvets.cache.datasets import CACHE_DATASETS
from evolvets.cache.store import cache_dir

#: The golden seed of the first shift1 sudden world, pinned by hand from the rule.
GOLDEN_SHIFT1_SUDDEN_0 = 2164398903049615463

#: Every GrammarConfig field name, in declaration order.
_CONFIG_FIELDS = tuple(field.name for field in fields(GrammarConfig))

#: Every GrammarConfig field with a replacement value that stays valid, so
#: mutating any one field alone can be shown to move the preset hash.
_FIELD_OVERRIDES = {
    "length": 2100,
    "period": 12,
    "warmup": 301,
    "min_segment": 199,
    "max_changes": 4,
    "level_range": (-6.0, 5.0),
    "slope_range": (-0.003, 0.002),
    "new_slope_range": (-0.02, 0.01),
    "season_amp_range": (0.5, 3.0),
    "phi_range": (0.1, 0.9),
    "sigma_range": (0.4, 1.0),
    "shift_range": (2.5, 5.0),
    "ramp_range": (25, 150),
    "max_bursts": 2,
    "burst_length_range": (12, 50),
    "burst_mult_range": (2.5, 4.0),
}


# --- Test 1: the registry ---------------------------------------------------------------------


def test_registry_covers_scan_datasets_in_order() -> None:
    """The mapping iterates in SCAN_DATASETS order and every name is its own key."""
    assert tuple(SCAN_PRESETS) == SCAN_DATASETS
    for name in SCAN_DATASETS:
        assert SCAN_PRESETS[name].name == name
        assert cache_dir(Path("cache_store"), name) == Path("cache_store") / name


def test_scan_datasets_are_disjoint_from_cache_and_sealed() -> None:
    assert not set(SCAN_DATASETS) & set(CACHE_DATASETS)
    assert not set(SCAN_DATASETS) & set(SEALED_DATASETS)


def test_registry_is_read_only() -> None:
    with pytest.raises(TypeError):
        SCAN_PRESETS["synthetic-shift1"] = SCAN_PRESETS["synthetic-shift2"]  # type: ignore[index]


# --- Test 2: the preset rules -------------------------------------------------------------------


@pytest.mark.parametrize("name", SCAN_DATASETS)
def test_families_are_a_subsequence_of_families_in_order(name: str) -> None:
    """Non-empty, duplicate-free, and in the same relative order as FAMILIES."""
    families = SCAN_PRESETS[name].families
    assert families
    assert len(set(families)) == len(families)
    positions = [FAMILIES.index(family) for family in families]
    assert positions == sorted(positions)
    assert set(families) <= set(FAMILIES)


@pytest.mark.parametrize("name", SCAN_DATASETS)
def test_length_period_and_warmup_stay_at_the_defaults(name: str) -> None:
    default = GrammarConfig()
    config = SCAN_PRESETS[name].config
    assert (config.length, config.period, config.warmup) == (
        default.length,
        default.period,
        default.warmup,
    )


def test_wide_is_the_default_grammar_over_all_families() -> None:
    wide = SCAN_PRESETS["synthetic-wide"]
    assert wide.families == FAMILIES
    assert wide.config == GrammarConfig()


def test_shift_presets_match_the_task_table_field_by_field() -> None:
    expected = {
        "synthetic-shift1": dict(
            max_changes=6,
            min_segment=120,
            shift_range=(3.0, 6.0),
            new_slope_range=(-0.02, 0.02),
            ramp_range=(10, 100),
            max_bursts=4,
            burst_length_range=(10, 60),
            burst_mult_range=(2.0, 5.0),
        ),
        "synthetic-shift2": dict(
            max_changes=12,
            min_segment=70,
            shift_range=(4.0, 8.0),
            new_slope_range=(-0.03, 0.03),
            ramp_range=(5, 50),
            max_bursts=6,
            burst_length_range=(5, 50),
            burst_mult_range=(3.0, 6.0),
        ),
        "synthetic-shift3": dict(
            max_changes=24,
            min_segment=48,
            shift_range=(5.0, 10.0),
            new_slope_range=(-0.05, 0.05),
            ramp_range=(4, 30),
            max_bursts=8,
            burst_length_range=(5, 30),
            burst_mult_range=(3.0, 8.0),
        ),
    }
    for name, table in expected.items():
        preset = SCAN_PRESETS[name]
        assert preset.families == ("sudden", "gradual", "recurring", "false_drift", "mixed")
        for field, value in table.items():
            assert getattr(preset.config, field) == value, (name, field)
        others = set(_CONFIG_FIELDS) - set(table) - {"length", "period", "warmup"}
        for field in others:
            assert getattr(preset.config, field) == getattr(GrammarConfig(), field), (name, field)


# --- Test 3: the severity ladder ----------------------------------------------------------------


def test_the_ladder_is_strictly_more_severe() -> None:
    ladder = [SCAN_PRESETS[name].config for name in SCAN_DATASETS[:3]]
    for earlier, later in pairwise(ladder):
        assert later.max_changes > earlier.max_changes
        assert later.max_bursts > earlier.max_bursts
        assert later.min_segment < earlier.min_segment
        assert later.shift_range[0] > earlier.shift_range[0]


def test_the_ladder_plants_strictly_more_changes() -> None:
    """Over 20 sudden worlds the mean change-point count rises rung by rung."""
    means = [
        sum(len(task.world.change_points) for task in scan_tasks(name, "sudden", 20)) / 20
        for name in SCAN_DATASETS[:3]
    ]
    assert means[0] < means[1] < means[2]


# --- Test 4: the seeds ---------------------------------------------------------------------------


def test_scan_seed_is_deterministic_and_in_range() -> None:
    assert scan_world_seed("synthetic-shift1", "sudden", 0) == GOLDEN_SHIFT1_SUDDEN_0
    seed = scan_world_seed("synthetic-wide", "mixed", 3)
    assert seed == scan_world_seed("synthetic-wide", "mixed", 3)
    assert 0 <= seed < 2**63


def test_scan_seeds_are_pairwise_distinct_and_disjoint_from_dev() -> None:
    scan = {
        scan_world_seed(name, family, index)
        for name in SCAN_DATASETS
        for family in SCAN_PRESETS[name].families
        for index in range(50)
    }
    assert len(scan) == sum(len(SCAN_PRESETS[name].families) for name in SCAN_DATASETS) * 50
    dev = {world_seed("dev", family, index) for family in FAMILIES for index in range(50)}
    assert not scan & dev


@pytest.mark.parametrize(
    "preset,family,index,match",
    [
        ("nope", "sudden", 0, "preset"),
        ("synthetic-shift1", "stationary", 0, "family"),
        ("synthetic-shift1", "sudden", -1, ">= 0"),
        ("synthetic-shift1", "sudden", True, "int"),
    ],
)
def test_scan_seed_rejects_bad_arguments(preset: str, family: str, index: int, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        scan_world_seed(preset, family, index)


# --- Test 5: the tasks ---------------------------------------------------------------------------

_PRESET_FAMILY_PAIRS = [
    (name, family) for name in SCAN_DATASETS for family in SCAN_PRESETS[name].families
]


@pytest.mark.parametrize(("name", "family"), _PRESET_FAMILY_PAIRS)
def test_scan_tasks_match_hand_built_tasks(name: str, family: str) -> None:
    """Each task is exactly make_task at its own seed, under the preset's config."""
    tasks = scan_tasks(name, family, 3)
    for index, task in enumerate(tasks):
        hand = make_task(family, scan_world_seed(name, family, index), SCAN_PRESETS[name].config)
        assert task.world.y == hand.world.y
        assert task.change_kinds == hand.change_kinds


def test_scan_tasks_of_zero_is_empty() -> None:
    for name in SCAN_DATASETS:
        for family in SCAN_PRESETS[name].families:
            assert scan_tasks(name, family, 0) == []


def test_wide_worlds_differ_from_the_dev_pool() -> None:
    """synthetic-wide is an independent replicate: index 0 is never the dev task."""
    for family in FAMILIES:
        assert scan_tasks("synthetic-wide", family, 1)[0].world.y != dev_tasks(family, 1)[0].world.y


# --- Test 6: the generator is unchanged ----------------------------------------------------------


def test_generator_version_stays_at_one() -> None:
    assert GENERATOR_VERSION == "1"


# --- Test 7: the hash ----------------------------------------------------------------------------


def test_preset_hash_is_stable_and_moves_with_every_field() -> None:
    preset = SCAN_PRESETS["synthetic-wide"]
    digest = preset_sha256(preset)
    assert preset_sha256(preset) == digest
    for field, value in _FIELD_OVERRIDES.items():
        edited = replace(preset, config=replace(preset.config, **{field: value}))
        assert preset_sha256(edited) != digest, field
        assert preset_to_dict(edited)["config"][field] == value


def test_preset_hash_moves_with_the_family_list() -> None:
    for preset in (SCAN_PRESETS["synthetic-wide"], SCAN_PRESETS["synthetic-shift2"]):
        digest = preset_sha256(preset)
        edited = replace(preset, families=preset.families[:-1])
        assert edited.families
        assert preset_sha256(edited) != digest
