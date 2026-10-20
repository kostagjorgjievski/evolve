"""Fresh-seed holdout replicates of the scan presets (TASK_016 PR C).

A holdout dataset replays one scan preset's families and grammar at scan
indexes the dev scan caches never use, ``[HOLDOUT_INDEX_START,
HOLDOUT_INDEX_STOP)``. The worlds come from the same preset-to-seed rule
as the dev scan datasets (:func:`~evolvets.arena.synthetic.presets.scan_world_seed`),
so a holdout replicate is drawn from the same distribution by construction,
at fresh seeds. Holdout scores are never used for selection: they score
accepted edits only, and practice-world generation and live-loop revision
must not draw from this range either (docs/ROADMAP.md entry 22).
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from evolvets.arena.synthetic.grammar import SyntheticTask, make_task
from evolvets.arena.synthetic.presets import SCAN_PRESETS, scan_world_seed

HOLDOUT_INDEX_START = 1000  # first scan index of every holdout replicate
HOLDOUT_INDEX_STOP = 2000  # exclusive; the holdout range is [START, STOP)


@dataclass(frozen=True)
class HoldoutPreset:
    """One holdout dataset: which scan preset it replicates, and from which index."""

    name: str  # the cache dataset name
    base: str  # the scan preset it replicates (a key of SCAN_PRESETS)
    index_start: int  # == HOLDOUT_INDEX_START


HOLDOUT_DATASETS: tuple[str, ...] = ("synthetic-shift2-holdout", "synthetic-shift3-holdout")
HOLDOUT_PRESETS: Mapping[str, HoldoutPreset] = MappingProxyType(
    {
        "synthetic-shift2-holdout": HoldoutPreset(
            name="synthetic-shift2-holdout",
            base="synthetic-shift2",
            index_start=HOLDOUT_INDEX_START,
        ),
        "synthetic-shift3-holdout": HoldoutPreset(
            name="synthetic-shift3-holdout",
            base="synthetic-shift3",
            index_start=HOLDOUT_INDEX_START,
        ),
    }
)


def holdout_series_id(family: str, i: int) -> str:
    """Return the series id of holdout index ``i`` of ``family``."""
    return f"{family}/{HOLDOUT_INDEX_START + i}"


def holdout_tasks(name: str, family: str, n: int) -> list[SyntheticTask]:
    """Return the first ``n`` holdout tasks of ``family`` under holdout dataset ``name``."""
    if name not in HOLDOUT_PRESETS:
        raise ValueError(f"dataset must be one of {list(HOLDOUT_DATASETS)}, got {name!r}")
    preset = HOLDOUT_PRESETS[name]
    base = SCAN_PRESETS[preset.base]
    if family not in base.families:
        raise ValueError(
            f"family must be one of {list(base.families)} for preset {preset.base!r}, "
            f"got {family!r}"
        )
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(
            f"n must be an int in [0, {HOLDOUT_INDEX_STOP - HOLDOUT_INDEX_START}], got {n!r}"
        )
    if n < 0 or n > HOLDOUT_INDEX_STOP - HOLDOUT_INDEX_START:
        raise ValueError(f"n must be in [0, {HOLDOUT_INDEX_STOP - HOLDOUT_INDEX_START}], got {n}")
    return [
        make_task(family, scan_world_seed(preset.base, family, preset.index_start + i), base.config)
        for i in range(n)
    ]
