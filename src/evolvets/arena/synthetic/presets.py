"""Seeded scan presets: harder synthetic worlds for the headroom scan (TASK_015).

Each preset names the cache dataset it becomes, the families it draws and the
:class:`~evolvets.arena.synthetic.grammar.GrammarConfig` it draws them with.
The presets were fixed in TASK_015 before any scan ran and never change
after, so every scan world is reproducible from this module alone. The shift
presets form a severity ladder of more, larger and faster changes;
``synthetic-wide`` is an independent wide replicate of the default grammar.
Scan seeds live in their own domain-separated space (the ``scan`` field of
the seed payload), disjoint from the dev and sealed pools.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from evolvets.arena.synthetic.grammar import (
    FAMILIES,
    GrammarConfig,
    SyntheticTask,
    config_to_dict,
    make_task,
)
from evolvets.ledger.canonical import canonical_bytes, sha256_hex

_SEED_MASK = 2**63 - 1
_PREFIX = b"evolvets/generator/v1"
#: The shift presets leave out ``stationary``, the family with nothing planted.
_SHIFT_FAMILIES = ("sudden", "gradual", "recurring", "false_drift", "mixed")


@dataclass(frozen=True)
class ScanPreset:
    """One named scan dataset: its families and the grammar they are drawn with."""

    name: str  # the cache dataset name
    families: tuple[str, ...]  # non-empty, a subsequence of FAMILIES in FAMILIES order
    config: GrammarConfig


def _shift(name: str, **overrides: Any) -> ScanPreset:
    """Build one rung of the severity ladder: ``overrides`` over the grammar defaults."""
    return ScanPreset(name=name, families=_SHIFT_FAMILIES, config=GrammarConfig(**overrides))


SCAN_DATASETS: tuple[str, ...] = (
    "synthetic-shift1",
    "synthetic-shift2",
    "synthetic-shift3",
    "synthetic-wide",
)
SCAN_PRESETS: Mapping[str, ScanPreset] = MappingProxyType(
    {
        "synthetic-shift1": _shift(
            "synthetic-shift1",
            max_changes=6,
            min_segment=120,
            shift_range=(3.0, 6.0),
            new_slope_range=(-0.02, 0.02),
            ramp_range=(10, 100),
            max_bursts=4,
            burst_length_range=(10, 60),
            burst_mult_range=(2.0, 5.0),
        ),
        "synthetic-shift2": _shift(
            "synthetic-shift2",
            max_changes=12,
            min_segment=70,
            shift_range=(4.0, 8.0),
            new_slope_range=(-0.03, 0.03),
            ramp_range=(5, 50),
            max_bursts=6,
            burst_length_range=(5, 50),
            burst_mult_range=(3.0, 6.0),
        ),
        "synthetic-shift3": _shift(
            "synthetic-shift3",
            max_changes=24,
            min_segment=48,
            shift_range=(5.0, 10.0),
            new_slope_range=(-0.05, 0.05),
            ramp_range=(4, 30),
            max_bursts=8,
            burst_length_range=(5, 30),
            burst_mult_range=(3.0, 8.0),
        ),
        "synthetic-wide": ScanPreset(
            name="synthetic-wide", families=FAMILIES, config=GrammarConfig()
        ),
    }
)


def _checked_preset(preset: str, family: str) -> ScanPreset:
    """Return ``SCAN_PRESETS[preset]`` after checking ``family`` belongs to it."""
    if preset not in SCAN_PRESETS:
        raise ValueError(f"preset must be one of {list(SCAN_DATASETS)}, got {preset!r}")
    spec = SCAN_PRESETS[preset]
    if family not in spec.families:
        raise ValueError(
            f"family must be one of {list(spec.families)} for preset {preset!r}, got {family!r}"
        )
    return spec


def scan_world_seed(preset: str, family: str, index: int) -> int:
    """Return the scan world seed of ``preset``/``family``/``index``, in ``[0, 2**63)``.

    The ``scan`` field keeps these seeds domain-separated from the dev and
    sealed pools, which ``world_seed`` draws with the pool name in its place.
    """
    _checked_preset(preset, family)
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"index must be an int, got {index!r}")
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")
    payload = b"\x00".join(
        (
            _PREFIX,
            b"scan",
            preset.encode("utf-8"),
            family.encode("utf-8"),
            str(index).encode("utf-8"),
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _SEED_MASK


def scan_tasks(preset: str, family: str, n: int) -> list[SyntheticTask]:
    """Return the first ``n`` scan tasks of ``family`` under ``preset``, in index order."""
    spec = _checked_preset(preset, family)
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError(f"n must be an int >= 0, got {n!r}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    return [
        make_task(family, scan_world_seed(preset, family, index), spec.config) for index in range(n)
    ]


def preset_to_dict(preset: ScanPreset) -> dict[str, Any]:
    """Return a JSON-ready dict describing ``preset``, for hashing and manifests."""
    return {
        "name": preset.name,
        "families": list(preset.families),
        "config": config_to_dict(preset.config),
    }


def preset_sha256(preset: ScanPreset) -> str:
    """Return the SHA-256 of ``preset``'s canonical JSON, so caches can pin it."""
    return sha256_hex(canonical_bytes(preset_to_dict(preset)))
