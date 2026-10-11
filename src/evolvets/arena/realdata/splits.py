"""Chronological splits with horizon-wide gaps between segments.

Segments are ranges of forecast origins. With ``gap = MAX_HORIZON``, every
origin ``t`` in a segment has all its targets ``t + h`` for ``h <= gap``
strictly before the next segment's start, so dev targets never fall in the
sealed segment. Experts may still read all history before their origin;
that is not leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MAX_HORIZON = 168
SPLIT_PERCENT = (
    20,
    35,
    30,
)  # warmup, dev, sealed, as integer percent of usable length; reserve gets the rest

SegmentName = Literal["warmup", "dev", "sealed", "reserve"]


class SplitError(ValueError):
    """A chronological split cannot be cut for this length and gap."""


@dataclass(frozen=True)
class Segment:
    """A half-open range ``[start, end)`` of forecast origins."""

    name: SegmentName
    start: int
    end: int  # exclusive


@dataclass(frozen=True)
class Split:
    """Four chronological segments with ``gap`` origins between neighbours."""

    length: int
    gap: int
    segments: tuple[Segment, Segment, Segment, Segment]  # warmup, dev, sealed, reserve


def chronological_split(length: int, gap: int = MAX_HORIZON) -> Split:
    """Cut ``[0, length)`` into warmup, dev, sealed and reserve segments.

    The usable length ``length - 3 * gap`` is split by :data:`SPLIT_PERCENT`
    with integer floor arithmetic, so the boundaries are deterministic.
    Raises :class:`SplitError` if any segment would be empty.
    """
    for name, value in (("length", length), ("gap", gap)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise SplitError(f"{name} must be an int, got {value!r}")
    if gap < 1:
        raise SplitError(f"gap must be >= 1, got {gap}")
    usable = length - 3 * gap
    warmup_len = usable * SPLIT_PERCENT[0] // 100
    dev_len = usable * SPLIT_PERCENT[1] // 100
    sealed_len = usable * SPLIT_PERCENT[2] // 100
    reserve_len = usable - warmup_len - dev_len - sealed_len
    if min(warmup_len, dev_len, sealed_len, reserve_len) <= 0:
        raise SplitError(
            f"cannot split length {length} with gap {gap}: a segment would be empty "
            f"(smallest valid length is {3 * gap + 5})"
        )
    warmup = Segment("warmup", 0, warmup_len)
    dev = Segment("dev", warmup.end + gap, warmup.end + gap + dev_len)
    sealed = Segment("sealed", dev.end + gap, dev.end + gap + sealed_len)
    reserve = Segment("reserve", sealed.end + gap, length)
    return Split(length=length, gap=gap, segments=(warmup, dev, sealed, reserve))
