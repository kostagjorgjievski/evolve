"""Chronological split tests: pinned boundaries and the gap invariant."""

from __future__ import annotations

import pytest

from evolvets.arena.realdata.splits import (
    MAX_HORIZON,
    SPLIT_PERCENT,
    Segment,
    Split,
    SplitError,
    chronological_split,
)


def _names(split: Split) -> tuple[str, str, str, str]:
    return tuple(segment.name for segment in split.segments)  # type: ignore[return-value]


def _etth1_split() -> Split:
    return chronological_split(17_420)


def _ecl_split() -> Split:
    return chronological_split(26_304)


def test_constants_are_pinned() -> None:
    assert MAX_HORIZON == 168
    assert SPLIT_PERCENT == (20, 35, 30)


def test_etth1_boundaries() -> None:
    split = _etth1_split()
    assert split == Split(
        length=17_420,
        gap=168,
        segments=(
            Segment("warmup", 0, 3383),
            Segment("dev", 3551, 9471),
            Segment("sealed", 9639, 14_713),
            Segment("reserve", 14_881, 17_420),
        ),
    )


def test_ecl_boundaries() -> None:
    split = _ecl_split()
    assert split == Split(
        length=26_304,
        gap=168,
        segments=(
            Segment("warmup", 0, 5160),
            Segment("dev", 5328, 14_358),
            Segment("sealed", 14_526, 22_266),
            Segment("reserve", 22_434, 26_304),
        ),
    )


@pytest.mark.parametrize("gap", [1, 7, 168])
def test_gap_invariant_for_all_small_lengths(gap: int) -> None:
    """Segments are ordered, non-empty, gap-separated, and origins stay inside."""
    for length in range(3 * gap + 5, 3 * gap + 401):
        split = chronological_split(length, gap)
        assert _names(split) == ("warmup", "dev", "sealed", "reserve")
        warmup, dev, sealed, reserve = split.segments
        assert split.length == length and split.gap == gap
        assert warmup.start == 0
        assert reserve.end == length
        previous = warmup
        for segment in (dev, sealed, reserve):
            assert previous.end <= segment.start
            assert segment.start - previous.end == gap
            assert segment.start < segment.end
            # every origin t in a segment has all targets t + h (h <= gap)
            # strictly before the next segment's start
            assert previous.end - 1 + gap < segment.start
            previous = segment


@pytest.mark.parametrize(
    ("length", "gap"),
    [
        (3 * 7 + 4, 7),  # usable length 4: a segment would be empty
        (10, 0),  # gap must be >= 1
        (True, 7),  # bools are not ints
        (1_000, True),
        (1_000.0, 7),  # floats are not ints
        ("1000", 7),
    ],
)
def test_invalid_arguments_raise(length: object, gap: object) -> None:
    with pytest.raises(SplitError):
        chronological_split(length, gap)  # type: ignore[arg-type]


def test_smallest_valid_length_splits_into_four_nonempty_segments() -> None:
    gap = 7
    split = chronological_split(3 * gap + 5, gap)
    assert all(segment.end > segment.start for segment in split.segments)
