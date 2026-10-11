"""LabelRelease: targets and history are handed out only once matured (PR A, test 1)."""

import pytest

from evolvets.arena.errors import ArenaError, LookAheadError
from evolvets.arena.labels import LabelRelease

VALUES = tuple(float(i) for i in range(10))


@pytest.mark.parametrize("now", [5, 6, 9])
def test_targets_at_and_after_the_boundary(now):
    """Origin 3 with H = 2 reads y_4, y_5 once now >= 5."""
    release = LabelRelease(VALUES, 2)
    assert release.targets(3, now) == (4.0, 5.0)


def test_targets_look_ahead_at_now_minus_one():
    """One step before the horizon passes, the request is refused."""
    release = LabelRelease(VALUES, 2)
    with pytest.raises(LookAheadError):
        release.targets(3, 4)


def test_targets_beyond_the_series():
    """A target the series does not contain is an arena problem, not look-ahead."""
    release = LabelRelease(VALUES, 2)
    with pytest.raises(LookAheadError):
        release.targets(3, 4)  # sanity: not yet matured
    with pytest.raises(ArenaError) as exc:
        release.targets(8, 10)
    assert not isinstance(exc.value, LookAheadError)
    with pytest.raises(ArenaError):
        release.targets(9, 20)


@pytest.mark.parametrize("bad", [True, -1, "3"])
def test_targets_rejects_non_int_arguments(bad):
    """True, -1 and a string are refused for both origin and now."""
    release = LabelRelease(VALUES, 2)
    with pytest.raises(ArenaError):
        release.targets(bad, 5)
    with pytest.raises(ArenaError):
        release.targets(3, bad)


def test_history_slices_and_guards():
    """history returns values[:end] up to one past now, and no further."""
    release = LabelRelease(VALUES, 2)
    assert release.history(6, 5) == VALUES[:6]  # end == now + 1 is allowed
    assert release.history(4, 5) == VALUES[:4]
    with pytest.raises(LookAheadError):
        release.history(7, 5)  # end == now + 2
    with pytest.raises(ArenaError):
        release.history(11, 20)  # past the end of the series
    with pytest.raises(ArenaError):
        release.history(-1, 5)


def test_constructor_rejects_bad_values_and_horizon():
    """NaN, inf, bools and horizon 0 are refused; ints are accepted."""
    with pytest.raises(ArenaError):
        LabelRelease((1.0, float("nan"), 2.0), 2)
    with pytest.raises(ArenaError):
        LabelRelease((1.0, float("inf"), 2.0), 2)
    with pytest.raises(ArenaError):
        LabelRelease((1.0, True, 2.0), 2)
    with pytest.raises(ArenaError):
        LabelRelease(VALUES, 0)
    with pytest.raises(ArenaError):
        LabelRelease(VALUES, True)
    assert len(LabelRelease((1, 2, 3), 1)) == 3  # ints and len() are fine
    assert LabelRelease((1, 2, 3), 1).targets(0, 1) == (2.0,)


def test_values_are_copied():
    """Mutating the sequence passed in cannot change what the release holds."""
    source = [1.0, 2.0, 3.0]
    release = LabelRelease(source, 1)
    source[2] = 99.0
    assert release.targets(1, 2) == (3.0,)
