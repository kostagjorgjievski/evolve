"""Seed tests: pool separation, salt rules, dev task derivation, headroom."""

import pytest

from evolvets.arena.synthetic.grammar import FAMILIES, make_task
from evolvets.arena.synthetic.oracle import true_best_forecast
from evolvets.arena.synthetic.seeds import dev_tasks, world_seed

_SALT = b"\x9f" * 32


def test_world_seed_deterministic_and_in_range():
    first = world_seed("dev", "sudden", 5)
    assert first == world_seed("dev", "sudden", 5)
    assert 0 <= first < 2**63


def test_dev_pool_takes_no_salt():
    with pytest.raises(ValueError, match="salt"):
        world_seed("dev", "sudden", 0, salt=b"x")


@pytest.mark.parametrize("salt", [b"", b"x" * 31])
def test_sealed_pool_needs_a_long_salt(salt):
    with pytest.raises(ValueError, match="32 bytes"):
        world_seed("sealed", "sudden", 0, salt=salt)
    assert 0 <= world_seed("sealed", "sudden", 0, salt=_SALT) < 2**63


def test_pools_families_and_salts_are_separated():
    """Same family and index: dev, sealed and different salts give different seeds."""
    dev = world_seed("dev", "sudden", 7)
    assert dev != world_seed("sealed", "sudden", 7, salt=_SALT)
    assert world_seed("sealed", "sudden", 7, salt=_SALT) != world_seed(
        "sealed", "sudden", 7, salt=b"\x01" * 32
    )
    for family in (f for f in FAMILIES if f != "sudden"):
        assert dev != world_seed("dev", family, 7)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"pool": "live", "family": "sudden", "index": 0}, "pool"),
        ({"pool": "dev", "family": "bogus", "index": 0}, "family"),
        ({"pool": "dev", "family": "sudden", "index": -1}, ">= 0"),
        ({"pool": "dev", "family": "sudden", "index": True}, "int"),
    ],
)
def test_world_seed_rejects_bad_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        world_seed(**kwargs)


def test_dev_seed_space_is_distinct_over_10000_indices():
    dev = {world_seed("dev", "sudden", index) for index in range(10_000)}
    assert len(dev) == 10_000
    for index in range(10_000):
        assert world_seed("sealed", "sudden", index, salt=_SALT) not in dev


def test_dev_tasks_match_hand_built_tasks():
    """dev_tasks is exactly the tasks built by hand from world_seed."""
    assert dev_tasks("sudden", 3) == [
        make_task("sudden", world_seed("dev", "sudden", index)) for index in range(3)
    ]


def test_true_best_forecast_beats_naive_on_dev_sudden_tasks():
    """Headroom sanity over 20 dev sudden tasks, H = 24, at h = 1 and h = 24."""
    horizon = 24
    for task in dev_tasks("sudden", 20):
        world = task.world
        oracle_abs = {1: [], 24: []}
        naive_abs = {1: [], 24: []}
        for origin in range(len(world.y) - horizon):
            forecast = true_best_forecast(world, origin, horizon)
            for h in (1, 24):
                oracle_abs[h].append(abs(forecast[h - 1] - world.y[origin + h]))
                naive_abs[h].append(abs(world.y[origin] - world.y[origin + h]))
        for h in (1, 24):
            oracle_mae = sum(oracle_abs[h]) / len(oracle_abs[h])
            naive_mae = sum(naive_abs[h]) / len(naive_abs[h])
            assert oracle_mae < naive_mae
