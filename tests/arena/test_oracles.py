"""Oracle tests: the switching DP, replay agreement, metrics, look-ahead (TASK_010 PR B).

The PR B list's tests 1 to 10 and 12 live here, numbered as in
docs/tasks/TASK_010_updater_baselines.md; test 11 (the AST guard) extends the
parametrised guard in tests/arena/test_evaluate.py with ``oracles.py``.
"""

from __future__ import annotations

import itertools
import math
import random
from hashlib import sha256
from pathlib import Path

import pytest

from evolvets.arena import oracles
from evolvets.arena.dev import load_dev_pool
from evolvets.arena.errors import ArenaError
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.arena.labels import LabelRelease
from evolvets.arena.oracles import ORACLE_SWITCHES, oracle_dataset, oracle_to_obj, switching_total
from evolvets.arena.replay import replay_series
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.experts.pool import EXPERT_NAMES
from evolvets.foundation.fake import FAKE_SPEC, FakeAdapter
from evolvets.foundation.store import build_foundation_cache
from evolvets.ledger.canonical import canonical_bytes
from evolvets.updater.baselines import BASELINE_IDS, BASELINES, DEFAULT_CONFIGS
from tests.arena.test_dev import SCORE_END, H, build_real_env
from tests.arena.test_evaluate import EqualWeights, OneHot

# --- fixtures -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_pool(tmp_path_factory: pytest.TempPathFactory):
    """The dev pool of a synthetic-dev cache with one series per family."""
    base = tmp_path_factory.mktemp("oracles-synthetic")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-dev", base, base, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    return load_dev_pool("synthetic-dev", cache_root, base, base)


@pytest.fixture(scope="module")
def pooled_env(tmp_path_factory: pytest.TempPathFactory):
    """A synthetic-dev cache plus a fake foundation cache built against it (TASK_009 helpers)."""
    base = tmp_path_factory.mktemp("oracles-pooled")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-dev", base, base, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    foundation_root = base / "foundation_cache"
    build_foundation_cache(
        "synthetic-dev",
        FakeAdapter(FAKE_SPEC.name),
        FAKE_SPEC,
        foundation_root,
        cache_root,
        base,
        base,
        7,
        per_family=1,
    )
    return base, cache_root, foundation_root


@pytest.fixture()
def real_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    """The 1200-row fixture ETTh1, prepared, with its cache built."""
    return build_real_env(tmp_path, monkeypatch)


def _mean(values: list[float]) -> float:
    """The left-to-right loop mean, as the oracle's metrics accumulate."""
    total = 0.0
    for value in values:
        total = total + value
    return total / len(values)


# --- Test 1: the DP on a hand-worked table ---


HAND_LOSSES = [
    [0.0, 0.5, 1.0, 4.0, 4.0, 4.0],  # expert 0: cheapest early
    [3.0, 2.0, 1.5, 1.0, 1.0, 3.0],  # expert 1: cheapest in the middle
    [2.0, 2.0, 2.0, 2.0, 2.5, 2.5],  # expert 2: flat
]


def test_switching_total_hand_worked_table() -> None:
    """m = 0, 1, 2, 5 on a 3 x 6 table, worked by hand:

    - m = 0: the row totals are 13.5, 11.5 and 13.0; the best fixed row is
      expert 1's 11.5.
    - m = 1: expert 0 for k = 0..2 (0 + 0.5 + 1.0 = 1.5), then expert 1 for
      k = 3..5 (1.0 + 1.0 + 3.0 = 5.0) totals 6.5.
    - m = 2: expert 0 (k = 0..2) = 1.5, expert 1 (k = 3..4) = 2.0, expert 2
      (k = 5) = 2.5 totals 6.0, which already reaches the per-origin minimum.
    - m = 5 >= n - 1: the per-origin minimum's total, 6.0.
    """
    assert switching_total(HAND_LOSSES, 0) == 11.5
    assert switching_total(HAND_LOSSES, 1) == 6.5
    assert switching_total(HAND_LOSSES, 2) == 6.0
    assert switching_total(HAND_LOSSES, 5) == 6.0
    row_total = 0.0
    for value in HAND_LOSSES[1]:
        row_total = row_total + value
    assert switching_total(HAND_LOSSES, 0) == row_total  # the best row's loop sum, exactly
    min_total = 0.0
    for k in range(6):
        low = HAND_LOSSES[0][k]
        for e in range(1, 3):
            low = HAND_LOSSES[e][k] if HAND_LOSSES[e][k] < low else low
        min_total = min_total + low
    assert switching_total(HAND_LOSSES, 5) == min_total


# --- Test 2: the DP against brute force ---


def test_switching_total_matches_brute_force() -> None:
    """For 50 random tables the DP equals the minimum over all E**n sequences."""
    rng = random.Random(3)
    for _ in range(50):
        n_experts = rng.randint(2, 4)
        n = rng.randint(1, 7)
        table = [[rng.uniform(0.0, 10.0) for _ in range(n)] for _ in range(n_experts)]
        brute = [math.inf] * (n + 1)  # brute[s]: best total over sequences with exactly s switches
        for sequence in itertools.product(range(n_experts), repeat=n):
            switches = 0
            for k in range(1, n):
                if sequence[k] != sequence[k - 1]:
                    switches = switches + 1
            total = 0.0
            for k in range(n):
                total = total + table[sequence[k]][k]
            if total < brute[switches]:
                brute[switches] = total
        for m in range(n + 1):
            expected = min(brute[: m + 1])
            assert abs(switching_total(table, m) - expected) <= 1e-12


# --- Test 3: monotonicity on the synthetic pool ---


def test_switch_mase_is_monotonic_in_m(synthetic_pool) -> None:
    """More switches never hurt; m = 0 is the best fixed expert; select tops m = 32."""
    result = oracle_dataset(synthetic_pool)
    assert result.switches == ORACLE_SWITCHES
    for entry in result.series:
        for i in range(1, len(entry.switch_mase)):
            assert entry.switch_mase[i] <= entry.switch_mase[i - 1], entry.series_id
        low = entry.expert_mase[0]
        for value in entry.expert_mase[1:]:
            low = value if value < low else low
        assert entry.switch_mase[0] == low, entry.series_id
        assert entry.select_mase <= entry.switch_mase[-1], entry.series_id


# --- Test 4: validation ---


@pytest.mark.parametrize("m", [-1, True, 1.0])
def test_switching_total_rejects_bad_switch_counts(m) -> None:
    with pytest.raises(ArenaError, match="max_switches"):
        switching_total(HAND_LOSSES, m)


def test_switching_total_rejects_bad_tables() -> None:
    with pytest.raises(ArenaError, match="at least one expert"):
        switching_total([], 0)
    with pytest.raises(ArenaError, match="at least one origin"):
        switching_total([[], []], 0)
    with pytest.raises(ArenaError, match="share one length"):
        switching_total([[1.0, 2.0], [1.0]], 0)
    with pytest.raises(ArenaError, match="finite"):
        switching_total([[1.0, float("nan")], [1.0, 2.0]], 0)


# --- Test 5: agreement with one-hot replays ---


@pytest.mark.parametrize("e", range(len(EXPERT_NAMES)))
def test_expert_mase_equals_the_one_hot_replay(e: int, synthetic_pool) -> None:
    """Playing expert e alone scores, per series, exactly expert_mase[e]."""
    replay = replay_dataset(lambda: OneHot(e), {}, 0, synthetic_pool)
    assert replay.status == "ok"
    oracle = oracle_dataset(synthetic_pool)
    assert [score.series_id for score in replay.series] == [
        entry.series_id for entry in oracle.series
    ]
    for score, entry in zip(replay.series, oracle.series, strict=True):
        assert score.mase == entry.expert_mase[e], entry.series_id


# --- Test 6: metrics ---


def test_metrics_keys_and_values(synthetic_pool) -> None:
    """The exact key set, and every metric as the left-to-right mean over series."""
    result = oracle_dataset(synthetic_pool)
    names = result.expert_names
    expected_keys = (
        {"n_series", "best_expert_mase", "global_expert_mase", "select_mase"}
        | {f"switch_mase_m{m}" for m in ORACLE_SWITCHES}
        | {f"expert_mase/{name}" for name in names}
    )
    assert set(result.metrics) == expected_keys
    entries = list(result.series)
    assert result.metrics["n_series"] == float(len(entries))
    assert result.metrics["best_expert_mase"] == _mean([min(e.expert_mase) for e in entries])
    per_expert = [_mean([entry.expert_mase[i] for entry in entries]) for i in range(len(names))]
    for i, name in enumerate(names):
        assert result.metrics[f"expert_mase/{name}"] == per_expert[i]
    assert result.metrics["global_expert_mase"] == min(per_expert)
    assert result.global_expert == names[per_expert.index(min(per_expert))]
    for index, m in enumerate(ORACLE_SWITCHES):
        assert result.metrics[f"switch_mase_m{m}"] == _mean(
            [entry.switch_mase[index] for entry in entries]
        )
    assert result.metrics["select_mase"] == _mean([entry.select_mase for entry in entries])
    assert result.metrics["best_expert_mase"] <= result.metrics["global_expert_mase"]


def test_pooled_pool_gains_the_foundation_expert(pooled_env) -> None:
    """A simple+foundation pool's oracles cover the fake model as a sixth expert."""
    base, cache_root, foundation_root = pooled_env
    pool = load_dev_pool(
        "synthetic-dev",
        cache_root,
        base,
        base,
        pool_spec="simple+foundation",
        foundation_root=foundation_root,
        foundation_models=("fake",),
    )
    result = oracle_dataset(pool)
    assert result.pool_spec == "simple+foundation"
    assert result.expert_names == (*EXPERT_NAMES, "fake")
    assert "expert_mase/fake" in result.metrics
    assert result.foundation_cache_ids == (pool.foundation[0].cache_id,)
    assert result.leakage_risk is False  # no model's corpus holds a synthetic dataset


# --- Test 7: no look-ahead ---


def test_no_look_ahead(real_env, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every oracle target is read exactly at maturity; history never runs past now + 1."""
    data_root, manifest_dir, cache_root = real_env
    pool = load_dev_pool("etth1", cache_root, data_root, manifest_dir)
    targets_seen: list[tuple[int, int]] = []
    history_seen: list[tuple[int, int]] = []
    original_targets = LabelRelease.targets
    original_history = LabelRelease.history

    def recording_targets(self: LabelRelease, origin: int, now: int) -> tuple[float, ...]:
        targets_seen.append((origin, now))
        return original_targets(self, origin, now)

    def recording_history(self: LabelRelease, end: int, now: int) -> tuple[float, ...]:
        history_seen.append((end, now))
        return original_history(self, end, now)

    monkeypatch.setattr(LabelRelease, "targets", recording_targets)
    monkeypatch.setattr(LabelRelease, "history", recording_history)
    result = oracle_dataset(pool)

    assert len(result.series) == len(pool.series)
    assert targets_seen
    for origin, now in targets_seen:
        assert origin + H == now, (origin, now)
    for end, now in history_seen:
        assert end <= now + 1, (end, now)
    # the deepest value index any call touches is the last scored origin's last target
    deepest = max(origin for origin, _now in targets_seen) + H
    assert deepest == SCORE_END - 1 + H
    assert deepest == len(pool.series[0].values) - 1


# --- Test 8: the oracles are comparators, not lower bounds ---


class TableForecasts:
    """A hand-built ``[origin][expert][h]`` table satisfying the replay's Forecasts protocol."""

    def __init__(self, rows: list[list[tuple[float, ...]]], series_id: str) -> None:
        self.series_id = series_id
        self.first_origin = 0
        self.last_origin = len(rows) - 1
        self._rows = [tuple(tuple(row) for row in origin_rows) for origin_rows in rows]

    def at(self, origin: int) -> tuple[tuple[float, ...], ...]:
        return self._rows[origin - self.first_origin]


def test_oracles_bound_selection_not_combination() -> None:
    """Equal weights on two straddling forecasts beat every single expert.

    Expert "high" always forecasts target + 1 and "low" target - 1, so every
    expert origin loss is exactly 1 and every switching total is positive,
    while the equal-weight combination's forecast is the target exactly and
    its errors are all zero: the oracles bound single-expert selection only.
    """
    horizon, n_origins = 3, 12
    values = [((i * 7) % 13) * 0.25 for i in range(n_origins + horizon)]  # dyadic, t +/- 1 exact
    rows = [
        [
            tuple(values[s + 1 + h] + 1.0 for h in range(horizon)),
            tuple(values[s + 1 + h] - 1.0 for h in range(horizon)),
        ]
        for s in range(n_origins)
    ]
    forecasts = TableForecasts(rows, "straddle")
    trace = replay_series(
        EqualWeights(),
        {},
        0,
        "straddle",
        forecasts,
        values,
        7,
        horizon,
        n_origins,
        expert_names=("high", "low"),
    )
    assert trace.errors
    for row in trace.errors:
        for error in row:
            assert error == 0.0

    losses = []
    for e in range(2):
        row = []
        for s in range(n_origins):
            total = 0.0
            for h in range(horizon):
                total = total + abs(rows[s][e][h] - values[s + 1 + h])
            row.append(total / horizon)
        losses.append(row)
    assert all(value == 1.0 for row in losses for value in row)
    for m in ORACLE_SWITCHES:
        assert switching_total(losses, m) > 0.0

    doc = oracles.__doc__ or ""
    assert "bound single-expert selection only" in doc
    assert "not a floor for weighted combinations" in doc


# --- Test 9: oracle_to_obj and the golden digest ---


ORACLE_OBJ_KEYS = {
    "dataset",
    "cache_id",
    "arena_version",
    "pool_spec",
    "foundation_cache_ids",
    "leakage_risk",
    "expert_names",
    "switches",
    "global_expert",
    "metrics",
    "series",
}


def test_oracle_to_obj_and_the_golden_digest(synthetic_pool) -> None:
    """Pin the canonical bytes of the synthetic-dev oracles (per_family = 1).

    Changing this digest means an oracle value changed: bump ARENA_VERSION
    and repin it.
    """
    result = oracle_dataset(synthetic_pool)
    obj = oracle_to_obj(result)
    assert set(obj) == ORACLE_OBJ_KEYS
    assert isinstance(obj["series"], list) and len(obj["series"]) == len(result.series)
    for entry in obj["series"]:
        assert set(entry) == {"series_id", "expert_mase", "switch_mase", "select_mase"}
        assert isinstance(entry["expert_mase"], list)
        assert isinstance(entry["switch_mase"], list)
    assert obj["switches"] == list(ORACLE_SWITCHES)
    assert obj["pool_spec"] == synthetic_pool.pool_spec == "simple"
    assert obj["foundation_cache_ids"] == []
    assert obj["leakage_risk"] is False  # a synthetic dataset is no model's pretraining data
    canonical_bytes(obj)
    digest = sha256(canonical_bytes(obj)).hexdigest()
    assert digest == "62b41507d8737af46cbb34d467a89f8046c6512d89d5795556f4cbc04e31027d"


# --- Test 10: version 3 changes no score ---


#: Version 2's digests of the golden replay and of the five baselines' replays
#: (tests/arena/test_evaluate.py test 13 and tests/updater/test_baselines_replay.py
#: before this PR repinned them). Stamping "2" back into the version-3 result
#: objects must reproduce them byte for byte: the bump changed the version
#: string and nothing else.
VERSION2_DIGESTS = {
    "equal": "a3dcc509e2a085c7ff473cdfa02c9bc434b6aa419f1975fda03a6a31b7405fd5",
    "ftl": "9083f89f3accf8ac65128d2ea8cdf1b4234f4fb434aa239e4c4cd206d0d2767b",
    "hedge": "ac63d96f53a32ecc8bca278d41a13a93d939866a8309fd0497a7012c9abcef82",
    "fixed_share": "fe76980b8586497c8570edb0de10121ce91a78375445a47122a5a0419b8e2bb4",
    "adahedge": "8da13ac2a3d465eeb4b7c2ab1c3d816d616ff5aaeeeb709c9ac241059c2149f2",
}


def test_version_3_changes_no_score(synthetic_pool) -> None:
    """Every version-3 replay object, stamped back to "2", hashes to version 2's digest."""
    golden = result_to_obj(replay_dataset(lambda: EqualWeights(), {}, 0, synthetic_pool))
    stamped = {**golden, "arena_version": "2"}
    assert sha256(canonical_bytes(stamped)).hexdigest() == VERSION2_DIGESTS["equal"]
    for id_ in BASELINE_IDS:
        replay = replay_dataset(BASELINES[id_], DEFAULT_CONFIGS[id_], 0, synthetic_pool)
        assert replay.status == "ok"
        stamped = {**result_to_obj(replay), "arena_version": "2"}
        assert sha256(canonical_bytes(stamped)).hexdigest() == VERSION2_DIGESTS[id_], id_


# --- Test 12: determinism ---


def test_two_oracle_dataset_calls_are_equal(synthetic_pool) -> None:
    a = oracle_dataset(synthetic_pool)
    b = oracle_dataset(synthetic_pool)
    assert a == b
