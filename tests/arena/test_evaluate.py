"""Replay-dataset tests: failures, no look-ahead, oracle, golden, fresh instances (tests 8-15)."""

from __future__ import annotations

import ast
import itertools
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena import ARENA_VERSION
from evolvets.arena.dev import load_dev_pool
from evolvets.arena.errors import ArenaError
from evolvets.arena.evaluate import ReplayResult, replay_dataset, result_to_obj
from evolvets.arena.labels import LabelRelease
from evolvets.arena.replay import series_seed
from evolvets.arena.synthetic import GENERATOR_VERSION
from evolvets.arena.versions import protocol_versions
from evolvets.cache import CACHE_VERSION
from evolvets.cache.store import cache_dir
from evolvets.ledger.canonical import canonical_bytes
from evolvets.ledger.payloads import PROTOCOL_KEYS, RunStart
from evolvets.ledger.schema import SCHEMA_VERSION, Budget
from tests.arena.test_dev import SCORE_END, SCORE_START, H, build_real_env

SRC = Path(__file__).parents[2] / "src" / "evolvets"
SYNTHETIC_IDS = [
    "stationary/0",
    "sudden/0",
    "gradual/0",
    "recurring/0",
    "false_drift/0",
    "mixed/0",
]


def equal_weights(n_experts: int, horizon: int) -> list[list[float]]:
    """Equal weight on every expert, per horizon step."""
    weight = 1.0 / n_experts
    return [[weight] * horizon for _ in range(n_experts)]


class EqualWeights:
    """Equal weight on every expert at every step."""

    def init(self, n_experts, horizon, config, seed):
        return None

    def update(self, state, feedback, context):
        return (equal_weights(len(context.expert_names), context.horizon), None)


class OneHot:
    """All weight on one expert at every step."""

    def __init__(self, expert):
        self.expert = expert

    def init(self, n_experts, horizon, config, seed):
        return None

    def update(self, state, feedback, context):
        rows = []
        for e in range(len(context.expert_names)):
            rows.append([1.0 if e == self.expert else 0.0] * context.horizon)
        return (rows, None)


class FailingUpdater:
    """Valid until its trigger: a failing init, a failing update, or invalid weights."""

    def __init__(self, mode: str, step: int = 7):
        self.mode = mode
        self.step = step

    def init(self, n_experts, horizon, config, seed):
        if self.mode == "init":
            raise ValueError("init boom")
        return None

    def update(self, state, feedback, context):
        if context.step == self.step:
            if self.mode == "update":
                raise RuntimeError("update boom")
            if self.mode == "invalid":
                return ([[2.0] * context.horizon for _ in context.expert_names], None)
        return (equal_weights(len(context.expert_names), context.horizon), None)


class ScriptedFactory:
    """Returns a new updater per call: maker k makes the k-th instance."""

    def __init__(self, makers):
        self.makers = list(makers)
        self.calls = 0

    def __call__(self):
        maker = self.makers[min(self.calls, len(self.makers) - 1)]
        self.calls += 1
        return maker()


@pytest.fixture()
def real_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The dev pool of the 1200-row fixture ETTh1."""
    data_root, manifest_dir, cache_root = build_real_env(tmp_path, monkeypatch)
    return load_dev_pool("etth1", cache_root, data_root, manifest_dir)


@pytest.fixture()
def synthetic_pool(tmp_path: Path):
    """The dev pool of a synthetic-dev cache with one series per family."""
    from evolvets.cache.datasets import public_inputs
    from evolvets.cache.store import build_cache

    cache_root = tmp_path / "cache_store"
    source, series = public_inputs("synthetic-dev", tmp_path, tmp_path, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    return load_dev_pool("synthetic-dev", cache_root, Path("data"), Path("data/manifests"))


# --- Test 8: failures -------------------------------------------------------------------------


def test_second_instance_failing_in_update(real_pool) -> None:
    """The pool stops at the failing series with empty metrics and scores."""
    factory = ScriptedFactory([EqualWeights, lambda: FailingUpdater("update", step=3)])
    result = replay_dataset(factory, {}, 0, real_pool)  # type: ignore[arg-type]
    assert isinstance(result, ReplayResult)
    assert result.status == "failed"
    assert result.failed_series == real_pool.series[1].series_id
    assert result.failed_step == 3
    assert result.error is not None and result.error.startswith("RuntimeError")
    assert result.metrics == {}
    assert result.series == ()
    assert result.arena_version == ARENA_VERSION


def test_first_instance_failing_in_init(real_pool) -> None:
    factory = ScriptedFactory([lambda: FailingUpdater("init")])
    result = replay_dataset(factory, {}, 0, real_pool)  # type: ignore[arg-type]
    assert result.status == "failed"
    assert result.failed_series == real_pool.series[0].series_id
    assert result.failed_step is None
    assert result.error is not None and result.error.startswith("ValueError")
    assert result.metrics == {} and result.series == ()


def test_invalid_weights_from_any_instance(real_pool) -> None:
    factory = ScriptedFactory([lambda: FailingUpdater("invalid", step=1)])
    result = replay_dataset(factory, {}, 0, real_pool)  # type: ignore[arg-type]
    assert result.status == "invalid_weights"
    assert result.failed_series == real_pool.series[0].series_id
    assert result.failed_step == 1
    assert result.metrics == {} and result.series == ()


def test_corrupted_cache_series_file_is_an_arena_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tampered series file is the cache's fault, not a failed result."""
    data_root, manifest_dir, cache_root = build_real_env(tmp_path, monkeypatch)
    series_file = cache_dir(cache_root, "etth1") / "series" / "0000.f64"
    raw = series_file.read_bytes()
    series_file.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))
    pool = load_dev_pool("etth1", cache_root, data_root, manifest_dir)
    with pytest.raises(ArenaError, match="sha256"):
        replay_dataset(lambda: EqualWeights(), {}, 0, pool)


# --- Test 9: no look-ahead end to end -----------------------------------------------------------


def test_no_look_ahead_end_to_end(real_pool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every target is read exactly at maturity; history never runs past now + 1."""
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
    result = replay_dataset(lambda: EqualWeights(), {}, 0, real_pool)  # type: ignore[arg-type]

    assert result.status == "ok"
    assert targets_seen
    for origin, now in targets_seen:
        assert origin + H == now, (origin, now)
    for end, now in history_seen:
        assert end <= now + 1, (end, now)
    # the MASE scale of every series is read at the scoring start
    assert history_seen == [(139, SCORE_START)] * len(real_pool.series)
    # the deepest value index any call touches is the last scored origin's last target
    deepest = max(origin for origin, _now in targets_seen) + H
    assert deepest == SCORE_END - 1 + H
    assert deepest == len(real_pool.series[0].values) - 1


# --- Test 10: oracle sanity ----------------------------------------------------------------------


def test_naive_is_worse_than_the_oracle_on_synthetic(synthetic_pool) -> None:
    result = replay_dataset(lambda: OneHot(0), {}, 0, synthetic_pool)  # type: ignore[arg-type]
    assert result.status == "ok"
    assert [score.series_id for score in result.series] == SYNTHETIC_IDS
    for score in result.series:
        assert score.oracle_mase is not None
        assert score.mase > score.oracle_mase, score.series_id
    assert result.metrics["regret"] > 0


# --- Test 11: protocol versions -------------------------------------------------------------------


def test_protocol_versions_pin_every_key() -> None:
    versions = protocol_versions()
    assert set(versions) == set(PROTOCOL_KEYS)
    assert versions == {
        "ledger": SCHEMA_VERSION,
        "arena": ARENA_VERSION,
        "generator": GENERATOR_VERSION,
        "cache_manifest": CACHE_VERSION,
    }
    RunStart(
        condition="dev-replay",
        seed=0,
        requires_predictions=False,
        config_blob="sha256:" + "0" * 64,
        git_commit="0" * 40,
        protocol_versions=versions,
        budget_limits=Budget(),
    )


# --- Test 12: result_to_obj ------------------------------------------------------------------


RESULT_KEYS = {
    "dataset",
    "cache_id",
    "arena_version",
    "updater_interface",
    "pool_spec",
    "foundation_cache_ids",
    "leakage_risk",
    "status",
    "error",
    "failed_series",
    "failed_step",
    "metrics",
    "series",
}


def test_result_to_obj_ok_and_failed(real_pool) -> None:
    ok = replay_dataset(lambda: EqualWeights(), {}, 0, real_pool)  # type: ignore[arg-type]
    obj = result_to_obj(ok)
    assert set(obj) == RESULT_KEYS
    assert list(obj)[:5] == [
        "dataset",
        "cache_id",
        "arena_version",
        "updater_interface",
        "pool_spec",
    ]
    assert obj["updater_interface"] == "weights-v1"
    assert obj["series"]
    assert set(obj["series"][0]) == {"series_id", "mase", "oracle_mase", "recovery"}
    assert isinstance(obj["series"][0]["recovery"], list)
    canonical_bytes(obj)  # must accept it

    failed = replay_dataset(
        ScriptedFactory([lambda: FailingUpdater("update", step=2)]),
        {},
        0,
        real_pool,  # type: ignore[arg-type]
    )
    failed_obj = result_to_obj(failed)
    assert set(failed_obj) == RESULT_KEYS
    assert failed_obj["series"] == [] and failed_obj["metrics"] == {}
    canonical_bytes(failed_obj)


# --- The updater interface (TASK_016 PR A) ----------------------------------------------------


class FactorySpy:
    """A zero-argument factory that counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> EqualWeights:
        self.calls += 1
        return EqualWeights()


@pytest.mark.parametrize("bad", ["weights-v2", "", None, 1])
def test_replay_dataset_checks_the_interface_first(bad, synthetic_pool) -> None:
    """A bad interface is refused before the factory, init or any cache read runs."""
    factory = FactorySpy()
    with pytest.raises(ArenaError, match="interface"):
        replay_dataset(factory, {}, 0, synthetic_pool, interface=bad)  # type: ignore[arg-type]
    assert factory.calls == 0


def test_updater_interface_is_stamped_on_every_result(real_pool) -> None:
    """ok, failed and invalid_weights results carry the interface they replayed under."""
    cases = [
        (lambda: EqualWeights(), "weights-v1", "ok"),
        (lambda: EqualWeights(), "forecast-v2", "ok"),  # weights stay valid under forecast-v2
        (lambda: FailingUpdater("update", step=2), "weights-v1", "failed"),
        (lambda: FailingUpdater("update", step=2), "forecast-v2", "failed"),
        (lambda: FailingUpdater("invalid", step=1), "weights-v1", "invalid_weights"),
        (lambda: FailingUpdater("invalid", step=1), "forecast-v2", "invalid_weights"),
    ]
    for factory, interface, status in cases:
        result = replay_dataset(factory, {}, 0, real_pool, interface=interface)  # type: ignore[arg-type]
        assert result.status == status
        assert result.updater_interface == interface
        assert result_to_obj(result)["updater_interface"] == interface


def test_arena_version_is_4() -> None:
    """Version 4 is the forecast-v2 interface bump; weights-only scores are unchanged.

    test_oracles.py proves the "unchanged" half by stamping version 4's result
    objects back to "3" (and dropping the interface key) and matching version
    3's digests.
    """
    assert ARENA_VERSION == "4"
    assert protocol_versions()["arena"] == "4"


# --- Test 13: golden -------------------------------------------------------------------------


def test_golden_synthetic_equal_weights(synthetic_pool) -> None:
    """Pin the equal-weights replay of the synthetic-dev pool (per_family = 1, seed 0).

    Changing this digest means some cached forecast, weight handling or metric
    value changed: bump ARENA_VERSION (or EXPERTS_VERSION / GENERATOR_VERSION
    when those are what changed) and repin the digest. Arena version 2
    repinned it once: the result object gained ``pool_spec``,
    ``foundation_cache_ids`` and ``leakage_risk``, which the digest covers;
    tests/arena/test_pool.py proves the numbers themselves are unchanged.
    Arena version 3 (the oracles) repinned it again: only the version string
    inside the hashed object changed, which test_oracles.py test 10 proves by
    stamping "2" back in and matching version 2's digest. Arena version 4
    (the forecast-v2 interface) repinned it once more: the result object
    gained ``updater_interface``, and test_oracles.py proves nothing else
    changed by stamping "3" back in, dropping that key, and matching
    version 3's digest.
    """
    result = replay_dataset(lambda: EqualWeights(), {}, 0, synthetic_pool)  # type: ignore[arg-type]
    assert result.status == "ok"
    digest = sha256(canonical_bytes(result_to_obj(result))).hexdigest()
    assert digest == "ffb5eef1675a218c010bad776d868a18720671edefb9d601378bcc981d3029b9"


# --- Test 14: determinism and AST guards ------------------------------------------------------


def test_two_replay_dataset_calls_are_equal(synthetic_pool) -> None:
    a = replay_dataset(lambda: EqualWeights(), {}, 7, synthetic_pool)  # type: ignore[arg-type]
    b = replay_dataset(lambda: EqualWeights(), {}, 7, synthetic_pool)  # type: ignore[arg-type]
    assert a == b


@pytest.mark.parametrize("name", ["metrics.py", "evaluate.py", "dev.py", "pool.py", "oracles.py"])
def test_no_float_sum_or_statistics(name: str) -> None:
    """The new arena modules never use builtin sum, math.fsum or statistics."""
    tree = ast.parse((SRC / "arena" / name).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == "sum":
                raise AssertionError(f"{name}: builtin sum()")
            if isinstance(node.func, ast.Attribute) and node.func.attr == "fsum":
                raise AssertionError(f"{name}: math.fsum()")
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("statistics")
        if isinstance(node, ast.ImportFrom):
            assert node.module is None or not (node.module or "").startswith("statistics")


# --- Test 15: a fresh instance per series ------------------------------------------------------


class RecordingUpdater:
    """Records its init args and every (feedback, context); returns equal weights."""

    def __init__(self):
        self.init_args: tuple[Any, ...] | None = None
        self.calls: list[tuple[Any, Any]] = []

    def init(self, n_experts, horizon, config, seed):
        self.init_args = (n_experts, horizon, dict(config), seed)
        return None

    def update(self, state, feedback, context):
        self.calls.append((feedback, context))
        return (equal_weights(len(context.expert_names), context.horizon), None)


class RecordingFactory:
    """Returns a fresh RecordingUpdater per call and remembers them in order."""

    def __init__(self):
        self.instances: list[RecordingUpdater] = []

    def __call__(self):
        instance = RecordingUpdater()
        self.instances.append(instance)
        return instance


def test_a_fresh_updater_per_series(real_pool) -> None:
    """One new instance per series, in pool order, each seeing only its own series."""
    factory = RecordingFactory()
    seed = 11
    replay_dataset(factory, {}, seed, real_pool)  # type: ignore[arg-type]
    assert len(factory.instances) == len(real_pool.series)
    for left, right in itertools.combinations(factory.instances, 2):
        assert left is not right
    for k, instance in enumerate(factory.instances):
        assert instance.init_args is not None
        _n_experts, _horizon, _config, received = instance.init_args
        assert received == series_seed(seed, real_pool.series[k].series_id)
        assert instance.calls[0][1].step == 0  # it saw its series from the first origin
        expected = real_pool.series[k].score_end - real_pool.manifest.series[k].first_origin
        assert len(instance.calls) == expected
