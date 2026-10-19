"""Foundation-pool tests: version-1 equality, joins, look-ahead, leakage (TASK_009 PR D).

The foundation experts here are fake-built caches (:class:`FakeAdapter` under
its ``fake`` spec, or a test-made spec for the leakage cases), so every test
runs in plain CI — no model library is installed or loaded. The sealed side
of the pool is tested in test_sealed_foundation.py (PR D's sealed split).
"""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from evolvets.arena.dev import load_dev_pool
from evolvets.arena.errors import ArenaError
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.arena.pool import (
    PoolForecasts,
    built_model_names,
    check_pool_spec,
    load_pool_forecasts,
    pool_expert_names,
)
from evolvets.arena.replay import replay_series
from evolvets.arena.versions import protocol_versions
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache, load_series
from evolvets.experts.pool import EXPERT_NAMES
from evolvets.foundation.fake import FAKE_SPEC, FakeAdapter
from evolvets.foundation.registry import MODELS, ModelSpec
from evolvets.foundation.store import build_foundation_cache, load_foundation_series
from evolvets.ledger.canonical import canonical_bytes
from tests.arena.test_dev import build_real_env

# The version-1 golden digest (tests/arena/test_evaluate.py before arena
# version 2): the simple pool under version 2 must reproduce it exactly once
# the version stamp and the new pool fields are stripped.
V1_GOLDEN_DIGEST = "83327d272555e2ef812af7f6adaf50d3f6cf8aa1d877f4043b1f904977ad58b3"
V2_EXTRA_KEYS = ("pool_spec", "foundation_cache_ids", "leakage_risk", "updater_interface")


class EqualWeights:
    """Equal weight on every expert at every step."""

    def init(self, n_experts: int, horizon: int, config: Any, seed: int) -> None:
        return None

    def update(self, state: Any, feedback: Any, context: Any) -> tuple[list[list[float]], None]:
        weight = 1.0 / len(context.expert_names)
        return ([[weight] * context.horizon for _ in context.expert_names], None)


class RecordingUpdater:
    """Records init's expert count and every context; returns equal weights."""

    def __init__(self) -> None:
        self.init_n_experts: int | None = None
        self.contexts: list[Any] = []

    def init(self, n_experts: int, horizon: int, config: Any, seed: int) -> None:
        self.init_n_experts = n_experts
        return None

    def update(self, state: Any, feedback: Any, context: Any) -> tuple[list[list[float]], None]:
        self.contexts.append(context)
        weight = 1.0 / len(context.expert_names)
        return ([[weight] * context.horizon for _ in context.expert_names], None)


def build_fake_cache(
    env: dict[str, Path], spec: ModelSpec = FAKE_SPEC, batch_size: int = 7
) -> None:
    """Build one fake foundation cache for synthetic-dev under the env's roots."""
    build_foundation_cache(
        "synthetic-dev",
        FakeAdapter(spec.name),
        spec,
        env["foundation_root"],
        env["cache_root"],
        env["data_root"],
        env["manifest_dir"],
        batch_size,
        per_family=1,
    )


def load_pooled(env: dict[str, Path], models: tuple[str, ...]) -> Any:
    """Load a pool for ``env``'s dataset with the named foundation models joined."""
    return load_dev_pool(
        env["dataset"],
        env["cache_root"],
        env["data_root"],
        env["manifest_dir"],
        pool_spec="simple+foundation",
        foundation_root=env["foundation_root"],
        foundation_models=models,
    )


@pytest.fixture()
def synthetic_env(tmp_path: Path) -> dict[str, Path]:
    """A synthetic-dev cache (one series per family) plus its data roots."""
    source, series = public_inputs("synthetic-dev", tmp_path, tmp_path, per_family=1)
    build_cache("synthetic-dev", source, series, tmp_path / "cache_store")
    return {
        "dataset": "synthetic-dev",
        "data_root": tmp_path,
        "manifest_dir": tmp_path,
        "cache_root": tmp_path / "cache_store",
        "foundation_root": tmp_path / "foundation_cache",
    }


@pytest.fixture(scope="module")
def real_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The fixture ETTh1, prepared and cached, with three fake foundation caches.

    ``fake``'s spec records no etth1 overlap (a published "no"); ``leaky``
    records "yes" and ``opaque`` "unknown", so the leakage flag has both a
    true and a false case on a real dataset.
    """
    base = tmp_path_factory.mktemp("etth1")
    monkeypatch = pytest.MonkeyPatch()
    try:
        data_root, manifest_dir, cache_root = build_real_env(base, monkeypatch)
    finally:
        monkeypatch.undo()
    env = {
        "dataset": "etth1",
        "data_root": data_root,
        "manifest_dir": manifest_dir,
        "cache_root": cache_root,
        "foundation_root": base / "foundation_cache",
    }
    for spec in (FAKE_SPEC, _custom_spec("leaky", "yes"), _custom_spec("opaque", "unknown")):
        build_foundation_cache(
            "etth1",
            FakeAdapter(spec.name),
            spec,
            env["foundation_root"],
            env["cache_root"],
            env["data_root"],
            env["manifest_dir"],
            batch_size=64,
        )
    return env


def _custom_spec(name: str, etth1: str) -> ModelSpec:
    """A test spec whose only difference from the fake is its etth1 overlap."""
    return replace(FAKE_SPEC, name=name, pretraining_overlap={"etth1": etth1, "ecl": "no"})


# --- Test 1: the simple pool is version 1, exactly -------------------------------------------


def test_simple_pool_reproduces_version_1(synthetic_env: dict[str, Path]) -> None:
    """A 'simple' replay under the current arena version equals version 1's golden numbers.

    Stripping the four keys later result objects added — the three pool fields
    of version 2 and the ``updater_interface`` key of version 4 — and stamping
    the version back to 1 must give version 1's golden digest byte for byte,
    so the pool work and the interface bump changed no forecast, weight or
    metric arithmetic.
    """
    pool = load_dev_pool(
        "synthetic-dev",
        synthetic_env["cache_root"],
        synthetic_env["data_root"],
        synthetic_env["manifest_dir"],
    )
    assert pool.pool_spec == "simple" and pool.foundation == ()
    result = replay_dataset(lambda: EqualWeights(), {}, 0, pool)
    assert result.status == "ok"
    assert result.pool_spec == "simple" and result.foundation_cache_ids == ()
    assert result.leakage_risk is False
    obj = result_to_obj(result)
    # version 4 is the forecast-v2 interface bump; this replay is weights-v1
    assert obj["arena_version"] == "4"
    assert protocol_versions()["arena"] == "4"
    v1_obj = {key: value for key, value in obj.items() if key not in V2_EXTRA_KEYS}
    v1_obj["arena_version"] = "1"
    assert sha256(canonical_bytes(v1_obj)).hexdigest() == V1_GOLDEN_DIGEST


# --- Test 2: the foundation pool reaches the updater as point forecasts ----------------------


def test_foundation_pool_forecasts_and_names(synthetic_env: dict[str, Path]) -> None:
    """expert_names lists 5 + N names; the extra rows equal the cached 0.5 quantiles."""
    build_fake_cache(synthetic_env)
    pool = load_pooled(synthetic_env, ("fake",))
    assert pool.pool_spec == "simple+foundation"
    assert len(pool.foundation) == 1 and pool.foundation[0].model == "fake"
    assert pool_expert_names(pool) == (*EXPERT_NAMES, "fake")

    updater = RecordingUpdater()
    result = replay_dataset(lambda: updater, {}, 3, pool)
    assert result.status == "ok"
    assert updater.init_n_experts == len(EXPERT_NAMES) + 1
    assert updater.contexts, "the updater saw no origins"
    assert result.foundation_cache_ids == (pool.foundation[0].cache_id,)

    first = pool.series[0]
    joined = load_pool_forecasts(pool, first.series_id)
    assert isinstance(joined, PoolForecasts)
    simple = load_series(pool.cache_root, pool.manifest, first.series_id)
    cache = pool.foundation[0]
    cached = load_foundation_series(cache.root, cache.manifest, first.series_id)
    origins = (first.score_start, first.score_start + 1, first.score_end - 1)
    for origin in origins:
        row = joined.at(origin)
        assert len(row) == len(EXPERT_NAMES) + 1
        assert row[: len(EXPERT_NAMES)] == simple.at(origin)
        assert row[-1] == cached.point(origin - cached.first_origin)
        # series[0] is replayed first, so its contexts sit at the front of the
        # recording, indexed from the cache's first origin (not score_start).
        context = updater.contexts[origin - joined.first_origin]
        assert context.expert_names == (*EXPERT_NAMES, "fake")
        assert context.forecasts == row


def test_pool_spec_and_models_are_validated(synthetic_env: dict[str, Path]) -> None:
    """Bad specs, a missing root, a missing cache and duplicated models are refused."""
    with pytest.raises(ArenaError, match="pool_spec"):
        check_pool_spec("foundation")
    env = synthetic_env
    simple = {
        "dataset": "synthetic-dev",
        "data_root": env["data_root"],
        "manifest_dir": env["manifest_dir"],
        "cache_root": env["cache_root"],
    }
    with pytest.raises(ArenaError, match="takes no foundation"):
        load_dev_pool(**simple, pool_spec="simple", foundation_root=env["foundation_root"])
    with pytest.raises(ArenaError, match="needs a foundation"):
        load_dev_pool(**simple, pool_spec="simple+foundation")
    with pytest.raises(ArenaError, match="fake"):
        load_pooled(env, ("fake",))
    build_fake_cache(env)
    with pytest.raises(ArenaError, match="twice"):
        load_pooled(env, ("fake", "fake"))
    with pytest.raises(ArenaError, match="is missing"):
        load_pooled(env, ("no-such-model",))


# --- Test 3: a foundation cache that does not join the simple cache is refused ---------------


def _edit_fake_manifest(env: dict[str, Path], edit) -> None:
    """Rewrite the fake cache's manifest with ``edit(obj)`` applied."""
    manifest_path = env["foundation_root"] / "synthetic-dev" / "fake" / "manifest.json"
    obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    edit(obj)
    manifest_path.write_text(json.dumps(obj), encoding="utf-8")


def test_mismatched_simple_cache_id_is_refused(synthetic_env: dict[str, Path]) -> None:
    """A foundation cache built against another simple cache never reaches a replay."""

    def edit(obj: Any) -> None:
        obj["simple_cache_id"] = "0" * 64

    build_fake_cache(synthetic_env)
    _edit_fake_manifest(synthetic_env, edit)
    with pytest.raises(ArenaError, match="was built against simple cache"):
        load_pooled(synthetic_env, ("fake",))


def test_series_entry_mismatch_is_refused(synthetic_env: dict[str, Path]) -> None:
    """A foundation cache whose series disagree with the simple cache's is refused."""

    def edit(obj: Any) -> None:
        obj["series"][0]["n_origins"] += 1

    build_fake_cache(synthetic_env)
    _edit_fake_manifest(synthetic_env, edit)
    with pytest.raises(ArenaError, match="n_origins"):
        load_pooled(synthetic_env, ("fake",))


# --- Test 4: the look-ahead property with the larger pool ------------------------------------


class CallSpy:
    """Records every (feedback, context) call; returns equal weights."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, Any]] = []

    def init(self, n_experts: int, horizon: int, config: Any, seed: int) -> None:
        return None

    def update(self, state: Any, feedback: Any, context: Any) -> tuple[list[list[float]], None]:
        self.calls.append((feedback, context))
        weight = 1.0 / len(context.expert_names)
        return ([[weight] * context.horizon for _ in context.expert_names], None)


def test_no_look_ahead_with_the_larger_pool(synthetic_env: dict[str, Path]) -> None:
    """Values after origin t change no forecast and no call at or before t.

    TASK_007's property, replayed over the joined pool: mutating every value
    after a cut leaves the pool's forecast rows at origins up to the cut (the
    cached point forecasts included) and every call the updater received at
    or before the cut untouched — the foundation rows cannot move, and no
    target is read before it matures.
    """
    build_fake_cache(synthetic_env)
    pool = load_pooled(synthetic_env, ("fake",))
    series = pool.series[1]
    joined = load_pool_forecasts(pool, series.series_id)
    cut = series.score_start + 3
    mutated = series.values[: cut + 1] + tuple(9e9 + i for i in range(len(series.values) - cut - 1))

    rows_before = [joined.at(origin) for origin in range(joined.first_origin, cut + 1)]
    names = pool_expert_names(pool)
    spy_a, spy_b = CallSpy(), CallSpy()
    replay_series(
        spy_a,
        {},
        5,
        series.series_id,
        joined,
        series.values,
        series.period,
        pool.manifest.horizon,
        cut + 1,
        expert_names=names,
    )
    replay_series(
        spy_b,
        {},
        5,
        series.series_id,
        joined,
        mutated,
        series.period,
        pool.manifest.horizon,
        cut + 1,
        expert_names=names,
    )
    rows_after = [joined.at(origin) for origin in range(joined.first_origin, cut + 1)]
    assert rows_before == rows_after
    # Every call at or before the cut is byte-identical, feedback rows
    # included: an origin's feedback arrives only once its whole horizon of
    # targets has matured, so nothing read at or before the cut touches a
    # value the mutation owns.
    n_recorded = cut - joined.first_origin + 1
    assert n_recorded >= 1
    assert spy_a.calls[:n_recorded] == spy_b.calls[:n_recorded]


# --- the forecast rows must match expert_names, or the replay is refused ---------------------


def test_replay_refuses_forecasts_that_do_not_match_expert_names(
    synthetic_env: dict[str, Path],
) -> None:
    """A 6-row pool replayed with 5 or 7 names raises ArenaError before init.

    The pool spec made the expert count a parameter, and nothing else ties
    ``expert_names`` to the forecast rows, so ``replay_series`` itself must
    refuse a mismatch before the updater runs: five names over six rows would
    silently drop the foundation expert and break the ``Context`` invariant,
    and seven names would crash the combination loop with a bare ``IndexError``
    instead of an ``ArenaError``.
    """
    build_fake_cache(synthetic_env)
    pool = load_pooled(synthetic_env, ("fake",))
    series = pool.series[0]
    joined = load_pool_forecasts(pool, series.series_id)
    assert len(joined.at(joined.first_origin)) == len(EXPERT_NAMES) + 1  # six rows
    for names in (None, EXPERT_NAMES, (*EXPERT_NAMES, "a", "b")):  # default, five, seven
        kwargs = {} if names is None else {"expert_names": names}
        updater = RecordingUpdater()
        with pytest.raises(ArenaError, match="rows at the first origin"):
            replay_series(
                updater,
                {},
                0,
                series.series_id,
                joined,
                series.values,
                series.period,
                pool.manifest.horizon,
                series.score_end,
                **kwargs,
            )
        assert updater.init_n_experts is None


# --- Test 6: the leakage flag ----------------------------------------------------------------


def test_leakage_risk_on_real_datasets(real_env: dict[str, Path]) -> None:
    """The flag is set for yes/unknown overlap on a real dataset, and only then."""
    simple = load_dev_pool(
        "etth1", real_env["cache_root"], real_env["data_root"], real_env["manifest_dir"]
    )
    assert replay_dataset(lambda: EqualWeights(), {}, 0, simple).leakage_risk is False

    clean = load_pooled(real_env, ("fake",))  # the fake's etth1 overlap is "no"
    result = replay_dataset(lambda: EqualWeights(), {}, 0, clean)
    assert result.status == "ok"
    assert result.leakage_risk is False
    assert result_to_obj(result)["leakage_risk"] is False

    for models, expected in (
        (("leaky",), True),  # the published corpus includes etth1
        (("opaque",), True),  # it cannot be decided from the published sources
        (("fake", "leaky"), True),  # one risky model in the pool is enough
    ):
        pool = load_pooled(real_env, models)
        result = replay_dataset(lambda: EqualWeights(), {}, 0, pool)
        assert result.status == "ok"
        assert result.leakage_risk is expected
        assert result_to_obj(result)["leakage_risk"] is expected


def test_leakage_risk_never_set_on_synthetic(synthetic_env: dict[str, Path]) -> None:
    """Synthetic results never carry the flag, whatever the models overlapped."""
    build_fake_cache(synthetic_env)
    pool = load_pooled(synthetic_env, ("fake",))
    result = replay_dataset(lambda: EqualWeights(), {}, 0, pool)
    assert result.status == "ok"
    assert result.leakage_risk is False


# --- the pool's expert list follows the registry's pinned set --------------------------------


def test_built_models_are_the_pinned_registry_entries() -> None:
    """The default foundation pool is the pinned registry models, in registry order."""
    pinned = tuple(name for name, spec in MODELS.items() if spec.hf_revision != "unpinned")
    assert built_model_names() == pinned == ("chronos2", "tirex", "timesfm25")
