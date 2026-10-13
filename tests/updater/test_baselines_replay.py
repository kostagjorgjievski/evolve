"""The baselines through the arena: dev replay, pooled replay, sealed path (TASK_010 PR A).

Tests 13 to 15 of the PR A list: every baseline replays the synthetic dev
pool to an ok result whose canonical bytes hash to a pinned digest, the
pooled spec reaches them as six experts, and the sealed server replays
``adahedge`` by reference with numbers identical to an in-process replay.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import date
from hashlib import sha256
from pathlib import Path

import pytest

from evolvets.arena.dev import load_dev_pool
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.arena.realdata.fetch import fetch
from evolvets.arena.realdata.prepare import prepare
from evolvets.arena.sealed.client import SealedPaths, SealedRequest, evaluate_sealed
from evolvets.arena.sealed.pools import build_sealed_cache, load_sealed_pool
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.foundation.fake import FAKE_SPEC, FakeAdapter
from evolvets.foundation.store import build_foundation_cache
from evolvets.ledger.canonical import canonical_bytes
from evolvets.updater.baselines import BASELINES, DEFAULT_CONFIGS, adahedge
from tests.arena.test_dev import ROWS
from tests.arena.test_realdata_prepare import _build_etth1_fixture, _install
from tests.arena.test_sealed_server import _baseline, _begin_run

# --- Test 13: every baseline through the arena ----------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_pool(tmp_path_factory: pytest.TempPathFactory):
    """The dev pool of a synthetic-dev cache with one series per family (seed 0 replays)."""
    base = tmp_path_factory.mktemp("baselines-synthetic")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-dev", base, base, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    return load_dev_pool("synthetic-dev", cache_root, base, base)


#: The digest of canonical_bytes(result_to_obj(result)) per baseline id, seed 0,
#: on the per_family = 1 synthetic-dev pool (the TASK_007 fixture), pinned on the
#: CI platform (Linux x86_64, CPython 3.11). Changing one requires a reason in the
#: PR: a baseline's behaviour changed. These digests are not ARENA_VERSION-governed,
#: but they will change when a later PR bumps ARENA_VERSION (the version string is
#: inside the hashed object); repin them there.
GOLDEN_DIGESTS = {
    "equal": "a3dcc509e2a085c7ff473cdfa02c9bc434b6aa419f1975fda03a6a31b7405fd5",
    "ftl": "9083f89f3accf8ac65128d2ea8cdf1b4234f4fb434aa239e4c4cd206d0d2767b",
    "hedge": "ac63d96f53a32ecc8bca278d41a13a93d939866a8309fd0497a7012c9abcef82",
    "fixed_share": "fe76980b8586497c8570edb0de10121ce91a78375445a47122a5a0419b8e2bb4",
    "adahedge": "8da13ac2a3d465eeb4b7c2ab1c3d816d616ff5aaeeeb709c9ac241059c2149f2",
}


@pytest.mark.parametrize("id_", sorted(GOLDEN_DIGESTS))
def test_baseline_replays_the_synthetic_pool(id_: str, synthetic_pool) -> None:
    """Every baseline at its default config replays synthetic-dev to ok, digest pinned."""
    result = replay_dataset(BASELINES[id_], DEFAULT_CONFIGS[id_], 0, synthetic_pool)
    assert result.status == "ok"
    digest = sha256(canonical_bytes(result_to_obj(result))).hexdigest()
    assert digest == GOLDEN_DIGESTS[id_]


# --- Test 14: the pooled spec reaches the baselines -----------------------------------------------


class InitSpy:
    """Wraps one baseline instance, recording the n_experts its init receives."""

    def __init__(self, inner: object) -> None:
        self.inner = inner
        self.init_n_experts: int | None = None

    def init(self, n_experts: int, horizon: int, config: object, seed: int) -> object:
        self.init_n_experts = n_experts
        return self.inner.init(n_experts, horizon, config, seed)  # type: ignore[attr-defined]

    def update(self, state: object, feedback: object, context: object) -> object:
        return self.inner.update(state, feedback, context)  # type: ignore[attr-defined]


@pytest.fixture(scope="module")
def pooled_env(tmp_path_factory: pytest.TempPathFactory):
    """A synthetic-dev cache plus a fake foundation cache built against it (TASK_009 helpers)."""
    base = tmp_path_factory.mktemp("baselines-pooled")
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


@pytest.mark.parametrize("id_", ["hedge", "adahedge"])
def test_pooled_replay_has_six_experts(id_: str, pooled_env) -> None:
    """The simple five plus the fake foundation model replay as six experts."""
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
    spies: list[InitSpy] = []

    def factory() -> InitSpy:
        spy = InitSpy(BASELINES[id_]())
        spies.append(spy)
        return spy

    result = replay_dataset(factory, DEFAULT_CONFIGS[id_], 0, pool)
    assert result.status == "ok"
    assert len(spies) == len(pool.series)
    assert all(spy.init_n_experts == 6 for spy in spies)


# --- Test 15: the sealed path ---------------------------------------------------------------------


@pytest.fixture(scope="module")
def sealed_env(tmp_path_factory: pytest.TempPathFactory):
    """The 1200-row fixture ETTh1, prepared, with its sealed cache built (TASK_008 setup)."""
    base = tmp_path_factory.mktemp("baselines-sealed")
    etth1 = _build_etth1_fixture(base / "fixtures", rows=ROWS)
    monkeypatch = pytest.MonkeyPatch()
    try:
        _install(monkeypatch, etth1=etth1)
        data_root, manifest_dir = base / "data", base / "manifests"
        fetch(etth1, data_root, manifest_dir, today=date(2026, 10, 11))
        prepare("etth1", data_root, manifest_dir)
        sealed_root = base / "sealed_cache"
        build_sealed_cache("etth1", data_root, manifest_dir, sealed_root)
        return data_root, manifest_dir, sealed_root
    finally:
        monkeypatch.undo()


def test_sealed_adahedge_matches_the_in_process_replay(sealed_env, tmp_path: Path) -> None:
    """The sealed server replays baselines:adahedge to the in-process numbers, exactly."""
    data_root, manifest_dir, sealed_root = sealed_env
    ledger_root = tmp_path / "ledgers"
    _begin_run(ledger_root, "run-1")
    receipt = evaluate_sealed(
        SealedRequest(
            dataset="etth1",
            updater_id="adahedge",
            updater_ref="evolvets.updater.baselines:adahedge",
            config={},
            seed=0,
        ),
        ledger_root=ledger_root,
        run_id="run-1",
        paths=SealedPaths(
            data_root=data_root, manifest_dir=manifest_dir, sealed_cache_root=sealed_root
        ),
    )
    assert receipt.status == "ok"
    baseline = _baseline(ledger_root, "run-1")
    pool = load_sealed_pool("etth1", data_root, manifest_dir, sealed_root)
    expected = replay_dataset(adahedge, {}, 0, pool)
    assert expected.status == "ok"
    assert baseline.metrics == expected.metrics


def test_importing_baselines_loads_only_the_updater_package() -> None:
    """A fresh interpreter importing baselines loads no other evolvets module."""
    code = (
        "import sys, json; import evolvets.updater.baselines; "
        "print(json.dumps(sorted(m for m in sys.modules if m.startswith('evolvets'))))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    modules = json.loads(out.stdout)
    assert modules == [
        "evolvets",
        "evolvets.updater",
        "evolvets.updater.baselines",
        "evolvets.updater.interfaces",
    ]
