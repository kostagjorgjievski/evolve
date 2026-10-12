"""Sealed-pool tests: salt, manifest, inputs, caches and pools (PR B tests 1-12 and guards).

The real sealed inputs, the sealed caches and the sealed pools are checked
against the 1200-row TASK_007 fixture (warmup [0, 139), dev [307, 550),
sealed [718, 926), reserve from 1094); the CLI tests live in
``tests/arena/test_sealed_cli.py``.
"""

from __future__ import annotations

import ast
import json
import math
import stat
from datetime import date
from pathlib import Path

import pytest

from evolvets.arena.dev import DevPool, load_dev_pool
from evolvets.arena.errors import ArenaError
from evolvets.arena.evaluate import replay_dataset
from evolvets.arena.realdata import REALDATA_VERSION
from evolvets.arena.realdata.fetch import fetch
from evolvets.arena.realdata.prepare import prepare, splits_manifest_path
from evolvets.arena.sealed.pools import (
    SALT_BYTES,
    SyntheticSealedManifest,
    build_sealed_cache,
    init_salt,
    load_sealed_pool,
    read_synthetic_manifest,
    salt_path,
    sealed_inputs,
    synthetic_manifest_path,
)
from evolvets.arena.synthetic import GENERATOR_VERSION
from evolvets.arena.synthetic.grammar import FAMILIES, make_task
from evolvets.arena.synthetic.oracle import true_best_forecast
from evolvets.arena.synthetic.seeds import dev_tasks, world_seed
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import cache_dir, cache_id
from evolvets.ledger.canonical import sha256_hex
from tests.arena.test_dev import ETTH1_IDS, ROWS, build_real_env
from tests.arena.test_evaluate import EqualWeights
from tests.arena.test_realdata_prepare import (
    _build_etth1_fixture,
    _install,
    _processed,
    _record_opens,
)

H = 24
SCALE_END, SEALED_START, SEALED_END = 139, 718, 926
SRC = Path(__file__).parents[2] / "src" / "evolvets"
_MANIFEST_KEYS = {
    "manifest_version",
    "pool",
    "generator_version",
    "salt_sha256",
    "salt_bytes",
    "per_family",
}


@pytest.fixture()
def salted_env(tmp_path: Path) -> tuple[Path, Path]:
    """An initialised sealed salt with one task per family."""
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    init_salt(data_root, manifest_dir, per_family=1)
    return data_root, manifest_dir


@pytest.fixture()
def real_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path, Path]:
    """The 1200-row fixture ETTh1, prepared, with public and sealed caches built."""
    data_root, manifest_dir, cache_root = build_real_env(tmp_path, monkeypatch)
    sealed_root = tmp_path / "sealed_cache"
    build_sealed_cache("etth1", data_root, manifest_dir, sealed_root)
    return data_root, manifest_dir, cache_root, sealed_root


def _sealed_values(data_root: Path, dataset: str = "etth1") -> dict[str, list[float]]:
    """The sealed file's values, keyed by series id."""
    obj = json.loads(_processed(data_root, dataset, "sealed").read_text(encoding="utf-8"))
    return {entry["series_id"]: entry["values"] for entry in obj["series"]}


def _rewrite_sealed(data_root: Path, manifest_dir: Path, mutate, pin_manifest: bool) -> None:
    """Rewrite ``etth1.sealed.json`` with ``mutate(obj)`` applied, optionally re-pinning it.

    With ``pin_manifest`` the split manifest's ``sealed_sha256`` is rewritten
    to the tampered file's hash, so the tamper reaches past the file check.
    """
    sealed_file = _processed(data_root, "etth1", "sealed")
    obj = json.loads(sealed_file.read_text(encoding="utf-8"))
    mutate(obj)
    sealed_file.write_text(json.dumps(obj), encoding="utf-8")
    if pin_manifest:
        manifest_file = splits_manifest_path(manifest_dir, "etth1")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest["sealed_sha256"] = sha256_hex(sealed_file.read_bytes())
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")


def _set_value(obj: object, series_index: int, index: int, value: float) -> None:
    """Set one sealed value in the parsed sealed file."""
    obj["series"][series_index]["values"][index] = value  # type: ignore[index]


def _swap_first_two_ids(obj: object) -> None:
    """Swap the series ids of the first two sealed entries."""
    entries = obj["series"]  # type: ignore[index]
    entries[0]["series_id"], entries[1]["series_id"] = (
        entries[1]["series_id"],
        entries[0]["series_id"],
    )


def _assert_no_salt_leak(exc: BaseException, salt: bytes) -> None:
    """Assert the salt bytes appear nowhere in ``exc``'s message."""
    assert salt.hex() not in str(exc)
    assert repr(salt) not in str(exc)


# --- Test 1: salt creation ----------------------------------------------------------------------


def test_init_salt_creates_and_is_idempotent(tmp_path: Path) -> None:
    """A 0o600 32-byte salt and a manifest with exactly the pinned keys."""
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    digest = init_salt(data_root, manifest_dir, per_family=1)
    salt_file = salt_path(data_root)
    salt = salt_file.read_bytes()
    assert len(salt) == SALT_BYTES
    assert stat.S_IMODE(salt_file.stat().st_mode) == 0o600
    assert digest == sha256_hex(salt)

    manifest_file = synthetic_manifest_path(manifest_dir)
    obj = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert set(obj) == _MANIFEST_KEYS
    assert obj["manifest_version"] == "1"
    assert obj["pool"] == "synthetic-sealed"
    assert obj["generator_version"] == GENERATOR_VERSION
    assert obj["salt_sha256"] == digest
    assert obj["salt_bytes"] == SALT_BYTES
    assert obj["per_family"] == 1

    manifest_before = manifest_file.read_bytes()
    assert init_salt(data_root, manifest_dir, per_family=1) == digest
    assert salt_file.read_bytes() == salt
    assert manifest_file.read_bytes() == manifest_before


# --- Test 2: salt refusals ----------------------------------------------------------------------


def test_init_salt_refusals_change_nothing(tmp_path: Path) -> None:
    """Every refusal is an ArenaError; the files on disk keep their bytes."""
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    init_salt(data_root, manifest_dir, per_family=1)
    salt_file, manifest_file = salt_path(data_root), synthetic_manifest_path(manifest_dir)
    salt = salt_file.read_bytes()
    manifest_bytes = manifest_file.read_bytes()

    with pytest.raises(ArenaError, match="per_family"):
        init_salt(data_root, manifest_dir, per_family=2)
    assert salt_file.read_bytes() == salt
    assert manifest_file.read_bytes() == manifest_bytes

    salt_file.unlink()
    with pytest.raises(ArenaError, match="never regenerate"):
        init_salt(data_root, manifest_dir, per_family=1)
    assert not salt_file.exists()
    assert manifest_file.read_bytes() == manifest_bytes

    salt_file.write_bytes(bytes(SALT_BYTES))  # 32 zero bytes: a different salt
    with pytest.raises(ArenaError, match="no longer matches"):
        init_salt(data_root, manifest_dir, per_family=1)
    assert salt_file.read_bytes() == bytes(SALT_BYTES)
    assert manifest_file.read_bytes() == manifest_bytes

    salt_file.write_bytes(salt[: SALT_BYTES - 1])
    with pytest.raises(ArenaError, match="exactly 32 bytes"):
        init_salt(data_root, manifest_dir, per_family=1)
    assert salt_file.read_bytes() == salt[: SALT_BYTES - 1]

    # recovery: the salt survived, the manifest was lost
    salt_file.write_bytes(salt)
    manifest_file.unlink()
    assert init_salt(data_root, manifest_dir, per_family=1) == sha256_hex(salt)
    assert manifest_file.read_bytes() == manifest_bytes


# --- Test 3: read_synthetic_manifest ------------------------------------------------------------


def test_read_synthetic_manifest_rejections(tmp_path: Path) -> None:
    """Missing, extra-key, bad-typed and stale manifests are refused."""
    manifest_dir = tmp_path / "manifests"
    with pytest.raises(ArenaError, match="is missing"):
        read_synthetic_manifest(manifest_dir)
    init_salt(tmp_path / "data", manifest_dir, per_family=1)
    manifest_file = synthetic_manifest_path(manifest_dir)
    obj = json.loads(manifest_file.read_text(encoding="utf-8"))

    def rewrite(**changes: object) -> None:
        edited = dict(obj)
        edited.update(changes)
        manifest_file.write_text(json.dumps(edited), encoding="utf-8")

    rewrite(extra=1)
    with pytest.raises(ArenaError, match="keys must be exactly"):
        read_synthetic_manifest(manifest_dir)
    rewrite(per_family=True)
    with pytest.raises(ArenaError, match="per_family"):
        read_synthetic_manifest(manifest_dir)
    rewrite(generator_version="999")
    with pytest.raises(ArenaError, match="generator_version"):
        read_synthetic_manifest(manifest_dir)
    rewrite(salt_sha256=obj["salt_sha256"][:63])
    with pytest.raises(ArenaError, match="64 lowercase hex"):
        read_synthetic_manifest(manifest_dir)
    rewrite()
    assert read_synthetic_manifest(manifest_dir) == SyntheticSealedManifest(
        generator_version=GENERATOR_VERSION,
        salt_sha256=obj["salt_sha256"],
        per_family=1,
    )


# --- Test 6: synthetic sealed_inputs ------------------------------------------------------------


def test_synthetic_sealed_inputs(salted_env: tuple[Path, Path], tmp_path: Path) -> None:
    """Six salted worlds, distinct from the dev pool and from any other salt."""
    data_root, manifest_dir = salted_env
    salt = salt_path(data_root).read_bytes()
    source, series = sealed_inputs("synthetic-sealed", data_root, manifest_dir, H)
    assert [entry.series_id for entry in series] == [f"{family}/0" for family in FAMILIES]
    for entry in series:
        family = entry.series_id.split("/")[0]
        world = make_task(family, world_seed("sealed", family, 0, salt)).world
        assert entry.period == world.spec.period
        assert entry.values == world.y
        assert entry.values != dev_tasks(family, 1)[0].world.y
    assert source == {
        "kind": "synthetic",
        "pool": "sealed",
        "generator_version": GENERATOR_VERSION,
        "per_family": "1",
        "salt_sha256": sha256_hex(salt),
    }

    other_root, other_manifests = tmp_path / "b" / "data", tmp_path / "b" / "manifests"
    init_salt(other_root, other_manifests, per_family=1)
    _source, other = sealed_inputs("synthetic-sealed", other_root, other_manifests, H)
    assert [entry.values for entry in other] != [entry.values for entry in series]


# --- Test 7: synthetic refusals -----------------------------------------------------------------


def test_synthetic_sealed_refusals_never_leak_the_salt(
    salted_env: tuple[Path, Path],
) -> None:
    """Salt and manifest faults are refused without the salt bytes in any message."""
    data_root, manifest_dir = salted_env
    salt = salt_path(data_root).read_bytes()
    manifest_file = synthetic_manifest_path(manifest_dir)
    manifest_bytes = manifest_file.read_bytes()

    salt_path(data_root).write_bytes(b"z" * SALT_BYTES)
    with pytest.raises(ArenaError, match="does not match") as excinfo:
        sealed_inputs("synthetic-sealed", data_root, manifest_dir, H)
    _assert_no_salt_leak(excinfo.value, salt)

    salt_path(data_root).unlink()
    with pytest.raises(ArenaError, match="is missing") as excinfo:
        sealed_inputs("synthetic-sealed", data_root, manifest_dir, H)
    _assert_no_salt_leak(excinfo.value, salt)
    salt_path(data_root).write_bytes(salt)

    obj = json.loads(manifest_bytes)
    obj["generator_version"] = "999"
    manifest_file.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ArenaError, match="generator_version") as excinfo:
        sealed_inputs("synthetic-sealed", data_root, manifest_dir, H)
    _assert_no_salt_leak(excinfo.value, salt)


# --- Test 4: real sealed_inputs -------------------------------------------------------------------


def test_real_sealed_inputs(real_env: tuple[Path, Path, Path, Path]) -> None:
    """Ids in manifest order, values cut at 950, and the pinned source dict."""
    data_root, manifest_dir, _cache_root, _sealed_root = real_env
    source, series = sealed_inputs("etth1", data_root, manifest_dir, H)
    assert [entry.series_id for entry in series] == ETTH1_IDS
    sealed = _sealed_values(data_root)
    for entry in series:
        assert entry.period == 24
        assert len(entry.values) == SEALED_END + H
        assert entry.values == tuple(sealed[entry.series_id][: SEALED_END + H])
    spec = json.loads(splits_manifest_path(manifest_dir, "etth1").read_text(encoding="utf-8"))
    assert source == {
        "kind": "realdata-sealed",
        "dataset": "etth1",
        "realdata_version": REALDATA_VERSION,
        "sealed_sha256": spec["sealed_sha256"],
        "public_sha256": spec["public_sha256"],
    }

    # nothing at index >= 950 is reachable: changing it (and re-pinning the
    # manifest) leaves the sealed inputs untouched
    before = [entry.values for entry in series]
    _rewrite_sealed(
        data_root, manifest_dir, lambda obj: _set_value(obj, 0, 1000, -999.0), pin_manifest=True
    )
    _source, again = sealed_inputs("etth1", data_root, manifest_dir, H)
    assert [entry.values for entry in again] == before


# --- Test 5: real tampering ----------------------------------------------------------------------


def test_real_sealed_tampering(real_env: tuple[Path, Path, Path, Path]) -> None:
    """Every tamper with the sealed file is an ArenaError."""
    data_root, manifest_dir, _cache_root, _sealed_root = real_env
    sealed_file = _processed(data_root, "etth1", "sealed")
    manifest_file = splits_manifest_path(manifest_dir, "etth1")
    sealed_bytes, manifest_bytes = sealed_file.read_bytes(), manifest_file.read_bytes()

    def restore() -> None:
        sealed_file.write_bytes(sealed_bytes)
        manifest_file.write_bytes(manifest_bytes)

    # a sealed value changed without updating the manifest: hash mismatch
    _rewrite_sealed(data_root, manifest_dir, lambda obj: _set_value(obj, 0, 800, 5.0), False)
    with pytest.raises(ArenaError, match="does not match the split manifest"):
        sealed_inputs("etth1", data_root, manifest_dir, H)
    restore()

    # a value before index 718 changed, the manifest re-pinned: prefix mismatch
    _rewrite_sealed(data_root, manifest_dir, lambda obj: _set_value(obj, 2, 5, 5.0), True)
    with pytest.raises(ArenaError, match="does not extend the public data"):
        sealed_inputs("etth1", data_root, manifest_dir, H)
    restore()

    # the sealed file deleted
    sealed_file.unlink()
    with pytest.raises(ArenaError, match="is missing"):
        sealed_inputs("etth1", data_root, manifest_dir, H)
    restore()

    # two series ids swapped, the manifest re-pinned
    _rewrite_sealed(data_root, manifest_dir, _swap_first_two_ids, True)
    with pytest.raises(ArenaError, match="starting"):
        sealed_inputs("etth1", data_root, manifest_dir, H)
    restore()


# --- Test 8: the real sealed pool -----------------------------------------------------------------


def test_real_sealed_pool(real_env: tuple[Path, Path, Path, Path], tmp_path: Path) -> None:
    """Sealed boundaries, values cut at the last scored target, no oracle."""
    data_root, manifest_dir, _cache_root, sealed_root = real_env
    pool = load_sealed_pool("etth1", data_root, manifest_dir, sealed_root)
    assert isinstance(pool, DevPool)
    assert pool.dataset == "etth1"
    assert pool.cache_root == sealed_root
    assert [entry.series_id for entry in pool.series] == ETTH1_IDS
    sealed = _sealed_values(data_root)
    for entry in pool.series:
        assert entry.period == 24
        assert entry.scale_end == SCALE_END
        assert entry.score_start == SEALED_START
        assert entry.score_end == SEALED_END
        assert entry.change_points == ()
        assert entry.oracle is None
        assert len(entry.values) == SEALED_END + H
        assert entry.values == tuple(sealed[entry.series_id][: SEALED_END + H])

    # an equal rebuild is a no-op with the same cache id
    build_sealed_cache("etth1", data_root, manifest_dir, sealed_root)
    assert cache_id(sealed_root, "etth1") == pool.cache_id

    # the sealed cache root is never the public cache root
    public_root = tmp_path / "nested" / "cache_store"
    with pytest.raises(ArenaError, match="public cache root"):
        build_sealed_cache("etth1", data_root, manifest_dir, public_root)
    with pytest.raises(ArenaError, match="public cache root"):
        load_sealed_pool("etth1", data_root, manifest_dir, public_root)


# --- Test 9: the synthetic sealed pool ------------------------------------------------------------


def test_synthetic_sealed_pool(salted_env: tuple[Path, Path], tmp_path: Path) -> None:
    """Warmup scoring start, salted change points and oracle."""
    data_root, manifest_dir = salted_env
    sealed_root = tmp_path / "sealed_cache"
    build_sealed_cache("synthetic-sealed", data_root, manifest_dir, sealed_root)
    pool = load_sealed_pool("synthetic-sealed", data_root, manifest_dir, sealed_root)
    salt = salt_path(data_root).read_bytes()
    assert [entry.series_id for entry in pool.series] == [f"{family}/0" for family in FAMILIES]
    for entry in pool.series:
        family = entry.series_id.split("/")[0]
        world = make_task(family, world_seed("sealed", family, 0, salt)).world
        assert entry.period == world.spec.period
        assert entry.scale_end == 300  # GrammarConfig().warmup
        assert entry.score_start == 300
        assert entry.score_end == 2000 - H
        assert entry.change_points == world.change_points
        assert entry.values == world.y
        assert entry.oracle is not None
        assert entry.oracle[0] == true_best_forecast(world, 300, H)


# --- Test 10: pool mismatches ---------------------------------------------------------------------


def test_sealed_pool_mismatches(
    real_env: tuple[Path, Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache from different data, an edited source, no cache, an unknown dataset."""
    data_root, manifest_dir, _cache_root, sealed_root = real_env

    shifted = _build_etth1_fixture(tmp_path / "fixtures", rows=ROWS, variant=1)
    _install(monkeypatch, etth1=shifted)
    other_data, other_manifests = tmp_path / "v1" / "data", tmp_path / "v1" / "manifests"
    fetch(shifted, other_data, other_manifests, today=date(2026, 10, 11))
    prepare("etth1", other_data, other_manifests)
    other_sealed = tmp_path / "v1" / "sealed_cache"
    build_sealed_cache("etth1", other_data, other_manifests, other_sealed)
    with pytest.raises(ArenaError):
        load_sealed_pool("etth1", data_root, manifest_dir, other_sealed)

    manifest_file = cache_dir(sealed_root, "etth1") / "manifest.json"
    original = manifest_file.read_bytes()
    obj = json.loads(original)
    obj["source"]["sealed_sha256"] = "0" * 64
    manifest_file.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ArenaError, match="recomputes"):
        load_sealed_pool("etth1", data_root, manifest_dir, sealed_root)
    manifest_file.write_bytes(original)

    with pytest.raises(ArenaError, match=r"manifest\.json"):
        load_sealed_pool("etth1", data_root, manifest_dir, tmp_path / "empty")
    with pytest.raises(ArenaError, match="unknown dataset"):
        load_sealed_pool("nope", data_root, manifest_dir, sealed_root)


# --- Test 11: the dev path stays blind ------------------------------------------------------------


def test_dev_path_stays_blind(
    real_env: tuple[Path, Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the sealed cache built, the dev path opens no sealed file, salt or sealed cache."""
    data_root, manifest_dir, cache_root, sealed_root = real_env
    opened: list[str] = []
    _record_opens(monkeypatch, opened)
    load_dev_pool("etth1", cache_root, data_root, manifest_dir)
    public_inputs("etth1", data_root, manifest_dir)
    assert opened, "the dev path should have opened its manifest, public file and cache"
    assert str(_processed(data_root, "etth1", "sealed")) not in opened
    assert not any(path.startswith(str(data_root / "sealed")) for path in opened)
    assert not any(path.startswith(str(sealed_root)) for path in opened)


# --- Test 12: determinism ------------------------------------------------------------------------


def test_replay_determinism_on_the_real_sealed_pool(
    real_env: tuple[Path, Path, Path, Path],
) -> None:
    """Two replays of the sealed pool are equal, with a finite positive MASE."""
    data_root, manifest_dir, _cache_root, sealed_root = real_env
    pool = load_sealed_pool("etth1", data_root, manifest_dir, sealed_root)
    first = replay_dataset(EqualWeights, {}, 0, pool)
    second = replay_dataset(EqualWeights, {}, 0, pool)
    assert first == second
    assert first.status == "ok"
    assert math.isfinite(first.metrics["mase"])
    assert first.metrics["mase"] > 0.0


# --- Test 13: AST guards ------------------------------------------------------------------------


def test_sealed_init_imports_nothing() -> None:
    """Importing the sealed package must load nothing: no import statements in __init__."""
    tree = ast.parse((SRC / "arena" / "sealed" / "__init__.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        assert not isinstance(node, (ast.Import, ast.ImportFrom))


def _docstring_ids(tree: ast.Module) -> set[int]:
    """Return the ``id()`` of every docstring constant node in ``tree``."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                ids.add(id(first.value))
    return ids


def test_no_sealed_path_strings_outside_the_sealed_package() -> None:
    """Only arena/sealed/ modules and runner/cli.py name the salt file or sealed cache root."""
    for path in sorted(SRC.rglob("*.py")):
        relative = path.relative_to(SRC).as_posix()
        if relative.startswith("arena/sealed/") or relative == "runner/cli.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstring_ids(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                assert "synthetic.salt" not in node.value, f"{relative}: {node.value!r}"
                assert "sealed_cache" not in node.value, f"{relative}: {node.value!r}"
