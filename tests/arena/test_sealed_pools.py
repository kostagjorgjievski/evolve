"""Sealed-pool tests: salt, manifest and synthetic inputs (PR B tests 1-3, 6-7 and guards).

The real sealed inputs, the sealed caches, the sealed pools and the CLI tests
land with the second PR of this stack.
"""

from __future__ import annotations

import ast
import json
import stat
from pathlib import Path

import pytest

from evolvets.arena.errors import ArenaError
from evolvets.arena.sealed.pools import (
    SALT_BYTES,
    SyntheticSealedManifest,
    init_salt,
    read_synthetic_manifest,
    salt_path,
    sealed_inputs,
    synthetic_manifest_path,
)
from evolvets.arena.synthetic import GENERATOR_VERSION
from evolvets.arena.synthetic.grammar import FAMILIES, make_task
from evolvets.arena.synthetic.seeds import dev_tasks, world_seed
from evolvets.ledger.canonical import sha256_hex

H = 24
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
