"""Pool tests: no look-ahead, shape, golden hash, and the static hygiene scan (11-14)."""

import ast
import math
from pathlib import Path

import pytest

from evolvets.arena.synthetic.grammar import make_task
from evolvets.arena.synthetic.seeds import world_seed
from evolvets.experts.interfaces import ExpertError
from evolvets.experts.pool import EXPERT_NAMES, first_origin, make_pool, run_pool
from evolvets.ledger.canonical import canonical_bytes, sha256_hex

SRC = Path(__file__).parents[2] / "src" / "evolvets" / "experts"


def _series(n):
    """A deterministic, varied series of ``n`` values."""
    return [0.3 * math.sin(i / 5.0) + (i % 24) * 0.25 + i * 0.001 for i in range(n)]


# --- Test 11: no look-ahead and the prefix property ---------------------------------


@pytest.mark.parametrize("cut", [287, 288, 300, 400])
def test_no_look_ahead(cut):
    """Values after the cut cannot change forecasts at or before it (==, not approx)."""
    values = _series(500)
    changed = values[: cut + 1] + [v * 0.5 - 3.0 for v in values[cut + 1 :]]
    base = run_pool(values, 24, 24)
    other = run_pool(changed, 24, 24)
    for origin in range(287, cut + 1):
        assert base[origin - 287] == other[origin - 287], f"origin {origin}"
    assert run_pool(values[: cut + 1], 24, 24) == base[: cut - 286]


# --- Test 12: run_pool shape and errors ----------------------------------------------


def test_run_pool_shape_and_order():
    """One entry per origin from first_origin, five experts per entry, in pool order."""
    values = _series(330)
    out = run_pool(values, 24, 7)
    assert len(out) == 330 - 287
    reference = make_pool(24)
    for v in values:
        for e in reference:
            e.observe(v)
    # The last entry must equal freshly-run experts at the last origin.
    expected = tuple(e.forecast(7) for e in reference)
    assert out[-1] == expected
    for entry in out:
        assert len(entry) == 5
        for forecasts in entry:
            assert len(forecasts) == 7
    # Expert order: entry 0 is naive (the last value), entry 1 seasonal_naive...
    last = values[287]
    assert out[0][0] == (last,) * 7


def test_run_pool_errors_and_first_origin():
    """A series that never reaches first_origin raises; first_origin(24) is 287."""
    assert first_origin(24) == 287
    with pytest.raises(ExpertError):
        run_pool(_series(287), 24, 24)
    with pytest.raises(ExpertError):
        run_pool(_series(100), 24, 24)


# --- Test 13: golden hash ------------------------------------------------------------

GOLDEN_POOL_SHA256 = "5435fe50173cf7a70fca2a675ec998d1b008a799e877eb75e0dadd80ea3f1908"


def test_golden_pool_hash():
    """Pin the SHA-256 of the dev 'mixed' series' pool output.

    Changing this hash means some forecast value changed, which requires
    bumping EXPERTS_VERSION (and, once cached, CACHE_VERSION).
    """
    task = make_task("mixed", world_seed("dev", "mixed", 0))
    out = run_pool(task.world.y, 24, 24)
    nested = [[list(entry) for entry in origin] for origin in out]
    assert sha256_hex(canonical_bytes(nested)) == GOLDEN_POOL_SHA256


# --- Test 14: static hygiene scan of src/evolvets/experts ----------------------------


def test_no_float_sum_and_no_foreign_imports():
    """Every experts module avoids builtin sum/fsum/statistics and non-experts imports."""
    modules = sorted(SRC.glob("*.py"))
    assert len(modules) >= 4
    for path in modules:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "sum":
                    raise AssertionError(f"{path.name}: builtin sum() on floats")
                if isinstance(node.func, ast.Attribute) and node.func.attr == "fsum":
                    raise AssertionError(f"{path.name}: math.fsum")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "statistics" or alias.name.startswith("statistics."):
                        raise AssertionError(f"{path.name}: statistics module")
                    _check_import(path.name, alias.name)
            if isinstance(node, ast.ImportFrom):
                if node.module is None or node.level > 0:
                    raise AssertionError(f"{path.name}: relative import")
                if node.module == "statistics" or node.module.startswith("statistics."):
                    raise AssertionError(f"{path.name}: statistics module")
                _check_import(path.name, node.module)


def _check_import(filename, module):
    """Fail unless an evolvets import stays inside evolvets.experts."""
    touches_evolvets = module == "evolvets" or module.startswith("evolvets.")
    inside_experts = module == "evolvets.experts" or module.startswith("evolvets.experts.")
    if touches_evolvets and not inside_experts:
        raise AssertionError(f"{filename}: import outside evolvets.experts: {module}")


def test_expert_names_order():
    """EXPERT_NAMES is the pool order used by the cache (PR B)."""
    assert EXPERT_NAMES == ("naive", "seasonal_naive", "ses", "holt_winters", "ar_ridge")
    assert tuple(e.name for e in make_pool(24)) == EXPERT_NAMES
