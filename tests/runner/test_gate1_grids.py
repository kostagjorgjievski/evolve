"""Gate 1 tuning grids and the fixed selection rule over dev records (TASK_011 PR B, part 1).

Tests 1 and 8 of the PR B list. The grids are checked for shape, order and
validity against the baselines' own ``init``; the selection rule is checked
on hand-built ledgers (a ``LedgerWriter`` plus ``BlobStore`` in ``tmp_path``)
so every rule — lowest MASE, exact-tie order, all-failed, sealed-stream and
oracle exclusion — is exercised without a tuning run.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.canonical import canonical_bytes, canonical_json
from evolvets.ledger.payloads import Baseline
from evolvets.ledger.schema import Record
from evolvets.ledger.store import LedgerReader, LedgerWriter
from evolvets.runner.gate1_grids import (
    ALPHA_GRID,
    ETA_GRID,
    GATE1_GRIDS,
    SELECTION_METRIC,
    select_configs,
)
from evolvets.updater.baselines import BASELINE_IDS, BASELINES, DEFAULT_CONFIGS
from evolvets.updater.learned import LEARNED_IDS


def blob_reader(root: Path, run_id: str) -> Callable[[str], object]:
    """A ``read_blob`` over one run's blob store, as ``select_configs`` takes it."""
    blobs = BlobStore(root / run_id / "blobs")

    def read(ref: str) -> object:
        return json.loads(blobs.get(ref).decode("utf-8"))

    return read


# --- Test 1: the grids ------------------------------------------------------------------------


def test_grids_shape_order_and_validity() -> None:
    """Sizes 1, 1, 11, 66, 1 in BASELINE_IDS order (80 points); no duplicates; inits pass."""
    assert list(GATE1_GRIDS) == list(BASELINE_IDS)
    sizes = {baseline: len(GATE1_GRIDS[baseline]) for baseline in BASELINE_IDS}
    assert sizes == {"equal": 1, "ftl": 1, "hedge": 11, "fixed_share": 66, "adahedge": 1}
    assert sum(sizes.values()) == 80
    assert [dict(config) for config in GATE1_GRIDS["hedge"]] == [{"eta": eta} for eta in ETA_GRID]
    assert [dict(config) for config in GATE1_GRIDS["fixed_share"]] == [
        {"eta": eta, "alpha": alpha} for eta in ETA_GRID for alpha in ALPHA_GRID
    ]  # eta outer
    assert SELECTION_METRIC == "mase"
    for baseline in BASELINE_IDS:
        grid = [dict(config) for config in GATE1_GRIDS[baseline]]
        assert len({canonical_json(config) for config in grid}) == len(grid)
        assert dict(DEFAULT_CONFIGS[baseline]) in grid
        for config in GATE1_GRIDS[baseline]:
            BASELINES[baseline]().init(5, 24, config, 0)  # no grid point is a known failure


# --- Test 8: the selection rule ---------------------------------------------------------------


def _write_hand_built(root: Path) -> list[Record]:
    """A run whose baseline records exercise every selection rule (and none else)."""
    blobs = BlobStore(root / "hand-run" / "blobs")

    def result_blob() -> str:
        return blobs.put(
            canonical_bytes({"result": {"dataset": "synthetic-dev", "pool_spec": "simple"}})
        )

    rows = (
        ("hedge", {"eta": 0.5}, 1.2, "ok"),
        ("hedge", {"eta": 1.0}, 1.1, "ok"),  # the lowest mase
        ("hedge", {"eta": 2.0}, 1.1, "ok"),  # an exact tie: the earlier record wins
        ("ftl", {}, None, "failed"),  # every grid point failed
        ("oracle", {}, 0.5, "ok"),  # not a baseline id: ignored
    )
    with LedgerWriter(root, "hand-run", "main") as writer:
        for updater_id, config, mase, status in rows:
            writer.append(
                "baseline",
                Baseline(
                    updater_id=updater_id,
                    hyperparams=dict(config),
                    pool="dev",
                    status=status,
                    metrics={} if mase is None else {"mase": mase},
                    per_task_blob=result_blob(),
                    error=None if status == "ok" else "ValueError: boom",
                ),
            )
    with LedgerWriter(root, "hand-run", "sealed") as writer:  # sealed stream: ignored
        writer.append(
            "baseline",
            Baseline(
                updater_id="hedge",
                hyperparams={},
                pool="sealed",
                status="ok",
                metrics={"mase": 0.1},
                per_task_blob=blobs.put(
                    canonical_bytes({"result": {"dataset": "etth1", "pool_spec": "simple"}})
                ),
                error=None,
            ),
        )
    return list(LedgerReader(root, "hand-run").records())


def test_select_configs_rules(tmp_path: Path) -> None:
    """Lowest mase wins, exact ties go to the lower seq, all-failed is None, rest is ignored."""
    root = tmp_path / "ledgers"
    records = _write_hand_built(root)
    selection = select_configs(records, blob_reader(root, "hand-run"))
    assert list(selection) == ["synthetic-dev"]  # the sealed stream's etth1 never appears
    simple = selection["synthetic-dev"]["simple"]
    assert list(simple) == ["hedge", "ftl"]
    pick = simple["hedge"]
    assert pick is not None
    assert pick["config"] == {"eta": 1.0}
    assert pick["mase"] == 1.1
    assert pick["record_id"] == records[1].record_id
    assert simple["ftl"] is None


def test_select_configs_updater_ids_filter(tmp_path: Path) -> None:
    """The updater_ids filter narrows the rule; learned ids select nothing here (TASK_014)."""
    root = tmp_path / "ledgers"
    records = _write_hand_built(root)
    read = blob_reader(root, "hand-run")
    selection = select_configs(records, read, updater_ids=("hedge",))
    simple = selection["synthetic-dev"]["simple"]
    assert list(simple) == ["hedge"]  # ftl drops out, not just loses to hedge
    pick = simple["hedge"]
    assert pick is not None
    assert pick["config"] == {"eta": 1.0}
    assert pick["record_id"] == records[1].record_id
    # The tune-side half of TASK_014's test 8: over a tune run's records —
    # baseline ids only — the learned filter selects nothing.
    assert select_configs(records, read, updater_ids=LEARNED_IDS) == {}
