"""The Gate 1 tuning grids and the fixed selection rule over dev records.

The grids are fixed in code before any real dev run, so a run's
``run_start`` pins them together with its git commit (TASK_011): Hedge
sweeps a learning-rate grid, Fixed-Share crosses it with a sharing-rate
grid, and the remaining baselines have nothing to tune. The selection
rule — lowest dev MASE, an earlier record on an exact tie — reads only
records and blobs, so the tuning run's summary and the report (PR C) can
both recompute it from the ledger alone.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from evolvets.ledger.payloads import Baseline
from evolvets.ledger.schema import Record
from evolvets.updater.baselines import BASELINE_IDS
from evolvets.updater.interfaces import ConfigValue

#: Hedge's learning-rate grid, ``2**-6 .. 2**4``.
ETA_GRID: tuple[float, ...] = (
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
)
#: Fixed-Share's sharing-rate grid, crossed with every eta of ``ETA_GRID``.
ALPHA_GRID: tuple[float, ...] = (0.0001, 0.001, 0.003, 0.01, 0.03, 0.1)

GATE1_GRIDS: Mapping[str, tuple[Mapping[str, ConfigValue], ...]] = MappingProxyType(
    {
        "equal": ({},),
        "ftl": ({},),
        "hedge": tuple({"eta": eta} for eta in ETA_GRID),
        "fixed_share": tuple(
            {"eta": eta, "alpha": alpha} for eta in ETA_GRID for alpha in ALPHA_GRID
        ),
        "adahedge": ({},),
    }
)
#: The metric the selection rule minimises over a grid's records.
SELECTION_METRIC = "mase"

Selection = dict[str, dict[str, dict[str, dict[str, object] | None]]]
# dataset -> pool spec -> baseline id -> {"config": {...}, "mase": float, "record_id": str} | None


@dataclass(frozen=True)
class _Pick:
    """The winning record of one (dataset, pool spec, baseline) so far."""

    config: dict[str, object]
    mase: float
    record_id: str


def select_configs(
    records: Sequence[Record],
    read_blob: Callable[[str], object],
    *,
    updater_ids: Sequence[str] = BASELINE_IDS,
) -> Selection:
    """Select each (dataset, pool spec, updater)'s tuned configuration from one run's records.

    Only main-stream dev ``baseline`` records with an updater id in
    ``updater_ids`` — the baselines' ids by default, the learned ids
    (``LEARNED_IDS``) for a probe run (TASK_014 PR C) — are considered; each
    record's blob ``result`` names the dataset and pool spec it belongs to
    (a failed result's blob carries them too). Among the ``ok`` records the
    lowest ``metrics["mase"]`` wins, an exact tie goes to the earlier
    record, and a triple with no ``ok`` record selects ``None``. Entries
    appear in first-seen order.
    """
    picks: dict[str, dict[str, dict[str, _Pick | None]]] = {}
    for record in records:
        if record.stream != "main" or record.kind != "baseline":
            continue
        payload = record.payload
        if not isinstance(payload, Baseline):
            continue
        if payload.pool != "dev" or payload.updater_id not in updater_ids:
            continue
        if payload.per_task_blob is None:
            continue
        blob = read_blob(payload.per_task_blob)
        if not isinstance(blob, dict):
            continue
        result = blob.get("result")
        if not isinstance(result, dict):
            continue
        dataset = result.get("dataset")
        pool_spec = result.get("pool_spec")
        if not isinstance(dataset, str) or not isinstance(pool_spec, str):
            continue
        by_baseline = picks.setdefault(dataset, {}).setdefault(pool_spec, {})
        if payload.updater_id not in by_baseline:
            by_baseline[payload.updater_id] = None
        pick = by_baseline[payload.updater_id]
        if payload.status != "ok":
            continue
        mase = payload.metrics.get(SELECTION_METRIC)
        if mase is None:
            continue
        if pick is None or mase < pick.mase:
            by_baseline[payload.updater_id] = _Pick(
                config=dict(payload.hyperparams), mase=mase, record_id=record.record_id
            )
    selection: Selection = {}
    for dataset, by_pool in picks.items():
        out_pool: dict[str, dict[str, dict[str, object] | None]] = {}
        for pool_spec, by_baseline in by_pool.items():
            out_baseline: dict[str, dict[str, object] | None] = {}
            for baseline_id, pick in by_baseline.items():
                out_baseline[baseline_id] = (
                    None
                    if pick is None
                    else {"config": pick.config, "mase": pick.mase, "record_id": pick.record_id}
                )
            out_pool[pool_spec] = out_baseline
        selection[dataset] = out_pool
    return selection
