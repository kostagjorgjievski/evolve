"""Holdout scoring of accepted edits (TASK_018 PR C: tests 1-7).

Each test first runs a tiny lineage into its own ledger — the engine's seams
stubbed exactly as ``test_phase1`` stubs them, but over a synthetic-shift3
fixture pool so the lineage's dataset family matches the holdout — and then
scores that lineage on ``synthetic-shift3-holdout``. The holdout's own
evaluation is the real subprocess evaluator where the test pins real
behaviour (the happy path's scores, the weights space's ``invalid_weights``)
and a stub where it pins structure (the pinned reference source, the linkage
keys); the lineages' evaluations are always stubbed, as in ``test_phase1``.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from pytest import CaptureFixture, MonkeyPatch

from evolvets.arena.dev import load_dev_pool
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.improver.artifacts import (
    REFERENCE_ARTIFACT_CONFIG,
    REFERENCE_ARTIFACT_SOURCE,
    WEIGHTS_REFERENCE_ARTIFACT_SOURCE,
    artifact_updater,
)
from evolvets.improver.interfaces import LLMResponse
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import RunStart
from evolvets.ledger.schema import Record
from evolvets.runner import phase1_holdout
from evolvets.runner.cli import main as cli_main
from evolvets.runner.phase1_eval import EvalOutcome
from evolvets.runner.phase1_holdout import HoldoutError, run_holdout
from tests.runner.test_phase1 import (
    WEIGHTS_CHILD,
    StubEvaluator,
    child,
    failed,
    install,
    lineage_run,
    lineage_spec,
    ok,
    records_of,
    reply,
    run_end_of,
    verify_run,
)

#: The holdout every test scores: shift3's replicate, matching the fixture lineages.
HOLDOUT = "synthetic-shift3-holdout"

CHILD_A = child("0.5")
CHILD_C = child("0.7")

#: A flat-forecast child of the weights reference: ``update`` answers the
#: combined forecast over horizons (the forecast-v2 shape) instead of weight
#: rows, so the arena scores it ``invalid_weights`` under weights-v1 — its
#: rule, not a new one (each "row" of a flat tuple is a float).
WEIGHTS_FLAT_CHILD = WEIGHTS_REFERENCE_ARTIFACT_SOURCE.replace(
    "    rows = []\n"
    "    for e in range(n_experts):\n"
    "        row = []\n"
    "        for h in range(horizon):\n"
    "            row.append(weights[h][e])\n"
    "        rows.append(tuple(row))\n"
    "    return (tuple(rows), state)",
    "    out = []\n"
    "    for h in range(horizon):\n"
    "        base = 0.0\n"
    "        for e in range(n_experts):\n"
    "            base = base + weights[h][e] * context.forecasts[e][h]\n"
    "        out.append(base)\n"
    "    return (tuple(out), state)",
)
assert WEIGHTS_FLAT_CHILD != WEIGHTS_REFERENCE_ARTIFACT_SOURCE  # the edit landed


@pytest.fixture(scope="module")
def shift3(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The shift3 fixture pools: (base, cache_root) with the scan and the holdout cache."""
    base = tmp_path_factory.mktemp("phase1-holdout")
    cache_root = base / "cache_store"
    for dataset in ("synthetic-shift3", HOLDOUT):
        source, series = public_inputs(dataset, base, base, per_family=1)
        build_cache(dataset, source, series, cache_root)
    return base, cache_root


def shift3_lineage(
    shift3: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    run_id: str,
    *,
    responses: Sequence[LLMResponse] = (),
    outcomes: Mapping[str, EvalOutcome] | None = None,
    default: EvalOutcome | Exception | None = None,
    **spec: Any,
) -> int:
    """Run one stubbed lineage on synthetic-shift3 into tmp_path's ledger."""
    install(monkeypatch, responses=responses, outcomes=outcomes, default=default)
    return lineage_run(
        shift3, tmp_path, replace(lineage_spec(**spec), dataset="synthetic-shift3"), run_id
    )


def holdout_run(
    shift3: tuple[Path, Path],
    tmp_path: Path,
    *,
    from_run: str = "lineage",
    run_id: str = "holdout",
    dataset: str = HOLDOUT,
    pool_spec: str = "simple",
    foundation_models: tuple[str, ...] | None = None,
) -> int:
    """Score one lineage on the shift3 holdout into tmp_path's ledger."""
    base, cache_root = shift3
    return run_holdout(
        from_run,
        dataset=dataset,
        pool_spec=pool_spec,
        foundation_models=foundation_models,
        ledger_root=tmp_path / "ledgers",
        cache_root=cache_root,
        data_root=base,
        manifest_dir=base,
        foundation_root=base / "foundation_cache",
        run_id=run_id,
    )


def patch_holdout(
    monkeypatch: MonkeyPatch, evaluator: StubEvaluator | None = None
) -> StubEvaluator | None:
    """Stub the holdout's git commit, and its evaluator when one is given."""
    monkeypatch.setattr(phase1_holdout, "current_git_commit", lambda repo: "0" * 40)
    if evaluator is not None:
        monkeypatch.setattr(phase1_holdout, "evaluate_source", evaluator)
    return evaluator


def baselines_of(records: Sequence[Record]) -> list[Record]:
    """The run's baseline records, in ledger order."""
    return [record for record in records if record.kind == "baseline"]


def blobs_of(tmp_path: Path, run_id: str, records: Sequence[Record]) -> list[dict[str, Any]]:
    """The baseline records' per-task blobs, decoded, in ledger order."""
    blobs = BlobStore(tmp_path / "ledgers" / run_id / "blobs")
    out = []
    for record in baselines_of(records):
        payload = record.payload
        assert payload.per_task_blob is not None
        out.append(json.loads(blobs.get(payload.per_task_blob).decode("utf-8")))
    return out


# --- Test 1: the happy path ----------------------------------------------------------


def test_happy_path_scores_start_and_each_accepted_edit(
    shift3: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """One baseline per accepted edit plus start, in acceptance order; exit 0, verified.

    The holdout's evaluation is the real evaluator, so the start artifact's
    score also pins a real replay over the holdout pool: bit-equal pooled
    metrics to an in-process ``replay_dataset`` call over the same pool.
    """
    shift3_lineage(
        shift3,
        tmp_path,
        monkeypatch,
        "lineage",
        responses=[reply(CHILD_A), reply(CHILD_C)],
        outcomes={
            REFERENCE_ARTIFACT_SOURCE: ok(1.0),
            CHILD_A: ok(0.8),  # climbs: acceptance 1
            CHILD_C: ok(0.7),  # climbs again off child A: acceptance 2
        },
        proposals_per_generation=1,
        max_evaluations=3,
    )
    patch_holdout(monkeypatch)
    assert holdout_run(shift3, tmp_path) == 0
    records = records_of(tmp_path, "holdout")
    assert [record.kind for record in records] == [
        "run_start",
        "baseline",
        "baseline",
        "baseline",
        "run_end",
    ]
    baselines = baselines_of(records)
    assert [record.payload.updater_id for record in baselines] == [
        "start",
        "accept-01",
        "accept-02",
    ]
    assert all(record.payload.pool == "dev" for record in baselines)
    assert all(record.payload.status == "ok" for record in baselines)
    assert all(record.budget.evaluations == 1 for record in baselines)
    start = records[0].payload
    assert isinstance(start, RunStart)
    assert start.condition == "phase1-holdout"
    assert start.seed == 0
    assert start.requires_predictions is False
    assert start.budget_limits.evaluations == 3.0  # sized for exactly the artifacts
    assert start.budget_limits.wall_seconds == pytest.approx(3 * (240.0 + 60.0))
    end = run_end_of(records)
    assert end.status == "completed"
    assert end.summary["artifacts"] == 3
    assert end.summary["accepted"] == 2
    assert end.summary["budget_spent"]["evaluations"] == 3.0
    verify_run(tmp_path, "holdout")
    # the start artifact's score is a real replay over the holdout pool
    base, cache_root = shift3
    pool = load_dev_pool(HOLDOUT, cache_root, base, base)
    result = replay_dataset(
        artifact_updater(REFERENCE_ARTIFACT_SOURCE),
        dict(REFERENCE_ARTIFACT_CONFIG),
        0,
        pool,
        interface="forecast-v2",
    )
    assert (
        blobs_of(tmp_path, "holdout", records)[0]["result"]["metrics"]
        == result_to_obj(result)["metrics"]
    )


# --- Test 2: refusals ---------------------------------------------------------------


def test_refuses_aborted_mismatched_and_missing_lineages(
    shift3: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """An aborted lineage, a family mismatch and a missing run are refused, writing nothing."""
    patch_holdout(monkeypatch)
    with pytest.raises(HoldoutError, match="does not verify"):
        holdout_run(shift3, tmp_path, from_run="nope")
    shift3_lineage(
        shift3,
        tmp_path,
        monkeypatch,
        "aborted",
        responses=[reply(CHILD_A)],
        outcomes={REFERENCE_ARTIFACT_SOURCE: failed("worker: crash at import")},
    )
    with pytest.raises(HoldoutError, match="ended 'aborted'"):
        holdout_run(shift3, tmp_path, from_run="aborted")
    shift3_lineage(
        shift3,
        tmp_path,
        monkeypatch,
        "lineage",
        responses=(),
        outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0)},
        max_evaluations=1,  # baseline only: finished, but no proposal ever ran
    )
    with pytest.raises(HoldoutError, match="ran dataset 'synthetic-shift3'"):
        holdout_run(shift3, tmp_path, dataset="synthetic-shift2-holdout")
    assert not (tmp_path / "ledgers" / "holdout").exists()


# --- Test 3: a lineage without proposals ---------------------------------------------


def test_lineage_without_proposals_scores_only_the_pinned_reference(
    shift3: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A lineage that proposed nothing is scored from the pinned reference, once."""
    shift3_lineage(
        shift3,
        tmp_path,
        monkeypatch,
        "lineage",
        responses=[reply(CHILD_A)],
        outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0)},
        llm_output_tokens=16_384 - 1,  # below one proposal's reservation: none runs
    )
    evaluator = patch_holdout(monkeypatch, StubEvaluator({}, ok(1.2)))
    assert holdout_run(shift3, tmp_path) == 0
    records = records_of(tmp_path, "holdout")
    baselines = baselines_of(records)
    assert [record.payload.updater_id for record in baselines] == ["start"]
    assert evaluator is not None
    request = evaluator.requests[0]
    assert request["source"] == REFERENCE_ARTIFACT_SOURCE
    assert request["config"] == dict(REFERENCE_ARTIFACT_CONFIG)
    assert request["seed"] == 0
    assert request["interface"] == "forecast-v2"
    assert request["dataset"] == HOLDOUT
    blob = blobs_of(tmp_path, "holdout", records)[0]
    assert blob["proposal_record_id"] is None  # no proposal parents it
    assert blob["generation"] == 0
    verify_run(tmp_path, "holdout")


# --- Test 4: the weights space on the holdout ----------------------------------------


def test_weights_lineage_scores_rows_ok_and_flats_invalid_weights(
    shift3: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A weights_only lineage replays under weights-v1; a flat child scores invalid_weights.

    The lineage's own dev evaluation is the stub, which says the flat child
    climbed; the holdout re-evaluates for real and records the truth — the
    arena's rule, not a new one — and the run still ends completed.
    """
    shift3_lineage(
        shift3,
        tmp_path,
        monkeypatch,
        "lineage",
        condition="weights_only",
        responses=[reply(WEIGHTS_CHILD), reply(WEIGHTS_FLAT_CHILD)],
        outcomes={
            WEIGHTS_REFERENCE_ARTIFACT_SOURCE: ok(1.0),
            WEIGHTS_CHILD: ok(0.8),
            WEIGHTS_FLAT_CHILD: ok(0.7),  # the stub accepts it; the holdout tells the truth
        },
        proposals_per_generation=1,
        max_evaluations=3,
    )
    patch_holdout(monkeypatch)
    assert holdout_run(shift3, tmp_path) == 0
    records = records_of(tmp_path, "holdout")
    baselines = baselines_of(records)
    assert [record.payload.updater_id for record in baselines] == [
        "start",
        "accept-01",
        "accept-02",
    ]
    assert [record.payload.status for record in baselines] == ["ok", "ok", "invalid_weights"]
    assert baselines[2].payload.metrics == {}
    assert baselines[2].payload.error is not None
    start = records[0].payload
    assert isinstance(start, RunStart)
    config: dict[str, Any] = json.loads(
        BlobStore(tmp_path / "ledgers" / "holdout" / "blobs").get(start.config_blob).decode("utf-8")
    )
    assert config == {
        "from_run": "lineage",
        "dataset": HOLDOUT,
        "pool_spec": "simple",
        "foundation_models": None,
        "interface": "weights-v1",
    }
    assert run_end_of(records).status == "completed"  # a non-ok score is a result, not an error
    verify_run(tmp_path, "holdout")


# --- Test 5: the linkage keys --------------------------------------------------------


def test_linkage_keys_point_back_at_the_lineage(
    shift3: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Every per-task blob names the lineage, its proposal and that proposal's generation."""
    shift3_lineage(
        shift3,
        tmp_path,
        monkeypatch,
        "lineage",
        responses=[reply(CHILD_A), reply(CHILD_C)],
        outcomes={
            REFERENCE_ARTIFACT_SOURCE: ok(1.0),
            CHILD_A: ok(0.8),
            CHILD_C: ok(0.7),
        },
        proposals_per_generation=1,
        max_evaluations=3,
    )
    patch_holdout(monkeypatch, StubEvaluator({}, ok(1.1)))
    assert holdout_run(shift3, tmp_path) == 0
    proposals = [record for record in records_of(tmp_path, "lineage") if record.kind == "proposal"]
    blobs = blobs_of(tmp_path, "holdout", records_of(tmp_path, "holdout"))
    assert [blob["lineage_run_id"] for blob in blobs] == ["lineage"] * 3
    assert [blob["proposal_record_id"] for blob in blobs] == [
        proposals[0].record_id,  # the start parents the first proposal
        proposals[0].record_id,  # acceptance 1 judged it
        proposals[1].record_id,  # acceptance 2 judged the generation-1 child
    ]
    assert [blob["generation"] for blob in blobs] == [0, 0, 1]


# --- Test 6: no improver-driven records, no improver imports -------------------------


def test_holdout_holds_no_improver_records_and_imports_no_improver(
    shift3: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The run holds only run_start, baselines and run_end; the module never imports improver."""
    shift3_lineage(
        shift3,
        tmp_path,
        monkeypatch,
        "lineage",
        responses=(),
        outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0)},
        max_evaluations=1,
    )
    patch_holdout(monkeypatch, StubEvaluator({}, ok(1.1)))
    assert holdout_run(shift3, tmp_path) == 0
    assert {record.kind for record in records_of(tmp_path, "holdout")} <= {
        "run_start",
        "baseline",
        "run_end",
    }
    # and the module's own imports name no evolvets.improver module, anywhere
    source = Path(phase1_holdout.__file__).read_text(encoding="utf-8")
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    assert not [
        name
        for name in names
        if name == "evolvets.improver" or name.startswith("evolvets.improver.")
    ]


# --- Test 7: the CLI -----------------------------------------------------------------


def _holdout_cli_args(shift3: tuple[Path, Path], tmp_path: Path, **overrides: str) -> list[str]:
    """One `phase1 holdout` invocation's arguments over the fixture pools."""
    base, cache_root = shift3
    args = {
        "from-run": "lineage",
        "dataset": HOLDOUT,
        "pool-spec": "simple",
        "ledger-root": str(tmp_path / "ledgers"),
        "cache-root": str(cache_root),
        "data-root": str(base),
        "manifest-dir": str(base),
        "foundation-root": str(base / "foundation_cache"),
    }
    args.update(overrides)
    flat: list[str] = ["phase1", "holdout"]
    for flag, value in args.items():
        flat += [f"--{flag}", value]
    return flat


def _baseline_only_lineage(
    shift3: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Run the cheapest finished lineage: baseline only, no proposal."""
    shift3_lineage(
        shift3,
        tmp_path,
        monkeypatch,
        "lineage",
        responses=(),
        outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0)},
        max_evaluations=1,
    )


def test_cli_scores_with_explicit_and_default_run_ids(
    shift3: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """--run-id names the run; without it a fresh phase1-holdout id is minted; both verify."""
    _baseline_only_lineage(shift3, tmp_path, monkeypatch)
    patch_holdout(monkeypatch, StubEvaluator({}, ok(1.2)))
    assert cli_main(_holdout_cli_args(shift3, tmp_path, **{"run-id": "cli-holdout"})) == 0
    assert "OK cli-holdout completed" in capsys.readouterr().out
    verify_run(tmp_path, "cli-holdout")
    assert cli_main(_holdout_cli_args(shift3, tmp_path)) == 0
    capsys.readouterr()
    minted = [
        path.name
        for path in sorted((tmp_path / "ledgers").iterdir())
        if path.name.startswith("phase1-holdout")
    ]
    assert len(minted) == 1  # the default --run-id: one fresh phase1-holdout run
    verify_run(tmp_path, minted[0])


def test_cli_missing_foundation_cache_fails_before_any_run(
    shift3: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """simple+foundation without a foundation cache: one ArenaError FAIL line, nothing written."""
    _baseline_only_lineage(shift3, tmp_path, monkeypatch)
    patch_holdout(monkeypatch, StubEvaluator({}, ok(1.2)))
    args = _holdout_cli_args(
        shift3,
        tmp_path,
        **{
            "pool-spec": "simple+foundation",
            "foundation-model": "fake",
            "run-id": "never",
        },
    )
    assert cli_main(args) == 1
    out = capsys.readouterr().out
    assert out.startswith("FAIL")
    assert "foundation cache" in out
    assert not (tmp_path / "ledgers" / "never").exists()
