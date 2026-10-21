"""The Phase 1 lineage engine (TASK_018 PR A2: tests 8-20).

Every test drives ``run_lineage`` with the evaluator stubbed by exact
artifact source and the LLM scripted by ``ScriptedClient`` — CI never calls
the API and never replays a pool here; test 1 of PR A1 pinned the
evaluator's numbers to ``replay_dataset``. What these tests pin instead is
the protocol: the ledger order propose -> predict -> allocate -> evaluate ->
accept, predictions durable before the evaluation they forecast, equal
budgets through one accounting, hill-climb acceptance after every evaluated
proposal — failures included — and the replay reproducing a source run's
control flow from its records at zero token cost. TASK_018 PR B added the
weights-only condition's tests (5-6 of that PR) after the replay section.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from pytest import CaptureFixture, MonkeyPatch

from evolvets.allocator.interfaces import Candidate
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.improver.artifacts import (
    REFERENCE_ARTIFACT_SOURCE,
    WEIGHTS_REFERENCE_ARTIFACT_SOURCE,
)
from evolvets.improver.client import LLMClientConfig
from evolvets.improver.interfaces import LLMResponse
from evolvets.improver.prompt import WEIGHTS_SYSTEM_PROMPT
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import QUANTILE_KEYS, RunEnd
from evolvets.ledger.schema import Budget, Record
from evolvets.ledger.store import LedgerReader
from evolvets.ledger.validate import validate_run
from evolvets.runner import phase1
from evolvets.runner.cli import main as cli_main
from evolvets.runner.phase1 import CONDITIONS, LineageSpec, Phase1Error, run_lineage
from evolvets.runner.phase1_eval import EvalOutcome
from evolvets.selfmodel.interfaces import HistoryPoint
from tests.improver.fake import ScriptedClient

#: The run summary's exactly pinned keys.
SUMMARY_KEYS = {
    "condition",
    "generations",
    "proposals",
    "predictions",
    "allocations",
    "evaluations",
    "accepted",
    "starting_mase",
    "best_mase",
    "budget_spent",
    "wall_seconds_total",
    "replay_of",
}


def child(base_value: str) -> str:
    """A grammatical mutation of the reference artifact: its base constant moved."""
    return REFERENCE_ARTIFACT_SOURCE.replace("base = 0.0", f"base = {base_value}")


CHILD_A = child("0.5")
CHILD_B = child("0.7")
CHILD_C = child("0.9")
CHILD_D = child("0.3")


def reply(
    source: str, *, output_tokens: int = 100, diagnosis: str = "move the base"
) -> LLMResponse:
    """One scripted completion: a well-formed {diagnosis, source} reply."""
    return LLMResponse(
        text=json.dumps({"diagnosis": diagnosis, "source": source}),
        input_tokens=500,
        output_tokens=output_tokens,
        stop_reason="end_turn",
    )


def ok(mase: float, wall: float = 0.05) -> EvalOutcome:
    """An ok outcome scoring ``mase``, charging ``wall``."""
    return EvalOutcome(
        status="ok",
        result_obj={"metrics": {"mase": mase}},
        series_wall_seconds=(0.01,),
        error=None,
        wall_seconds=wall,
    )


def timed_out(wall: float = 3.0) -> EvalOutcome:
    """A timeout outcome, as the evaluator reports one."""
    return EvalOutcome(
        status="timeout",
        result_obj=None,
        series_wall_seconds=(),
        error="3 s wall deadline exceeded: worker killed",
        wall_seconds=wall,
    )


def failed(error: str = "ValueError: boom") -> EvalOutcome:
    """A failed outcome with a surfaced error."""
    return EvalOutcome(
        status="failed", result_obj=None, series_wall_seconds=(), error=error, wall_seconds=0.05
    )


class StubEvaluator:
    """``evaluate_source`` by exact artifact source; every call's source recorded.

    The default outcome (an ``EvalOutcome``, or an ``Exception`` to raise)
    serves every source without an entry, so a test that mis-scripts a child
    still gets a recorded evaluation rather than a silent skip. ``requests``
    keeps each whole worker request, so a test can pin the interface, config
    and pool a condition evaluated under.
    """

    def __init__(
        self, outcomes: Mapping[str, EvalOutcome], default: EvalOutcome | Exception
    ) -> None:
        self.outcomes = dict(outcomes)
        self.default = default
        self.calls: list[str] = []
        self.requests: list[dict[str, Any]] = []

    def __call__(
        self, request: dict[str, Any], *, timeout_seconds: float, memory_gib: float
    ) -> EvalOutcome:
        del timeout_seconds, memory_gib
        source = str(request["source"])
        self.calls.append(source)
        self.requests.append(dict(request))
        outcome = self.outcomes.get(source, self.default)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class Sight:
    """What the spy allocator was handed: candidates and history per choose()."""

    candidates: list[list[Candidate]] = field(default_factory=list)
    histories: list[list[HistoryPoint]] = field(default_factory=list)


class SpyAllocator:
    """A real allocator wrapped so every choose() is recorded."""

    def __init__(self, inner: Any, sight: Sight) -> None:
        self._inner = inner
        self._sight = sight

    def identify(self) -> dict[str, str]:
        """The wrapped allocator's identity."""
        return self._inner.identify()

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: Any,
    ) -> Any:
        """Record, then delegate."""
        self._sight.candidates.append(list(candidates))
        self._sight.histories.append(list(history))
        return self._inner.choose(candidates, history, evaluations_left, rng)


@dataclass
class Seams:
    """The patched engine seams a test can inspect: the evaluator and the client."""

    evaluator: StubEvaluator
    clients: list[ScriptedClient]
    sight: Sight | None


def install(
    monkeypatch: MonkeyPatch,
    *,
    responses: Sequence[LLMResponse] = (),
    outcomes: Mapping[str, EvalOutcome] | None = None,
    default: EvalOutcome | Exception | None = None,
    sight: Sight | None = None,
    key: bool = True,
) -> Seams:
    """Patch the engine's seams: git commit, evaluator, client and allocator factories."""
    monkeypatch.setattr(phase1, "current_git_commit", lambda repo: "0" * 40)
    evaluator = StubEvaluator(outcomes or {}, default if default is not None else ok(1.5))
    monkeypatch.setattr(phase1, "evaluate_source", evaluator)
    clients: list[ScriptedClient] = []
    client = ScriptedClient(*responses)

    def client_factory(config: LLMClientConfig) -> ScriptedClient:
        clients.append(client)
        return client

    monkeypatch.setattr(phase1, "HttpsLLMClient", client_factory)
    if key:
        monkeypatch.setenv("EVOLVETS_LLM_API_KEY", "test-key")
    if sight is not None:
        real_make_allocator = phase1.make_allocator

        def allocator_factory(allocator_id: str, **kwargs: Any) -> SpyAllocator:
            return SpyAllocator(real_make_allocator(allocator_id, **kwargs), sight)

        monkeypatch.setattr(phase1, "make_allocator", allocator_factory)
    return Seams(evaluator=evaluator, clients=clients, sight=sight)


def lineage_spec(
    *,
    condition: str = "predictive",
    seed: int = 0,
    proposals_per_generation: int = 2,
    eval_slots: int = 1,
    max_evaluations: int = 3,
    max_generations: int = 8,
    llm_output_tokens: int = 200_000,
    llm_input_tokens: int = 2_000_000,
    max_wall_seconds: float = 100_000.0,
    eval_timeout_seconds: float = 240.0,
    eval_memory_gib: float = 2.0,
    replay_of: str | None = None,
) -> LineageSpec:
    """A runnable spec over the fixture pool; the client config is the default."""
    known = CONDITIONS.get(condition)  # unknown ids flow through to run_lineage's refusal
    is_llm = known is not None and known.proposer_id == "llm_edit"
    return LineageSpec(
        condition=condition,
        dataset="synthetic-dev",
        pool_spec="simple",
        foundation_models=None,
        seed=seed,
        proposals_per_generation=proposals_per_generation,
        eval_slots=eval_slots,
        max_evaluations=max_evaluations,
        max_generations=max_generations,
        budget_limits=Budget(
            llm_input_tokens=llm_input_tokens,
            llm_output_tokens=llm_output_tokens,
            wall_seconds=max_wall_seconds,
            evaluations=max_evaluations,
        ),
        eval_timeout_seconds=eval_timeout_seconds,
        eval_memory_gib=eval_memory_gib,
        client_config=LLMClientConfig() if is_llm else None,
        replay_of=replay_of,
    )


@pytest.fixture(scope="module")
def pool(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """The synthetic-dev fixture pool (one series per family): (base, cache_root)."""
    base = tmp_path_factory.mktemp("phase1-lineage")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-dev", base, base, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    return base, cache_root


def lineage_run(pool: tuple[Path, Path], tmp_path: Path, spec: LineageSpec, run_id: str) -> int:
    """Run one lineage into ``tmp_path``'s ledger; the exit code of run_lineage."""
    base, cache_root = pool
    return run_lineage(
        spec,
        ledger_root=tmp_path / "ledgers",
        cache_root=cache_root,
        data_root=base,
        manifest_dir=base,
        foundation_root=base / "foundation_cache",
        run_id=run_id,
    )


def records_of(tmp_path: Path, run_id: str) -> list[Record]:
    """One run's records, chain-verified while reading."""
    return list(LedgerReader(tmp_path / "ledgers", run_id).records())


def run_end_of(records: Sequence[Record]) -> RunEnd:
    """The run's run_end payload."""
    payload = records[-1].payload
    assert isinstance(payload, RunEnd)
    return payload


def kinds_of(records: Sequence[Record]) -> list[str]:
    """The run's record kinds, in ledger order."""
    return [record.kind for record in records]


def verify_run(tmp_path: Path, run_id: str) -> None:
    """The chain and blobs verify, and validate_run finds no violation."""
    reader = LedgerReader(tmp_path / "ledgers", run_id)
    assert reader.verify(check_blobs=True).ok
    assert validate_run(list(reader.records())) == []


# --- Test 8: the end-to-end tiny lineage ---------------------------------------


def test_end_to_end_tiny_lineage_order_and_exit(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """2 generations x 2 proposals x 1 slot: exact ledger order, clean validate, exit 0."""
    seams = install(
        monkeypatch,
        responses=[reply(CHILD_A), reply(CHILD_B), reply(CHILD_C), reply(CHILD_D)],
        outcomes={
            REFERENCE_ARTIFACT_SOURCE: ok(1.0),
            CHILD_A: ok(0.8),
            CHILD_B: ok(1.2),
            CHILD_C: ok(1.4),
            CHILD_D: ok(1.3),
        },
    )
    assert lineage_run(pool, tmp_path, lineage_spec(max_evaluations=3), "run8") == 0
    records = records_of(tmp_path, "run8")
    assert kinds_of(records) == [
        "run_start",
        "baseline",
        "proposal",
        "proposal",
        "prediction",
        "prediction",
        "allocation",
        "evaluation",
        "acceptance",
        "proposal",
        "proposal",
        "prediction",
        "prediction",
        "allocation",
        "evaluation",
        "acceptance",
        "run_end",
    ]
    verify_run(tmp_path, "run8")
    summary = run_end_of(records).summary
    assert set(summary) == SUMMARY_KEYS
    assert run_end_of(records).status == "budget_exhausted"
    assert summary["condition"] == "predictive"
    assert summary["generations"] == 2
    assert summary["proposals"] == 4
    assert summary["predictions"] == 4
    assert summary["allocations"] == 2
    assert summary["evaluations"] == 2
    assert summary["accepted"] == 1  # only the better generation-0 child climbs
    assert summary["starting_mase"] == pytest.approx(1.0)
    assert summary["best_mase"] == pytest.approx(0.8)
    assert summary["budget_spent"]["evaluations"] == 3.0  # baseline + two slots
    assert summary["replay_of"] is None
    assert seams.evaluator.calls[0] == REFERENCE_ARTIFACT_SOURCE  # the baseline ran first


# --- Test 9: the hill-climb rule -------------------------------------------------


def test_hill_climb_accepts_better_and_keeps_worse_out(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A better child advances the lineage (its record, its source); a worse one does not."""
    seams = install(
        monkeypatch,
        responses=[reply(CHILD_A), reply(CHILD_B), reply(CHILD_C), reply(CHILD_D)],
        outcomes={
            REFERENCE_ARTIFACT_SOURCE: ok(1.0),
            CHILD_A: ok(0.8),
            CHILD_B: ok(1.2),
            CHILD_C: ok(0.9),  # worse than the accepted 0.8
            CHILD_D: ok(1.3),
        },
    )
    assert lineage_run(pool, tmp_path, lineage_spec(), "run9") == 0
    records = records_of(tmp_path, "run9")
    proposals = [r for r in records if r.kind == "proposal"]
    acceptances = [r for r in records if r.kind == "acceptance"]
    accepted, rejected = (r.payload for r in acceptances)
    assert accepted.accepted is True
    assert accepted.new_artifact is not None
    assert rejected.accepted is False
    assert rejected.new_artifact is None
    # the accepted child is the next generation's parent: record ref, blob ref, prompt
    child_a_ref = proposals[0].payload.child_artifact
    assert accepted.new_artifact == child_a_ref
    assert proposals[2].refers_to == [proposals[0].record_id]
    assert proposals[2].payload.parent_artifact == child_a_ref
    assert len(seams.clients) == 1
    _system, user = seams.clients[0].prompts[2]  # the generation-1 prompt
    assert "base = 0.5" in user  # it edits child A's source, not the reference's
    summary = run_end_of(records).summary
    assert summary["accepted"] == 1
    assert summary["best_mase"] == pytest.approx(0.8)  # the worse child did not regress it


# --- Test 10: a non-ok evaluation -------------------------------------------------


def test_timeout_evaluation_is_recorded_and_skipped_by_history(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A timed-out child: evaluation timeout, acceptance False, prediction exists, no history."""
    sight = Sight()
    install(
        monkeypatch,
        responses=[reply(CHILD_A), reply(CHILD_B), reply(CHILD_C), reply(CHILD_D)],
        outcomes={
            REFERENCE_ARTIFACT_SOURCE: ok(1.0),
            CHILD_A: timed_out(),
            CHILD_C: ok(1.4),
            CHILD_D: ok(1.3),
        },
        sight=sight,
    )
    assert lineage_run(pool, tmp_path, lineage_spec(), "run10") == 0
    records = records_of(tmp_path, "run10")
    evaluation = next(r for r in records if r.kind == "evaluation")
    assert evaluation.payload.status == "timeout"
    assert evaluation.payload.metrics == {}
    assert evaluation.payload.error is not None
    acceptance = next(r for r in records if r.kind == "acceptance")
    assert acceptance.payload.accepted is False
    assert acceptance.payload.new_artifact is None
    # the run continued, but the timed-out proposal joined no fit history
    assert sight.histories[0] == []
    assert sight.histories[1] == []  # generation 0's only evaluation was not ok
    # its prediction was still written, before the evaluation that timed out
    assert records.index(next(r for r in records if r.kind == "prediction")) < records.index(
        evaluation
    )
    prediction_ids = {r.refers_to[0] for r in records if r.kind == "prediction"}
    assert evaluation.refers_to[0] in prediction_ids


# --- Test 11: equal budget through evaluate_all -----------------------------------


def test_evaluate_all_charges_exactly_the_evaluation_budget(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """max_evaluations=3: baseline plus both generation-0 candidates, then exhaustion."""
    install(
        monkeypatch,
        responses=[reply(CHILD_A), reply(CHILD_B), reply(CHILD_C), reply(CHILD_D)],
        outcomes={
            REFERENCE_ARTIFACT_SOURCE: ok(1.0),
            CHILD_A: ok(0.8),
            CHILD_B: ok(1.2),
            CHILD_C: ok(1.4),
            CHILD_D: ok(1.3),
        },
    )
    spec = lineage_spec(condition="evaluate_all", max_evaluations=3)
    assert lineage_run(pool, tmp_path, spec, "run11a") == 0
    records = records_of(tmp_path, "run11a")
    summary = run_end_of(records).summary
    assert summary["budget_spent"]["evaluations"] == 3.0  # exactly the budget, no more
    assert summary["evaluations"] == 2
    assert run_end_of(records).status == "budget_exhausted"
    verify_run(tmp_path, "run11a")


def test_evaluate_all_second_generation_evaluates_one_of_two(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """max_evaluations=4: the second generation evaluates only 1 of its 2 candidates.

    The task's test 11 reads "max_evaluations=3 ... the second generation
    evaluates only 1 of its 2 candidates", but the baseline is charged
    evaluations=1, so three evaluations close the run after one generation;
    four are needed for the second generation to run at all. This sub-case
    pins the intended truncation; flagged in the PR.
    """
    install(
        monkeypatch,
        responses=[reply(CHILD_A), reply(CHILD_B), reply(CHILD_C), reply(CHILD_D)],
        outcomes={
            REFERENCE_ARTIFACT_SOURCE: ok(1.0),
            CHILD_A: ok(0.8),
            CHILD_B: ok(1.2),
            CHILD_C: ok(1.4),
            CHILD_D: ok(1.3),
        },
    )
    spec = lineage_spec(condition="evaluate_all", max_evaluations=4)
    assert lineage_run(pool, tmp_path, spec, "run11b") == 0
    records = records_of(tmp_path, "run11b")
    allocations = [r for r in records if r.kind == "allocation"]
    assert len(allocations[0].payload.chosen) == 2  # both candidates fitted
    assert len(allocations[1].payload.chosen) == 1  # one evaluation slot left
    assert run_end_of(records).summary["budget_spent"]["evaluations"] == 4.0
    verify_run(tmp_path, "run11b")


# --- Test 12: generation-0 gains reach the fit history -----------------------------


def test_generation_zero_gains_reach_the_allocator_history(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The second generation's allocator sees the evaluated generation-0 gain."""
    sight = Sight()
    install(
        monkeypatch,
        responses=[reply(CHILD_A), reply(CHILD_B), reply(CHILD_C), reply(CHILD_D)],
        outcomes={
            REFERENCE_ARTIFACT_SOURCE: ok(1.0),
            CHILD_A: ok(0.8),
            CHILD_B: ok(1.2),
            CHILD_C: ok(0.7),
            CHILD_D: ok(1.3),
        },
        sight=sight,
    )
    assert lineage_run(pool, tmp_path, lineage_spec(), "run12") == 0
    assert len(sight.histories) == 2
    seen_by_generation_one = sight.histories[1]  # what generation 1's choose() got
    assert len(seen_by_generation_one) == 1
    point = seen_by_generation_one[0]
    assert point.features.generation == 0
    assert point.features.is_llm is True
    assert point.gain == pytest.approx(1.0 - 0.8)  # against the baseline MASE
    # the predictive allocator's candidates carry the five predicted quantiles
    assert all(set(candidate.quantiles) == set(QUANTILE_KEYS) for candidate in sight.candidates[0])


# --- Test 13: prediction-blind rivals ---------------------------------------------


def test_prediction_blind_rival_gets_empty_quantiles_but_predictions_exist(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Condition fixed: candidates carry {} quantiles; every proposal still gets a prediction."""
    sight = Sight()
    install(
        monkeypatch,
        responses=[reply(CHILD_A), reply(CHILD_B)],
        outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0), CHILD_A: ok(1.5), CHILD_B: ok(1.4)},
        sight=sight,
    )
    spec = lineage_spec(condition="fixed", max_evaluations=2)
    assert lineage_run(pool, tmp_path, spec, "run13") == 0
    records = records_of(tmp_path, "run13")
    assert all(candidate.quantiles == {} for candidate in sight.candidates[0])
    proposals = [r for r in records if r.kind == "proposal"]
    predictions = [r for r in records if r.kind == "prediction"]
    assert len(predictions) == len(proposals) == 2
    assert {tuple(prediction.refers_to) for prediction in predictions} == {
        (proposal.record_id,) for proposal in proposals
    }
    verify_run(tmp_path, "run13")


# --- Test 14: the output-token budget binds mid-round ------------------------------


def test_output_token_budget_binds_between_proposals(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A limit paying two reservations but not the third: exactly two proposals run."""
    install(
        monkeypatch,
        responses=[
            reply(CHILD_A, output_tokens=12000),
            reply(CHILD_B, output_tokens=12000),
            reply(CHILD_C, output_tokens=12000),
        ],
        outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0), CHILD_A: ok(0.8)},
    )
    spec = lineage_spec(proposals_per_generation=3, max_evaluations=10, llm_output_tokens=30_000)
    assert lineage_run(pool, tmp_path, spec, "run14") == 0
    records = records_of(tmp_path, "run14")
    summary = run_end_of(records).summary
    assert summary["proposals"] == 2  # the third reservation did not fit
    assert summary["predictions"] == 2
    assert summary["allocations"] == 1
    assert summary["evaluations"] == 1
    assert summary["budget_spent"]["llm_output_tokens"] == 24_000.0
    assert run_end_of(records).status == "budget_exhausted"
    verify_run(tmp_path, "run14")


def test_output_token_budget_below_one_reservation_ends_after_baseline(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A limit below one proposal's reservation: budget_exhausted right after the baseline."""
    install(
        monkeypatch,
        responses=[reply(CHILD_A)],
        outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0), CHILD_A: ok(0.8)},
    )
    spec = lineage_spec(llm_output_tokens=16_384 - 1)  # below (1 + 1) * max_tokens
    assert lineage_run(pool, tmp_path, spec, "run14b") == 0
    records = records_of(tmp_path, "run14b")
    summary = run_end_of(records).summary
    assert summary["proposals"] == 0
    assert summary["evaluations"] == 0
    assert summary["budget_spent"]["evaluations"] == 1.0  # the baseline alone
    verify_run(tmp_path, "run14b")


# --- Test 15: the wall budget binds before an evaluation ---------------------------


def test_wall_budget_binds_before_an_evaluation_that_cannot_pay(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A stubbed evaluator charging real wall: the second evaluation is refused, not run."""
    install(
        monkeypatch,
        outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0, wall=10.0)},
        default=ok(1.5, wall=10.0),
        key=False,  # random_param needs no key
    )
    spec = lineage_spec(
        condition="random_param",
        proposals_per_generation=1,
        max_evaluations=10,
        max_wall_seconds=175.0,
        eval_timeout_seconds=100.0,  # each evaluation reserves 100 + 60 seconds
    )
    assert lineage_run(pool, tmp_path, spec, "run15") == 0
    records = records_of(tmp_path, "run15")
    summary = run_end_of(records).summary
    assert summary["evaluations"] == 1  # one child evaluation; the next could not pay
    assert summary["budget_spent"]["wall_seconds"] == pytest.approx(20.0, abs=1.0)
    assert run_end_of(records).status == "budget_exhausted"
    verify_run(tmp_path, "run15")


# --- Test 16: the baseline itself fails --------------------------------------------


@pytest.mark.parametrize("outcome", [failed("worker: crash at import"), timed_out()])
def test_failed_baseline_aborts_the_run(
    pool: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    outcome: EvalOutcome,
) -> None:
    """A non-ok baseline ends the run aborted, exit 2, with no lineage records at all."""
    install(monkeypatch, outcomes={REFERENCE_ARTIFACT_SOURCE: outcome})
    assert lineage_run(pool, tmp_path, lineage_spec(), "run16") == 2
    records = records_of(tmp_path, "run16")
    assert kinds_of(records) == ["run_start", "baseline", "run_end"]
    assert run_end_of(records).status == "aborted"
    summary = run_end_of(records).summary
    for key in (
        "generations",
        "proposals",
        "predictions",
        "allocations",
        "evaluations",
        "accepted",
    ):
        assert summary[key] == 0
    assert summary["starting_mase"] is None
    assert summary["best_mase"] is None
    assert str(summary["error"]).startswith(f"baseline {outcome.status}: ")
    verify_run(tmp_path, "run16")


# --- Test 17: a crash mid-generation ------------------------------------------------


def test_crash_mid_generation_ends_aborted_and_verifies(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """An evaluator raising mid-generation: aborted run_end, exit 2, clean ledger."""
    install(
        monkeypatch,
        responses=[reply(CHILD_A), reply(CHILD_B)],
        outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0)},
        default=RuntimeError("boom: evaluator died"),
    )
    assert lineage_run(pool, tmp_path, lineage_spec(), "run17") == 2
    records = records_of(tmp_path, "run17")
    assert run_end_of(records).status == "aborted"
    assert "RuntimeError: boom: evaluator died" in str(run_end_of(records).summary["error"])
    verify_run(tmp_path, "run17")


# --- Tests 18 and 19: replay --------------------------------------------------------

#: Within one generation every LLM call carries the same prompt (same parent,
#: same observation), and ``recorded_responses`` serves each prompt key's
#: replies first-in first-out — so a replay reproduces a round's distinct
#: replies in recorded order, and a same-reply round trivially in order; the
#: replay tests script same-reply rounds, the shape they pin.


def _source_and_replay(
    pool: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    *,
    source_spec: LineageSpec,
    replay_spec: LineageSpec,
    responses: Sequence[LLMResponse],
    outcomes: Mapping[str, EvalOutcome],
    default: EvalOutcome | Exception | None = None,
) -> tuple[list[Record], list[Record]]:
    """Run the source live, then replay it; the two runs' records."""
    install(monkeypatch, responses=responses, outcomes=outcomes, default=default)
    assert lineage_run(pool, tmp_path, source_spec, "source") == 0
    install(monkeypatch, outcomes=outcomes, default=default)  # the replay needs the evaluator
    assert lineage_run(pool, tmp_path, replay_spec, "replay") == 0
    return records_of(tmp_path, "source"), records_of(tmp_path, "replay")


def test_replay_reproduces_the_source_at_zero_token_cost(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A tiny lineage replays from its records: zero tokens, identical digests, matches."""
    outcomes = {
        REFERENCE_ARTIFACT_SOURCE: ok(1.0),
        CHILD_A: ok(0.8),
        CHILD_C: ok(0.9),  # the generation-1 candidate, rejected against 0.8
    }
    responses = [
        reply(CHILD_A),  # generation 0: both calls share one prompt, one reply
        reply(CHILD_A),
        reply(CHILD_C),  # generation 1: the parent and history changed the prompt
        reply(CHILD_C),
    ]
    source_records, replay_records = _source_and_replay(
        pool,
        tmp_path,
        monkeypatch,
        source_spec=lineage_spec(max_evaluations=3),
        replay_spec=lineage_spec(max_evaluations=3, replay_of="source"),
        responses=responses,
        outcomes=outcomes,
    )
    source_summary = run_end_of(source_records).summary
    replay_summary = run_end_of(replay_records).summary
    assert replay_summary["budget_spent"]["llm_input_tokens"] == 0.0
    assert replay_summary["budget_spent"]["llm_output_tokens"] == 0.0
    assert replay_summary["matches_source_run"] is True
    assert replay_summary["timeout_status_mismatches"] == 0
    for key in ("generations", "proposals", "predictions", "evaluations", "accepted"):
        assert replay_summary[key] == source_summary[key]
    source_children = [r.payload.child_artifact for r in source_records if r.kind == "proposal"]
    replay_children = [r.payload.child_artifact for r in replay_records if r.kind == "proposal"]
    assert replay_children == source_children  # identical artifact digests
    verify_run(tmp_path, "replay")


def test_replay_of_mid_round_exhaustion_matches(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A source that exhausted tokens between proposals replays exactly its counts."""
    outcomes = {REFERENCE_ARTIFACT_SOURCE: ok(1.0), CHILD_A: ok(0.8)}
    responses = [
        reply(CHILD_A, output_tokens=12000),
        reply(CHILD_A, output_tokens=12000),
        reply(CHILD_A, output_tokens=12000),
    ]
    source_records, replay_records = _source_and_replay(
        pool,
        tmp_path,
        monkeypatch,
        source_spec=lineage_spec(
            proposals_per_generation=3,
            max_evaluations=10,
            llm_output_tokens=30_000,  # two reservations, not the third
        ),
        replay_spec=lineage_spec(
            proposals_per_generation=3,
            max_evaluations=10,
            llm_output_tokens=200_000,  # the replay's calls charge nothing
            replay_of="source",
        ),
        responses=responses,
        outcomes=outcomes,
    )
    source_summary = run_end_of(source_records).summary
    replay_summary = run_end_of(replay_records).summary
    assert source_summary["proposals"] == 2  # the source's round ended early
    for key in ("generations", "proposals", "predictions", "evaluations"):
        assert replay_summary[key] == source_summary[key]
    assert replay_summary["budget_spent"]["llm_output_tokens"] == 0.0
    assert replay_summary["matches_source_run"] is True
    verify_run(tmp_path, "replay")


def test_replay_of_wall_exhaustion_between_allocation_and_evaluation(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A source that exhausted wall after its allocation replays exactly its evaluations.

    Test 15's shape: every evaluation charges 10 s of wall, so the source's
    second chosen proposal fails its 100 + 60 s reservation after the
    allocation that chose it. The replay's wall limit can afford that
    evaluation, but it must not run it — the source's per-generation
    evaluation count, not its chosen count, drives the round — and it ends
    with the source's own status.
    """
    outcomes = {REFERENCE_ARTIFACT_SOURCE: ok(1.0, wall=10.0)}
    both = dict(
        condition="random_param",
        proposals_per_generation=1,
        max_evaluations=10,
        max_wall_seconds=175.0,
        eval_timeout_seconds=100.0,  # each evaluation reserves 100 + 60 seconds
    )
    source_records, replay_records = _source_and_replay(
        pool,
        tmp_path,
        monkeypatch,
        source_spec=lineage_spec(**both),
        replay_spec=lineage_spec(replay_of="source", **both),
        responses=(),
        outcomes=outcomes,
        default=ok(1.5, wall=10.0),
    )
    source_summary = run_end_of(source_records).summary
    replay_summary = run_end_of(replay_records).summary
    assert source_summary["generations"] == 2  # the second round allocated, then could not pay
    assert source_summary["evaluations"] == 1
    for key in ("generations", "proposals", "predictions", "evaluations"):
        assert replay_summary[key] == source_summary[key]
    assert replay_summary["matches_source_run"] is True
    assert replay_summary["timeout_status_mismatches"] == 0
    assert run_end_of(replay_records).status == run_end_of(source_records).status
    verify_run(tmp_path, "source")
    verify_run(tmp_path, "replay")


# --- TASK_018 PR B: the weights-only condition (tests 5-6) -------------------------


#: One params edit of the weights reference: the share step's effective rate doubled.
WEIGHTS_CHILD = WEIGHTS_REFERENCE_ARTIFACT_SOURCE.replace(
    "row[e] * math.exp(-eta * (losses[e] - low))",
    "row[e] * math.exp(-2.0 * eta * (losses[e] - low))",
)
#: A flat-forecast child: the forecast-v2 reference's module as a reply, the
#: shape that ignored the rows rule. The arena scores it invalid_weights.
FLAT_CHILD = REFERENCE_ARTIFACT_SOURCE


def invalid_weights() -> EvalOutcome:
    """An invalid_weights outcome, as the arena scores a flat reply under weights-v1."""
    return EvalOutcome(
        status="invalid_weights",
        result_obj=None,
        series_wall_seconds=(),
        error="invalid_weights: TypeError: weights must be a list or a tuple, got ForecastOutput",
        wall_seconds=0.05,
    )


def test_weights_only_rows_child_climbs_and_flat_child_is_invalid(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A rows-output child is evaluated ok and accepted; a flat child is invalid_weights.

    The flat reply is recorded honestly (evaluation invalid_weights,
    acceptance False) and joins no fit history — the mechanism that keeps the
    object space honest — while the rows child climbs and becomes the next
    generation's parent.
    """
    sight = Sight()
    seams = install(
        monkeypatch,
        responses=[reply(WEIGHTS_CHILD), reply(FLAT_CHILD)],
        outcomes={
            WEIGHTS_REFERENCE_ARTIFACT_SOURCE: ok(1.0),
            WEIGHTS_CHILD: ok(0.8),
            FLAT_CHILD: invalid_weights(),
        },
        sight=sight,
    )
    spec = lineage_spec(condition="weights_only", proposals_per_generation=1, max_evaluations=3)
    assert lineage_run(pool, tmp_path, spec, "run-weights") == 0
    records = records_of(tmp_path, "run-weights")
    # the baseline evaluated the weights reference at its config under weights-v1
    baseline_request = seams.evaluator.requests[0]
    assert baseline_request["source"] == WEIGHTS_REFERENCE_ARTIFACT_SOURCE
    assert baseline_request["config"] == {"eta": 1.0, "alpha": 0.01}
    assert baseline_request["interface"] == "weights-v1"
    assert seams.clients[0].prompts[0][0] == WEIGHTS_SYSTEM_PROMPT  # the weights contract
    evaluations = [r for r in records if r.kind == "evaluation"]
    acceptances = [r for r in records if r.kind == "acceptance"]
    assert [e.payload.status for e in evaluations] == ["ok", "invalid_weights"]
    assert [a.payload.accepted for a in acceptances] == [True, False]
    assert acceptances[0].payload.new_artifact is not None
    assert acceptances[1].payload.new_artifact is None
    # the flat child joined no fit history; the rows child's generation-0 gain did
    assert sight.histories[0] == []
    assert len(sight.histories[1]) == 1
    assert sight.histories[1][0].gain == pytest.approx(1.0 - 0.8)
    proposals = [r for r in records if r.kind == "proposal"]
    assert proposals[1].payload.parent_artifact == proposals[0].payload.child_artifact
    verify_run(tmp_path, "run-weights")


def test_weights_only_registry_predictive_with_visible_quantiles(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """CONDITIONS["weights_only"] is the treatment's shape over the weights-v1 space.

    Same predictive allocator, quantiles visible: every candidate the spy
    allocator was handed carries the five predicted quantile keys, so the
    ablation differs from the treatment in the object space and nothing else.
    """
    condition = CONDITIONS["weights_only"]
    assert condition.proposer_id == "llm_edit"
    assert condition.allocator_id == "predictive"
    assert condition.interface == "weights-v1"
    assert condition.reference_updater_id == "weights_reference"
    assert condition.system_prompt == WEIGHTS_SYSTEM_PROMPT
    sight = Sight()
    install(
        monkeypatch,
        responses=[reply(WEIGHTS_CHILD), reply(WEIGHTS_CHILD)],
        outcomes={WEIGHTS_REFERENCE_ARTIFACT_SOURCE: ok(1.0), WEIGHTS_CHILD: ok(0.9)},
        sight=sight,
    )
    spec = lineage_spec(condition="weights_only", max_evaluations=2)
    assert lineage_run(pool, tmp_path, spec, "run-weights-registry") == 0
    assert all(set(candidate.quantiles) == set(QUANTILE_KEYS) for candidate in sight.candidates[0])
    verify_run(tmp_path, "run-weights-registry")


# --- Test 20: the CLI ---------------------------------------------------------------


def _cli_args(pool: tuple[Path, Path], tmp_path: Path, **overrides: str) -> list[str]:
    """One `phase1 lineage` invocation's arguments over the fixture pool."""
    base, cache_root = pool
    args = {
        "condition": "predictive",
        "dataset": "synthetic-dev",
        "pool-spec": "simple",
        "ledger-root": str(tmp_path / "ledgers"),
        "cache-root": str(cache_root),
        "data-root": str(base),
        "manifest-dir": str(base),
        "foundation-root": str(base / "foundation_cache"),
        "run-id": "cli-run",
        "max-evaluations": "3",
        "max-wall-seconds": "100000",
        "eval-timeout-seconds": "240",
        "eval-memory-gib": "2",
    }
    args.update(overrides)
    flat: list[str] = ["phase1", "lineage"]
    for flag, value in args.items():
        flat += [f"--{flag}", value]
    return flat


def test_cli_refuses_unknown_condition(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """An unknown condition is one FAIL line, exit 1, and no ledger written."""
    install(monkeypatch, outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0)})
    assert cli_main(_cli_args(pool, tmp_path, condition="nope")) == 1
    assert "FAIL" in capsys.readouterr().out
    assert not (tmp_path / "ledgers" / "cli-run" / "main.jsonl").exists()


def test_cli_llm_edit_needs_the_key(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """llm_edit without EVOLVETS_LLM_API_KEY fails before any ledger write."""
    install(monkeypatch, outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0)})
    monkeypatch.delenv("EVOLVETS_LLM_API_KEY", raising=False)
    assert cli_main(_cli_args(pool, tmp_path)) == 1
    assert "EVOLVETS_LLM_API_KEY" in capsys.readouterr().out
    assert not (tmp_path / "ledgers" / "cli-run" / "main.jsonl").exists()


def test_cli_random_param_needs_no_key(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """random_param runs to a designed end with the key absent, exit 0."""
    install(monkeypatch, outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0)}, key=False)
    monkeypatch.delenv("EVOLVETS_LLM_API_KEY", raising=False)
    args = _cli_args(pool, tmp_path, condition="random_param", **{"max-evaluations": "1"})
    assert cli_main(args) == 0
    assert "OK cli-run budget_exhausted" in capsys.readouterr().out
    verify_run(tmp_path, "cli-run")


def test_cli_config_blob_names_the_env_var_not_the_key(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The config blob carries api_key_env's name; the key's value reaches no record or blob."""
    monkeypatch.setenv("EVOLVETS_LLM_API_KEY", "super-secret-key-value")
    install(
        monkeypatch,
        responses=[
            reply(CHILD_A),
            reply(CHILD_B),
            reply(CHILD_C),
        ],  # the CLI default is 3 per round
        outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0), CHILD_A: ok(0.8)},
    )
    assert cli_main(_cli_args(pool, tmp_path, **{"max-evaluations": "2"})) == 0
    run_dir = tmp_path / "ledgers" / "cli-run"
    start = records_of(tmp_path, "cli-run")[0]
    assert start.kind == "run_start"
    config_blob = BlobStore(run_dir / "blobs").get(start.payload.config_blob).decode("utf-8")
    assert "EVOLVETS_LLM_API_KEY" in config_blob
    assert "super-secret-key-value" not in config_blob
    everything = (run_dir / "main.jsonl").read_text(encoding="utf-8")
    for path in sorted((run_dir / "blobs").rglob("*")):
        if path.is_file():
            everything += path.read_text(encoding="utf-8", errors="ignore")
    assert "super-secret-key-value" not in everything


def test_cli_refuses_a_wall_limit_that_cannot_reserve_the_baseline(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    """A wall limit below timeout + margin is refused before run_start."""
    install(monkeypatch, outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0)})
    args = _cli_args(pool, tmp_path, **{"max-wall-seconds": "100", "eval-timeout-seconds": "1800"})
    assert cli_main(args) == 1
    assert "cannot reserve" in capsys.readouterr().out
    assert not (tmp_path / "ledgers" / "cli-run" / "main.jsonl").exists()


def test_phase1_error_is_raised_for_api_callers(
    pool: tuple[Path, Path], tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """run_lineage itself raises Phase1Error on a bad spec, writing nothing."""
    install(monkeypatch, outcomes={REFERENCE_ARTIFACT_SOURCE: ok(1.0)})
    with pytest.raises(Phase1Error, match="unknown condition"):
        lineage_run(pool, tmp_path, lineage_spec(condition="nope"), "never")
    assert not (tmp_path / "ledgers" / "never").exists()
