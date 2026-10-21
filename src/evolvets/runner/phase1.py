"""The Phase 1 lineage engine and its conditions (TASK_018 PR A2, entry 19).

One lineage is propose -> predict -> allocate -> evaluate -> accept, one
generation at a time, over one dev pool under one budget accounting. The
proposer's edit, the self-model's quantiles (durable before any evaluation,
AGENTS.md 2.3), the allocator's choice and the subprocess evaluation land in
the ledger in exactly that order — the order ``validate_run`` checks — and
every evaluated proposal, failures included, gets exactly one hill-climb
acceptance decision. The runner is instrument code: it touches dev pools
only through ``load_dev_pool``, and only it holds a ledger path — the
evaluation worker never sees one.

Two histories run side by side, and the asymmetry is deliberate. The
*prompt* history (``observation_object``) skips generation-0 proposals,
because the starting artifact is no proposal's child; the *fit* history the
self-model and the allocators see holds every proposal whose evaluation was
``ok``, generation 0 included, gained against the parent's MASE at proposal
time — a model fitted on one round of data beats one fitted on none.

The budget rule is *reserve before acting*: no proposal or evaluation starts
unless its worst-case charge still fits under every ``run_start`` budget
limit, so summed charges never overshoot a limit and a ``budget_exhausted``
run still passes ``validate_run`` — the exit-0 and verify-clean contracts
hold together. A replay (``replay_of``) re-runs a source lineage from its
recorded responses — zero tokens, real evaluations — driven by the source's
own per-generation proposal, chosen and evaluation counts rather than by its
guards, which cannot bind at the same points at zero token cost, and reports
``matches_source_run`` over the aligned records.
"""

from __future__ import annotations

import dataclasses
import json
import os
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolvets.allocator.allocators import append_allocation, make_allocator
from evolvets.allocator.interfaces import Candidate
from evolvets.arena.dev import load_dev_pool
from evolvets.arena.pool import POOL_SPECS
from evolvets.arena.synthetic.holdout import HOLDOUT_DATASETS
from evolvets.cache.datasets import DEV_DATASETS
from evolvets.improver.artifacts import (
    REFERENCE_ARTIFACT_CONFIG,
    REFERENCE_ARTIFACT_SOURCE,
    artifact_family,
    const_change_stats,
    diff_lines,
)
from evolvets.improver.client import (
    HttpsLLMClient,
    LLMClientConfig,
    RecordedLLMClient,
)
from evolvets.improver.interfaces import LLMClient, ProposalFeatures
from evolvets.improver.prompt import IMPROVER_SYSTEM_PROMPT
from evolvets.improver.propose import propose_and_record, recorded_responses
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import (
    Acceptance,
    Allocation,
    Baseline,
    Evaluation,
    Proposal,
    RunEnd,
    RunStart,
)
from evolvets.ledger.schema import Budget, Record
from evolvets.ledger.store import LedgerReader
from evolvets.runner.phase1_eval import EvalOutcome, evaluate_source
from evolvets.runner.runs import Run, current_git_commit
from evolvets.selfmodel.interfaces import HistoryPoint
from evolvets.selfmodel.quantiles import ConformalRidgeSelfModel, append_prediction
from evolvets.updater.interfaces import ConfigValue

#: The worst-case input tokens reserved per proposal: an order of magnitude
#: above the largest prompt the harness builds, so no single call charges
#: more input than the reservation.
PROPOSAL_INPUT_RESERVE_TOKENS = 200_000

#: The wall reserved by one random_param proposal (no network, no tokens).
PROPOSAL_WALL_RESERVE_SECONDS = 10.0

#: Parent-side slack on top of the deadline when reserving one evaluation:
#: the child is killed at the deadline, so its charge is bounded by the
#: deadline plus this overhead.
EVAL_WALL_MARGIN_SECONDS = 60.0

#: Repair rounds per proposal, the value every reservation is sized against.
MAX_REPAIR_ROUNDS = 1

#: The allocators whose candidates carry the self-model's quantiles; every
#: prediction-blind rival's candidates carry ``{}`` (PR B adds weights_only).
_QUANTILE_VISIBLE = frozenset({"predictive", "shuffled", "weights_only"})

#: The acceptance rule's name in the ledger.
_RULE = "hill_climb_v1"

#: The budget sums the run_end summary reports.
_BUDGET_SUMS = ("llm_input_tokens", "llm_output_tokens", "wall_seconds", "evaluations")

#: The headroom a replay's own wall limit gets over the source's spend.
_REPLAY_WALL_HEADROOM_SECONDS = 60.0

_SEED_LIMIT = 2**63


class Phase1Error(RuntimeError):
    """Raised when a lineage cannot start: a bad spec, client or replay source."""


@dataclass(frozen=True)
class Condition:
    """One experimental condition: proposer, allocator, object space, prompts."""

    proposer_id: str  # "llm_edit" | "random_param"
    allocator_id: str  # one of ALLOCATOR_IDS
    interface: str  # "forecast-v2" | "weights-v1"
    reference_updater_id: str  # "reference" | "weights_reference"
    system_prompt: str  # IMPROVER_SYSTEM_PROMPT in PR A


def _forecast_condition(allocator_id: str) -> Condition:
    """One forecast-v2 LLM condition over ``allocator_id``: the treatment's shape."""
    return Condition(
        proposer_id="llm_edit",
        allocator_id=allocator_id,
        interface="forecast-v2",
        reference_updater_id="reference",
        system_prompt=IMPROVER_SYSTEM_PROMPT,
    )


#: The Phase 1 conditions of the forecast-v2 object space: the treatment
#: (``predictive``), its prediction-blind rivals, its ablation, and the
#: non-LLM track of the treatment. PR B adds ``weights_only``.
CONDITIONS: dict[str, Condition] = {
    allocator_id: _forecast_condition(allocator_id)
    for allocator_id in (
        "predictive",
        "evaluate_all",
        "fixed",
        "random",
        "bandit_ucb",
        "bo_gp",
        "shuffled",
    )
}
CONDITIONS["random_param"] = Condition(
    proposer_id="random_param",
    allocator_id="predictive",
    interface="forecast-v2",
    reference_updater_id="reference",
    system_prompt=IMPROVER_SYSTEM_PROMPT,
)


@dataclass(frozen=True)
class LineageSpec:
    """What one lineage runs, and under which limits."""

    condition: str
    dataset: str
    pool_spec: str
    foundation_models: tuple[str, ...] | None
    seed: int
    proposals_per_generation: int
    eval_slots: int
    max_evaluations: int
    max_generations: int
    budget_limits: Budget
    eval_timeout_seconds: float
    eval_memory_gib: float
    client_config: LLMClientConfig | None  # None iff random_param
    replay_of: str | None  # run id when replaying


def _reference_artifact(condition: Condition) -> tuple[str, dict[str, ConfigValue]]:
    """The starting artifact of the condition's object space: (source, config)."""
    del condition  # PR B switches this on the interface; forecast-v2 has one reference
    return REFERENCE_ARTIFACT_SOURCE, dict(REFERENCE_ARTIFACT_CONFIG)


def _require_int_at_least(spec: LineageSpec, name: str) -> None:
    """Check one of the spec's int knobs is an int >= 1 (bools excluded)."""
    value = getattr(spec, name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Phase1Error(f"{name} must be an int >= 1, got {value!r}")


def _validate_spec(spec: LineageSpec) -> Condition:
    """Refuse a spec the run cannot honour, before anything is loaded."""
    if spec.condition not in CONDITIONS:
        raise Phase1Error(
            f"unknown condition {spec.condition!r}; expected one of {sorted(CONDITIONS)}"
        )
    condition = CONDITIONS[spec.condition]
    if spec.dataset not in DEV_DATASETS:
        raise Phase1Error(f"unknown dataset {spec.dataset!r}; expected one of {list(DEV_DATASETS)}")
    if spec.dataset in HOLDOUT_DATASETS:
        raise Phase1Error(
            f"{spec.dataset} is a holdout dataset: holdouts score accepted edits only "
            "and are never selected on"
        )
    if spec.pool_spec not in POOL_SPECS:
        raise Phase1Error(
            f"unknown pool spec {spec.pool_spec!r}; expected one of {list(POOL_SPECS)}"
        )
    if isinstance(spec.seed, bool) or not isinstance(spec.seed, int):
        raise Phase1Error(f"seed must be an int in [0, {_SEED_LIMIT}), got {spec.seed!r}")
    if not 0 <= spec.seed < _SEED_LIMIT:
        raise Phase1Error(f"seed must be an int in [0, {_SEED_LIMIT}), got {spec.seed}")
    for name in (
        "proposals_per_generation",
        "eval_slots",
        "max_evaluations",
        "max_generations",
    ):
        _require_int_at_least(spec, name)
    if not isinstance(spec.budget_limits, Budget):
        raise Phase1Error("budget_limits must be a Budget instance")
    for name, value in (
        ("eval_timeout_seconds", spec.eval_timeout_seconds),
        ("eval_memory_gib", spec.eval_memory_gib),
        ("budget_limits.wall_seconds", spec.budget_limits.wall_seconds),
        ("budget_limits.llm_input_tokens", spec.budget_limits.llm_input_tokens),
        ("budget_limits.llm_output_tokens", spec.budget_limits.llm_output_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise Phase1Error(f"{name} must be a number > 0, got {value!r}")
    if spec.budget_limits.evaluations != spec.max_evaluations:
        raise Phase1Error(
            "budget_limits.evaluations must equal max_evaluations "
            f"({spec.max_evaluations}): the reservation rule stops at "
            "max_evaluations and validate_run stops at the limit"
        )
    if (spec.client_config is None) != (condition.proposer_id == "random_param"):
        raise Phase1Error(
            "client_config must be None exactly for the random_param proposer, "
            f"got {type(spec.client_config).__name__} for {condition.proposer_id!r}"
        )
    return condition


def _can_reserve_proposal(
    condition: Condition, spec: LineageSpec, limits: Budget, spent: Budget
) -> bool:
    """Whether one more proposal's worst-case charge fits under every limit."""
    remaining_wall = limits.wall_seconds - spent.wall_seconds
    if condition.proposer_id == "llm_edit":
        assert spec.client_config is not None  # _validate_spec guarantees it
        calls = 1 + MAX_REPAIR_ROUNDS
        return (
            limits.llm_output_tokens - spent.llm_output_tokens
            >= calls * spec.client_config.max_tokens
            and limits.llm_input_tokens - spent.llm_input_tokens >= PROPOSAL_INPUT_RESERVE_TOKENS
            and remaining_wall
            >= calls * spec.client_config.max_attempts * spec.client_config.timeout_seconds
        )
    return remaining_wall >= PROPOSAL_WALL_RESERVE_SECONDS


def _can_reserve_evaluation(spec: LineageSpec, limits: Budget, spent: Budget) -> bool:
    """Whether one more evaluation fits: a slot left and the deadline reservable."""
    return spent.evaluations < spec.max_evaluations and (
        limits.wall_seconds - spent.wall_seconds
        >= spec.eval_timeout_seconds + EVAL_WALL_MARGIN_SECONDS
    )


def _run_config(spec: LineageSpec) -> dict[str, object]:
    """The ``run_start`` config blob: the whole spec plus the client's fields.

    The client block carries the API key's environment-variable *name* only;
    the value never reaches a blob, record or log.
    """
    client: dict[str, object] | None = None
    if spec.client_config is not None:
        config = spec.client_config
        client = {
            "base_url": config.base_url,
            "model": config.model,
            "api_key_env": config.api_key_env,
            "timeout_seconds": config.timeout_seconds,
            "max_attempts": config.max_attempts,
            "backoff_base_seconds": config.backoff_base_seconds,
            "backoff_cap_seconds": config.backoff_cap_seconds,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
        }
    return {
        "experiment": "phase1",
        "condition": spec.condition,
        "dataset": spec.dataset,
        "pool_spec": spec.pool_spec,
        "foundation_models": (
            list(spec.foundation_models) if spec.foundation_models is not None else None
        ),
        "seed": spec.seed,
        "proposals_per_generation": spec.proposals_per_generation,
        "eval_slots": spec.eval_slots,
        "max_evaluations": spec.max_evaluations,
        "max_generations": spec.max_generations,
        "budget_limits": dataclasses.asdict(spec.budget_limits),
        "eval_timeout_seconds": spec.eval_timeout_seconds,
        "eval_memory_gib": spec.eval_memory_gib,
        "replay_of": spec.replay_of,
        "llm": client,
    }


def _artifact_text(blobs: BlobStore, ref: str) -> str:
    """The artifact source stored under ``ref`` (blobs hold the canonical JSON of it)."""
    text = json.loads(blobs.get(ref).decode("utf-8"))
    assert isinstance(text, str)  # the proposer stored the source as a JSON string
    return text


@dataclass(frozen=True)
class _SourceRun:
    """The source lineage of a replay: its records, status, plan and wall total."""

    records: tuple[Record, ...]
    wall_seconds_total: float
    status: str  # the source's run_end status; a finished replay mirrors it
    plan: tuple[tuple[int, int, int], ...]  # (proposals, chosen, evaluations) per generation


def _read_source_run(ledger_root: Path, spec: LineageSpec) -> _SourceRun:
    """Read the replay's source run and its per-generation plan, refusing mismatches.

    The source must be a finished lineage of the same condition, dataset,
    pool spec and seed; its plan is one ``(proposals, chosen, evaluations)``
    triple per allocation, in ledger order, where ``evaluations`` counts the
    evaluation records that followed that allocation — a round whose wall
    reservation failed after its allocation ran fewer evaluations than it
    chose, and the replay must not run what the source never did — and its
    wall total sizes the replay's own wall limit.
    """
    records = tuple(LedgerReader(ledger_root, spec.replay_of or "").records())
    starts = [record for record in records if record.kind == "run_start"]
    ends = [record for record in records if record.kind == "run_end"]
    if len(starts) != 1 or not isinstance(starts[0].payload, RunStart):
        raise Phase1Error(f"replay source {spec.replay_of!r} has no single run_start")
    if not ends or not isinstance(ends[-1].payload, RunEnd):
        raise Phase1Error(f"replay source {spec.replay_of!r} has no run_end")
    if ends[-1].payload.status not in ("completed", "budget_exhausted"):
        raise Phase1Error(
            f"replay source {spec.replay_of!r} ended {ends[-1].payload.status!r}; "
            "only finished lineages replay"
        )
    start = starts[0].payload
    if start.condition != f"phase1-{spec.condition}":
        raise Phase1Error(
            f"replay source {spec.replay_of!r} ran condition {start.condition!r}, "
            f"not phase1-{spec.condition!r}"
        )
    if start.seed != spec.seed:
        raise Phase1Error(
            f"replay source {spec.replay_of!r} ran seed {start.seed}, not {spec.seed}"
        )
    blobs = BlobStore(Path(ledger_root) / (spec.replay_of or "") / "blobs")
    config = json.loads(blobs.get(start.config_blob).decode("utf-8"))
    for key in ("dataset", "pool_spec"):
        source_value = config.get(key) if isinstance(config, dict) else None
        if source_value != getattr(spec, key):
            raise Phase1Error(
                f"replay source {spec.replay_of!r} ran {key} {source_value!r}, "
                f"not {getattr(spec, key)!r}"
            )
    plan: list[list[int]] = []  # [proposals, chosen, evaluations], finalized below
    pending = 0
    for record in records:
        if record.kind == "proposal":
            pending += 1
        elif record.kind == "allocation":
            payload = record.payload
            assert isinstance(payload, Allocation)
            plan.append([pending, len(payload.chosen), 0])
            pending = 0
        elif record.kind == "evaluation" and plan:
            plan[-1][2] += 1
    if pending:
        raise Phase1Error(
            f"replay source {spec.replay_of!r} ends with {pending} proposal(s) no "
            "allocation judged; cannot replay"
        )
    wall = ends[-1].payload.summary.get("wall_seconds_total", 0.0)
    if isinstance(wall, bool) or not isinstance(wall, (int, float)):
        wall = 0.0
    return _SourceRun(
        records=records,
        wall_seconds_total=float(wall),
        status=ends[-1].payload.status,
        plan=tuple((proposals, chosen, ran) for proposals, chosen, ran in plan),
    )


def _features(
    parent: str, child: str, generation: int, parent_gain: float, is_llm: bool
) -> ProposalFeatures:
    """One proposal's features: family, diff size, constant stats and context."""
    const_changes, max_rel = const_change_stats(parent, child)
    return ProposalFeatures(
        family=artifact_family(parent, child),
        changed_lines=len(diff_lines(parent, child)),
        const_changes=const_changes,
        max_rel_const_change=max_rel,
        generation=generation,
        parent_gain=parent_gain,
        is_llm=is_llm,
    )


def _compare_runs(source: Sequence[Record], replay: Sequence[Record]) -> tuple[bool, int]:
    """``(matches, timeout_status_mismatches)`` over the aligned source/replay records.

    Aligned and compared: every proposal's child-artifact digest, every
    allocation's chosen sequence — as the child-artifact digests it names,
    since record ids are fresh in every run — and every evaluation's status
    and metrics. A count mismatch already makes ``matches`` false; zip pairs
    what aligned, and an evaluation that differs on exactly a ``timeout``
    also counts toward the mismatch tally (timeouts are wall-dependent,
    reported as-is).
    """

    def of(records: Sequence[Record], kind: str) -> list[Record]:
        return [record for record in records if record.kind == kind]

    def children(records: Sequence[Record]) -> dict[str, str]:
        """One run's proposal record ids mapped to their child-artifact digests."""
        return {
            record.record_id: record.payload.child_artifact
            for record in records
            if record.kind == "proposal" and isinstance(record.payload, Proposal)
        }

    matches = True
    for kind in ("proposal", "allocation", "evaluation"):
        if len(of(source, kind)) != len(of(replay, kind)):
            matches = False
    for left, right in zip(of(source, "proposal"), of(replay, "proposal"), strict=False):
        left_payload, right_payload = left.payload, right.payload
        assert isinstance(left_payload, Proposal) and isinstance(right_payload, Proposal)
        if left_payload.child_artifact != right_payload.child_artifact:
            matches = False
    source_children, replay_children = children(source), children(replay)
    for left, right in zip(of(source, "allocation"), of(replay, "allocation"), strict=False):
        left_payload, right_payload = left.payload, right.payload
        assert isinstance(left_payload, Allocation) and isinstance(right_payload, Allocation)
        if [source_children.get(record_id) for record_id in left_payload.chosen] != [
            replay_children.get(record_id) for record_id in right_payload.chosen
        ]:
            matches = False
    timeout_mismatches = 0
    for left, right in zip(of(source, "evaluation"), of(replay, "evaluation"), strict=False):
        left_payload, right_payload = left.payload, right.payload
        assert isinstance(left_payload, Evaluation) and isinstance(right_payload, Evaluation)
        if left_payload.status != right_payload.status:
            matches = False
            if (left_payload.status == "timeout") != (right_payload.status == "timeout"):
                timeout_mismatches += 1
        elif left_payload.metrics != right_payload.metrics:
            matches = False
    return matches, timeout_mismatches


@dataclass
class _LineageState:
    """The mutable lineage state: the current artifact, the fit history, the records."""

    current_source: str
    current_record_id: str | None
    current_mase: float
    current_gain: float
    records: list[Record]
    history: list[HistoryPoint]


class _Engine:
    """One run's engine: the baseline, the generation loop, the summary."""

    def __init__(
        self,
        run: Run,
        spec: LineageSpec,
        condition: Condition,
        limits: Budget,
        paths: tuple[Path, Path, Path, Path],
        client: LLMClient | None,
        rng: random.Random,
        started: float,
    ) -> None:
        self._run = run
        self._spec = spec
        self._condition = condition
        self._limits = limits
        self._cache_root, self._data_root, self._manifest_dir, self._foundation_root = paths
        self._client = client
        self._rng = rng
        self._started = started
        self._reference_source, self._reference_config = _reference_artifact(condition)
        self.state = _LineageState(
            current_source=self._reference_source,
            current_record_id=None,
            current_mase=0.0,
            current_gain=0.0,
            records=[],
            history=[],
        )
        self._starting_mase: float | None = None
        #: the generations the loop finished; None while, or because, it stopped early
        self._finished_generations: int | None = None
        self._stop_reason: str | None = None

    # --- the generation-0 baseline ------------------------------------------------

    def baseline(self) -> bool:
        """Evaluate the starting artifact, record its baseline; False aborts the run.

        A lineage cannot start without a parent score, so a baseline that is
        not ``ok`` ends the run ``aborted`` — the ledger has no ``failed`` run
        status — with the End summary (all counts zero, null scores) plus the
        reason as ``error``.
        """
        outcome = self._evaluate(self._reference_source)
        record = self._run.append(
            "baseline",
            Baseline(
                updater_id=self._condition.reference_updater_id,
                hyperparams=dict(self._reference_config),
                pool="dev",
                status=outcome.status,
                metrics=self._metrics(outcome),
                per_task_blob=self._per_task_blob(outcome),
                error=outcome.error,
            ),
            budget=Budget(evaluations=1, wall_seconds=outcome.wall_seconds),
        )
        self.state.records.append(record)
        if outcome.status != "ok":
            self._run.end(
                "aborted",
                self._summary(error=f"baseline {outcome.status}: {outcome.error}"),
            )
            return False
        mase = self._metrics(outcome)["mase"]
        self._starting_mase = mase
        self.state.current_mase = mase
        return True

    # --- the generation loop --------------------------------------------------------

    def run_generations(self, source: _SourceRun | None) -> None:
        """Run the live generations, or the source run's generations on a replay."""
        if source is None:
            generation = 0
            while (
                self._run.spent.evaluations < self._spec.max_evaluations
                and generation < self._spec.max_generations
            ):
                if not self._generation(generation, self._spec.proposals_per_generation, None):
                    return
                generation += 1
            # Only the generation cap completes the run. A loop exit on
            # evaluations — exactly spent at the last slot or never entered
            # after the baseline ate the budget — is exhaustion, like any
            # mid-round stop.
            if self._run.spent.evaluations < self._spec.max_evaluations:
                self._finished_generations = generation
            return
        for generation, (proposals, chosen, evaluations) in enumerate(source.plan):
            if not self._generation(generation, proposals, chosen, evaluations):
                return
        self._finished_generations = len(source.plan)

    def _generation(
        self,
        generation: int,
        target: int,
        chosen_cap: int | None,
        evaluations_cap: int | None = None,
    ) -> bool:
        """One propose -> predict -> allocate -> evaluate round; False ends on budget.

        ``chosen_cap`` truncates the allocator's choice (the source round's
        chosen count on a replay); ``None`` truncates to the evaluations the
        budget still allows. ``evaluations_cap`` stops the round's evaluation
        loop at the evaluations the source round actually ran (``None`` on a
        live run): a chosen proposal the source never evaluated — its
        reservation failed after the allocation — is left unevaluated too.
        """
        parent_mase_at_proposal = self.state.current_mase
        made = self._propose(generation, target)
        if not made:
            self._stop_reason = f"generation {generation}: no proposal's reservation fit"
            return False
        quantiles = self._predict(made)
        chosen = self._allocate(made, quantiles, chosen_cap)
        return self._evaluate_and_accept(made, chosen, parent_mase_at_proposal, evaluations_cap)

    def _propose(self, generation: int, count: int) -> list[tuple[Record, ProposalFeatures]]:
        """Make up to ``count`` proposals, each reserved first; fewer when budget binds."""
        made: list[tuple[Record, ProposalFeatures]] = []
        parent_source = self.state.current_source
        is_llm = self._condition.proposer_id == "llm_edit"
        for _ in range(count):
            if not _can_reserve_proposal(
                self._condition, self._spec, self._limits, self._run.spent
            ):
                return made  # the round ends with the proposals made so far
            record = propose_and_record(
                self._run,
                proposer_id=self._condition.proposer_id,
                parent_source=self.state.current_source,
                parent_record_id=self.state.current_record_id,
                generation=generation,
                parent_gain=self.state.current_gain,
                history_records=self.state.records,
                rng=self._rng,
                client=self._client,
                client_config=self._spec.client_config,
                max_repair_rounds=MAX_REPAIR_ROUNDS,
            )
            self.state.records.append(record)
            payload = record.payload
            assert isinstance(payload, Proposal)
            child = _artifact_text(self._run.blobs, payload.child_artifact)
            made.append(
                (
                    record,
                    _features(parent_source, child, generation, self.state.current_gain, is_llm),
                )
            )
        return made

    def _predict(
        self, made: Sequence[tuple[Record, ProposalFeatures]]
    ) -> dict[str, dict[str, float]]:
        """Predict every proposal's gain quantiles, each durable before any evaluation."""
        model = ConformalRidgeSelfModel(self.state.history)
        identify = model.identify()
        quantiles: dict[str, dict[str, float]] = {}
        for record, features in made:
            started = time.monotonic()
            prediction = dict(model.predict(features))
            quantiles[record.record_id] = prediction
            self.state.records.append(
                append_prediction(
                    self._run,
                    predictor=identify,
                    proposal_record_id=record.record_id,
                    quantiles=prediction,
                    wall_seconds=time.monotonic() - started,
                )
            )
        return quantiles

    def _allocate(
        self,
        made: Sequence[tuple[Record, ProposalFeatures]],
        quantiles: Mapping[str, Mapping[str, float]],
        chosen_cap: int | None,
    ) -> tuple[str, ...]:
        """Allocate once, record the decision, return the truncated chosen ids."""
        visible = self._condition.allocator_id in _QUANTILE_VISIBLE
        candidates = [
            Candidate(
                record_id=record.record_id,
                features=features,
                quantiles=quantiles[record.record_id] if visible else {},
            )
            for record, features in made
        ]
        # Budget fields are floats; the counts are whole, so the int is exact
        evaluations_left = int(self._spec.max_evaluations - self._run.spent.evaluations)
        allocator = make_allocator(self._condition.allocator_id, k=self._spec.eval_slots)
        started = time.monotonic()
        decision = allocator.choose(candidates, self.state.history, evaluations_left, self._rng)
        identify = allocator.identify()
        wall = time.monotonic() - started
        cap = evaluations_left if chosen_cap is None else min(chosen_cap, evaluations_left)
        chosen = tuple(decision.chosen[:cap])
        self.state.records.append(
            append_allocation(
                self._run,
                allocator=identify,
                candidate_ids=[record.record_id for record, _ in made],
                chosen_ids=list(chosen),
                scores=dict(decision.scores),
                wall_seconds=wall,
            )
        )
        return chosen

    def _evaluate_and_accept(
        self,
        made: Sequence[tuple[Record, ProposalFeatures]],
        chosen: Sequence[str],
        parent_mase_at_proposal: float,
        evaluations_cap: int | None = None,
    ) -> bool:
        """Evaluate each chosen proposal in order, accept it if it climbs.

        On a replay ``evaluations_cap`` ends the loop after the source
        round's evaluations. False ends the run ``budget_exhausted`` after
        the bookkeeping of the evaluations already done.
        """
        by_id = {record.record_id: (record, features) for record, features in made}
        for index, record_id in enumerate(chosen):
            if evaluations_cap is not None and index >= evaluations_cap:
                break  # the source chose this proposal but never afforded its evaluation
            if not _can_reserve_evaluation(self._spec, self._limits, self._run.spent):
                self._stop_reason = (
                    f"evaluation of {record_id}: the wall budget cannot reserve "
                    f"{self._spec.eval_timeout_seconds:g}s + {EVAL_WALL_MARGIN_SECONDS:g}s"
                )
                return False
            proposal, features = by_id[record_id]
            proposal_payload = proposal.payload
            assert isinstance(proposal_payload, Proposal)
            child_source = _artifact_text(self._run.blobs, proposal_payload.child_artifact)
            outcome = self._evaluate(child_source)
            evaluation = self._run.append(
                "evaluation",
                Evaluation(
                    pool="dev",
                    status=outcome.status,
                    metrics=self._metrics(outcome),
                    per_task_blob=self._per_task_blob(outcome),
                    error=outcome.error,
                ),
                refers_to=[proposal.record_id],
                budget=Budget(evaluations=1, wall_seconds=outcome.wall_seconds),
            )
            self.state.records.append(evaluation)
            child_mase = self._metrics(outcome).get("mase") if outcome.status == "ok" else None
            parent_mase_at_evaluation = self.state.current_mase
            accepted = child_mase is not None and child_mase < parent_mase_at_evaluation
            self.state.records.append(
                self._run.append(
                    "acceptance",
                    Acceptance(
                        accepted=accepted,
                        rule=_RULE,
                        new_artifact=proposal_payload.child_artifact if accepted else None,
                    ),
                    refers_to=[proposal.record_id, evaluation.record_id],
                )
            )
            if child_mase is not None:
                # generation 0 included; failures and timeouts join no fit history
                self.state.history.append(
                    HistoryPoint(features=features, gain=parent_mase_at_proposal - child_mase)
                )
            if accepted:
                assert child_mase is not None
                self.state.current_source = child_source
                self.state.current_record_id = proposal.record_id
                self.state.current_mase = child_mase
                self.state.current_gain = max(
                    -1.0, min(1.0, parent_mase_at_evaluation - child_mase)
                )
        return True

    # --- evaluation plumbing, the end status and the summary -------------------------

    def _evaluate(self, source: str) -> EvalOutcome:
        """Evaluate one artifact source in the subprocess under the spec's limits."""
        return evaluate_source(
            self._request(source),
            timeout_seconds=self._spec.eval_timeout_seconds,
            memory_gib=self._spec.eval_memory_gib,
        )

    def _request(self, source: str) -> dict[str, object]:
        """One worker request: this source at the reference config under the interface."""
        foundation = (
            list(self._spec.foundation_models) if self._spec.foundation_models is not None else []
        )
        return {
            "source": source,
            "config": dict(self._reference_config),
            "seed": self._spec.seed,
            "interface": self._condition.interface,
            "dataset": self._spec.dataset,
            "pool_spec": self._spec.pool_spec,
            "foundation_models": foundation,
            "cache_root": str(self._cache_root),
            "data_root": str(self._data_root),
            "manifest_dir": str(self._manifest_dir),
            "foundation_root": str(self._foundation_root),
        }

    @staticmethod
    def _metrics(outcome: EvalOutcome) -> dict[str, float]:
        """The pooled metrics of an ``ok`` outcome, ``{}`` otherwise."""
        if outcome.status != "ok" or outcome.result_obj is None:
            return {}
        metrics = outcome.result_obj.get("metrics", {})
        return dict(metrics) if isinstance(metrics, dict) else {}

    def _per_task_blob(self, outcome: EvalOutcome) -> str:
        """The blob ref of the outcome's per-task detail: wall per series, the result."""
        return self._run.put_blob(
            {
                "series_wall_seconds": list(outcome.series_wall_seconds),
                "result": outcome.result_obj,
            }
        )

    def _summary(
        self, *, error: str | None = None, source: _SourceRun | None = None
    ) -> dict[str, object]:
        """The run_end summary: counts, scores, budget sums, wall, replay comparison."""
        kinds = [record.kind for record in self.state.records]
        spent = self._run.spent
        summary: dict[str, object] = {
            "condition": self._spec.condition,
            "generations": kinds.count("allocation"),
            "proposals": kinds.count("proposal"),
            "predictions": kinds.count("prediction"),
            "allocations": kinds.count("allocation"),
            "evaluations": kinds.count("evaluation"),
            "accepted": sum(
                1
                for record in self.state.records
                if isinstance(record.payload, Acceptance) and record.payload.accepted
            ),
            "starting_mase": self._starting_mase,
            "best_mase": self.state.current_mase if self._starting_mase is not None else None,
            "budget_spent": {field: getattr(spent, field) for field in _BUDGET_SUMS},
            "wall_seconds_total": time.monotonic() - self._started,
            "replay_of": self._spec.replay_of,
        }
        if error is not None:
            summary["error"] = error
        if source is not None:
            matches, timeout_mismatches = _compare_runs(source.records, self.state.records)
            summary["matches_source_run"] = matches
            summary["timeout_status_mismatches"] = timeout_mismatches
            if self._stop_reason is not None:
                summary["error"] = self._stop_reason
        return summary

    def finish(self, source: _SourceRun | None) -> int:
        """End the run and return the exit code; exhaustion is a designed end, so 0.

        A replay that finished the source's plan ends with the source's own
        status — a replay of an exhausted lineage reads ``budget_exhausted``
        like its source — and one that could not afford the plan ends
        ``budget_exhausted`` like any run stopped early.
        """
        if source is not None and self._finished_generations is not None:
            status = source.status
        else:
            status = "completed" if self._finished_generations is not None else "budget_exhausted"
        self._run.end(status, self._summary(source=source))
        return 0


def run_lineage(
    spec: LineageSpec,
    *,
    ledger_root: Path,
    cache_root: Path,
    data_root: Path,
    manifest_dir: Path,
    foundation_root: Path,
    run_id: str,
) -> int:
    """Run one lineage inside a recorded run and return the process exit code.

    0 for ``completed`` and ``budget_exhausted`` (exhaustion is the designed
    end of a lineage), 2 otherwise; every refusal before ``run_start`` — a
    bad spec, a missing key, an unloadable pool, a config that cannot
    reserve the baseline, an unusable replay source, a dirty tree, a reused
    run id — raises :class:`Phase1Error`, :class:`~evolvets.arena.errors.ArenaError`
    or :class:`~evolvets.runner.runs.RunError` with nothing written, and any
    crash after it ends the run ``aborted`` through the ``Run`` context.
    """
    started = time.monotonic()
    condition = _validate_spec(spec)
    source: _SourceRun | None = None
    client: LLMClient | None = None
    if spec.replay_of is not None:
        source = _read_source_run(Path(ledger_root), spec)
        if condition.proposer_id == "llm_edit":
            blobs = BlobStore(Path(ledger_root) / spec.replay_of / "blobs")
            client = RecordedLLMClient(recorded_responses(blobs, source.records))
    elif condition.proposer_id == "llm_edit":
        assert spec.client_config is not None
        if os.environ.get(spec.client_config.api_key_env) is None:
            raise Phase1Error(
                f"the environment variable {spec.client_config.api_key_env} is not set; "
                f"the {spec.condition} condition cannot run without it"
            )
        client = HttpsLLMClient(spec.client_config)
    # the pool loads before the run starts: no run starts that cannot run
    load_dev_pool(
        spec.dataset,
        Path(cache_root),
        Path(data_root),
        Path(manifest_dir),
        pool_spec=spec.pool_spec,
        foundation_root=Path(foundation_root) if spec.pool_spec == "simple+foundation" else None,
        foundation_models=(
            spec.foundation_models if spec.pool_spec == "simple+foundation" else None
        ),
    )
    limits = spec.budget_limits
    if source is not None:
        limits = dataclasses.replace(
            spec.budget_limits,
            wall_seconds=(
                spec.budget_limits.wall_seconds
                + source.wall_seconds_total
                + _REPLAY_WALL_HEADROOM_SECONDS
            ),
        )
    if not _can_reserve_evaluation(spec, limits, Budget()):
        raise Phase1Error(
            "the budget cannot reserve the generation-0 baseline: max wall "
            f"{spec.budget_limits.wall_seconds:g}s is below eval_timeout_seconds "
            f"{spec.eval_timeout_seconds:g}s + {EVAL_WALL_MARGIN_SECONDS:g}s"
        )
    run = Run.start(
        Path(ledger_root),
        run_id,
        condition=f"phase1-{spec.condition}",
        seed=spec.seed,
        config=_run_config(spec),
        budget_limits=limits,
        git_commit=current_git_commit(Path.cwd()),
        requires_predictions=True,
    )
    try:
        with run:
            engine = _Engine(
                run,
                spec,
                condition,
                limits,
                (Path(cache_root), Path(data_root), Path(manifest_dir), Path(foundation_root)),
                client,
                random.Random(spec.seed),
                started,
            )
            if not engine.baseline():
                return 2
            engine.run_generations(source)
            return engine.finish(source)
    except Exception:
        # The Run context has already ended the run aborted with the
        # exception's name and message in its summary: the crash is in the
        # ledger, and the caller learns it through run_end, not a traceback.
        return 2
