"""The allocator slot and its rivals (TASK_017 PR C, roadmap entry 18).

The allocator is where the self-model's predictions turn into compute:
``predictive`` ranks candidates by a chosen gain quantile, and every rival —
evaluate-everything, fixed, random and the shuffled ablation here, the bandit
and Bayesian optimisation in PR C2 — answers the same ``choose`` call so (f)
can run all conditions at equal budget. Allocators select; they never charge
tokens, and the runner charges every condition the same budget for the same
candidates, which is what "equal budget" means here (AGENTS.md 2.5).

An allocator's whole world is its arguments: candidates (features and, where
the condition allows it, predicted quantiles), the history of evaluated
proposals, the evaluations the budget still allows and a seeded rng. Nothing
here imports, reads or could reach a pool, a dataset or the holdout — that is
what keeps selection off the holdout. Only ``random`` and ``shuffled`` touch
the rng; the rest are deterministic functions of their arguments, so a
condition's decisions replay exactly from its records.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import Protocol

from evolvets.allocator.interfaces import Allocator, AllocatorError, Candidate, Decision
from evolvets.ledger.payloads import QUANTILE_KEYS, Allocation, Payload
from evolvets.ledger.schema import Budget, Record
from evolvets.selfmodel.interfaces import HistoryPoint

#: Every allocator an improver run may use, in config order. PR C2
#: (feat/allocator-rivals) adds the last two to make_allocator's dispatch.
ALLOCATOR_IDS: tuple[str, ...] = (
    "evaluate_all",
    "fixed",
    "random",
    "predictive",
    "shuffled",
    "bandit_ucb",
    "bo_gp",
)

#: The two edit families, the bandit's arms, in tie-break order (PR C2).
FAMILY_ORDER: tuple[str, ...] = ("params", "logic")

#: The score of a bandit arm no history point has ever been drawn from (PR C2).
ARM_UNSEEN_SCORE = 1e9

#: The allocators PR C1 implements; make_allocator dispatches over exactly these.
_CORE_IDS = ALLOCATOR_IDS[:5]

#: Both the core allocators and PR C2's rivals are at version 1 of their behaviour.
_ALLOCATOR_VERSION = "1"


class _RecordingRun(Protocol):
    """What append_allocation needs of the run recording it: one append.

    The runner's ``Run`` satisfies this structurally; the allocator names the
    shape rather than the class, as the improver and the self-model do, so no
    runner import ever crosses into this package.
    """

    def append(
        self,
        kind: str,
        payload: Payload,
        *,
        refers_to: Sequence[str] = (),
        budget: Budget | None = None,
    ) -> Record: ...


def _require_positive_int(name: str, value: int) -> int:
    """Check an int of at least 1 (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value!r}")
    return value


def _require_positive_float(name: str, value: float) -> float:
    """Check a finite number strictly above zero (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return float(value)


def _choice_size(candidates: Sequence[Candidate], evaluations_left: int, slots: int | None) -> int:
    """Validate choose()'s shared preconditions and return how many to pick.

    ``slots`` is the allocator's own per-generation evaluation budget; ``None``
    means the allocator ignores ``k`` (``evaluate_all``), whose only caps are
    the evaluations left and the candidate count.
    """
    if not candidates:
        raise AllocatorError("choose needs at least one candidate")
    if evaluations_left < 0:
        raise AllocatorError(f"evaluations_left must be >= 0, got {evaluations_left}")
    cap = len(candidates) if slots is None else min(slots, len(candidates))
    return min(cap, evaluations_left)


def _membership_scores(candidates: Sequence[Candidate], chosen: Sequence[str]) -> dict[str, float]:
    """1.0 for every chosen candidate, 0.0 for the rest — no ranking signal."""
    chosen_set = set(chosen)
    return {
        candidate.record_id: 1.0 if candidate.record_id in chosen_set else 0.0
        for candidate in candidates
    }


def _quantile_values(
    candidates: Sequence[Candidate], quantile: str, allocator_id: str
) -> list[float]:
    """One candidate's predicted quantile per candidate, in candidate order.

    Every candidate must carry all ``QUANTILE_KEYS``: a condition running
    without predictions must fail loudly here, not silently rank noise.
    """
    values: list[float] = []
    for candidate in candidates:
        if not set(QUANTILE_KEYS) <= set(candidate.quantiles):
            raise AllocatorError(
                f"candidate {candidate.record_id} does not carry the quantiles "
                f"{list(QUANTILE_KEYS)}; the {allocator_id!r} allocator needs "
                "predictions"
            )
        values.append(float(candidate.quantiles[quantile]))
    return values


def _top_order(values: Sequence[float], size: int) -> list[int]:
    """The ``size`` best indexes by ``(-value, index)``: ties keep candidate order."""
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    return order[:size]


class _EvaluateAllAllocator:
    """The Dream-RSI-style baseline: evaluate everything the budget still allows."""

    def identify(self) -> dict[str, str]:
        """This allocator's id and version."""
        return {"id": "evaluate_all", "version": _ALLOCATOR_VERSION}

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision:
        """The first ``evaluations_left`` candidates in candidate order — no selection."""
        size = _choice_size(candidates, evaluations_left, None)
        chosen = tuple(candidate.record_id for candidate in candidates[:size])
        return Decision(chosen=chosen, scores=_membership_scores(candidates, chosen))


class _FixedAllocator:
    """Always the first ``k`` candidates: the no-intelligence baseline."""

    def __init__(self, k: int) -> None:
        self._k = k

    def identify(self) -> dict[str, str]:
        """This allocator's id and version."""
        return {"id": "fixed", "version": _ALLOCATOR_VERSION}

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision:
        """The first candidates in order, ignoring features, quantiles and history."""
        size = _choice_size(candidates, evaluations_left, self._k)
        chosen = tuple(candidate.record_id for candidate in candidates[:size])
        return Decision(chosen=chosen, scores=_membership_scores(candidates, chosen))


class _RandomAllocator:
    """A seeded uniform choice among the candidates: the luck baseline."""

    def __init__(self, k: int) -> None:
        self._k = k

    def identify(self) -> dict[str, str]:
        """This allocator's id and version."""
        return {"id": "random", "version": _ALLOCATOR_VERSION}

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision:
        """A uniformly sampled candidate set, listed in candidate order."""
        size = _choice_size(candidates, evaluations_left, self._k)
        indexes = sorted(rng.sample(range(len(candidates)), size))
        chosen = tuple(candidates[index].record_id for index in indexes)
        return Decision(chosen=chosen, scores=_membership_scores(candidates, chosen))


class _PredictiveAllocator:
    """The self-model allocator: rank by one predicted gain quantile."""

    def __init__(self, k: int, quantile: str) -> None:
        self._k = k
        self._quantile = quantile

    def identify(self) -> dict[str, str]:
        """This allocator's id and version."""
        return {"id": "predictive", "version": _ALLOCATOR_VERSION}

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision:
        """The top candidates by the chosen quantile, ties keeping candidate order."""
        size = _choice_size(candidates, evaluations_left, self._k)
        values = _quantile_values(candidates, self._quantile, "predictive")
        order = _top_order(values, size)
        return Decision(
            chosen=tuple(candidates[index].record_id for index in order),
            scores={
                candidate.record_id: value
                for candidate, value in zip(candidates, values, strict=True)
            },
        )


class _ShuffledAllocator:
    """The ablation: predictive's rule over prediction-shaped noise.

    The quantile values are permuted across candidates before the ranking, and
    the permuted values are what is scored and recorded. Same allocator, same
    rule, destroyed information: the Phase 1 report's check that prediction
    *content* matters, not prediction-shaped noise.
    """

    def __init__(self, k: int, quantile: str) -> None:
        self._k = k
        self._quantile = quantile

    def identify(self) -> dict[str, str]:
        """This allocator's id and version."""
        return {"id": "shuffled", "version": _ALLOCATOR_VERSION}

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision:
        """The top candidates by the permuted values, ties keeping candidate order."""
        size = _choice_size(candidates, evaluations_left, self._k)
        values = _quantile_values(candidates, self._quantile, "shuffled")
        rng.shuffle(values)
        order = _top_order(values, size)
        return Decision(
            chosen=tuple(candidates[index].record_id for index in order),
            scores={
                candidate.record_id: value
                for candidate, value in zip(candidates, values, strict=True)
            },
        )


def make_allocator(
    allocator_id: str,
    *,
    k: int = 1,  # per-generation evaluation slots, >= 1
    quantile: str = "q50",  # the predictive allocators' score key
    bandit_c: float = 1.0,  # UCB exploration rate, > 0
    gp_gamma: float = 1.0,  # RBF scale, > 0
    gp_noise: float = 0.01,  # observation noise variance, > 0
    gp_beta: float = 2.0,  # GP-UCB exploration weight, > 0
    gp_prior_std: float = 1.0,  # prior std with no history, > 0
    history_cap: int = 100,  # rivals fit at most this many latest points, >= 1
) -> Allocator:
    """Build the allocator named ``allocator_id`` with every parameter validated.

    The whole parameter set is checked here — the rivals' included — so PR C2
    adds dispatch branches only. ``bandit_ucb`` and ``bo_gp`` are the rivals
    and arrive in PR C2 (``feat/allocator-rivals``); this PR implements
    ``evaluate_all``, ``fixed``, ``random``, ``predictive`` and ``shuffled``.
    """
    if allocator_id not in ALLOCATOR_IDS:
        raise AllocatorError(f"unknown allocator {allocator_id!r}; expected one of {ALLOCATOR_IDS}")
    slots = _require_positive_int("k", k)
    if quantile not in QUANTILE_KEYS:
        raise ValueError(f"quantile must be one of {list(QUANTILE_KEYS)}, got {quantile!r}")
    _require_positive_float("bandit_c", bandit_c)
    _require_positive_float("gp_gamma", gp_gamma)
    _require_positive_float("gp_noise", gp_noise)
    _require_positive_float("gp_beta", gp_beta)
    _require_positive_float("gp_prior_std", gp_prior_std)
    _require_positive_int("history_cap", history_cap)
    if allocator_id == "evaluate_all":
        return _EvaluateAllAllocator()
    if allocator_id == "fixed":
        return _FixedAllocator(slots)
    if allocator_id == "random":
        return _RandomAllocator(slots)
    if allocator_id == "predictive":
        return _PredictiveAllocator(slots, quantile)
    if allocator_id == "shuffled":
        return _ShuffledAllocator(slots, quantile)
    raise AllocatorError(
        f"{allocator_id!r} arrives in PR C2 (feat/allocator-rivals); PR C1 implements {_CORE_IDS}"
    )


def append_allocation(
    run: _RecordingRun,
    *,
    allocator: Mapping[str, str],
    candidate_ids: Sequence[str],
    chosen_ids: Sequence[str],
    scores: Mapping[str, float],
    wall_seconds: float,
) -> Record:
    """Append one ``allocation`` record and return it.

    The payload checks its own shape (an ``{id, version}`` allocator, UUID
    candidate ids, no duplicates, ``chosen`` a subset of the candidates, score
    keys among the candidates); ``refers_to`` is the candidate ids in
    candidate order, so the record links to every proposal it judged. The
    record charges wall time only — allocators select, they never charge
    tokens, and the runner charges every condition the same budget for the
    same candidates (AGENTS.md 2.5).
    """
    payload = Allocation(
        allocator=dict(allocator),
        candidates=list(candidate_ids),
        chosen=list(chosen_ids),
        scores=dict(scores),
    )
    return run.append(
        "allocation",
        payload,
        refers_to=list(candidate_ids),
        budget=Budget(wall_seconds=wall_seconds),
    )
