"""The allocator slot and every rival of (f)'s conditions (TASK_017 PR C).

Seven implementations share one ``choose`` contract: the self-model
allocator (``predictive``) and its information-destroying ablation
(``shuffled`` — same rule, permuted scores, the Phase 1 report's check that
prediction *content* matters), the blind baselines (``evaluate_all``, the
Dream-RSI-style no-selection baseline, ``fixed``, ``random``) and the
standard rivals (``bandit_ucb`` over the two edit families, ``bo_gp`` as
GP-UCB over the eight proposal features). No allocator imports, reads or
could reach any pool, dataset or holdout: candidates and history are all
any of them ever sees, and that is what keeps selection off the holdout
(AGENTS.md 2.1).

The bandit and the GP are hand-rolled in the ``experts/ridge.py`` style —
Gaussian elimination with partial pivoting, explicit left-to-right loops —
because the equal-budget claim would not survive a linear-algebra
dependency, so there is none (AGENTS.md 2.5). Allocators select; they never
charge tokens: ``append_allocation`` charges wall time only, and (f) charges
every condition the same budget for the same candidates, which is what
"equal budget" means here. Only ``random`` and ``shuffled`` touch the
injected ``random.Random``; the rest are deterministic functions of their
arguments, and the caller seeds the rng from the run config (AGENTS.md 2.6).
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from typing import ClassVar, Protocol

from evolvets.allocator.interfaces import Allocator, AllocatorError, Candidate, Decision
from evolvets.improver.interfaces import feature_vector
from evolvets.ledger.payloads import QUANTILE_KEYS, Allocation, Payload
from evolvets.ledger.schema import Budget, Record
from evolvets.selfmodel.interfaces import HistoryPoint

#: The allocators a condition may run, in config order.
ALLOCATOR_IDS: tuple[str, ...] = (
    "evaluate_all",
    "fixed",
    "random",
    "predictive",
    "shuffled",
    "bandit_ucb",
    "bo_gp",
)

#: The bandit arms, also the tie-break order of equal arm scores.
FAMILY_ORDER: tuple[str, ...] = ("params", "logic")

#: The score of an arm no history point has ever come from: unexplored arms go first.
ARM_UNSEEN_SCORE = 1e9

#: Every allocator is at version 1 of its own behaviour.
_ALLOCATOR_VERSION = "1"


def _require_slot_count(name: str, value: int) -> int:
    """Check an int of at least 1 (bools excluded) and return it."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be an int >= 1, got {value!r}")
    return value


def _require_positive(name: str, value: float) -> float:
    """Check a finite number that is strictly positive (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number > 0, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return float(value)


def _solve(matrix_in: list[list[float]], rhs_in: list[float]) -> list[float]:
    """Solve a square system by Gaussian elimination with partial pivoting.

    The pivot is the largest absolute value in the column at or below the
    diagonal, ties going to the lowest row index — the ``experts/ridge.py``
    pattern. ``A = K + gp_noise * I`` is positive definite, so a zero pivot
    cannot occur; the guard stays so the failure is named, not silent.
    """
    a = [row[:] for row in matrix_in]
    b = rhs_in[:]
    n = len(b)
    for col in range(n):
        pivot_row = col
        best = abs(a[col][col])
        for row in range(col + 1, n):
            magnitude = abs(a[row][col])
            if magnitude > best:
                best = magnitude
                pivot_row = row
        if best == 0.0:
            raise AllocatorError("the GP system is singular")
        if pivot_row != col:
            a[col], a[pivot_row] = a[pivot_row], a[col]
            b[col], b[pivot_row] = b[pivot_row], b[col]
        for row in range(col + 1, n):
            factor = a[row][col] / a[col][col]
            for c in range(col, n):
                a[row][c] = a[row][c] - factor * a[col][c]
            b[row] = b[row] - factor * b[col]
    w = [0.0] * n
    for i in range(n - 1, -1, -1):
        total = b[i]
        for j in range(i + 1, n):
            total = total - a[i][j] * w[j]
        w[i] = total / a[i][i]
    return w


def _squared_distance(x: Sequence[float], y: Sequence[float]) -> float:
    """The squared Euclidean distance, one explicit left-to-right accumulation."""
    total = 0.0
    for a, b in zip(x, y, strict=True):
        diff = a - b
        total = total + diff * diff
    return total


def _arm_of(family: str) -> str:
    """The bandit arm of a family; a family outside ``FAMILY_ORDER`` cannot be scored."""
    if family not in FAMILY_ORDER:
        raise AllocatorError(f"family {family!r} is not one of {list(FAMILY_ORDER)}")
    return family


def _one_zero_scores(candidates: Sequence[Candidate], chosen: tuple[str, ...]) -> dict[str, float]:
    """1.0 for every chosen id and 0.0 for the rest: the blind allocators' score shape."""
    picked = set(chosen)
    return {
        candidate.record_id: (1.0 if candidate.record_id in picked else 0.0)
        for candidate in candidates
    }


def _quantile_scores(candidates: Sequence[Candidate], quantile: str) -> list[float]:
    """The score list of the predictive allocators, in candidate order.

    Every candidate must carry all ``QUANTILE_KEYS``: a condition without
    predictions must not silently run on partial ones.
    """
    scores: list[float] = []
    for candidate in candidates:
        if not set(QUANTILE_KEYS) <= set(candidate.quantiles):
            raise AllocatorError(
                f"candidate {candidate.record_id} must carry all of {list(QUANTILE_KEYS)} "
                f"for the predictive allocators, got {sorted(candidate.quantiles)}"
            )
        scores.append(candidate.quantiles[quantile])
    return scores


def _top_indexes(scores: Sequence[float], size: int) -> tuple[int, ...]:
    """The indexes of the ``size`` best scores, ties keeping candidate order."""
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return tuple(order[:size])


class _BaseAllocator:
    """identify() and the choose() guard shared by every allocator."""

    _id: ClassVar[str]

    def identify(self) -> dict[str, str]:
        """This allocator's id and version."""
        return {"id": self._id, "version": _ALLOCATOR_VERSION}

    def _check(self, candidates: Sequence[Candidate], evaluations_left: int) -> None:
        """The shared guard: at least one candidate, and a non-negative budget."""
        if not candidates:
            raise AllocatorError("choose needs at least one candidate")
        if evaluations_left < 0:
            raise AllocatorError(f"evaluations_left must be >= 0, got {evaluations_left!r}")


class _SlottedAllocator(_BaseAllocator):
    """An allocator that respects k per-generation evaluation slots."""

    def __init__(self, k: int) -> None:
        self._k = k

    def _slots(self, evaluations_left: int, n_candidates: int) -> int:
        """min(k, evaluations_left, candidates): the cap every slotted allocator respects."""
        return min(self._k, evaluations_left, n_candidates)


class _EvaluateAll(_BaseAllocator):
    """The Dream-RSI-style baseline: the first evaluations_left candidates, no selection."""

    _id = "evaluate_all"

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision:
        """The first min(evaluations_left, candidates) ids in candidate order."""
        self._check(candidates, evaluations_left)
        del history, rng  # no selection and no randomness: nothing else is ever looked at
        size = min(evaluations_left, len(candidates))
        chosen = tuple(candidate.record_id for candidate in candidates[:size])
        return Decision(chosen=chosen, scores=_one_zero_scores(candidates, chosen))


class _Fixed(_SlottedAllocator):
    """The fixed allocator: the first k candidates, whatever their features say."""

    _id = "fixed"

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision:
        """The first min(k, evaluations_left, candidates) ids in candidate order."""
        self._check(candidates, evaluations_left)
        del history, rng  # features, quantiles and history are ignored entirely
        size = self._slots(evaluations_left, len(candidates))
        chosen = tuple(candidate.record_id for candidate in candidates[:size])
        return Decision(chosen=chosen, scores=_one_zero_scores(candidates, chosen))


class _Random(_SlottedAllocator):
    """The random allocator: a uniformly picked subset, listed in candidate order."""

    _id = "random"

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision:
        """rng.sample picks the set; the ids are listed in candidate order."""
        self._check(candidates, evaluations_left)
        del history
        size = self._slots(evaluations_left, len(candidates))
        picked = set(rng.sample(range(len(candidates)), size))
        chosen = tuple(
            candidate.record_id for index, candidate in enumerate(candidates) if index in picked
        )
        return Decision(chosen=chosen, scores=_one_zero_scores(candidates, chosen))


class _Predictive(_SlottedAllocator):
    """The self-model allocator: rank by the predicted gain quantile."""

    _id = "predictive"

    def __init__(self, k: int, quantile: str) -> None:
        super().__init__(k)
        self._quantile = quantile

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision:
        """The top candidates by their quantile, ties keeping candidate order."""
        self._check(candidates, evaluations_left)
        del history, rng
        values = _quantile_scores(candidates, self._quantile)
        size = self._slots(evaluations_left, len(candidates))
        chosen = tuple(candidates[i].record_id for i in _top_indexes(values, size))
        scores = {
            candidate.record_id: value for candidate, value in zip(candidates, values, strict=True)
        }
        return Decision(chosen=chosen, scores=scores)


class _Shuffled(_SlottedAllocator):
    """The ablation: the predictive rule over scores permuted across candidates."""

    _id = "shuffled"

    def __init__(self, k: int, quantile: str) -> None:
        super().__init__(k)
        self._quantile = quantile

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision:
        """The predictive rule over the quantile list after rng.shuffle permutes it."""
        self._check(candidates, evaluations_left)
        del history
        values = _quantile_scores(candidates, self._quantile)
        rng.shuffle(values)
        size = self._slots(evaluations_left, len(candidates))
        chosen = tuple(candidates[i].record_id for i in _top_indexes(values, size))
        scores = {
            candidate.record_id: value for candidate, value in zip(candidates, values, strict=True)
        }
        return Decision(chosen=chosen, scores=scores)


class _BanditUcb(_SlottedAllocator):
    """A UCB bandit whose two arms are the edit families of FAMILY_ORDER."""

    _id = "bandit_ucb"

    def __init__(self, k: int, bandit_c: float, history_cap: int) -> None:
        super().__init__(k)
        self._bandit_c = bandit_c
        self._history_cap = history_cap

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision:
        """Candidates arm by arm, best arm score first, within an arm in candidate order."""
        self._check(candidates, evaluations_left)
        del rng
        counts = {arm: 0 for arm in FAMILY_ORDER}
        totals = {arm: 0.0 for arm in FAMILY_ORDER}
        for point in list(history)[-self._history_cap :]:
            arm = _arm_of(point.features.family)
            counts[arm] = counts[arm] + 1
            totals[arm] = totals[arm] + point.gain
        n_total = sum(counts.values())
        arm_scores: dict[str, float] = {}
        for arm in FAMILY_ORDER:
            if counts[arm] == 0:
                arm_scores[arm] = ARM_UNSEEN_SCORE
            else:
                mean = totals[arm] / counts[arm]
                arm_scores[arm] = mean + self._bandit_c * math.sqrt(
                    2 * math.log(n_total + 1) / counts[arm]
                )
        size = self._slots(evaluations_left, len(candidates))
        arm_order = sorted(
            FAMILY_ORDER, key=lambda arm: (-arm_scores[arm], FAMILY_ORDER.index(arm))
        )
        chosen: list[str] = []
        for arm in arm_order:
            if len(chosen) >= size:
                break
            for candidate in candidates:
                if len(chosen) >= size:
                    break
                if candidate.features.family == arm:
                    chosen.append(candidate.record_id)
        scores = {
            candidate.record_id: arm_scores[_arm_of(candidate.features.family)]
            for candidate in candidates
        }
        return Decision(chosen=tuple(chosen), scores=scores)


class _BoGp(_SlottedAllocator):
    """GP-UCB over the eight proposal features, with a fixed exploration weight."""

    _id = "bo_gp"

    def __init__(
        self,
        k: int,
        gp_gamma: float,
        gp_noise: float,
        gp_beta: float,
        gp_prior_std: float,
        history_cap: int,
    ) -> None:
        super().__init__(k)
        self._gp_gamma = gp_gamma
        self._gp_noise = gp_noise
        self._gp_beta = gp_beta
        self._gp_prior_std = gp_prior_std
        self._history_cap = history_cap

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision:
        """The top candidates by mean + gp_beta * std of the GP posterior at their features."""
        self._check(candidates, evaluations_left)
        del rng
        points = list(history)[-self._history_cap :]
        scores: dict[str, float] = {}
        if not points:
            # no history: mean = 0.0 and std = gp_prior_std for every candidate
            scores = {
                candidate.record_id: self._gp_beta * self._gp_prior_std for candidate in candidates
            }
        else:
            xs = [feature_vector(point.features) for point in points]
            ys = [point.gain for point in points]
            n = len(xs)
            a = [
                [math.exp(-self._gp_gamma * _squared_distance(xs[i], xs[j])) for j in range(n)]
                for i in range(n)
            ]
            for i in range(n):
                a[i][i] = a[i][i] + self._gp_noise
            alpha = _solve(a, ys)
            for candidate in candidates:
                x = feature_vector(candidate.features)
                k_star = [math.exp(-self._gp_gamma * _squared_distance(x, xi)) for xi in xs]
                v = _solve(a, k_star)
                mean = 0.0
                k_star_v = 0.0
                for i in range(n):
                    mean = mean + k_star[i] * alpha[i]
                    k_star_v = k_star_v + k_star[i] * v[i]
                variance = 1.0 - k_star_v
                # the floor keeps the square root real near duplicate points
                std = math.sqrt(variance) if variance > 0.0 else 0.0
                scores[candidate.record_id] = mean + self._gp_beta * std
        size = self._slots(evaluations_left, len(candidates))
        order = sorted(
            range(len(candidates)),
            key=lambda i: (-scores[candidates[i].record_id], i),
        )
        chosen = tuple(candidates[i].record_id for i in order[:size])
        return Decision(chosen=chosen, scores=scores)


def make_allocator(
    allocator_id: str,
    *,
    k: int = 1,
    quantile: str = "q50",
    bandit_c: float = 1.0,
    gp_gamma: float = 1.0,
    gp_noise: float = 0.01,
    gp_beta: float = 2.0,
    gp_prior_std: float = 1.0,
    history_cap: int = 100,
) -> Allocator:
    """Build the allocator ``allocator_id`` with its knobs checked.

    ``k`` is the per-generation evaluation slot count; ``quantile`` the
    predictive allocators' score key; ``bandit_c`` the UCB exploration rate;
    ``gp_gamma``/``gp_noise``/``gp_beta``/``gp_prior_std`` the GP's RBF
    scale, observation noise variance, UCB exploration weight and no-history
    prior std; ``history_cap`` the number of latest history points the rivals
    fit on. Every argument is validated whatever the id, so a bad knob in a
    run config fails at construction, never mid-lineage.
    """
    slots = _require_slot_count("k", k)
    if quantile not in QUANTILE_KEYS:
        raise ValueError(f"quantile must be one of {list(QUANTILE_KEYS)}, got {quantile!r}")
    exploration = _require_positive("bandit_c", bandit_c)
    gamma = _require_positive("gp_gamma", gp_gamma)
    noise = _require_positive("gp_noise", gp_noise)
    beta = _require_positive("gp_beta", gp_beta)
    prior_std = _require_positive("gp_prior_std", gp_prior_std)
    cap = _require_slot_count("history_cap", history_cap)
    if allocator_id == "evaluate_all":
        return _EvaluateAll()
    if allocator_id == "fixed":
        return _Fixed(slots)
    if allocator_id == "random":
        return _Random(slots)
    if allocator_id == "predictive":
        return _Predictive(slots, quantile)
    if allocator_id == "shuffled":
        return _Shuffled(slots, quantile)
    if allocator_id == "bandit_ucb":
        return _BanditUcb(slots, exploration, cap)
    if allocator_id == "bo_gp":
        return _BoGp(slots, gamma, noise, beta, prior_std, cap)
    raise AllocatorError(f"unknown allocator {allocator_id!r}; expected one of {ALLOCATOR_IDS}")


class _RecordingRun(Protocol):
    """What append_allocation needs of the run recording it: one append.

    The runner's ``Run`` satisfies this structurally; the allocator names the
    shape rather than the class, as the self-model does, so no runner import
    ever crosses into this package.
    """

    def append(
        self,
        kind: str,
        payload: Payload,
        *,
        refers_to: Sequence[str] = (),
        budget: Budget | None = None,
    ) -> Record: ...


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

    ``candidates``, ``chosen`` and ``scores`` land in the payload as given —
    it rejects duplicates, non-subsets and score keys outside the candidates —
    and the record refers to every candidate it judged, in candidate order.
    The record charges wall time only: allocators select, they never call an
    LLM, and (f) charges every condition the same budget for the same
    candidates (AGENTS.md 2.5).
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
