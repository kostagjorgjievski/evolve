"""The allocator's shared types (TASK_017 PR C, roadmap entry 18).

The allocator decides, each generation, which of the improver's proposals are
spent an evaluation on: it sees the candidates' features and — for the
self-model conditions only — their predicted gain quantiles, the history of
already-evaluated proposals, how many evaluations the budget still allows and
a seeded rng. That is all it ever sees: no allocator imports, reads or could
reach any pool, dataset or holdout, which is what keeps selection off the
holdout. Candidates without predictions carry empty ``quantiles`` — a
condition whose allocator must not see predictions passes exactly that.
"""

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from evolvets.improver.interfaces import ProposalFeatures
from evolvets.selfmodel.interfaces import HistoryPoint


class AllocatorError(RuntimeError):
    """Raised when an allocator is asked for something it cannot do."""


@dataclass(frozen=True)
class Candidate:
    """One proposal an allocator may choose to evaluate."""

    record_id: str
    features: ProposalFeatures
    quantiles: Mapping[str, float]  # {} for conditions whose allocator must not see predictions


@dataclass(frozen=True)
class Decision:
    """One allocation: what to evaluate, in order, and the scores that drove it."""

    chosen: tuple[str, ...]  # record ids, in evaluation order
    scores: Mapping[str, float]  # one finite float per candidate it scored


class Allocator(Protocol):
    """Anything that picks which candidates to evaluate."""

    def identify(self) -> dict[str, str]: ...

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision: ...
