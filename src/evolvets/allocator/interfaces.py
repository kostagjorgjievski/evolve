"""The allocator's shared types (TASK_017 PR C, roadmap entry 18).

The allocator decides which proposed edits are worth one of the evaluation
slots still left. Everything it ever sees arrives as arguments — the
candidates' features, the self-model quantiles the condition allows it to
see, and the lineage's evaluated history — so no allocator can import, read
or reach any pool, dataset or holdout: that is what keeps selection off the
holdout (AGENTS.md 2.1). The quantiles are empty for conditions whose
allocator must not see predictions; ``predictive`` refuses them there
rather than running on nothing.
"""

from __future__ import annotations

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
    """One proposed edit as the allocator sees it: identity, features, quantiles."""

    record_id: str
    features: ProposalFeatures
    quantiles: Mapping[str, float]  # {} for conditions whose allocator must not see predictions


@dataclass(frozen=True)
class Decision:
    """One allocation: what to evaluate, in order, and the score that produced it."""

    chosen: tuple[str, ...]  # record ids, in evaluation order
    scores: Mapping[str, float]  # one finite float per candidate it scored


class Allocator(Protocol):
    """Anything that ranks candidate edits under an evaluation budget."""

    def identify(self) -> dict[str, str]: ...

    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision: ...
