"""The improver's shared types (TASK_017 PR A1, roadmap entry 16).

The improver proposes edits to updater artifacts (``improver/artifacts.py``
pins what an artifact is); every proposal is described, before it is
evaluated, by the features of this module. ``LLMClient`` is the boundary the
PR A2 harness talks to an LLM through — the real client, the recorded-replay
client and the test fake all satisfy it — and nothing here touches the
network, a pool, a dataset or the ledger.
"""

from dataclasses import dataclass
from typing import Protocol

IMPROVER_VERSION = "1"


@dataclass(frozen=True)
class LLMResponse:
    """One successful completion: the text and the token counts to charge."""

    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


class LLMClient(Protocol):
    """Anything that answers one prompt with one completion."""

    def complete(self, system: str, user: str) -> LLMResponse: ...


@dataclass(frozen=True)
class ProposalFeatures:
    """What a proposal looks like before it is evaluated."""

    family: str  # "params" (only numeric constants changed) | "logic" (anything else)
    changed_lines: int  # >= 0, diff body lines
    const_changes: int  # >= 0, numeric constants that differ from the parent
    max_rel_const_change: float  # in [0, 1]; 0.0 when no constant changed
    generation: int  # >= 0
    parent_gain: float  # this artifact's own realised gain, clipped to [-1, 1]; 0.0 at generation 0
    is_llm: bool


#: The eight feature names, in feature_vector's order.
FEATURE_NAMES: tuple[str, ...] = (
    "generation",
    "family_params",
    "family_logic",
    "changed_lines",
    "const_changes",
    "max_rel_const_change",
    "parent_gain",
    "is_llm",
)


def feature_vector(features: ProposalFeatures) -> tuple[float, ...]:
    """The eight features as floats, in ``FEATURE_NAMES`` order.

    Every value is finite and in ``[0, 1]`` except ``parent_gain``, which is
    clipped to ``[-1, 1]``; ``max_rel_const_change`` is clipped to ``[0, 1]``,
    and the count features saturate at 50 generations, 20 diff lines and 5
    changed constants.
    """
    gain = features.parent_gain
    if gain < -1.0:
        gain = -1.0
    elif gain > 1.0:
        gain = 1.0
    rel = features.max_rel_const_change
    if rel < 0.0:
        rel = 0.0
    elif rel > 1.0:
        rel = 1.0
    return (
        min(features.generation, 50) / 50,
        1.0 if features.family == "params" else 0.0,
        1.0 if features.family == "logic" else 0.0,
        min(features.changed_lines, 20) / 20,
        min(features.const_changes, 5) / 5,
        rel,
        gain,
        1.0 if features.is_llm else 0.0,
    )
