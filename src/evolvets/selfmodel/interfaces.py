"""The self-model's shared types (TASK_017 PR B, roadmap entry 17).

The self-model predicts, before any evaluation starts, the distribution of
one proposal's dev MASE gain: five quantiles per proposal, written to the
ledger and flushed before the evaluation it forecasts. The features are the
improver's shared :class:`~evolvets.improver.interfaces.ProposalFeatures`;
nothing here touches a pool, a dataset, the holdout or the network.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from evolvets.improver.interfaces import ProposalFeatures

METRIC_DEV_MASE_GAIN = "dev_mase_gain"

#: The prior gain quantiles, from the headroom scan's per-edit magnitudes.
DEFAULT_PRIOR: Mapping[str, float] = MappingProxyType(
    {"q05": -0.02, "q25": -0.005, "q50": 0.002, "q75": 0.01, "q95": 0.04}
)


@dataclass(frozen=True)
class HistoryPoint:
    """One evaluated proposal as the self-model sees it."""

    features: ProposalFeatures
    gain: float  # realised dev MASE gain of that proposal


class SelfModel(Protocol):
    """Anything that turns proposal features into gain quantiles."""

    def identify(self) -> dict[str, str]: ...

    def predict(self, features: ProposalFeatures) -> Mapping[str, float]: ...
