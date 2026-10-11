"""Arena error types: bad arena inputs, premature target requests, failing updaters."""

from typing import Literal

FailureKind = Literal["failed", "invalid_weights"]  # the ledger's EVALUATION_STATUSES minus timeout


class ArenaError(Exception):
    """Raised when the arena is given an invalid input: a bad pool, cache or argument."""


class LookAheadError(ArenaError):
    """Raised when a target or history is requested before its horizon has matured."""


class UpdaterFailure(Exception):
    """Raised when an updater misbehaves; stops the replay of its series."""

    kind: FailureKind
    message: str  # at most 1000 characters
    series_id: str
    step: int | None  # None when init failed

    def __init__(self, kind: FailureKind, message: str, series_id: str, step: int | None) -> None:
        super().__init__(kind, message, series_id, step)
        self.kind = kind
        self.message = message
        self.series_id = series_id
        self.step = step
