"""Kind-specific payload dataclasses and their validation.

Every payload field is required: unknown or missing fields are rejected, and
non-finite floats are rejected anywhere inside a payload, at construction
time rather than at write time. Payload dataclasses that nest a ``Budget``
are listed in ``_BUDGET_FIELDS`` explicitly (no annotation introspection, so
future-annotations styles cannot break the conversion silently).
"""

import dataclasses
import itertools
import math
import re
import uuid
from typing import Any

from evolvets.ledger.schema import Budget

PROTOCOL_KEYS = frozenset({"ledger", "arena", "generator", "cache_manifest"})
RUN_END_STATUSES = frozenset({"completed", "aborted", "budget_exhausted"})
PROPOSAL_TARGETS = frozenset({"updater", "improver"})
EVALUATION_POOLS = frozenset({"dev", "sealed", "live"})
EVALUATION_STATUSES = frozenset({"ok", "failed", "timeout", "invalid_weights"})
QUANTILE_KEYS = ("q05", "q25", "q50", "q75", "q95")
PROPOSER_KEYS = frozenset({"id", "version", "model", "temperature"})
AGENT_ID_KEYS = frozenset({"id", "version"})
PLANTED_DEFECT_KEYS = frozenset({"defect_id", "family", "params"})

_BLOB_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class PayloadError(ValueError):
    """Raised when a payload does not match its kind's schema."""


def _require_str(name: str, value: Any) -> None:
    """Check that ``value`` is a str."""
    if not isinstance(value, str):
        raise PayloadError(f"{name} must be a string, got {type(value).__name__}")


def _require_nonempty_str(name: str, value: Any) -> None:
    """Check that ``value`` is a non-empty str."""
    _require_str(name, value)
    if not value:
        raise PayloadError(f"{name} must not be empty")


def _require_blob_ref(name: str, value: Any) -> None:
    """Check that ``value`` is a ``sha256:<64 hex>`` blob reference."""
    if not isinstance(value, str) or not _BLOB_RE.match(value):
        raise PayloadError(f"{name} must be a blob ref 'sha256:<64 hex>', got {value!r}")


def _require_nullable_blob_ref(name: str, value: Any) -> None:
    """Check a blob reference that may be null when the content is absent."""
    if value is not None:
        _require_blob_ref(name, value)


def _require_literal(name: str, value: Any, options: frozenset[str]) -> None:
    """Check that ``value`` is one of ``options``."""
    if value not in options:
        raise PayloadError(f"{name} must be one of {sorted(options)}, got {value!r}")


def _require_int(name: str, value: Any) -> None:
    """Check that ``value`` is an int (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise PayloadError(f"{name} must be an int, got {value!r}")


def _require_positive_int(name: str, value: Any) -> None:
    """Check that ``value`` is an int of at least 1 (bools excluded)."""
    _require_int(name, value)
    if value < 1:
        raise PayloadError(f"{name} must be >= 1, got {value!r}")


def _require_float(name: str, value: Any) -> None:
    """Check that ``value`` is a finite number (bools excluded)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PayloadError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise PayloadError(f"{name} must be finite")


def _require_bool(name: str, value: Any) -> None:
    """Check that ``value`` is a bool."""
    if not isinstance(value, bool):
        raise PayloadError(f"{name} must be a bool, got {value!r}")


def _require_uuid_str(name: str, value: Any) -> None:
    """Check that ``value`` is a UUID string, as all record ids are."""
    _require_str(name, value)
    try:
        uuid.UUID(value)
    except ValueError as exc:
        raise PayloadError(f"{name} must be a UUID string, got {value!r}") from exc


def _require_exact_keys(name: str, value: Any, keys: frozenset[str]) -> None:
    """Check that ``value`` is a dict with exactly ``keys``."""
    if not isinstance(value, dict) or set(value) != set(keys):
        raise PayloadError(f"{name} must be a dict with exactly the keys {sorted(keys)}")


def _require_str_keys(name: str, value: Any) -> None:
    """Check that ``value`` is a dict whose keys are strings."""
    if not isinstance(value, dict):
        raise PayloadError(f"{name} must be a dict, got {type(value).__name__}")
    for key in value:
        if not isinstance(key, str):
            raise PayloadError(f"{name} keys must be strings, got {key!r}")


def _require_float_dict(name: str, value: Any) -> None:
    """Check that ``value`` is a dict of str to finite floats."""
    _require_str_keys(name, value)
    for key, item in value.items():
        _require_float(f"{name}[{key}]", item)


def _require_agent_id(name: str, value: Any) -> None:
    """Check an ``{id, version}`` agent identifier dict."""
    _require_exact_keys(name, value, AGENT_ID_KEYS)
    _require_str(f"{name}.id", value["id"])
    _require_str(f"{name}.version", value["version"])


def _check_finite(name: str, obj: Any) -> None:
    """Reject NaN and infinities anywhere inside a JSON-like structure."""
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise PayloadError(f"{name} must not contain non-finite floats")
    elif isinstance(obj, dict):
        for key, value in obj.items():
            _check_finite(f"{name}.{key}", value)
    elif isinstance(obj, (list, tuple)):
        for i, item in enumerate(obj):
            _check_finite(f"{name}[{i}]", item)


@dataclasses.dataclass(frozen=True)
class RunStart:
    """First record of a run: condition, config and protocol versions."""

    condition: str
    seed: int
    requires_predictions: bool
    config_blob: str
    git_commit: str
    protocol_versions: dict[str, str]
    budget_limits: Budget

    def __post_init__(self) -> None:
        _require_nonempty_str("condition", self.condition)
        _require_int("seed", self.seed)
        _require_bool("requires_predictions", self.requires_predictions)
        _require_blob_ref("config_blob", self.config_blob)
        _require_nonempty_str("git_commit", self.git_commit)
        _require_exact_keys("protocol_versions", self.protocol_versions, PROTOCOL_KEYS)
        for key, value in self.protocol_versions.items():
            _require_str(f"protocol_versions[{key}]", value)
        if not isinstance(self.budget_limits, Budget):
            raise PayloadError("budget_limits must be a Budget instance")
        _check_finite("run_start", dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class RunEnd:
    """Last record of a run: how it ended and a free-form summary."""

    status: str
    summary: dict[str, Any]

    def __post_init__(self) -> None:
        _require_literal("status", self.status, RUN_END_STATUSES)
        _require_str_keys("summary", self.summary)
        _check_finite("run_end", dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class Proposal:
    """One candidate change proposed by the improver."""

    target: str
    parent_artifact: str
    child_artifact: str
    diff_blob: str
    observation_blob: str
    prompt_blob: str | None
    response_blob: str | None
    diagnosis: str
    proposer: dict[str, Any]
    improver_version: str
    planted_defect: dict[str, Any] | None

    def __post_init__(self) -> None:
        _require_literal("target", self.target, PROPOSAL_TARGETS)
        _require_blob_ref("parent_artifact", self.parent_artifact)
        _require_blob_ref("child_artifact", self.child_artifact)
        _require_blob_ref("diff_blob", self.diff_blob)
        _require_blob_ref("observation_blob", self.observation_blob)
        _require_nullable_blob_ref("prompt_blob", self.prompt_blob)
        _require_nullable_blob_ref("response_blob", self.response_blob)
        _require_str("diagnosis", self.diagnosis)
        _require_exact_keys("proposer", self.proposer, PROPOSER_KEYS)
        _require_str("proposer.id", self.proposer["id"])
        _require_str("proposer.version", self.proposer["version"])
        # model and temperature are null for non-LLM proposers (random
        # mutation, search); inventing values would contaminate the record.
        if self.proposer["model"] is not None:
            _require_str("proposer.model", self.proposer["model"])
        if self.proposer["temperature"] is not None:
            _require_float("proposer.temperature", self.proposer["temperature"])
        _require_str("improver_version", self.improver_version)
        if self.planted_defect is not None:
            _require_exact_keys("planted_defect", self.planted_defect, PLANTED_DEFECT_KEYS)
            _require_str("planted_defect.defect_id", self.planted_defect["defect_id"])
            _require_str("planted_defect.family", self.planted_defect["family"])
            _require_str_keys("planted_defect.params", self.planted_defect["params"])
        _check_finite("proposal", dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class Prediction:
    """Self-model quantiles for the gain of one proposal."""

    predictor: dict[str, str]
    metric: str
    horizon_generations: int
    quantiles: dict[str, float]

    def __post_init__(self) -> None:
        _require_agent_id("predictor", self.predictor)
        _require_nonempty_str("metric", self.metric)
        _require_positive_int("horizon_generations", self.horizon_generations)
        _require_exact_keys("quantiles", self.quantiles, frozenset(QUANTILE_KEYS))
        _require_float_dict("quantiles", self.quantiles)
        ordered = [self.quantiles[key] for key in QUANTILE_KEYS]
        if any(later < earlier for earlier, later in itertools.pairwise(ordered)):
            raise PayloadError(
                f"quantiles must be monotone non-decreasing in {list(QUANTILE_KEYS)}, got {ordered}"
            )


@dataclasses.dataclass(frozen=True)
class Allocation:
    """The allocator's choice among candidate proposals."""

    allocator: dict[str, str]
    candidates: list[str]
    chosen: list[str]
    scores: dict[str, float]

    def __post_init__(self) -> None:
        _require_agent_id("allocator", self.allocator)
        if not isinstance(self.candidates, list) or not self.candidates:
            raise PayloadError("candidates must be a non-empty list of record ids")
        if not isinstance(self.chosen, list):
            raise PayloadError("chosen must be a list of record ids")
        for record_id in self.candidates + self.chosen:
            _require_uuid_str("candidates/chosen entry", record_id)
        if len(set(self.candidates)) != len(self.candidates):
            raise PayloadError("candidates must not contain duplicates")
        if len(set(self.chosen)) != len(self.chosen):
            raise PayloadError("chosen must not contain duplicates")
        if not set(self.chosen) <= set(self.candidates):
            raise PayloadError("chosen must be a subset of candidates")
        _require_float_dict("scores", self.scores)
        if not set(self.scores) <= set(self.candidates):
            raise PayloadError("scores keys must be candidate record ids")


@dataclasses.dataclass(frozen=True)
class Evaluation:
    """The result of evaluating one proposal in one pool."""

    pool: str
    status: str
    metrics: dict[str, float]
    per_task_blob: str | None
    error: str | None

    def __post_init__(self) -> None:
        _require_literal("pool", self.pool, EVALUATION_POOLS)
        _require_literal("status", self.status, EVALUATION_STATUSES)
        _require_float_dict("metrics", self.metrics)
        _require_nullable_blob_ref("per_task_blob", self.per_task_blob)
        # Failures are part of the record: an unexplained failure cannot be
        # audited, and a success carrying an error would mislead the reader.
        if self.status == "ok":
            if self.error is not None:
                raise PayloadError("error must be None when status is 'ok'")
        elif not isinstance(self.error, str) or not self.error:
            raise PayloadError("error must be a non-empty string when status is not 'ok'")


@dataclasses.dataclass(frozen=True)
class Acceptance:
    """Whether a proposal was accepted after its evaluation."""

    accepted: bool
    rule: str
    new_artifact: str | None

    def __post_init__(self) -> None:
        _require_bool("accepted", self.accepted)
        _require_nonempty_str("rule", self.rule)
        _require_nullable_blob_ref("new_artifact", self.new_artifact)


@dataclasses.dataclass(frozen=True)
class Baseline:
    """A hand-designed updater's score in one pool, or why it has none."""

    updater_id: str
    hyperparams: dict[str, Any]
    pool: str
    status: str  # one of EVALUATION_STATUSES
    metrics: dict[str, float]  # {} unless status == "ok"
    per_task_blob: str | None
    error: str | None  # None iff status == "ok"

    def __post_init__(self) -> None:
        _require_nonempty_str("updater_id", self.updater_id)
        _require_str_keys("hyperparams", self.hyperparams)
        _require_literal("pool", self.pool, EVALUATION_POOLS)
        _require_literal("status", self.status, EVALUATION_STATUSES)
        _require_float_dict("metrics", self.metrics)
        _require_nullable_blob_ref("per_task_blob", self.per_task_blob)
        # A run that did not finish has no score to report, and a failure
        # without an explanation cannot be audited; same pairing as Evaluation.
        if self.status != "ok" and self.metrics != {}:
            raise PayloadError("metrics must be empty when status is not 'ok'")
        if self.status == "ok":
            if self.error is not None:
                raise PayloadError("error must be None when status is 'ok'")
        elif not isinstance(self.error, str) or not self.error:
            raise PayloadError("error must be a non-empty string when status is not 'ok'")
        _check_finite("baseline", dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class Commitment:
    """A timestamped public commitment to future behaviour."""

    subject: str
    content_hash: str
    external_timestamp: str | None

    def __post_init__(self) -> None:
        _require_nonempty_str("subject", self.subject)
        _require_blob_ref("content_hash", self.content_hash)
        if self.external_timestamp is not None:
            _require_str("external_timestamp", self.external_timestamp)


@dataclasses.dataclass(frozen=True)
class Correction:
    """A correction of exactly one earlier record; the original stays put."""

    reason: str
    replacement: dict[str, Any] | None

    def __post_init__(self) -> None:
        _require_nonempty_str("reason", self.reason)
        if self.replacement is not None:
            _require_str_keys("replacement", self.replacement)
        _check_finite("correction", dataclasses.asdict(self))


Payload = (
    RunStart
    | RunEnd
    | Proposal
    | Prediction
    | Allocation
    | Evaluation
    | Acceptance
    | Baseline
    | Commitment
    | Correction
)

PAYLOAD_KINDS: dict[str, type[Payload]] = {
    "run_start": RunStart,
    "run_end": RunEnd,
    "proposal": Proposal,
    "prediction": Prediction,
    "allocation": Allocation,
    "evaluation": Evaluation,
    "acceptance": Acceptance,
    "baseline": Baseline,
    "commitment": Commitment,
    "correction": Correction,
}

_BUDGET_FIELDS: dict[type[Payload], frozenset[str]] = {
    RunStart: frozenset({"budget_limits"}),
}


def validate_kind_payload(kind: str, payload: object) -> None:
    """Check that ``payload`` is the dataclass registered for ``kind``."""
    cls = PAYLOAD_KINDS.get(kind)
    if cls is None:
        raise PayloadError(f"unknown kind {kind!r}")
    if not isinstance(payload, cls):
        raise PayloadError(f"payload for kind {kind!r} must be {cls.__name__}")


def payload_from_dict(kind: str, data: dict[str, Any]) -> Payload:
    """Rebuild the payload for ``kind``, rejecting unknown or missing fields."""
    cls = PAYLOAD_KINDS.get(kind)
    if cls is None:
        raise PayloadError(f"unknown kind {kind!r}")
    names = {field.name for field in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise PayloadError(f"unknown fields for kind {kind!r}: {sorted(unknown)}")
    missing = names - set(data)
    if missing:
        raise PayloadError(f"missing fields for kind {kind!r}: {sorted(missing)}")
    kwargs = dict(data)
    for field_name in _BUDGET_FIELDS.get(cls, frozenset()):
        if isinstance(kwargs[field_name], dict):
            kwargs[field_name] = Budget(**kwargs[field_name])
    return cls(**kwargs)


def payload_to_dict(payload: Payload) -> dict[str, Any]:
    """Return the JSON-ready dict form of ``payload``."""
    return dataclasses.asdict(payload)
