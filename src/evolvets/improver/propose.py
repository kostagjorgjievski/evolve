"""The proposers: LLM edits and random constant mutations, recorded (TASK_017 PR A2).

``propose_and_record`` makes one proposal and appends its ``proposal``
record: every blob is written first (crash safety — an interrupted proposal
leaves at most unreferenced, content-addressed blobs, never a record pointing
at a missing blob), then the record, charged through the budget accounting.
The LLM proposer repairs malformed replies once per ``max_repair_rounds`` and
falls back to a recorded no-op when every round fails — the failed calls are
still part of the record (AGENTS.md 2.8). A call that raises
``LLMClientError`` (exhausted retries, a non-retryable 4xx, a missing key)
ends the rounds the same way: its audit — no text, the error message — and
its cost are recorded before the fallback no-op is written, so a paid call
never goes uncharged (AGENTS.md 2.5, 2.8). The random proposer needs no
client at all: ``proposer.model = null`` is exactly the owner's non-LLM
track, on which the self-model (PR B) and the allocators (PR C) are tested.

Every prompt and response is a blob, so a lineage replays from its records:
``recorded_responses`` maps each call's prompt key to its recorded outcome
and ``RecordedLLMClient`` serves them without a network call, charging zero
tokens and zero wall. Repair prompts are distinct keys, so a whole call
sequence replays — including a failed one: an audit with no text marks a
call that raised, and it replays as the recorded error, so the same fallback
draft is re-derived at zero cost. The API key never appears in a config blob, record, blob or
log: configuration carries the environment variable's name, and the client
reads the value at call time.
"""

from __future__ import annotations

import ast
import json
import random
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from evolvets.improver.artifacts import (
    diff_lines,
    normalized_source,
    validate_artifact_source,
)
from evolvets.improver.client import (
    LLMCall,
    LLMClientConfig,
    LLMClientError,
    RecordedLLMFailure,
    prompt_key,
)
from evolvets.improver.interfaces import IMPROVER_VERSION, LLMClient, LLMResponse
from evolvets.improver.observations import observation_object
from evolvets.improver.prompt import IMPROVER_SYSTEM_PROMPT, prompt_object, user_prompt
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.payloads import Payload, Proposal
from evolvets.ledger.schema import Budget, Record

#: The proposers an improver run may use, in config order.
PROPOSER_IDS: tuple[str, ...] = ("llm_edit", "random_param")

#: Both proposers are at version 1 of their own behaviour.
_PROPOSER_VERSION = "1"


class ProposerError(RuntimeError):
    """Raised when a proposer is asked for something it cannot do."""


@dataclass(frozen=True)
class ProposalDraft:
    """One proposal before it is recorded: the edit, the why, and the proposer identity."""

    parent_source: str
    child_source: str
    diagnosis: str
    proposer: dict[str, Any]  # exactly PROPOSER_KEYS


class _AuditedClient(Protocol):
    """An LLM client that also exposes the audit trail of its latest complete()."""

    def complete(self, system: str, user: str) -> LLMResponse: ...

    @property
    def last_call(self) -> LLMCall: ...


class _RecordingRun(Protocol):
    """What the proposer needs of the run recording it: blobs and appends.

    The runner's ``Run`` satisfies this structurally; the improver names the
    shape rather than the class because the runner reaches arena modules the
    improver must never reach, even in a typing-only import (import-linter).
    """

    @property
    def blobs(self) -> BlobStore: ...

    def put_blob(self, obj: object) -> str: ...

    def append(
        self,
        kind: str,
        payload: Payload,
        *,
        refers_to: Sequence[str] = (),
        budget: Budget | None = None,
    ) -> Record: ...


def _parse_reply(text: str) -> tuple[str, str]:
    """(diagnosis, source) from a reply, or a ValueError naming what is wrong with it.

    The substring from the first ``{`` to the last ``}`` must be a JSON object
    with exactly the keys ``diagnosis`` and ``source``, both strings, and the
    source must satisfy the artifact grammar.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("the reply contains no JSON object")
    try:
        parsed: object = json.loads(text[start : end + 1])
    except json.JSONDecodeError as error:
        raise ValueError(f"the reply is not valid JSON: {error}") from error
    if not isinstance(parsed, dict):
        raise ValueError("the reply must be one JSON object")
    if set(parsed) != {"diagnosis", "source"}:
        raise ValueError(
            "the reply object must have exactly the keys diagnosis and source, "
            f"got {sorted(parsed)}"
        )
    diagnosis = parsed["diagnosis"]
    source = parsed["source"]
    if not isinstance(diagnosis, str):
        raise ValueError("diagnosis must be a string")
    if not isinstance(source, str):
        raise ValueError("source must be a string")
    validate_artifact_source(source)  # raises ArtifactError, a ValueError
    return diagnosis, source


def _audit_object(call: LLMCall, response: LLMResponse) -> dict[str, object]:
    """The per-call response audit: the transport trail plus the reply itself."""
    return {
        "attempts": [
            {"ok": attempt.ok, "status": attempt.status, "error": attempt.error}
            for attempt in call.attempts
        ],
        "waits_seconds": list(call.waits_seconds),
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "stop_reason": response.stop_reason,
        "text": response.text,
    }


def _error_audit_object(call: LLMCall | None, message: str) -> dict[str, object]:
    """The audit of a call that raised: the transport trail it left, no reply.

    ``text`` is ``None`` — the mark a failed call leaves in the response
    blob; ``error`` carries the exception's message. A client that raised
    before attempting anything (``call`` is ``None``, as on the replay path)
    is audited with zero counts; the failed call's tokens and wall are
    charged by the caller from ``call`` itself.
    """
    audited = (
        call
        if call is not None
        else LLMCall(
            attempts=(), waits_seconds=(), input_tokens=0, output_tokens=0, wall_seconds=0.0
        )
    )
    return {
        "attempts": [
            {"ok": attempt.ok, "status": attempt.status, "error": attempt.error}
            for attempt in audited.attempts
        ],
        "waits_seconds": list(audited.waits_seconds),
        "input_tokens": audited.input_tokens,
        "output_tokens": audited.output_tokens,
        "stop_reason": None,
        "text": None,
        "error": message,
    }


def _llm_proposal(
    client: _AuditedClient,
    config: LLMClientConfig,
    parent_source: str,
    observation: dict[str, object],
    max_repair_rounds: int,
) -> tuple[ProposalDraft, list[dict[str, object]], list[dict[str, object]], Budget]:
    """Call the LLM (with repairs); the draft, per-call prompts and audits, the cost."""
    system = IMPROVER_SYSTEM_PROMPT
    base_user = user_prompt(parent_source, observation)
    user = base_user
    prompts: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    input_tokens = 0
    output_tokens = 0
    wall_seconds = 0.0
    diagnosis: str | None = None
    child: str | None = None
    failure = "no call was made"
    calls = 0
    for _ in range(1 + max_repair_rounds):
        try:
            response = client.complete(system, user)
        except LLMClientError as error:
            # a failed call is still part of the record: its audit and its cost
            # are kept, no further call is made, and the fallback below writes
            # the no-op (AGENTS.md 2.5 equal budgets, 2.8 report failures)
            calls += 1
            prompts.append(prompt_object(config, system, user))
            audits.append(_error_audit_object(error.call, str(error)))
            if error.call is not None:
                input_tokens += error.call.input_tokens
                output_tokens += error.call.output_tokens
                wall_seconds += error.call.wall_seconds
            failure = str(error)
            break
        calls += 1
        call = client.last_call
        prompts.append(prompt_object(config, system, user))
        audits.append(_audit_object(call, response))
        input_tokens += call.input_tokens
        output_tokens += call.output_tokens
        wall_seconds += call.wall_seconds
        try:
            diagnosis, source = _parse_reply(response.text)
            child = normalized_source(source)
            break
        except ValueError as error:  # a malformed reply, or a source outside the grammar
            failure = str(error)
            user = (
                base_user
                + "\n\nYour previous reply could not be used: "
                + failure
                + ". Reply with only the JSON object described."
            )
    if child is None or diagnosis is None:
        # the fallback no-op: the cost is still charged, the calls are still recorded
        child = parent_source
        diagnosis = f"proposer fallback after {calls} call(s): {failure}"
    draft = ProposalDraft(
        parent_source=parent_source,
        child_source=child,
        diagnosis=diagnosis,
        proposer={
            "id": "llm_edit",
            "version": _PROPOSER_VERSION,
            "model": config.model,
            "temperature": config.temperature,
        },
    )
    budget = Budget(
        llm_input_tokens=input_tokens,
        llm_output_tokens=output_tokens,
        wall_seconds=wall_seconds,
    )
    return draft, prompts, audits, budget


def _numeric_constant_nodes(tree: ast.Module) -> list[ast.Constant]:
    """The numeric constant nodes of a tree in document order (bools excluded)."""
    nodes: list[ast.Constant] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if (
                isinstance(child, ast.Constant)
                and isinstance(child.value, (int, float))
                and not isinstance(child.value, bool)
            ):
                nodes.append(child)
            visit(child)

    visit(tree)
    return nodes


def _twelve_significant_digits(value: float) -> float:
    """``value`` rounded to 12 significant digits, the mutation's numeric precision."""
    return float(f"{value:.12g}")


def random_propose(parent_source: str, rng: random.Random) -> ProposalDraft:
    """One random edit: one numeric constant picked uniformly and scaled by U(0.5, 2.0).

    The constants live in the AST, so the child is rebuilt with ``ast.unparse``
    (no text surgery) and validated before it is returned. A parent with no
    numeric constant is a caller error: the reference artifact has several.
    """
    tree = ast.parse(parent_source)
    constants = _numeric_constant_nodes(tree)
    if not constants:
        raise ProposerError("random_propose needs a parent with at least one numeric constant")
    index = rng.choice(range(len(constants)))
    old = cast("int | float", constants[index].value)
    new = _twelve_significant_digits(old * rng.uniform(0.5, 2.0))
    constants[index].value = new  # the node is reused by ast.unparse below
    child = ast.unparse(tree)
    validate_artifact_source(child)
    return ProposalDraft(
        parent_source=parent_source,
        child_source=child,
        diagnosis=f"random constant mutation: constant {index} from {old} to {new}",
        proposer={
            "id": "random_param",
            "version": _PROPOSER_VERSION,
            "model": None,
            "temperature": None,
        },
    )


def propose_and_record(
    run: _RecordingRun,
    *,
    proposer_id: str,
    parent_source: str,
    parent_record_id: str | None,  # the proposal whose child_artifact is parent_source
    generation: int,
    parent_gain: float,  # in the pinned signature for the harness; the observation carries metrics
    history_records: Sequence[Record],
    rng: random.Random,
    client: LLMClient | None = None,  # required by "llm_edit"
    client_config: LLMClientConfig | None = None,
    max_repair_rounds: int = 1,
) -> Record:
    """Propose one edit of ``parent_source`` and append its ``proposal`` record.

    Blobs first (parent and child sources, the joined diff lines, the
    observation, and — for the LLM proposer — the per-call prompt and response
    audit objects), then the record, charged with the proposer's tokens and
    wall time; the random proposer charges wall time only. ``refers_to`` links
    the parent proposal, or nothing at generation 0.
    """
    del parent_gain  # the pinned signature carries it for the harness; nothing records it in A2
    if proposer_id not in PROPOSER_IDS:
        raise ProposerError(f"unknown proposer {proposer_id!r}; expected one of {PROPOSER_IDS}")
    observation = observation_object(run.blobs, history_records, generation)
    if proposer_id == "llm_edit":
        if client is None or client_config is None:
            raise ProposerError("the llm_edit proposer requires a client and a client config")
        if not hasattr(client, "last_call"):
            raise ProposerError(
                "the llm_edit proposer requires a client that exposes its call audit (last_call)"
            )
        draft, prompts, audits, budget = _llm_proposal(
            cast("_AuditedClient", client),
            client_config,
            parent_source,
            observation,
            max_repair_rounds,
        )
        prompt_blob: str | None = run.put_blob(prompts)
        response_blob: str | None = run.put_blob(audits)
    else:
        start = time.monotonic()
        draft = random_propose(parent_source, rng)
        budget = Budget(wall_seconds=time.monotonic() - start)
        prompt_blob = None
        response_blob = None
    parent_ref = run.put_blob(draft.parent_source)
    child_ref = run.put_blob(draft.child_source)
    diff_ref = run.put_blob("\n".join(diff_lines(draft.parent_source, draft.child_source)))
    observation_ref = run.put_blob(observation)
    return run.append(
        "proposal",
        Proposal(
            target="updater",
            parent_artifact=parent_ref,
            child_artifact=child_ref,
            diff_blob=diff_ref,
            observation_blob=observation_ref,
            prompt_blob=prompt_blob,
            response_blob=response_blob,
            diagnosis=draft.diagnosis,
            proposer=dict(draft.proposer),
            improver_version=IMPROVER_VERSION,
            planted_defect=None,
        ),
        refers_to=[parent_record_id] if parent_record_id is not None else [],
        budget=budget,
    )


def recorded_responses(
    blobs: BlobStore, records: Iterable[Record]
) -> dict[str, LLMResponse | RecordedLLMFailure]:
    """Map every recorded call's prompt key to its outcome, from the proposals' blobs.

    Each proposal's prompt blob is the list of its calls' prompt objects and
    its response blob the matching list of audits. An audit with a text is a
    successful call and maps to its :class:`LLMResponse`; an audit with no
    text marks a call that raised :class:`LLMClientError` and maps to a
    :class:`RecordedLLMFailure` carrying the recorded message, so a failed
    proposal replays — through :class:`RecordedLLMClient` raising that
    message — to the same fallback draft at zero cost.
    """
    responses: dict[str, LLMResponse | RecordedLLMFailure] = {}
    for record in records:
        if record.kind != "proposal" or not isinstance(record.payload, Proposal):
            continue
        prompt_blob = record.payload.prompt_blob
        response_blob = record.payload.response_blob
        if prompt_blob is None or response_blob is None:
            continue
        prompts = cast(
            "list[dict[str, object]]", json.loads(blobs.get(prompt_blob).decode("utf-8"))
        )
        audits = cast(
            "list[dict[str, object]]", json.loads(blobs.get(response_blob).decode("utf-8"))
        )
        for prompt, audit in zip(prompts, audits, strict=True):
            key = prompt_key(str(prompt["system"]), str(prompt["user"]))
            if audit["text"] is None:  # the mark of a call that raised
                responses[key] = RecordedLLMFailure(str(audit["error"]))
            else:
                responses[key] = LLMResponse(
                    text=str(audit["text"]),
                    input_tokens=int(cast("int", audit["input_tokens"])),
                    output_tokens=int(cast("int", audit["output_tokens"])),
                    stop_reason=cast("str | None", audit["stop_reason"]),
                )
    return responses
