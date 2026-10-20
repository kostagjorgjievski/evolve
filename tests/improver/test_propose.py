"""Tests for the proposers, the observation object and the prompts (TASK_017 PR A2: tests 17-24).

The LLM path runs end to end against ``ScriptedClient`` (CI never calls the
API) into a real run directory, so blob references, the payload, the budget
and ``validate_run`` are all checked the way (f)'s lineage loop will see
them. The observation fixtures build records by hand — including a sealed
evaluation that must never reach a prompt — and the replay test proves a
proposal re-derives from its recorded responses at zero cost. The review of
PR A2 added one more end-to-end path: a real ``HttpsLLMClient`` over a
scripted ``post`` that raises mid-proposal must still be recorded, charged
and replayable.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

from evolvets.improver.artifacts import (
    REFERENCE_ARTIFACT_SOURCE,
    artifact_family,
    diff_lines,
    validate_artifact_source,
)
from evolvets.improver.client import (
    HttpsLLMClient,
    LLMClientConfig,
    LLMClientError,
    RecordedLLMClient,
    RecordedLLMFailure,
    prompt_key,
)
from evolvets.improver.interfaces import IMPROVER_VERSION, LLMResponse
from evolvets.improver.observations import ImproverError, observation_object
from evolvets.improver.prompt import IMPROVER_SYSTEM_PROMPT, prompt_object, user_prompt
from evolvets.improver.propose import (
    ProposerError,
    propose_and_record,
    random_propose,
    recorded_responses,
)
from evolvets.ledger.blobs import BlobStore
from evolvets.ledger.canonical import canonical_bytes, canonical_json, sha256_hex
from evolvets.ledger.payloads import Acceptance, Evaluation, Proposal, RunStart
from evolvets.ledger.schema import ZERO_HASH, Budget, Record, make_record
from evolvets.ledger.store import LedgerReader
from evolvets.ledger.validate import validate_run
from evolvets.runner.runs import Run
from tests.improver.fake import (
    ScriptedClient,
    ScriptedMonotonic,
    ScriptedPost,
    ScriptedSleep,
)

COMMIT = "a" * 40
KEY = "sk-live-secret-never-recorded"
CLIENT_CONFIG = LLMClientConfig()  # the pinned defaults, including api_key_env's name
RUN_CONFIG: dict[str, Any] = {
    "proposer": "llm_edit",
    "llm": {
        "base_url": "https://open.bigmodel.cn/api/anthropic",
        "model": "glm-5.3",
        "api_key_env": "EVOLVETS_LLM_API_KEY",
        "temperature": 1.0,
        "max_tokens": 8192,
    },
}
LIMITS = Budget(llm_input_tokens=1e9, llm_output_tokens=1e9, wall_seconds=1e9)

#: One params edit of the reference: the combination's starting level.
EDITED_SOURCE = REFERENCE_ARTIFACT_SOURCE.replace("base = 0.0", "base = 0.5")


def reply(diagnosis: str, source: str) -> str:
    """A model reply holding exactly the JSON object the contract asks for."""
    return json.dumps({"diagnosis": diagnosis, "source": source})


def empty_observation() -> dict[str, Any]:
    """The observation over an empty lineage: what generation 0 sees."""
    return {"generation": 0, "parent_metrics": {}, "history": []}


def start_run(root: Path, run_id: str = "improver-1") -> Run:
    """A run with the improver's config and limits far above anything a test charges."""
    return Run.start(
        root,
        run_id,
        condition="improver-harness",
        seed=7,
        config=RUN_CONFIG,
        budget_limits=LIMITS,
        git_commit=COMMIT,
    )


def records_of(root: Path, run_id: str) -> list[Record]:
    return list(LedgerReader(root, run_id).records())


def blob_json(run_blobs: BlobStore, ref: str | None) -> Any:
    assert ref is not None
    return json.loads(run_blobs.get(ref).decode("utf-8"))


def propose(
    run: Run,
    *,
    client: Any = None,
    proposer_id: str = "llm_edit",
    **overrides: Any,
) -> Record:
    """propose_and_record with the test defaults for every keyword."""
    arguments: dict[str, Any] = {
        "proposer_id": proposer_id,
        "parent_source": REFERENCE_ARTIFACT_SOURCE,
        "parent_record_id": None,
        "generation": 0,
        "parent_gain": 0.0,
        "history_records": [],
        "rng": random.Random(0),
        "client": client,
        "client_config": CLIENT_CONFIG,
        "max_repair_rounds": 1,
    }
    arguments.update(overrides)
    return propose_and_record(run, **arguments)


# --- Test 17: llm_edit end to end --------------------------------------------------


def test_run_exposes_its_blob_store(tmp_path: Path) -> None:
    """The property the proposer reads history artifacts through: the store put_blob writes."""
    run = start_run(tmp_path / "ledgers", "improver-blobs")
    ref = run.put_blob("an artifact")
    assert json.loads(run.blobs.get(ref).decode("utf-8")) == "an artifact"
    run.end("completed", {})


def test_llm_edit_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One proposal record per generation, every field as specified, blobs resolvable, run valid."""
    monkeypatch.setenv("EVOLVETS_LLM_API_KEY", KEY)
    root = tmp_path / "ledgers"
    run = start_run(root)
    first = propose(
        run,
        client=ScriptedClient(
            LLMResponse(
                reply("start the combination from the mean", EDITED_SOURCE), 100, 50, "end_turn"
            )
        ),
    )
    second_source = EDITED_SOURCE.replace("base = 0.5", "base = 0.75")
    second = propose(
        run,
        client=ScriptedClient(
            LLMResponse(reply("raise the level further", second_source), 200, 80, "end_turn")
        ),
        parent_source=EDITED_SOURCE,
        parent_record_id=first.record_id,
        generation=1,
        parent_gain=0.01,
        history_records=records_of(root, "improver-1"),
    )
    run.end("completed", {"proposals": 2})

    records = records_of(root, "improver-1")
    assert [record.kind for record in records] == ["run_start", "proposal", "proposal", "run_end"]
    assert first.refers_to == []  # generation 0 edits the starting artifact
    assert second.refers_to == [first.record_id]  # generation 1 names its parent proposal
    payload = first.payload
    assert isinstance(payload, Proposal)
    assert payload.target == "updater"
    assert payload.proposer == {
        "id": "llm_edit",
        "version": "1",
        "model": "glm-5.3",
        "temperature": 1.0,
    }
    assert payload.improver_version == IMPROVER_VERSION
    assert payload.planted_defect is None
    assert payload.diagnosis == "start the combination from the mean"
    run_blobs = run.blobs
    assert blob_json(run_blobs, payload.parent_artifact) == REFERENCE_ARTIFACT_SOURCE
    assert blob_json(run_blobs, payload.child_artifact) == EDITED_SOURCE
    assert blob_json(run_blobs, payload.diff_blob) == "\n".join(
        diff_lines(REFERENCE_ARTIFACT_SOURCE, EDITED_SOURCE)
    )
    assert blob_json(run_blobs, payload.observation_blob) == empty_observation()
    prompt_blob = blob_json(run_blobs, payload.prompt_blob)
    assert prompt_blob == [
        prompt_object(
            CLIENT_CONFIG,
            IMPROVER_SYSTEM_PROMPT,
            user_prompt(REFERENCE_ARTIFACT_SOURCE, empty_observation()),
        )
    ]
    response_blob = blob_json(run_blobs, payload.response_blob)
    assert response_blob == [
        {
            "attempts": [{"ok": True, "status": None, "error": None}],
            "waits_seconds": [],
            "input_tokens": 100,
            "output_tokens": 50,
            "stop_reason": "end_turn",
            "text": reply("start the combination from the mean", EDITED_SOURCE),
        }
    ]
    assert first.budget == Budget(llm_input_tokens=100, llm_output_tokens=50, wall_seconds=0.0)
    assert second.budget == Budget(llm_input_tokens=200, llm_output_tokens=80, wall_seconds=0.0)
    assert run.spent.llm_input_tokens == 300  # both calls' tokens are charged
    assert LedgerReader(root, "improver-1").verify(check_blobs=True).ok
    assert validate_run(records) == []
    run_start = records[0].payload
    assert isinstance(run_start, RunStart)
    config = blob_json(run_blobs, run_start.config_blob)
    assert config["llm"]["api_key_env"] == "EVOLVETS_LLM_API_KEY"  # the name, never a value
    assert KEY not in json.dumps(config)
    assert KEY not in json.dumps(prompt_blob) + json.dumps(response_blob)


# --- Test 18: the repair round -----------------------------------------------------


def test_repair_round_then_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A first unusable reply, then a good one: both calls audited, both prompts recorded."""
    monkeypatch.setenv("EVOLVETS_LLM_API_KEY", KEY)
    run = start_run(tmp_path / "ledgers", "improver-repair")
    client = ScriptedClient(
        LLMResponse("i am not json at all", 10, 5, None),
        LLMResponse(reply("absorb the bias faster", EDITED_SOURCE), 20, 8, "end_turn"),
    )
    record = propose(run, client=client)
    run.end("completed", {})

    payload = record.payload
    assert isinstance(payload, Proposal)
    assert payload.diagnosis == "absorb the bias faster"
    assert record.budget == Budget(llm_input_tokens=30, llm_output_tokens=13, wall_seconds=0.0)
    prompts = blob_json(run.blobs, payload.prompt_blob)
    assert len(prompts) == 2  # the blob holds the first prompt, and the repair's
    base_user = user_prompt(REFERENCE_ARTIFACT_SOURCE, empty_observation())
    assert prompts[0]["user"] == base_user
    assert prompts[1]["user"] == (
        base_user
        + "\n\nYour previous reply could not be used: the reply contains no JSON object."
        + " Reply with only the JSON object described."
    )
    assert prompts[0]["system"] == prompts[1]["system"] == IMPROVER_SYSTEM_PROMPT
    audits = blob_json(run.blobs, payload.response_blob)
    assert [audit["text"] for audit in audits] == [
        "i am not json at all",
        reply("absorb the bias faster", EDITED_SOURCE),
    ]
    assert [audit["input_tokens"] for audit in audits] == [10, 20]
    assert len(client.prompts) == 2
    assert client.prompts[1][1].startswith(client.prompts[0][1])  # the repair extends the prompt


# --- Test 19: the fallback no-op ---------------------------------------------------


def test_fallback_after_failed_rounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two unusable replies: a no-op child, the fallback diagnosis, and the cost still charged."""
    monkeypatch.setenv("EVOLVETS_LLM_API_KEY", KEY)
    run = start_run(tmp_path / "ledgers", "improver-fallback")
    invalid_source = (
        "def init(n_experts, horizon, config, seed):\n"
        "    import os\n"
        "    return {}\n"
        "\n"
        "\n"
        "def update(state, feedback, context):\n"
        "    return ({}, state)\n"
    )
    client = ScriptedClient(
        LLMResponse(reply("here is an import", invalid_source), 10, 5, "end_turn"),
        LLMResponse("no braces either", 20, 8, None),
    )
    record = propose(run, client=client)
    run.end("completed", {})

    payload = record.payload
    assert isinstance(payload, Proposal)
    assert payload.child_artifact == payload.parent_artifact  # the no-op child is the parent
    assert blob_json(run.blobs, payload.child_artifact) == REFERENCE_ARTIFACT_SOURCE
    assert blob_json(run.blobs, payload.diff_blob) == ""
    assert (
        payload.diagnosis == "proposer fallback after 2 call(s): the reply contains no JSON object"
    )
    assert record.budget == Budget(llm_input_tokens=30, llm_output_tokens=13, wall_seconds=0.0)
    assert len(blob_json(run.blobs, payload.prompt_blob)) == 2
    assert len(blob_json(run.blobs, payload.response_blob)) == 2


# --- Review fix (PR A2): the client raises during the proposal rounds ---------------


def messages_body(text: str, input_tokens: int, output_tokens: int) -> str:
    """A well-formed 2xx Messages body whose text block is the model reply."""
    return json.dumps(
        {
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "stop_reason": "end_turn",
        }
    )


def rate_limited_body(input_tokens: int, output_tokens: int = 0) -> str:
    """A 429 body that reports usage (rate limits still bill their input tokens)."""
    return json.dumps({"usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}})


#: The repair prompt the unusable first reply of the exhausting client earns.
REPAIR_SUFFIX = (
    "\n\nYour previous reply could not be used: the reply contains no JSON object."
    " Reply with only the JSON object described."
)


def exhausting_repair_client() -> tuple[HttpsLLMClient, ScriptedPost]:
    """A real client whose first reply cannot parse and whose repair call exhausts.

    The first call answers a well-formed 200 (usage 5000 in / 3000 out) whose
    text is not JSON; the repair call meets two 429s (``max_attempts = 2``),
    the first reporting 5000 input tokens, and raises. The scripted clock
    makes the two calls' chargeable wall 1.0 + 9.0 seconds (the one wait
    excluded), so the charged budget is exactly 10000 in / 3000 out / 10.0 s.
    """
    post = ScriptedPost(
        (200, messages_body("i am not json at all", 5000, 3000)),
        (429, rate_limited_body(5000)),
        (429, "rate limited"),
    )
    client = HttpsLLMClient(
        LLMClientConfig(max_attempts=2),
        post=post,
        sleep=ScriptedSleep(),
        monotonic=ScriptedMonotonic(0.0, 1.0, 10.0, 21.0),
    )
    return client, post


def test_client_error_during_repair_is_recorded_and_charged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review's case (a): an unusable 200, then a repair that exhausts on 429s.

    The proposal is still recorded — fallback no-op child, both calls' audits
    in the response blob (the failed one with no text and the error message),
    both calls' tokens charged including the 429 body's usage — and the run
    validates. Nothing paid escapes the ledger (AGENTS.md 2.5, 2.8).
    """
    monkeypatch.setenv("EVOLVETS_LLM_API_KEY", KEY)
    root = tmp_path / "ledgers"
    run = start_run(root, "improver-exhausted")
    client, post = exhausting_repair_client()
    record = propose(run, client=client)
    run.end("completed", {})

    payload = record.payload
    assert isinstance(payload, Proposal)
    assert payload.child_artifact == payload.parent_artifact  # the fallback no-op
    assert blob_json(run.blobs, payload.child_artifact) == REFERENCE_ARTIFACT_SOURCE
    assert blob_json(run.blobs, payload.diff_blob) == ""
    assert payload.diagnosis == (
        "proposer fallback after 2 call(s): giving up after 2 attempts: HTTP 429 rate limited"
    )
    # 5000 + 3000 from the unusable 200, 5000 from the 429 body that reported usage
    assert record.budget == Budget(
        llm_input_tokens=10000, llm_output_tokens=3000, wall_seconds=10.0
    )
    assert run.spent.llm_input_tokens == 10000
    assert run.spent.llm_output_tokens == 3000
    assert len(post.sent) == 3  # one attempt for the 200, two for the 429s
    base_user = user_prompt(REFERENCE_ARTIFACT_SOURCE, empty_observation())
    prompts = blob_json(run.blobs, payload.prompt_blob)
    assert [prompt["user"] for prompt in prompts] == [base_user, base_user + REPAIR_SUFFIX]
    audits = blob_json(run.blobs, payload.response_blob)
    assert audits[0] == {
        "attempts": [{"ok": True, "status": 200, "error": None}],
        "waits_seconds": [],
        "input_tokens": 5000,
        "output_tokens": 3000,
        "stop_reason": "end_turn",
        "text": "i am not json at all",
    }
    assert audits[1] == {
        "attempts": [
            {"ok": False, "status": 429, "error": "rate limited"},
            {"ok": False, "status": 429, "error": "rate limited"},
        ],
        "waits_seconds": [2.0],
        "input_tokens": 5000,
        "output_tokens": 0,
        "stop_reason": None,
        "text": None,  # the mark of a call that raised
        "error": "giving up after 2 attempts: HTTP 429 rate limited",
    }
    assert validate_run(records_of(root, "improver-exhausted")) == []
    assert KEY not in json.dumps(prompts) + json.dumps(audits)


def test_first_call_error_is_recorded_and_charged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review's case (b): the first call itself exhausts — one fallback record, charged."""
    monkeypatch.setenv("EVOLVETS_LLM_API_KEY", KEY)
    root = tmp_path / "ledgers"
    run = start_run(root, "improver-first-exhausted")
    post = ScriptedPost((429, rate_limited_body(700, 300)), (429, "rate limited"))
    client = HttpsLLMClient(
        LLMClientConfig(max_attempts=2),
        post=post,
        sleep=ScriptedSleep(),
        monotonic=ScriptedMonotonic(0.0, 5.0),  # wall 5.0 minus the one 2.0 wait
    )
    record = propose(run, client=client)
    run.end("completed", {})

    payload = record.payload
    assert isinstance(payload, Proposal)
    assert payload.child_artifact == payload.parent_artifact
    assert payload.diagnosis == (
        "proposer fallback after 1 call(s): giving up after 2 attempts: HTTP 429 rate limited"
    )
    assert record.budget == Budget(llm_input_tokens=700, llm_output_tokens=300, wall_seconds=3.0)
    assert run.spent.llm_input_tokens == 700
    base_user = user_prompt(REFERENCE_ARTIFACT_SOURCE, empty_observation())
    prompts = blob_json(run.blobs, payload.prompt_blob)
    assert [prompt["user"] for prompt in prompts] == [base_user]  # no repair round happened
    audits = blob_json(run.blobs, payload.response_blob)
    assert len(audits) == 1
    assert audits[0]["text"] is None
    assert audits[0]["error"] == "giving up after 2 attempts: HTTP 429 rate limited"
    assert audits[0]["input_tokens"] == 700
    assert validate_run(records_of(root, "improver-first-exhausted")) == []


def test_missing_key_first_call_records_a_zero_cost_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error with no audit at all (``call`` is ``None``): zero counts, still recorded."""
    monkeypatch.delenv("EVOLVETS_LLM_API_KEY", raising=False)
    root = tmp_path / "ledgers"
    run = start_run(root, "improver-no-key")
    post = ScriptedPost()
    client = HttpsLLMClient(
        LLMClientConfig(), post=post, sleep=ScriptedSleep(), monotonic=ScriptedMonotonic()
    )
    record = propose(run, client=client)
    run.end("completed", {})

    payload = record.payload
    assert isinstance(payload, Proposal)
    assert payload.child_artifact == payload.parent_artifact
    assert (
        payload.diagnosis == "proposer fallback after 1 call(s): "
        "the environment variable EVOLVETS_LLM_API_KEY is not set"
    )
    assert record.budget == Budget()  # nothing was attempted, so nothing is charged
    assert run.spent.llm_input_tokens == 0
    assert post.sent == []  # the key was missing: no request ever left
    audits = blob_json(run.blobs, payload.response_blob)
    assert audits == [
        {
            "attempts": [],
            "waits_seconds": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "stop_reason": None,
            "text": None,
            "error": "the environment variable EVOLVETS_LLM_API_KEY is not set",
        }
    ]
    assert validate_run(records_of(root, "improver-no-key")) == []


def test_replay_of_a_failed_proposal_reproduces_the_fallback_for_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review's case (c): the failed proposal replays to the same child and diagnosis.

    ``recorded_responses`` maps the repair prompt to a ``RecordedLLMFailure``
    (the successful first call maps to its response), and ``RecordedLLMClient``
    raises the recorded message, so the fallback draft is re-derived with zero
    charged tokens and zero wall.
    """
    monkeypatch.setenv("EVOLVETS_LLM_API_KEY", KEY)
    root = tmp_path / "ledgers"
    run = start_run(root, "improver-failed-original")
    original = propose(run, client=exhausting_repair_client()[0])
    run.end("completed", {})

    base_user = user_prompt(REFERENCE_ARTIFACT_SOURCE, empty_observation())
    responses = recorded_responses(run.blobs, records_of(root, "improver-failed-original"))
    assert len(responses) == 2
    assert responses[prompt_key(IMPROVER_SYSTEM_PROMPT, base_user)] == LLMResponse(
        "i am not json at all", 5000, 3000, "end_turn"
    )
    assert responses[prompt_key(IMPROVER_SYSTEM_PROMPT, base_user + REPAIR_SUFFIX)] == (
        RecordedLLMFailure("giving up after 2 attempts: HTTP 429 rate limited")
    )

    replay_run = start_run(root, "improver-failed-replay")
    replay = propose(replay_run, client=RecordedLLMClient(responses))
    replay_run.end("completed", {})

    original_payload, replay_payload = original.payload, replay.payload
    assert isinstance(original_payload, Proposal)
    assert isinstance(replay_payload, Proposal)
    assert blob_json(replay_run.blobs, replay_payload.child_artifact) == blob_json(
        run.blobs, original_payload.child_artifact
    )
    assert replay_payload.diagnosis == original_payload.diagnosis
    assert replay.budget == Budget()  # the replay of a failure is free too
    assert replay_run.spent.llm_input_tokens == 0
    assert replay_run.spent.wall_seconds == 0.0
    assert len(blob_json(replay_run.blobs, replay_payload.prompt_blob)) == 2  # both calls reissued
    assert validate_run(records_of(root, "improver-failed-replay")) == []


# --- Test 20: a genuine no-op ------------------------------------------------------


def test_genuine_noop_reply(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A valid reply that returns the parent unchanged is a normal proposal, not a fallback."""
    monkeypatch.setenv("EVOLVETS_LLM_API_KEY", KEY)
    run = start_run(tmp_path / "ledgers", "improver-noop")
    client = ScriptedClient(
        LLMResponse(
            reply("the artifact is already the best I can propose", REFERENCE_ARTIFACT_SOURCE),
            15,
            6,
            "end_turn",
        )
    )
    record = propose(run, client=client)
    run.end("completed", {})

    payload = record.payload
    assert isinstance(payload, Proposal)
    assert payload.child_artifact == payload.parent_artifact
    assert blob_json(run.blobs, payload.diff_blob) == ""
    assert payload.diagnosis == "the artifact is already the best I can propose"
    assert record.budget == Budget(llm_input_tokens=15, llm_output_tokens=6, wall_seconds=0.0)


# --- Test 21: random_param ---------------------------------------------------------


def test_random_propose_is_seed_deterministic() -> None:
    """The same seed gives the same child; another seed differs; the child validates."""
    first = random_propose(REFERENCE_ARTIFACT_SOURCE, random.Random(42))
    again = random_propose(REFERENCE_ARTIFACT_SOURCE, random.Random(42))
    other = random_propose(REFERENCE_ARTIFACT_SOURCE, random.Random(43))
    assert first.child_source == again.child_source
    assert first.child_source != other.child_source
    assert first.child_source != REFERENCE_ARTIFACT_SOURCE
    validate_artifact_source(first.child_source)
    assert first.parent_source == REFERENCE_ARTIFACT_SOURCE
    assert first.proposer == {
        "id": "random_param",
        "version": "1",
        "model": None,
        "temperature": None,
    }
    assert "random constant mutation" in first.diagnosis


def test_random_propose_without_constants_raises() -> None:
    """A parent with no numeric constant is a caller error (the reference has several)."""
    bare = (
        "def init(n_experts, horizon, config, seed):\n"
        "    return {}\n"
        "\n"
        "\n"
        "def update(state, feedback, context):\n"
        "    return ({}, state)\n"
    )
    with pytest.raises(ProposerError, match="numeric constant"):
        random_propose(bare, random.Random(0))


def test_random_param_record_charges_no_tokens(tmp_path: Path) -> None:
    """The non-LLM track: no client, null model and temperature, zero tokens, no prompt blobs."""
    run = start_run(tmp_path / "ledgers", "improver-random")
    first = propose(run, proposer_id="random_param")
    second = propose(
        run,
        proposer_id="random_param",
        rng=random.Random(1),
        generation=1,
        parent_record_id=first.record_id,
        parent_source=blob_json(run.blobs, first.payload.child_artifact),
    )
    run.end("completed", {})

    for record in (first, second):
        payload = record.payload
        assert isinstance(payload, Proposal)
        assert payload.proposer == {
            "id": "random_param",
            "version": "1",
            "model": None,
            "temperature": None,
        }
        assert payload.prompt_blob is None
        assert payload.response_blob is None
        assert record.budget.llm_input_tokens == 0
        assert record.budget.llm_output_tokens == 0
        assert record.budget.wall_seconds >= 0
    assert (
        blob_json(run.blobs, first.payload.child_artifact)
        == random_propose(REFERENCE_ARTIFACT_SOURCE, random.Random(0)).child_source
    )
    assert second.refers_to == [first.record_id]
    assert validate_run(records_of(tmp_path / "ledgers", "improver-random")) == []


def test_unknown_proposer_and_missing_client_raise(tmp_path: Path) -> None:
    """An unknown id, or llm_edit without a client and config, is a caller error."""
    run = start_run(tmp_path / "ledgers", "improver-errors")
    with pytest.raises(ProposerError, match="unknown proposer"):
        propose(run, proposer_id="gaussian_walk")
    with pytest.raises(ProposerError, match="client"):
        propose(run, client=None)
    run.end("completed", {})


# --- Test 22: replay from recorded responses ---------------------------------------


def test_replay_reproduces_the_proposal_for_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A RecordedLLMClient replays the run's responses: same child, same diagnosis, zero cost."""
    monkeypatch.setenv("EVOLVETS_LLM_API_KEY", KEY)
    root = tmp_path / "ledgers"
    run = start_run(root, "improver-original")
    original = propose(
        run,
        client=ScriptedClient(LLMResponse(reply("mean start", EDITED_SOURCE), 100, 50, "end_turn")),
    )
    run.end("completed", {})

    responses = recorded_responses(run.blobs, records_of(root, "improver-original"))
    assert len(responses) == 1
    replay_run = start_run(root, "improver-replay")  # a sibling run, replayed from the records
    replay = propose(replay_run, client=RecordedLLMClient(responses))
    replay_run.end("completed", {})

    original_payload, replay_payload = original.payload, replay.payload
    assert isinstance(original_payload, Proposal)
    assert isinstance(replay_payload, Proposal)
    assert blob_json(replay_run.blobs, replay_payload.child_artifact) == blob_json(
        run.blobs, original_payload.child_artifact
    )
    assert replay_payload.diagnosis == original_payload.diagnosis
    assert replay.budget == Budget()  # zero tokens and zero wall: the replay is an audit
    assert replay_run.spent.llm_input_tokens == 0

    unknown_user = user_prompt(REFERENCE_ARTIFACT_SOURCE, empty_observation())
    unknown = prompt_key(IMPROVER_SYSTEM_PROMPT, unknown_user)
    with pytest.raises(LLMClientError) as excinfo:
        RecordedLLMClient({}).complete(IMPROVER_SYSTEM_PROMPT, unknown_user)
    assert sha256_hex(unknown.encode("utf-8")) in str(excinfo.value)  # names the missing key


# --- Test 23: the observation object ------------------------------------------------


def fixture_record(
    kind: str,
    payload: Any,
    *,
    seq: int,
    refers_to: list[str] | None = None,
    stream: str = "main",
) -> Record:
    """One hand-built record; the fixture never goes through a store, so hashes are free."""
    return make_record(
        stream=stream,
        seq=seq,
        prev_hash=ZERO_HASH if seq == 0 else "ab" * 32,
        run_id="obs-fixture",
        kind=kind,
        payload=payload,
        refers_to=refers_to or [],
        created_at=f"2026-10-20T12:{seq // 60:02d}:{seq % 60:02d}.000000Z",
    )


def put_artifact(blobs: BlobStore, text: str) -> str:
    """Store an artifact source the way a run does: canonical JSON of the text."""
    return blobs.put(canonical_bytes(text))


def chain_artifact(step: int) -> str:
    """The reference with the combination's starting level set to ``step / 100``."""
    return REFERENCE_ARTIFACT_SOURCE.replace("base = 0.0", f"base = {step / 100}")


def proposal_payload(parent: str, child: str, diagnosis: str) -> Proposal:
    """One proposal payload; the diff and observation blobs are not under test here."""
    return Proposal(
        target="updater",
        parent_artifact=parent,
        child_artifact=child,
        diff_blob=parent,
        observation_blob=parent,
        prompt_blob=None,
        response_blob=None,
        diagnosis=diagnosis,
        proposer={"id": "llm_edit", "version": "1", "model": None, "temperature": None},
        improver_version="1",
        planted_defect=None,
    )


def evaluation_payload(mase: float) -> Evaluation:
    """One ok dev evaluation with the given pooled MASE."""
    return Evaluation(
        pool="dev", status="ok", metrics={"mase": mase}, per_task_blob=None, error=None
    )


def acceptance_payload(accepted: bool) -> Acceptance:
    """One acceptance decision under the lineage's rule."""
    return Acceptance(accepted=accepted, rule="beats-parent-on-dev", new_artifact=None)


def test_observation_object_exact(tmp_path: Path) -> None:
    """Two generations, one failed evaluation, one accepted, one rejected: the exact object."""
    blobs = BlobStore(tmp_path / "blobs")
    reference = put_artifact(blobs, REFERENCE_ARTIFACT_SOURCE)
    first_child = put_artifact(blobs, chain_artifact(1))
    failed_child = put_artifact(blobs, chain_artifact(2))
    rejected_child = put_artifact(blobs, chain_artifact(3))
    p0 = fixture_record("proposal", proposal_payload(reference, first_child, "d0"), seq=0)
    e0 = fixture_record("evaluation", evaluation_payload(0.8), seq=1, refers_to=[p0.record_id])
    a0 = fixture_record(
        "acceptance", acceptance_payload(True), seq=2, refers_to=[p0.record_id, e0.record_id]
    )
    p1 = fixture_record(
        "proposal",
        proposal_payload(first_child, failed_child, "d1"),
        seq=3,
        refers_to=[p0.record_id],
    )
    e1 = fixture_record(
        "evaluation",
        Evaluation(pool="dev", status="failed", metrics={}, per_task_blob=None, error="crashed"),
        seq=4,
        refers_to=[p1.record_id],
    )
    p2 = fixture_record(
        "proposal",
        proposal_payload(first_child, rejected_child, "d2"),
        seq=5,
        refers_to=[p0.record_id],
    )
    e2 = fixture_record("evaluation", evaluation_payload(0.78), seq=6, refers_to=[p2.record_id])
    a2 = fixture_record(
        "acceptance", acceptance_payload(False), seq=7, refers_to=[p2.record_id, e2.record_id]
    )
    records = [p0, e0, a0, p1, e1, p2, e2, a2]

    observed = observation_object(blobs, records, generation=2)

    assert observed == {
        "generation": 2,
        "parent_metrics": {"mase": 0.8},  # the accepted artifact's own dev evaluation
        "history": [
            {
                "generation": 1,  # p2's parent is p0, the generation-0 proposal
                "family": artifact_family(chain_artifact(1), chain_artifact(3)),
                "gain": 0.8 - 0.78,  # parent minus child mase; positive is better
                "accepted": False,  # a2 rejected p2
            }
        ],
    }
    # p0 has no history entry: its parent, the starting artifact, has no dev
    # evaluation of its own in the lineage; p1 has none: its evaluation failed
    # generation 0: nothing has happened yet
    assert observation_object(blobs, [], generation=0) == {
        "generation": 0,
        "parent_metrics": {},
        "history": [],
    }
    # a sealed score can never reach a prompt, whatever stream it lives in
    sealed = fixture_record(
        "evaluation",
        Evaluation(
            pool="sealed", status="ok", metrics={"mase": 0.5}, per_task_blob=None, error=None
        ),
        seq=8,
        refers_to=[p2.record_id],
        stream="sealed",
    )
    with pytest.raises(ImproverError, match="sealed"):
        observation_object(blobs, [*records, sealed], generation=2)


def test_observation_history_is_capped_at_twenty(tmp_path: Path) -> None:
    """24 accepted generations: the history holds the last 20, gains from the mase pairs."""
    blobs = BlobStore(tmp_path / "blobs")
    artifacts = [put_artifact(blobs, REFERENCE_ARTIFACT_SOURCE)] + [
        put_artifact(blobs, chain_artifact(step)) for step in range(1, 25)
    ]
    records: list[Record] = []
    parent_id: str | None = None
    for step in range(24):
        proposal = fixture_record(
            "proposal",
            proposal_payload(artifacts[step], artifacts[step + 1], f"d{step}"),
            seq=step * 3,
            refers_to=[parent_id] if parent_id is not None else None,
        )
        evaluation = fixture_record(
            "evaluation",
            evaluation_payload(1.0 - 0.01 * step),
            seq=step * 3 + 1,
            refers_to=[proposal.record_id],
        )
        acceptance = fixture_record(
            "acceptance",
            acceptance_payload(True),
            seq=step * 3 + 2,
            refers_to=[proposal.record_id, evaluation.record_id],
        )
        records.extend((proposal, evaluation, acceptance))
        parent_id = proposal.record_id

    observed = observation_object(blobs, records, generation=24)

    history = observed["history"]
    assert isinstance(history, list)
    assert len(history) == 20  # 23 evaluated proposals exist; the last 20 are kept
    assert history[0]["generation"] == 4  # generations 4..23
    for entry, generation in zip(history, range(4, 24), strict=True):
        assert entry["generation"] == generation
        assert entry["family"] == "params"  # every edit in the chain changed only a constant
        assert entry["accepted"] is True
        # the same subtraction the implementation does, derived in the test
        assert entry["gain"] == (1.0 - 0.01 * (generation - 1)) - (1.0 - 0.01 * generation)
    assert observed["parent_metrics"] == {"mase": 1.0 - 0.01 * 23}


# --- Test 24: the prompts -----------------------------------------------------------


def test_prompts() -> None:
    """The user message embeds the observation's canonical JSON and the parent source verbatim."""
    observation = {
        "generation": 2,
        "parent_metrics": {"mase": 0.8},
        "history": [{"generation": 1, "family": "params", "gain": 0.02, "accepted": False}],
    }
    user = user_prompt(REFERENCE_ARTIFACT_SOURCE, observation)
    assert canonical_json(observation) in user
    assert REFERENCE_ARTIFACT_SOURCE in user
    assert user.index(canonical_json(observation)) < user.index(REFERENCE_ARTIFACT_SOURCE)
    for needle in (
        "exactly one JSON object",
        '"diagnosis"',
        '"source"',
        "init(n_experts, horizon, config, seed)",
        "update(state, feedback, context)",
        "No imports",
        "math is provided",
        "x.pop(k)",
        "x.append(v)",
        "raise ValueError(<one argument>)",
        "f-string",
        "No while",
        "comprehension",
        "state object",
        "discarded",
        "dev pooled MASE",
    ):
        assert needle in IMPROVER_SYSTEM_PROMPT, needle
    config = LLMClientConfig(model="glm-5.3", temperature=0.7, max_tokens=1024)
    assert prompt_object(config, "sys", "usr") == {
        "system": "sys",
        "user": "usr",
        "model": "glm-5.3",
        "temperature": 0.7,
        "max_tokens": 1024,
    }
