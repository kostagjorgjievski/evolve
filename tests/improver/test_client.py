"""Tests for the LLM client (TASK_017 PR A2: tests 12-16).

Every test scripts the transport, the sleep and the clock, so nothing here
touches a network or waits in real time. The retry policy — 429, >= 500,
transport errors and malformed 2xx bodies retried with recorded-but-uncharged
backoff; any other 4xx failing immediately with at most 500 body characters
and never the key — is pinned attempt by attempt.
"""

from __future__ import annotations

import dataclasses
import json
import urllib.error

import pytest

from evolvets.improver.client import (
    ANTHROPIC_VERSION,
    HttpsLLMClient,
    LLMAttempt,
    LLMCall,
    LLMClientConfig,
    LLMClientError,
    RecordedLLMClient,
    RecordedLLMFailure,
    RecordedRepliesExhausted,
    prompt_key,
)
from evolvets.improver.interfaces import LLMResponse
from evolvets.ledger.canonical import sha256_hex
from tests.improver.fake import ScriptedMonotonic, ScriptedPost, ScriptedSleep

KEY_ENV = "EVOLVETS_LLM_TEST_KEY"


def ok_body(text: str = "hello", input_tokens: int = 10, output_tokens: int = 5) -> str:
    """A well-formed 2xx body: one text block, usage, a stop reason."""
    return json.dumps(
        {
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "stop_reason": "end_turn",
        }
    )


def make_client(
    monotonic_values: tuple[float, float], **config
) -> tuple[HttpsLLMClient, ScriptedPost, ScriptedSleep]:
    """A client over scripted transports, with the key set for the test."""
    post, sleep = ScriptedPost(), ScriptedSleep()
    client = HttpsLLMClient(
        LLMClientConfig(api_key_env=KEY_ENV, **config),
        post=post,
        sleep=sleep,
        monotonic=ScriptedMonotonic(*monotonic_values),
    )
    return client, post, sleep


# --- Test 12: a success after one retry -------------------------------------------


def test_success_with_retry_after_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 429 then a 200: one wait of backoff_base_seconds, uncharged; both attempts audited."""
    monkeypatch.setenv(KEY_ENV, "sk-test-key")
    good = json.dumps(
        {
            "content": [
                {"type": "text", "text": "first"},
                {"type": "other", "data": 1},
                {"type": "text", "text": "second"},
            ],
            "usage": {"input_tokens": 11, "output_tokens": 7},
            "stop_reason": "end_turn",
        }
    )
    rate_limited = json.dumps({"usage": {"input_tokens": 3, "output_tokens": 2}})
    client, post, sleep = make_client((0.0, 10.0))
    post.replies.extend([(429, rate_limited), (200, good)])

    response = client.complete("be the improver", "edit this module")

    assert response.text == "firstsecond"  # the text blocks, in order, concatenated
    assert response.input_tokens == 11
    assert response.output_tokens == 7
    assert response.stop_reason == "end_turn"
    assert sleep.waits == [2.0]  # one wait of backoff_base_seconds
    audit = client.last_call
    assert audit is not None
    assert audit.attempts == (
        LLMAttempt(ok=False, status=429, error="rate limited"),
        LLMAttempt(ok=True, status=200, error=None),
    )
    assert audit.waits_seconds == (2.0,)
    assert audit.wall_seconds == 8.0  # 10.0 monotonic elapsed minus the one wait
    assert audit.input_tokens == 14  # the 429 reported usage too; both are summed
    assert audit.output_tokens == 9
    assert len(post.sent) == 2
    first = post.sent[0]
    assert first.url == "https://open.bigmodel.cn/api/anthropic/v1/messages"
    assert first.key == "sk-test-key"
    assert first.version == ANTHROPIC_VERSION
    assert first.timeout == 300.0
    body = json.loads(first.body)
    assert body["model"] == "glm-5.3"
    assert body["system"] == "be the improver"
    assert body["messages"] == [{"role": "user", "content": "edit this module"}]
    assert body["temperature"] == 1.0
    assert body["max_tokens"] == 8192
    assert first.body == json.dumps(body, sort_keys=True, separators=(",", ":"))


def test_success_with_retry_after_transport_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transport error then a 200 retries the same way: a failed attempt with no status."""
    monkeypatch.setenv(KEY_ENV, "sk-test-key")
    client, post, sleep = make_client((0.0, 4.0))
    post.replies.extend([urllib.error.URLError("no route to host"), (200, ok_body())])

    response = client.complete("system", "user")

    assert response.text == "hello"
    assert sleep.waits == [2.0]
    audit = client.last_call
    assert audit is not None
    assert audit.attempts[0].ok is False
    assert audit.attempts[0].status is None
    assert "no route to host" in str(audit.attempts[0].error)
    assert audit.attempts[1] == LLMAttempt(ok=True, status=200, error=None)


# --- Test 13: exhaustion ----------------------------------------------------------


def test_exhaustion_raises_with_the_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five 503s (max_attempts = 5) exhaust: four waits base * [1, 2, 4, 8], the status named."""
    monkeypatch.setenv(KEY_ENV, "sk-test-key")
    client, post, sleep = make_client((0.0, 100.0), max_attempts=5)
    post.replies.extend([(503, "upstream down")] * 5)

    with pytest.raises(LLMClientError) as excinfo:
        client.complete("system", "user")

    assert "giving up after 5 attempts" in str(excinfo.value)
    assert "503" in str(excinfo.value)  # the message names the status
    assert sleep.waits == [2.0, 4.0, 8.0, 16.0]
    audit = excinfo.value.call
    assert audit is not None
    assert len(audit.attempts) == 5
    assert all(attempt.ok is False and attempt.status == 503 for attempt in audit.attempts)
    assert audit.waits_seconds == (2.0, 4.0, 8.0, 16.0)
    assert audit.wall_seconds == 100.0 - 30.0  # elapsed minus every wait
    assert audit.input_tokens == 0  # no attempt reported usage


def test_backoff_waits_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The k-th wait is min(cap, base * 2 ** (k - 1)); with a small cap every wait after
    the first two equals it."""
    monkeypatch.setenv(KEY_ENV, "sk-test-key")
    client, post, sleep = make_client((0.0, 50.0), max_attempts=4, backoff_cap_seconds=5.0)
    post.replies.extend([(503, "down")] * 4)

    with pytest.raises(LLMClientError):
        client.complete("system", "user")

    assert sleep.waits == [2.0, 4.0, 5.0]


# --- Test 14: a non-retryable 4xx -------------------------------------------------


def test_client_error_is_immediate_and_never_leaks_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 400 raises at once: the status, the first 500 body characters, no waits, no key."""
    monkeypatch.setenv(KEY_ENV, "sk-test-secret-do-not-log")
    body = "e" * 550 + "TAIL"
    client, post, sleep = make_client((0.0, 1.0))
    post.replies.append((400, body))

    with pytest.raises(LLMClientError) as excinfo:
        client.complete("system", "user")

    message = str(excinfo.value)
    assert "400" in message
    assert "e" * 500 in message  # at most the first 500 characters
    assert "TAIL" not in message  # and never beyond them
    assert "sk-test-secret" not in message  # the key lives in a header, not the body
    assert sleep.waits == []  # immediate: no retry, no wait
    audit = excinfo.value.call
    assert audit is not None
    assert audit.attempts == (LLMAttempt(ok=False, status=400, error="client error"),)
    assert audit.waits_seconds == ()
    assert "sk-test-secret" not in json.dumps(dataclasses.asdict(audit))
    assert post.sent[0].key == "sk-test-secret-do-not-log"  # it was sent, never echoed


# --- Test 15: malformed 2xx bodies are retried ------------------------------------


def _malformed_bodies() -> list[tuple[str, str]]:
    """(label, body) for every way a 2xx body can fail the expected shape."""
    return [
        ("not-json", "oops {"),
        ("no-content", json.dumps({"usage": {"input_tokens": 1, "output_tokens": 1}})),
        ("no-text-block", json.dumps({"content": [{"type": "tool_use"}]})),
        (
            "no-usage",
            json.dumps({"content": [{"type": "text", "text": "hi"}]}),
        ),
    ]


_MALFORMED = _malformed_bodies()


@pytest.mark.parametrize(("label", "bad"), _MALFORMED, ids=[label for label, _ in _MALFORMED])
def test_malformed_2xx_is_retried_then_succeeds(
    label: str, bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2xx body that is not the expected JSON is a failed attempt; the next 200 succeeds."""
    monkeypatch.setenv(KEY_ENV, "sk-test-key")
    client, post, sleep = make_client((0.0, 5.0))
    post.replies.extend([(200, bad), (200, ok_body("recovered"))])

    response = client.complete("system", "user")

    assert response.text == "recovered"
    assert sleep.waits == [2.0]
    audit = client.last_call
    assert audit is not None
    assert audit.attempts[0].ok is False
    assert audit.attempts[0].status == 200
    assert audit.attempts[1].ok is True
    assert audit.attempts[1].status == 200


# --- Test 16: the missing key -----------------------------------------------------


def test_missing_key_names_the_variable_and_posts_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the environment variable, complete raises naming it and never posts."""
    monkeypatch.delenv("EVOLVETS_LLM_API_KEY", raising=False)
    post, sleep = ScriptedPost(), ScriptedSleep()
    client = HttpsLLMClient(
        LLMClientConfig(),
        post=post,
        sleep=sleep,
        monotonic=ScriptedMonotonic(0.0, 1.0),
    )

    with pytest.raises(LLMClientError, match="EVOLVETS_LLM_API_KEY"):
        client.complete("system", "user")

    assert post.sent == []
    assert sleep.waits == []
    assert client.last_call is None


def test_max_attempts_must_be_at_least_one() -> None:
    """The one constraint the config pins: max_attempts >= 1 (bools excluded)."""
    with pytest.raises(ValueError, match="max_attempts"):
        LLMClientConfig(max_attempts=0)
    with pytest.raises(ValueError, match="max_attempts"):
        LLMClientConfig(max_attempts=True)
    assert LLMClientConfig(max_attempts=1).max_attempts == 1


# --- TASK_018 PR D: the per-key FIFO recorded client --------------------------------


def zero_cost_audit() -> LLMCall:
    """The audit every served recorded response leaves: one ok attempt, nothing charged."""
    return LLMCall(
        attempts=(LLMAttempt(ok=True, status=None, error=None),),
        waits_seconds=(),
        input_tokens=0,
        output_tokens=0,
        wall_seconds=0.0,
    )


def test_three_distinct_replies_replay_in_recorded_order() -> None:
    """PR D test 1: replies served per key first-in first-out, each leaving a zero-cost audit.

    The three proposals of one generation share one prompt and get three
    distinct replies (owner decision #165); the replay serves them in
    recorded order — never reversed, repeated or collapsed to the last.
    """
    system, user = "be the improver", "edit this module"
    replies = [
        LLMResponse("first child", 1, 1, "end_turn"),
        LLMResponse("second child", 2, 2, "end_turn"),
        LLMResponse("third child", 3, 3, "end_turn"),
    ]
    client = RecordedLLMClient({prompt_key(system, user): list(replies)})
    for reply in replies:
        assert client.complete(system, user) == reply
        assert client.last_call == zero_cost_audit()


def test_exhausted_key_raises_naming_its_sha256_and_leaves_no_audit() -> None:
    """PR D test 2: the call after a key's last reply is loud, and other keys still serve."""
    system, user = "be the improver", "edit this module"
    key = prompt_key(system, user)
    other = prompt_key(system, "a prompt of another generation")
    client = RecordedLLMClient(
        {
            key: [LLMResponse("one", 1, 1, "end_turn"), LLMResponse("two", 1, 1, "end_turn")],
            other: [LLMResponse("elsewhere", 1, 1, "end_turn")],
        }
    )
    client.complete(system, user)
    client.complete(system, user)
    with pytest.raises(RecordedRepliesExhausted) as excinfo:
        client.complete(system, user)
    assert sha256_hex(key.encode("utf-8")) in str(excinfo.value)  # names the used-up key
    assert client.last_call is None  # reset on entry; the replay attempted nothing
    assert client.complete(system, "a prompt of another generation").text == "elsewhere"


def test_exhaustion_is_not_an_llm_client_error() -> None:
    """PR D test 3: pinned so the proposer can never swallow exhaustion into a no-op.

    A key mapped to an empty sequence exhausts on its first call — the same
    loud path, not a silent reuse of anything.
    """
    system, user = "be the improver", "edit this module"
    client = RecordedLLMClient({prompt_key(system, user): []})
    with pytest.raises(RecordedRepliesExhausted):
        client.complete(system, user)
    assert not isinstance(RecordedRepliesExhausted("used up"), LLMClientError)


def test_recorded_failure_consumes_its_own_slot() -> None:
    """PR D test 4: a key recorded [failure, response] raises first, serves second."""
    system, user = "be the improver", "edit this module"
    key = prompt_key(system, user)
    response = LLMResponse("the recovery", 4, 2, "end_turn")
    client = RecordedLLMClient({key: [RecordedLLMFailure("HTTP 429: rate limited"), response]})
    with pytest.raises(LLMClientError, match="HTTP 429: rate limited"):
        client.complete(system, user)
    assert client.last_call is None  # the replay attempted nothing
    assert client.complete(system, user) == response  # the failure consumed its own slot


def test_constructor_copies_every_sequence() -> None:
    """PR D test 6: the client serves from its own deques; the caller's lists never change."""
    system, user = "be the improver", "edit this module"
    key = prompt_key(system, user)
    other = prompt_key(system, "a prompt of another generation")
    replies = [LLMResponse("first", 1, 1, "end_turn"), LLMResponse("second", 1, 1, "end_turn")]
    failures = [RecordedLLMFailure("boom")]
    handed_over = {key: replies, other: failures}
    client = RecordedLLMClient(handed_over)
    client.complete(system, user)
    client.complete(system, user)
    with pytest.raises(LLMClientError, match="boom"):
        client.complete(system, "a prompt of another generation")
    assert replies == [
        LLMResponse("first", 1, 1, "end_turn"),
        LLMResponse("second", 1, 1, "end_turn"),
    ]  # bit-unchanged, both entries still there
    assert failures == [RecordedLLMFailure("boom")]
    assert handed_over == {key: replies, other: failures}
