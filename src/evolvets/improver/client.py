"""The LLM client: standard-library HTTPS, retries and the audit trail (TASK_017 PR A2).

One :meth:`HttpsLLMClient.complete` is one POST to the Anthropic-compatible
Messages endpoint (``{base_url}/v1/messages``) with the pinned headers. The
transport, the sleep and the clock are injectable, so tests script every
response without a network — CI never calls the API; the real client runs only
on sp-1, in (f)'s lineage runs. Rate limits, server errors, transport errors
and malformed 2xx bodies are retried with exponential backoff; each wait is
recorded in the audit but never charged as compute (owner decision #143.4).
Any other 4xx fails immediately: the message carries the server's body, never
the key, which lives only in a request header and is read from the
environment at call time — a configuration names the *variable*
(``EVOLVETS_LLM_API_KEY``), never a value.

``prompt_key`` names a prompt by the canonical JSON of its two texts, and
``RecordedLLMClient`` serves each key's recorded outcomes first-in
first-out (owner decision #165: the three proposals of one generation share
one prompt and get three distinct replies): the replay path, which makes no
network call and charges zero tokens and zero wall. A recorded call that
raised is served as :class:`RecordedLLMFailure`: ``complete`` re-raises the
recorded ``LLMClientError`` message, with no audit of its own (the replay
attempted nothing), so the proposer re-derives the same fallback draft at
zero cost. A key whose replies are used up raises
:class:`RecordedRepliesExhausted` — deliberately not an ``LLMClientError``,
so the proposer cannot swallow it into a fallback no-op: a replay that asks
for a reply its records do not hold is a harness defect (AGENTS.md 2.8),
never a fabricated proposal.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from typing import cast

from evolvets.improver.interfaces import LLMResponse
from evolvets.ledger.canonical import canonical_json, sha256_hex

#: The Messages API version header the endpoint expects.
ANTHROPIC_VERSION = "2023-06-01"


@dataclasses.dataclass(frozen=True)
class LLMClientConfig:
    """Where and how to call the LLM; the key's environment-variable name, never a value."""

    base_url: str = "https://open.bigmodel.cn/api/anthropic"
    model: str = "glm-5.3"
    api_key_env: str = "EVOLVETS_LLM_API_KEY"
    timeout_seconds: float = 300.0
    max_attempts: int = 6  # >= 1: attempts per complete(), retries included
    backoff_base_seconds: float = 2.0
    backoff_cap_seconds: float = 120.0
    max_tokens: int = 8192
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError(f"max_attempts must be an int >= 1, got {self.max_attempts!r}")


@dataclasses.dataclass(frozen=True)
class LLMAttempt:
    """One HTTP attempt: whether it produced a usable response, and why not."""

    ok: bool
    status: int | None
    error: str | None


@dataclasses.dataclass(frozen=True)
class LLMCall:
    """The audit trail of one complete(): attempts, waits, tokens, chargeable wall time."""

    attempts: tuple[LLMAttempt, ...]
    waits_seconds: tuple[float, ...]
    input_tokens: int
    output_tokens: int
    wall_seconds: float  # monotonic elapsed minus every backoff wait


class LLMClientError(RuntimeError):
    """A call failed; ``call`` carries its audit, which never contains the key."""

    def __init__(self, message: str, call: LLMCall | None = None) -> None:
        super().__init__(message)
        self.call = call


class RecordedRepliesExhausted(RuntimeError):
    """A prompt key's recorded replies are used up: a harness defect, never a silent reuse.

    Deliberately not an :class:`LLMClientError` subclass, so the proposer's
    ``except LLMClientError`` cannot swallow exhaustion into a fallback
    no-op — a replay that asks for a reply its records do not hold is
    reported as one (AGENTS.md 2.8).
    """


def _clipped(body: str | None) -> str:
    """The first 500 characters of a body, for an error message (never the key)."""
    if body is None:
        return ""
    return body[:500]


def _usage_counts(body_text: str | None) -> tuple[int, int] | None:
    """(input_tokens, output_tokens) if the body carries a well-formed usage, else None.

    Any attempt that reported usage contributes its tokens to the audit, even
    a failed one (a 429 body often carries them).
    """
    if body_text is None:
        return None
    try:
        parsed: object = json.loads(body_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    usage = parsed.get("usage")
    if not isinstance(usage, dict):
        return None
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    for value in (input_tokens, output_tokens):
        if isinstance(value, bool) or not isinstance(value, int):
            return None
    return (cast("int", input_tokens), cast("int", output_tokens))


def _parse_success(body_text: str) -> tuple[LLMResponse, None] | tuple[None, str]:
    """The response built from a 2xx body, or the reason the body is not usable."""
    try:
        parsed: object = json.loads(body_text)
    except json.JSONDecodeError:
        return None, "the body is not JSON"
    if not isinstance(parsed, dict):
        return None, "the body is not a JSON object"
    content = parsed.get("content")
    if not isinstance(content, list):
        return None, "content is missing or not a list"
    texts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if not isinstance(text, str):
                return None, "a text block carries no string text"
            texts.append(text)
    if not texts:
        return None, "content has no text block"
    usage = _usage_counts(body_text)
    if usage is None:
        return None, "usage is missing or malformed"
    stop_reason = parsed.get("stop_reason")
    return (
        LLMResponse(
            text="".join(texts),
            input_tokens=usage[0],
            output_tokens=usage[1],
            stop_reason=stop_reason if isinstance(stop_reason, str) else None,
        ),
        None,
    )


def _urllib_post(url: str, key: str, version: str, body: str, timeout: float) -> tuple[int, str]:
    """POST ``body`` to ``url`` with the pinned headers — the default transport.

    An ``HTTPError`` (any 4xx/5xx) is returned as its status and body so the
    retry policy sees it; every other :class:`OSError` propagates and counts
    as a transport error.
    """
    request = urllib.request.Request(
        url,
        data=body.encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": version,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return int(error.code), error.read().decode("utf-8", "replace")


def prompt_key(system: str, user: str) -> str:
    """The canonical JSON of ``{"system", "user"}``: the identity of one prompt."""
    return canonical_json({"system": system, "user": user})


class HttpsLLMClient:
    """The real client: one POST per attempt, retries with backoff, a full audit."""

    def __init__(
        self,
        config: LLMClientConfig,
        *,
        post: Callable[..., tuple[int, str]] | None = None,  # tests inject; default urllib
        sleep: Callable[[float], None] | None = None,  # tests inject; default time.sleep
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._post = post if post is not None else _urllib_post
        self._sleep = sleep if sleep is not None else time.sleep
        self._monotonic = monotonic
        #: the audit of the last complete(), set on every exit path that attempted one
        self.last_call: LLMCall | None = None

    def complete(self, system: str, user: str) -> LLMResponse:
        """POST the prompt once per attempt until a usable 2xx body, then return it.

        Retried, after the recorded wait, on HTTP 429, HTTP >= 500, transport
        errors and 2xx bodies that are not the expected JSON; any other 4xx
        raises immediately with the status and the body's first 500
        characters. ``last_call`` always holds the audit of this call.
        """
        config = self._config
        key = os.environ.get(config.api_key_env)
        if key is None:
            raise LLMClientError(f"the environment variable {config.api_key_env} is not set")
        body = json.dumps(
            {
                "max_tokens": config.max_tokens,
                "messages": [{"role": "user", "content": user}],
                "model": config.model,
                "system": system,
                "temperature": config.temperature,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        url = config.base_url.rstrip("/") + "/v1/messages"
        attempts: list[LLMAttempt] = []
        waits: list[float] = []
        input_tokens = 0
        output_tokens = 0
        failure = "no attempt was made"
        start = self._monotonic()
        for index in range(config.max_attempts):
            status: int | None
            body_text: str | None
            try:
                status, body_text = self._post(
                    url, key, ANTHROPIC_VERSION, body, config.timeout_seconds
                )
            except OSError as error:
                attempts.append(LLMAttempt(ok=False, status=None, error=str(error)))
                failure = f"transport error: {error}"
            else:
                usage = _usage_counts(body_text)
                if usage is not None:
                    input_tokens += usage[0]
                    output_tokens += usage[1]
                if 200 <= status < 300:
                    response, reason = _parse_success(body_text)
                    if response is not None:
                        attempts.append(LLMAttempt(ok=True, status=status, error=None))
                        self.last_call = self._audit(
                            attempts, waits, input_tokens, output_tokens, start
                        )
                        return response
                    attempts.append(LLMAttempt(ok=False, status=status, error=reason))
                    failure = f"HTTP {status}: {reason}"
                elif status == 429:
                    attempts.append(LLMAttempt(ok=False, status=status, error="rate limited"))
                    failure = f"HTTP {status} rate limited"
                elif status >= 500:
                    attempts.append(LLMAttempt(ok=False, status=status, error="server error"))
                    failure = f"HTTP {status} server error"
                elif 400 <= status < 500:
                    attempts.append(LLMAttempt(ok=False, status=status, error="client error"))
                    self.last_call = self._audit(
                        attempts, waits, input_tokens, output_tokens, start
                    )
                    raise LLMClientError(
                        f"HTTP {status}: {_clipped(body_text)}", call=self.last_call
                    )
                else:
                    attempts.append(LLMAttempt(ok=False, status=status, error="unexpected status"))
                    failure = f"HTTP {status} unexpected status"
            if index + 1 == config.max_attempts:
                break
            wait = min(config.backoff_cap_seconds, config.backoff_base_seconds * 2**index)
            waits.append(wait)
            self._sleep(wait)
        self.last_call = self._audit(attempts, waits, input_tokens, output_tokens, start)
        raise LLMClientError(
            f"giving up after {config.max_attempts} attempts: {failure}", call=self.last_call
        )

    def _audit(
        self,
        attempts: list[LLMAttempt],
        waits: list[float],
        input_tokens: int,
        output_tokens: int,
        start: float,
    ) -> LLMCall:
        """The call's audit: monotonic elapsed minus every recorded wait (never negative)."""
        wall = self._monotonic() - start - sum(waits)
        return LLMCall(
            attempts=tuple(attempts),
            waits_seconds=tuple(waits),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            wall_seconds=wall if wall > 0.0 else 0.0,
        )


@dataclasses.dataclass(frozen=True)
class RecordedLLMFailure:
    """A recorded call that raised ``LLMClientError``: its message, replayed verbatim."""

    message: str


class RecordedLLMClient:
    """Serves each prompt key's recorded outcomes first-in first-out; no network call.

    One key can hold a whole sequence of outcomes — the three proposals of a
    generation share one prompt and get three distinct replies — and each
    :meth:`complete` consumes the front one, in recorded order. A replayed
    call charges zero tokens and zero wall (``last_call`` is a zero-cost
    audit), so a lineage re-run from recorded responses costs nothing — the
    replay is an audit, not a run. A key mapped to a
    :class:`RecordedLLMFailure` raises the recorded error instead, which is
    how a proposal whose call failed replays to the same fallback draft.
    """

    def __init__(self, responses: Mapping[str, Sequence[LLMResponse | RecordedLLMFailure]]) -> None:
        # every key's sequence is copied into the client's own deque, so the
        # caller's structures are never consumed or mutated
        self._responses: dict[str, deque[LLMResponse | RecordedLLMFailure]] = {
            key: deque(outcomes) for key, outcomes in responses.items()
        }
        #: the audit of the last complete(): a zero-cost synthetic success
        self.last_call: LLMCall | None = None

    def complete(self, system: str, user: str) -> LLMResponse:
        """Return the key's next recorded response in order, or raise.

        ``last_call`` is reset on entry and set only when a response is
        served — a replay that raised or found nothing attempted nothing, so
        there is no audit to expose. A recorded failure raises its recorded
        message; a key whose replies are used up raises
        :class:`RecordedRepliesExhausted` naming the key's SHA-256; an
        unknown key raises ``LLMClientError`` naming it.
        """
        key = prompt_key(system, user)
        self.last_call = None
        queue = self._responses.get(key)
        if queue is None:
            raise LLMClientError(
                "no recorded response for prompt key sha256:" + sha256_hex(key.encode("utf-8"))
            )
        if not queue:
            raise RecordedRepliesExhausted(
                "the recorded replies for prompt key sha256:"
                + sha256_hex(key.encode("utf-8"))
                + " are used up"
            )
        outcome = queue.popleft()
        if isinstance(outcome, RecordedLLMFailure):
            raise LLMClientError(outcome.message)
        self.last_call = LLMCall(
            attempts=(LLMAttempt(ok=True, status=None, error=None),),
            waits_seconds=(),
            input_tokens=0,
            output_tokens=0,
            wall_seconds=0.0,
        )
        return outcome
