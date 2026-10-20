"""Fakes for the improver harness tests: scripted clients and transports.

``ScriptedClient`` stands in for an LLM client (CI never calls the API): it
answers scripted responses in order, records every prompt, and exposes a
zero-wait ``last_call`` mirroring each response's tokens, which is what the
proposer charges. ``ScriptedPost``/``ScriptedSleep``/``ScriptedMonotonic``
script ``HttpsLLMClient``'s three injectables so retry, backoff and
wall-time accounting are tested without a network and without sleeping.
"""

from dataclasses import dataclass, field

from evolvets.improver.client import LLMAttempt, LLMCall
from evolvets.improver.interfaces import LLMResponse


@dataclass
class SentPost:
    """One post() call exactly as the fake transport saw it."""

    url: str
    key: str
    version: str
    body: str
    timeout: float


class ScriptedPost:
    """post(url, key, version, body, timeout) scripted with replies, in order.

    A reply is a ``(status, body_text)`` tuple or an exception instance to
    raise (a transport error); every call is recorded in ``sent``.
    """

    def __init__(self, *replies: object) -> None:
        self.replies = list(replies)
        self.sent: list[SentPost] = []

    def __call__(
        self, url: str, key: str, version: str, body: str, timeout: float
    ) -> tuple[int, str]:
        self.sent.append(SentPost(url, key, version, body, timeout))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        status, body_text = reply  # type: ignore[misc]
        return status, body_text


@dataclass
class ScriptedSleep:
    """sleep(seconds) that records the waits instead of sleeping."""

    waits: list[float] = field(default_factory=list)

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


class ScriptedMonotonic:
    """monotonic() returning scripted values in order (each call consumes one)."""

    def __init__(self, *values: float) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        return self.values.pop(0)


class ScriptedClient:
    """An LLMClient answering scripted responses in order, recording every prompt.

    ``last_call`` mirrors the response's tokens with a single successful
    attempt and zero wait and wall, so the proposer charges exactly what the
    scripted response says it should.
    """

    def __init__(self, *responses: LLMResponse) -> None:
        self.responses = list(responses)
        self.prompts: list[tuple[str, str]] = []
        self.last_call: LLMCall | None = None

    def complete(self, system: str, user: str) -> LLMResponse:
        self.prompts.append((system, user))
        response = self.responses.pop(0)
        self.last_call = LLMCall(
            attempts=(LLMAttempt(ok=True, status=None, error=None),),
            waits_seconds=(),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            wall_seconds=0.0,
        )
        return response
