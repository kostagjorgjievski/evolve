"""The improver's prompts: the system contract and the user message (TASK_017 PR A2).

The system prompt pins the reply contract (exactly one JSON object with a
``diagnosis`` and the complete new module ``source``) and the artifact rules
of ``improver/artifacts.py`` in words the model can follow — the validator
enforces them either way, so a reply that ignores them is simply discarded.
The user message is the fixed instruction block, the observation's canonical
JSON and the full parent source, each part clearly delimited, so a prompt is
a pure function of what the lineage has shown. ``prompt_object`` adds the
sampling parameters; its canonical bytes are the prompt blob, and
``prompt_key`` (client.py) names it.
"""

from __future__ import annotations

from collections.abc import Mapping

from evolvets.improver.client import LLMClientConfig
from evolvets.ledger.canonical import canonical_json

IMPROVER_SYSTEM_PROMPT = """You improve updater artifacts: small Python modules that combine \
and adapt cached expert forecasts online. Your task: edit the given forecast-v2 updater module \
so that its dev pooled MASE falls.

Reply with exactly one JSON object and nothing else — no prose, no markdown, no code fences:
{"diagnosis": "<a short string: what you changed and why>", "source": "<the complete new module \
source>"}

The artifact rules. Anything violating them is discarded:
- The module holds exactly two top-level functions, init(n_experts, horizon, config, seed) and \
update(state, feedback, context), and nothing else at module level (an optional docstring \
excepted).
- No imports at all. math is provided as a global: math.exp, math.sqrt, math.log, math.isfinite \
and math.fabs are the only math functions.
- Allowed calls: the pinned builtins abs, min, max, float, int, bool, len, range, tuple, list, \
dict, sorted, enumerate, zip, isinstance, str, ValueError; nested functions you define and call \
by name; and exactly the two one-argument method calls x.pop(k) and x.append(v). No other method \
call exists.
- The only raise form is raise ValueError(<one argument>); the message may be built from string \
constants and str() concatenation, but no f-string.
- Allowed operators: + - * / % //, unary - + not, comparisons, and, or, and the conditional \
expression (a if c else b). ** is not allowed.
- Allowed statements: def, for, if, return, plain and augmented assignment, pass, docstrings, \
and allowed calls as statements. No while, no try, no lambda, no comprehension (list, set, dict \
or generator), no walrus :=, no break or continue, no class, no delete, no global or nonlocal, \
no async.
- All state travels through the state object init returns and update threads through; the \
module itself stores nothing.
- Every numeric constant must be finite.

The "source" you reply with must be the complete module — never a diff, a fragment or an \
explanation."""


def user_prompt(parent_source: str, observation: Mapping[str, object]) -> str:
    """The user message: the instruction block, the observation's canonical JSON, the source."""
    return (
        "Improve the updater module below and reply with exactly one JSON object "
        '{"diagnosis": ..., "source": ...} and nothing else.\n'
        "\n=== OBSERVATION (canonical JSON) ===\n"
        + canonical_json(observation)
        + "\n\n=== CURRENT UPDATER SOURCE (edit this module) ===\n"
        + parent_source
        + "\n=== END OF SOURCE ==="
    )


def prompt_object(config: LLMClientConfig, system: str, user: str) -> dict[str, object]:
    """The prompt blob object: the two texts and the sampling parameters sent with them."""
    return {
        "system": system,
        "user": user,
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
