"""Tests for the improver's artifact object space (TASK_017 PR A1: tests 1-11).

The artifact grammar is pinned by rejection cases that each name their
construct, one accept-and-run case that exercises every allowed construct, a
sandbox check that the loader's namespace is exactly the pinned builtins and
``math``, and the reference artifact: bias_share translated into the grammar,
which must replay bit-equal to the class original. The rejection cases
include the escape routes closed after the PR A1 review — rebinding a
pinned name to smuggle a method under an allowed call name, handing a
method to a builtin through a keyword argument, and (round 2)
``__builtins__``: the sandbox's builtins dict, loadable as a name and
rebindable by a subscript store, which is why no dunder name is anywhere.
The unit drives use ``tests/updater/test_forecast.py``'s helpers; the replay
test builds the same synthetic-dev fixture pool as
``tests/updater/test_baselines_replay.py``.
"""

from __future__ import annotations

import ast
import math
import random
import re
from hashlib import sha256
from typing import Any

import pytest

from evolvets.arena.dev import load_dev_pool
from evolvets.arena.evaluate import replay_dataset, result_to_obj
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.improver.artifacts import (
    ARTIFACT_BUILTINS,
    MATH_MEMBERS,
    REFERENCE_ARTIFACT_CONFIG,
    REFERENCE_ARTIFACT_SOURCE,
    ArtifactError,
    artifact_family,
    artifact_updater,
    const_change_stats,
    diff_lines,
    normalized_source,
    validate_artifact_source,
)
from evolvets.improver.interfaces import FEATURE_NAMES, ProposalFeatures, feature_vector
from evolvets.ledger.canonical import canonical_bytes
from evolvets.updater.forecast import FORECAST_DEFAULT_CONFIGS, bias_share
from evolvets.updater.interfaces import ForecastOutput
from tests.updater.test_forecast import GOLDEN_DIGEST, matured, step_context

CONFIG: dict[str, Any] = dict(REFERENCE_ARTIFACT_CONFIG)


def _reference() -> Any:
    """A fresh adapter around the reference artifact's namespace."""
    return artifact_updater(REFERENCE_ARTIFACT_SOURCE)()


# --- Test 1: the reference artifact is valid and pinned ---------------------------

#: The SHA-256 of REFERENCE_ARTIFACT_SOURCE, pinned by TASK_017 (PR A1). The
#: reference is the lineage's starting point: every Phase 1 proposal is an edit
#: of it, so its bytes are part of the experiment's definition, like a seed or
#: a config. The equivalence test below (test 2) only means something against
#: an immutable starting text; changing this digest needs a reason in the PR.
REFERENCE_SHA256 = "3334293f80615f44897e2fd01aa937225b32ea9af763067ba9ff46addd650f3c"


def test_reference_artifact_is_valid_and_pinned() -> None:
    """The reference validates, is a fixed point of normalized_source, and its bytes are pinned."""
    validate_artifact_source(REFERENCE_ARTIFACT_SOURCE)
    assert normalized_source(REFERENCE_ARTIFACT_SOURCE) == REFERENCE_ARTIFACT_SOURCE
    assert sha256(REFERENCE_ARTIFACT_SOURCE.encode("utf-8")).hexdigest() == REFERENCE_SHA256


@pytest.fixture(scope="module")
def synthetic_pool(tmp_path_factory: pytest.TempPathFactory):
    """The dev pool of a synthetic-dev cache with one series per family (TASK_007 fixture)."""
    base = tmp_path_factory.mktemp("improver-artifacts")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-dev", base, base, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    return load_dev_pool("synthetic-dev", cache_root, base, base)


def test_reference_artifact_replays_bit_equal_to_bias_share(synthetic_pool) -> None:
    """The reference replays synthetic-dev exactly as bias_share: metrics, every MASE, one digest.

    The artifact grammar forces the same operations in the same order as
    updater/forecast.py and _share_step, so the translation is not merely
    close: it is bit-equal (the same canonical digest), which is what the float
    discipline of updater/baselines.py was for. That is the anchor that lets a
    lineage start at the artifact and still speak the Gate 1 world's numbers.
    """
    reference = replay_dataset(
        artifact_updater(REFERENCE_ARTIFACT_SOURCE),
        REFERENCE_ARTIFACT_CONFIG,
        0,
        synthetic_pool,
        interface="forecast-v2",
    )
    baseline = replay_dataset(
        bias_share,
        FORECAST_DEFAULT_CONFIGS["bias_share"],
        0,
        synthetic_pool,
        interface="forecast-v2",
    )
    assert reference.status == "ok"
    assert reference.metrics == baseline.metrics
    assert [score.mase for score in reference.series] == [score.mase for score in baseline.series]
    assert result_to_obj(reference) == result_to_obj(baseline)
    digest = sha256(canonical_bytes(result_to_obj(reference))).hexdigest()
    assert digest == GOLDEN_DIGEST  # tests/updater/test_forecast.py pinned it for bias_share


# --- Test 3: config parity --------------------------------------------------------


def _bad_init_cases() -> list[tuple[str, dict[str, Any], int, int, Any, str]]:
    """(label, config, n_experts, horizon, seed, message needle) for every init refusal."""
    return [
        ("missing-key", {k: v for k, v in CONFIG.items() if k != "alpha"}, 2, 2, 0, "config keys"),
        ("extra-key", CONFIG | {"gamma": 0.5}, 2, 2, 0, "config keys"),
        ("eta-zero", CONFIG | {"eta": 0.0}, 2, 2, 0, "eta"),
        ("eta-bool", CONFIG | {"eta": True}, 2, 2, 0, "eta"),
        ("alpha-negative", CONFIG | {"alpha": -0.1}, 2, 2, 0, "alpha"),
        ("alpha-above-one", CONFIG | {"alpha": 1.5}, 2, 2, 0, "alpha"),
        ("rate-negative", CONFIG | {"bias_rate": -0.1}, 2, 2, 0, "bias_rate"),
        ("rate-above-one", CONFIG | {"bias_rate": 1.5}, 2, 2, 0, "bias_rate"),
        ("rate-bool", CONFIG | {"bias_rate": True}, 2, 2, 0, "bias_rate"),
        ("experts-zero", CONFIG, 0, 2, 0, "n_experts"),
        ("horizon-zero", CONFIG, 2, 0, 0, "horizon"),
        ("seed-bool", CONFIG, 2, 2, True, "seed"),
    ]


_INIT_CASES = _bad_init_cases()


@pytest.mark.parametrize(
    ("label", "config", "n_experts", "horizon", "seed", "needle"),
    _INIT_CASES,
    ids=[label for label, *_ in _INIT_CASES],
)
def test_reference_init_rejects(
    label: str, config: dict[str, Any], n_experts: int, horizon: int, seed: Any, needle: str
) -> None:
    """The artifact's init refuses every bad argument bias_share's init refuses."""
    with pytest.raises(ValueError, match=needle):
        _reference().init(n_experts, horizon, config, seed)


def test_reference_seed_value_is_ignored() -> None:
    """Two different int seeds build identical states and drive identical outputs."""
    first, second = _reference(), _reference()
    state_first = first.init(2, 2, CONFIG, 0)
    state_second = second.init(2, 2, CONFIG, 987654)
    assert state_first == state_second
    context = step_context(2, 2, 0)
    out_first, next_first = first.update(state_first, (), context)
    out_second, next_second = second.update(state_second, (), context)
    assert out_first == out_second
    assert next_first == next_second


# --- Test 4: update parity edges --------------------------------------------------


def test_reference_missing_step_error_is_exact() -> None:
    """A Matured whose step never had a bias applied raises exactly forecast.py:114's error."""
    updater = _reference()
    state = updater.init(2, 2, CONFIG, 0)
    bogus = matured([[-0.1, 0.2], [0.3, -0.4]], [0.0, 0.0], step=5)
    with pytest.raises(ValueError) as excinfo:
        updater.update(state, (bogus,), step_context(2, 2, 0))
    assert str(excinfo.value) == "no update call ever applied a bias at step 5"


def test_reference_two_states_driven_interleaved() -> None:
    """Two states of one adapter driven interleaved match two adapters driven separately."""
    rng_a, rng_b = random.Random(1), random.Random(2)
    n_experts, horizon, steps = 2, 2, 8
    shared = _reference()
    state_a = shared.init(n_experts, horizon, CONFIG, 0)
    state_b = shared.init(n_experts, horizon, CONFIG, 1)
    alone_a, alone_b = _reference(), _reference()
    alone_state_a = alone_a.init(n_experts, horizon, CONFIG, 0)
    alone_state_b = alone_b.init(n_experts, horizon, CONFIG, 1)
    _, state_a = shared.update(state_a, (), step_context(n_experts, horizon, 0))
    _, state_b = shared.update(state_b, (), step_context(n_experts, horizon, 0))
    _, alone_state_a = alone_a.update(alone_state_a, (), step_context(n_experts, horizon, 0))
    _, alone_state_b = alone_b.update(alone_state_b, (), step_context(n_experts, horizon, 0))
    for k in range(1, steps + 1):
        rows_a = [[rng_a.uniform(-1.0, 1.0) for _ in range(horizon)] for _ in range(n_experts)]
        rows_b = [[rng_b.uniform(-1.0, 1.0) for _ in range(horizon)] for _ in range(n_experts)]
        forecasts_a = tuple(tuple(row) for row in rows_a)
        forecasts_b = tuple(tuple(row) for row in rows_b)
        feedback_a = (matured(rows_a, [0.0] * horizon, step=k - 1),)
        feedback_b = (matured(rows_b, [0.1] * horizon, step=k - 1),)
        interleaved_a, state_a = shared.update(
            state_a, feedback_a, step_context(n_experts, horizon, k, forecasts_a)
        )
        interleaved_b, state_b = shared.update(
            state_b, feedback_b, step_context(n_experts, horizon, k, forecasts_b)
        )
        separate_a, alone_state_a = alone_a.update(
            alone_state_a, feedback_a, step_context(n_experts, horizon, k, forecasts_a)
        )
        separate_b, alone_state_b = alone_b.update(
            alone_state_b, feedback_b, step_context(n_experts, horizon, k, forecasts_b)
        )
        assert interleaved_a == separate_a
        assert interleaved_b == separate_b


# --- Tests 5 and 6: the grammar ---------------------------------------------------


def _artifact(
    init_body: str = "    return {}\n",
    update_body: str = "    return ({}, state)\n",
    module_tail: str = "",
) -> str:
    """An artifact with the given indented bodies, plus an optional module tail."""
    return (
        f"def init(n_experts, horizon, config, seed):\n{init_body}"
        f"\n\ndef update(state, feedback, context):\n{update_body}{module_tail}"
    )


def _rejects() -> list[tuple[str, str, str]]:
    """(label, source, message needle) for every construct the grammar rejects."""
    return [
        ("import-os", _artifact(module_tail="import os\n"), "import"),
        ("import-math", _artifact(module_tail="import math\n"), "import"),
        ("from-math-import", _artifact(module_tail="from math import exp\n"), "import"),
        (
            "while",
            _artifact(update_body="    while True:\n        pass\n    return ({}, state)\n"),
            "while",
        ),
        (
            "try",
            _artifact(
                update_body="    try:\n        pass\n    except ValueError:\n"
                "        pass\n    return ({}, state)\n"
            ),
            "try",
        ),
        (
            "raise-runtime-error",
            _artifact(update_body='    raise RuntimeError("boom")\n'),
            "RuntimeError",
        ),
        ("bare-raise", _artifact(update_body="    raise\n"), "bare raise"),
        (
            "raise-from",
            _artifact(update_body='    raise ValueError("x") from None\n'),
            "from",
        ),
        (
            "lambda",
            _artifact(update_body="    double = lambda x: x + x\n    return ({}, state)\n"),
            "lambda",
        ),
        (
            "list-comprehension",
            _artifact(update_body="    xs = [x for x in range(3)]\n    return ({}, state)\n"),
            "comprehension",
        ),
        (
            "walrus",
            _artifact(update_body="    if (n := 1) > 0:\n        pass\n    return ({}, state)\n"),
            "walrus",
        ),
        (
            "break",
            _artifact(
                update_body="    for x in range(3):\n        break\n    return ({}, state)\n"
            ),
            "break",
        ),
        ("class", _artifact(module_tail="class Box:\n    pass\n"), "class"),
        (
            "power",
            _artifact(update_body="    value = 2 ** 3\n    return ({}, state)\n"),
            "**",
        ),
        (
            "call-sum",
            _artifact(update_body="    value = sum([1, 2])\n    return ({}, state)\n"),
            "sum",
        ),
        (
            "call-open",
            _artifact(update_body='    value = open("x")\n    return ({}, state)\n'),
            "open",
        ),
        (
            "call-print",
            _artifact(update_body='    value = print("x")\n    return ({}, state)\n'),
            "print",
        ),
        (
            "dunder-attribute",
            _artifact(update_body="    value = state.__dict__\n    return ({}, state)\n"),
            "__dict__",
        ),
        (
            "math-pow",
            _artifact(update_body="    value = math.pow(2.0, 3.0)\n    return ({}, state)\n"),
            "pow",
        ),
        (
            "math-pop",
            _artifact(update_body="    value = math.pop(1)\n    return ({}, state)\n"),
            "math.pop",
        ),
        (
            "method-sort",
            _artifact(update_body="    xs = [1.0]\n    xs.sort()\n    return ({}, state)\n"),
            "sort",
        ),
        (
            "method-join",
            _artifact(update_body='    value = "".join(["a"])\n    return ({}, state)\n'),
            "join",
        ),
        (
            "pop-two-arguments",
            _artifact(update_body='    value = {}.pop("a", 1)\n    return ({}, state)\n'),
            "pop",
        ),
        (
            "nonfinite-constant",
            _artifact(update_body="    value = 1e400\n    return ({}, state)\n"),
            "finite",
        ),
        ("module-assignment", _artifact(module_tail="X = 1\n"), "assignment"),
        ("third-top-level-def", _artifact(module_tail="def helper(x):\n    return x\n"), "helper"),
        (
            "missing-update",
            "def init(n_experts, horizon, config, seed):\n    return {}\n",
            "update",
        ),
        ("missing-init", "def update(state, feedback, context):\n    return ({}, state)\n", "init"),
        (
            "f-string",
            _artifact(update_body='    value = f"x{n_experts}"\n    return ({}, state)\n'),
            "f-string",
        ),
        (
            "call-through-expression",
            _artifact(update_body='    value = (state["f"])(1)\n    return ({}, state)\n'),
            "expression",
        ),
        # --- the rebind and method-as-value routes (PR A1 review) ------------------
        # A name the call rule resolves to is never rebindable, by any binding form.
        (
            "format-string-bind",
            _artifact(
                update_body='    len = "{0.__init__.__globals__[sys].modules}".format\n'
                "    return ({}, state)\n"
            ),
            "an assignment must not bind the pinned name 'len'",
        ),
        (
            "rebind-builtin-assign",
            _artifact(
                update_body="    sorted = config.clear\n    sorted()\n    return ({}, state)\n"
            ),
            "an assignment must not bind the pinned name 'sorted'",
        ),
        (
            "rebind-builtin-for-target",
            _artifact(
                update_body="    for len in [config.clear]:\n        pass\n    return ({}, state)\n"
            ),
            "a for-loop target must not bind the pinned name 'len'",
        ),
        (
            "rebind-builtin-tuple-target",
            _artifact(
                update_body="    (abs, x) = (config.clear, 1)\n    abs()\n    return ({}, state)\n"
            ),
            "an assignment must not bind the pinned name 'abs'",
        ),
        (
            "rebind-builtin-augassign",
            _artifact(update_body="    len += 1\n    return ({}, state)\n"),
            "an augmented assignment must not bind the pinned name 'len'",
        ),
        (
            "rebind-own-function",
            _artifact(
                update_body="    def helper():\n        pass\n"
                "    helper = config.clear\n    helper()\n    return ({}, state)\n"
            ),
            "an assignment must not bind the pinned name 'helper'",
        ),
        (
            "rebind-math",
            _artifact(
                update_body="    math = config\n    value = math.exp(0.5)\n    return ({}, state)\n"
            ),
            "an assignment must not bind the pinned name 'math'",
        ),
        (
            "nested-param-shadows",
            _artifact(
                update_body="    def helper(sorted):\n        pass\n"
                "    helper(config.clear)\n    return ({}, state)\n"
            ),
            "the parameter 'sorted' of helper shadows a pinned name",
        ),
        (
            "nested-def-named-builtin",
            _artifact(update_body="    def len(x):\n        return x\n    return ({}, state)\n"),
            "a nested def must not be named 'len'",
        ),
        (
            "nested-def-shadows-init",
            _artifact(update_body="    def init(x):\n        return x\n    return ({}, state)\n"),
            "the nested def 'init' shadows another function of the artifact",
        ),
        (
            "duplicate-nested-def",
            _artifact(
                update_body="    def step(x):\n        pass\n\n"
                "    def step(y):\n        pass\n    return ({}, state)\n"
            ),
            "the nested def 'step' shadows another function of the artifact",
        ),
        (
            "cross-function-helper-call",
            _artifact(
                init_body="    helper(1)\n    return {}\n",
                update_body="    def helper(x):\n        pass\n    return ({}, state)\n",
            ),
            "the call to 'helper' is not allowed",
        ),
        # Methods are not values, and reflection has no route in at all.
        (
            "format-attribute-load",
            _artifact(update_body='    value = "{0.x}".format\n    return ({}, state)\n'),
            ".format is not allowed anywhere",
        ),
        (
            "format-map-attribute-load",
            _artifact(update_body='    value = "{0.x}".format_map\n    return ({}, state)\n'),
            ".format_map is not allowed anywhere",
        ),
        (
            "pop-as-value",
            _artifact(update_body="    xs = [1.0]\n    value = xs.pop\n    return ({}, state)\n"),
            ".pop is a method call, not a value",
        ),
        (
            "append-as-value",
            _artifact(
                update_body="    xs = [1.0]\n    value = xs.append\n    return ({}, state)\n"
            ),
            ".append is a method call, not a value",
        ),
        (
            "keyword-argument",
            _artifact(
                update_body="    value = sorted([1], key=config.get)\n    return ({}, state)\n"
            ),
            "keyword arguments in a call are not allowed",
        ),
        # --- the __builtins__ route (PR A1 review, round 2) ------------------------
        # __builtins__ resolves to the sandbox's builtins dict, so loading it
        # would let an artifact rebind any pinned builtin by a subscript store
        # (config.clear running as len()) or keep cross-series state in the
        # dict. No dunder name may be loaded or bound, by any form.
        (
            "builtins-subscript-store",
            _artifact(
                init_body='    __builtins__["len"] = config.clear\n    len()\n    return {}\n'
            ),
            "the dunder name '__builtins__' may not be loaded",
        ),
        (
            "builtins-alias-load",
            _artifact(update_body="    b = __builtins__\n    return ({}, state)\n"),
            "the dunder name '__builtins__' may not be loaded",
        ),
        (
            "builtins-bare-load",
            _artifact(update_body="    value = __builtins__\n    return ({}, state)\n"),
            "the dunder name '__builtins__' may not be loaded",
        ),
        (
            "builtins-name-bind",
            _artifact(update_body="    __builtins__ = {}\n    return ({}, state)\n"),
            "an assignment must not bind the dunder name '__builtins__'",
        ),
        (
            "builtins-for-target",
            _artifact(
                update_body="    for __builtins__ in [1]:\n        pass\n    return ({}, state)\n"
            ),
            "a for-loop target must not bind the dunder name '__builtins__'",
        ),
        (
            "builtins-nested-param",
            _artifact(
                update_body="    def helper(__builtins__):\n        pass\n    return ({}, state)\n"
            ),
            "the parameter '__builtins__' of helper is a dunder name",
        ),
        (
            "builtins-nested-def-name",
            _artifact(
                update_body="    def __len__(x):\n        return x\n    return ({}, state)\n"
            ),
            "a nested def must not be named '__len__'",
        ),
    ]


_REJECT_CASES = _rejects()


def test_artifact_error_is_a_value_error() -> None:
    """ArtifactError subclasses ValueError, so the arena reports it as a failed replay."""
    assert issubclass(ArtifactError, ValueError)


@pytest.mark.parametrize(
    ("label", "source", "needle"),
    _REJECT_CASES,
    ids=[label for label, _, _ in _REJECT_CASES],
)
def test_validator_rejects(label: str, source: str, needle: str) -> None:
    """Every construct outside the grammar is rejected, its name in the message."""
    with pytest.raises(ArtifactError, match=re.escape(needle)):
        validate_artifact_source(source)


#: The PR A1 review payload: ``str.format`` bound to a pinned name, used to
#: walk ``__init__.__globals__`` chains from ``context`` into ``sys.modules``.
#: It is ordinary Python and runs happily in an unrestricted namespace — a
#: read path from artifact code into modules it must never see — so the
#: grammar has to refuse it before any of that matters.
FORMAT_SYS_MODULES_PAYLOAD = """def init(n_experts, horizon, config, seed):
    return {"h": horizon}


def update(state, feedback, context):
    len = "{0.__init__.__globals__[dataclass].__globals__[sys].modules}".format
    t = len(context)
    max = t.find
    state["mods"] = t[max("evolvets.arena"):]
    return ([0.0] * state["h"], state)
"""


def test_format_sys_modules_payload_is_rejected_before_exec() -> None:
    """The format/sys.modules payload is refused at validation, before compile or exec.

    The payload executes cleanly if it is ever run, so the only way this can
    raise the validator's own error is that validation ran first: the walk
    names the first pinned-name binding, and the loader refuses the source
    without ever compiling it.
    """
    with pytest.raises(ArtifactError, match=re.escape("pinned name 'len'")):
        validate_artifact_source(FORMAT_SYS_MODULES_PAYLOAD)
    with pytest.raises(ArtifactError, match=re.escape("pinned name 'len'")):
        artifact_updater(FORMAT_SYS_MODULES_PAYLOAD)


#: The PR A1 round-2 review payload: ``__builtins__`` is the sandbox's builtins
#: dict, so binding ``len`` in it by a subscript store re-opens every method
#: call (``config.clear`` runs as ``len()``) and the dict is state outside the
#: state object. The dunder-name rule refuses it first.
BUILTINS_REBIND_PAYLOAD = """def init(n_experts, horizon, config, seed):
    __builtins__["len"] = config.clear
    len()
    return {"h": horizon}


def update(state, feedback, context):
    return ([0.0] * state["h"], state)
"""


def test_builtins_rebind_payload_is_rejected_before_exec() -> None:
    """The __builtins__ rebind payload is refused at validation, before compile or exec.

    The payload empties the caller's config as soon as ``init`` runs (round 2
    of review demonstrated it), so the only way this can raise the validator's
    own error is that validation ran first: the loader refuses the source
    without ever compiling it, so no adapter — and no ``init`` — ever exists.
    """
    with pytest.raises(ArtifactError, match=re.escape("the dunder name '__builtins__'")):
        validate_artifact_source(BUILTINS_REBIND_PAYLOAD)
    with pytest.raises(ArtifactError, match=re.escape("the dunder name '__builtins__'")):
        artifact_updater(BUILTINS_REBIND_PAYLOAD)


def test_helper_scope_is_per_function() -> None:
    """Helpers are callable anywhere in the function that defines them; the two top-level
    defs stay callable from each other; a helper of one is not callable from the other."""
    siblings = _artifact(
        update_body=(
            "    def first(x):\n"
            "        return second(x)\n"
            "\n"
            "    def second(y):\n"
            "        return y\n"
            "    value = first(1)\n"
            "    return ({}, state)\n"
        )
    )
    validate_artifact_source(siblings)
    top_level = _artifact(update_body="    state = init(1, 1, {}, 0)\n    return ({}, state)\n")
    validate_artifact_source(top_level)  # init and update live in the one sandbox namespace


#: One artifact exercising every allowed construct: the pinned builtins, math's
#: members, nested helpers (called by name, for their side effects), every
#: allowed statement and expression form, and both raise shapes.
KITCHEN_SINK = '''"""A module docstring."""


def init(n_experts, horizon, config, seed):
    "Exercise every allowed construct once."

    def checked(name, value):
        "A nested helper, called by name."
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(name + " must be a number")
        if not math.isfinite(value):
            raise ValueError(name + " must be finite")
        return float(value)

    d = {}
    xs = []
    row = [1.0, 2.0]
    pair = (1, 2.5)
    d["constants"] = (0, 1, -2, 3.5, True, False, "text")
    d["tuple"] = pair
    d["subscript"] = row[1]
    d["slice"] = row[0:1]
    d["step_slice"] = row[0:2:1]
    d["arithmetic"] = (1 + 2) - 0.5 * 4 / 2 % 3 // 1
    d["unary"] = (-row[0], +1, not True)
    d["compare"] = (0 < 1, 1 <= 1, 2 > 1, 2 >= 2, 1 == 1, 1 != 2)
    d["identity"] = True is True
    d["membership"] = ("a" in "abc", "z" not in "abc")
    d["logic"] = (True and False) or not False
    d["ifexp"] = "yes" if n_experts >= 1 else "no"
    d["math"] = (math.exp(0.0), math.sqrt(4.0), math.log(1.0))
    d["isfinite"] = math.isfinite(1.0)
    d["fabs"] = math.fabs(-1.0)
    d["builtins"] = (
        abs(-1.0),
        min(1.0, 2.0),
        max(1.0, 2.0),
        float(1),
        int(1.0),
        bool(0),
        len(pair),
        range(2),
        tuple(row),
        list(pair),
        dict(),
        sorted(row),
        enumerate(pair),
        zip(pair, row),
        isinstance(1, bool),
        str(1),
        ValueError("never raised"),
    )
    d["eta"] = checked("eta", config["eta"])
    if len(d) == -1:
        raise ValueError("eta must be > 0")
    if len(d) == -2:
        raise ValueError("at step " + str(seed))
    for value in row:
        xs.append(value)
    scores = {"a": 1.0}
    scores.pop("a")
    if not xs:
        pass
    total = 0.0
    for index, value in enumerate(xs):
        total = total + value
    for left, right in zip(xs, row):
        total += left * right
    state = {"d": d, "xs": xs, "total": total, "seen": sorted(xs)}
    state["extra"] = 0
    state["extra"] += 1
    return state


def update(state, feedback, context):
    "Run the same kinds of expressions on the update side."

    def note(state, value):
        "A side-effecting helper invoked as a statement."
        state["last"] = value

    note(state, context.step)
    out = []
    for h in range(context.horizon):
        base = 0.0
        for e in range(len(context.forecasts)):
            base = base + context.forecasts[e][h]
        out.append(base)
    return (tuple(out), state)
'''


def test_kitchen_sink_validates_and_runs() -> None:
    """An artifact using every allowed construct validates, loads and runs in the sandbox."""
    validate_artifact_source(KITCHEN_SINK)
    updater = artifact_updater(KITCHEN_SINK)()
    state = updater.init(2, 2, {"eta": 1.0}, 0)
    assert state["total"] == 1.0 + 2.0 + 1.0 * 1.0 + 2.0 * 2.0
    output, state = updater.update(state, (), step_context(2, 2, 0))
    assert isinstance(output, ForecastOutput)
    assert output.values == (1.0, 1.0)  # both experts forecast 0.5 at each horizon
    assert state["last"] == 0


def test_rows_output_passes_through_unchanged() -> None:
    """An artifact answering with weight rows gets them back unchanged (the (f) weights track)."""
    source = _artifact(update_body="    return (((0.5, 0.5), (0.5, 0.5)), state)\n")
    output, _ = artifact_updater(source)().update({}, (), step_context(2, 2, 0))
    assert output == ((0.5, 0.5), (0.5, 0.5))


# --- Test 7: the sandbox ----------------------------------------------------------


def test_sandbox_namespace_is_exactly_the_pinned_set() -> None:
    """The exec namespace holds only __builtins__, math, init and update; the 17 pinned names."""
    updater = artifact_updater(REFERENCE_ARTIFACT_SOURCE)()
    namespace = updater.namespace
    assert set(namespace) == {"__builtins__", "math", "init", "update"}
    builtins = namespace["__builtins__"]
    assert isinstance(builtins, dict)
    assert set(builtins) == set(ARTIFACT_BUILTINS)
    assert set(ARTIFACT_BUILTINS) == {
        "abs",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "max",
        "min",
        "range",
        "sorted",
        "str",
        "tuple",
        "ValueError",
        "zip",
    }
    assert namespace["math"] is math
    for banned in ("__import__", "open", "eval", "exec", "print", "repr", "type", "set", "sum"):
        assert banned not in builtins


def test_each_adapter_gets_its_own_namespace() -> None:
    """Defence in depth: every adapter execs into its own namespace and builtins dict.

    The grammar cannot reach ``__builtins__`` (the dunder-name rule), but a
    future grammar change that re-opened a route into it must still not let
    one series' run read or write another's: the builtins dicts are separate
    objects, one per adapter, so a mutation of one adapter's dict never
    appears in another's — and both adapters still run the same source.
    """
    factory = artifact_updater(REFERENCE_ARTIFACT_SOURCE)
    first, second = factory(), factory()
    assert first.namespace is not second.namespace
    first_builtins = first.namespace["__builtins__"]
    second_builtins = second.namespace["__builtins__"]
    assert isinstance(first_builtins, dict)
    assert isinstance(second_builtins, dict)
    assert first_builtins is not second_builtins
    first_builtins["leak"] = 111  # the cross-series channel of round 2, seeded directly
    assert "leak" not in second_builtins
    first_state = first.init(2, 2, CONFIG, 0)
    second_state = second.init(2, 2, CONFIG, 0)
    assert first_state == second_state  # the untouched namespace answers identically


# --- Test 8: normalisation --------------------------------------------------------


def test_normalisation_is_ast_canonical() -> None:
    """Two differently formatted AST-equal programs normalise to one source; dumps are equal."""
    spaced = (
        "def  init( n_experts , horizon , config , seed ) :\n"
        "    x   =  1.0\n"
        "    return  {}\n"
        "\n"
        "\n"
        "\n"
        "def update(state, feedback, context):\n"
        "    # a comment, dropped by the parse\n"
        "    y = x\n"
        "    return ({}, state)\n"
    )
    tight = (
        "def init(n_experts, horizon, config, seed):\n"
        "    x = 1.0\n"
        "    return {}\n"
        "\n"
        "\n"
        "def update(state, feedback, context):\n"
        "    y = x\n"
        "    return ({}, state)\n"
    )
    assert normalized_source(spaced) == normalized_source(tight)
    assert ast.dump(ast.parse(normalized_source(spaced))) == ast.dump(ast.parse(spaced))


# --- Test 9: diffs ----------------------------------------------------------------


DIFF_PARENT = (
    "def init(n_experts, horizon, config, seed):\n"
    "    rate = 0.1\n"
    "    return {}\n"
    "\n"
    "\n"
    "def update(state, feedback, context):\n"
    "    return ({}, state)\n"
)


def test_diff_lines() -> None:
    """A one-constant change gives the unified body lines; identical sources give []."""
    child = DIFF_PARENT.replace("0.1", "0.2")
    assert diff_lines(DIFF_PARENT, child) == [
        "@@ -1,5 +1,5 @@",
        " def init(n_experts, horizon, config, seed):",
        "-    rate = 0.1",
        "+    rate = 0.2",
        "     return {}",
        " ",
        " ",
    ]
    assert diff_lines(DIFF_PARENT, DIFF_PARENT) == []


# --- Test 10: family and constants ------------------------------------------------


def test_family_and_const_change_stats() -> None:
    """params vs logic, hand-computed counts and relative changes, and the fallbacks."""
    params_child = DIFF_PARENT.replace("0.1", "0.12")
    logic_child = DIFF_PARENT.replace("rate = 0.1", "scale = 0.1")
    assert artifact_family(DIFF_PARENT, params_child) == "params"
    assert artifact_family(DIFF_PARENT, logic_child) == "logic"
    assert artifact_family(DIFF_PARENT, DIFF_PARENT) == "params"
    changed, rel = const_change_stats(DIFF_PARENT, params_child)
    assert changed == 1
    assert rel == min(abs(0.12 - 0.1) / max(abs(0.1), 1e-12), 1.0)
    assert const_change_stats(DIFF_PARENT, DIFF_PARENT) == (0, 0.0)
    doubled = DIFF_PARENT.replace("0.1", "0.2")
    assert const_change_stats(DIFF_PARENT, doubled)[1] == 1.0
    two = DIFF_PARENT.replace("rate = 0.1", "row = [2.0]")
    four = DIFF_PARENT.replace("rate = 0.1", "row = [4.0]")
    assert const_change_stats(two, four) == (1, 1.0)
    extra = DIFF_PARENT.replace("    return ({}, state)", "    extra = 5.0\n    return ({}, state)")
    assert const_change_stats(DIFF_PARENT, extra) == (2, 1.0)


# --- Test 11: the feature vector --------------------------------------------------


def test_feature_vector_hand_checked() -> None:
    """One full ProposalFeatures maps to its eight values, and every clip holds."""
    features = ProposalFeatures(
        family="params",
        changed_lines=5,
        const_changes=2,
        max_rel_const_change=0.25,
        generation=7,
        parent_gain=0.3,
        is_llm=True,
    )
    assert feature_vector(features) == (7 / 50, 1.0, 0.0, 5 / 20, 2 / 5, 0.25, 0.3, 1.0)
    clipped = ProposalFeatures(
        family="logic",
        changed_lines=30,
        const_changes=9,
        max_rel_const_change=7.5,
        generation=500,
        parent_gain=-2.0,
        is_llm=False,
    )
    assert feature_vector(clipped) == (1.0, 0.0, 1.0, 1.0, 1.0, 1.0, -1.0, 0.0)
    assert len(FEATURE_NAMES) == 8
    assert FEATURE_NAMES == (
        "generation",
        "family_params",
        "family_logic",
        "changed_lines",
        "const_changes",
        "max_rel_const_change",
        "parent_gain",
        "is_llm",
    )
    assert frozenset({"exp", "sqrt", "log", "isfinite", "fabs"}) == MATH_MEMBERS
