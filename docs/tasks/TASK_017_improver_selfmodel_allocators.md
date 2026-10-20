Status: active

# Task 017: the improver harness, the self-model and the allocators (Phase 1 c-e)

Read `AGENTS.md` and `docs/ROADMAP.md` first. The owner asked for this task on #150, in parallel with TASK_016 PR C: this file becomes the active task only when TASK_016 is `done` (the developer works the lowest-numbered active task, so PR C lands first). It covers items (c), (d) and (e) of #142: the LLM improver, the self-model and the allocator with its rivals. It is a stack of three PRs opened in order, stopping after each for review. Nothing in this task runs on sp-1, calls the real API or touches `.github/`.

## Why

- **Owner, #150 (2026-10-20):** plan (c)-(e) now; the object is the forecast-v2 updater (arena 4) with `bias_share` as the reference; the holdout replicates judge kept edits and are never used for selection; small PR stacks; CI uses a fake LLM client only; the self-model and allocator must work with a non-LLM proposer (`proposer.model = null`) so a quota problem never blocks them. The run CLI and detached launcher (f), the report (g) and the meta level (h) are the next task.
- **Owner, #143 (2026-10-20), all six items answered:**
  1. `glm-5.3` (Zhipu bigmodel.cn), one model for every condition, no second model.
  2. Standard-library HTTPS against the Anthropic-compatible Messages endpoint the developer agent already uses (`https://open.bigmodel.cn/api/anthropic`, `POST /v1/messages`), base URL and model in configuration. CI and tests use a fake client only.
  3. Long runs on sp-1 read the key from the node's environment or `.env`; no GitHub secret; the key never enters the ledger, logs or blobs. Prompts carry updater code and synthetic dev-pool summaries and may go to the API.
  4. No dollar cost, but rate and usage limits: budgets are tokens and calls in `run_start.budget_limits`; backoff and retry on rate-limit responses with the waiting time recorded and not charged as compute; a pilot comes first (shift3, 1 seed, the main conditions).
  5. Lineages run one at a time on sp-1, detached.
  6. Every prompt and response is a ledger blob; "reproducible" means re-running the lineage from the recorded responses with no API call; a fresh re-run with live calls is a new run.
- **Owner, #142:** baselines at equal budget are evaluate-everything with replay selection, a fixed allocator, random, a bandit, Bayesian optimisation and a weights-only object space (the last is a condition of (f), not an allocator). Metrics are best-so-far dev MASE gain per unit of compute, self-model calibration and the safe-inheritance rate.

This task gives Phase 1 its improver (PR A: the LLM client, the artifact object space, the proposers, the `proposal` records), its self-model (PR B: per-proposal gain quantiles and `prediction` records flushed before evaluation) and its allocator slot with every rival (PR C: `allocation` records). The lineage loop that drives all three — propose, predict, allocate, evaluate, accept, repeat, with equal budget accounting and holdout scoring of accepted edits — is (f).

## Decisions already made (do not reopen)

- **Ledger schema v1 is frozen and is not changed.** Every kind Phase 1 needs already exists (`proposal`, `prediction`, `allocation`, `evaluation`, `acceptance`), with `PROPOSAL_KEYS` (including a nullable `model` and `temperature`), `QUANTILE_KEYS`, `REQUIRED_REFERS_TO` and the prediction-before-evaluation rule of `validate_run` already in place. No payload kind, field, status, `PROTOCOL_KEYS` entry or ordering rule changes in this task.
- **The arena is not touched.** `ARENA_VERSION` stays `"4"`; `replay_dataset`, `ForecastOutput`, `FORECAST_ABS_LIMIT` and the invalid-forecast rules are used as TASK_016 left them. No `protocol/` PR and no `protocol-change` label.
- **TASK_016:** `bias_share` is the reference forecast-v2 updater and the lineage's starting point; a fresh updater is used per series; the holdout datasets (`synthetic-shift2-holdout`, `synthetic-shift3-holdout`) are never used for selection.
- **The object space** is forecast-v2 updater modules written in the updater discipline: module-level `init` and `update`, arithmetic in explicit left-to-right loops, no imports except `math`'s pure functions (the baselines themselves use `math.exp`), state threaded through the state object. PR A pins this as a validator, so it is enforced, not requested.
- **Reproducibility:** every prompt and response is a blob; replay from recorded responses is bit-equal on artifacts and charges zero tokens; LLM outputs are never seeded — they are recorded.

## Rules for all three PRs

- **No new dependency.** Standard library only (`urllib.request`, `json`, `ast`, `difflib`, `time`, `math`, `random`).
- **CI never calls the API.** The real client is exercised only on sp-1 by (f)'s runs. Tests use a fake client and a fake `post`/`sleep` injection.
- **The key never appears in a config blob, record, blob or log.** Configuration carries the *name* of the environment variable (`EVOLVETS_LLM_API_KEY`), never a value.
- **Import contracts.** `improver/`, `selfmodel/` and `allocator/` never import `evolvets.arena.sealed`, `evolvets.arena.live`, `evolvets.arena.synthetic`, `evolvets.arena.realdata`, `evolvets.cache` or `evolvets.foundation`. `.importlinter` is unchanged and must pass. `updater/` is imported (interfaces only) but never imports back.
- **Seeds from config.** Every stochastic component (the random proposer, the random and shuffled allocators) takes a `random.Random` the caller seeds from the run config.
- **Mypy strict.** Each new module joins the strict override list in `pyproject.toml` in the same PR, with a `TASK_017` comment. Do not weaken any override.
- **Tests** live in `tests/improver/`, `tests/selfmodel/` and `tests/allocator/` (new directories) and use fixtures in `tmp_path` only.
- Do not edit `.github/`, `.importlinter`, `AGENTS.md`, `docs/agents/` or anything under `arena/`, `ledger/`, `cache/`, `updater/` (this task only adds to `improver/`, `selfmodel/`, `allocator/`, `tests/` and `pyproject.toml`).

---

## PR A: `feat/improver-harness`

Item (c): the object space (updater artifacts), the LLM client, the proposers and the `proposal` records.

### Files and interfaces

`src/evolvets/improver/interfaces.py` (new):

```python
IMPROVER_VERSION = "1"


@dataclass(frozen=True)
class LLMResponse:
    """One successful completion: the text and the token counts to charge."""

    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> LLMResponse: ...


@dataclass(frozen=True)
class ProposalFeatures:
    """What a proposal looks like before it is evaluated."""

    family: str  # "params" (only numeric constants changed) | "logic" (anything else)
    changed_lines: int  # >= 0, diff body lines
    const_changes: int  # >= 0, numeric constants that differ from the parent
    max_rel_const_change: float  # in [0, 1]; 0.0 when no constant changed
    generation: int  # >= 0
    parent_gain: float  # this artifact's own realised gain, clipped to [-1, 1]; 0.0 at generation 0
    is_llm: bool


FEATURE_NAMES: tuple[str, ...]  # 8 names, the order below
def feature_vector(features: ProposalFeatures) -> tuple[float, ...]
```

`feature_vector` returns, in order: `min(generation, 50) / 50`; `1.0` if `family == "params"` else `0.0`; `1.0` if `family == "logic"` else `0.0`; `min(changed_lines, 20) / 20`; `min(const_changes, 5) / 5`; `max_rel_const_change`; `parent_gain`; `1.0` if `is_llm` else `0.0`. Every value is finite and in `[0, 1]` except `parent_gain`, which is in `[-1, 1]`.

`src/evolvets/improver/artifacts.py` (new; imports only the standard library, `evolvets.updater.interfaces` and `improver/interfaces.py`'s types):

```python
class ArtifactError(ValueError): ...

REFERENCE_ARTIFACT_CONFIG: Mapping[str, float] = {"eta": 1.0, "alpha": 0.01, "bias_rate": 0.1}
REFERENCE_ARTIFACT_SOURCE: str  # the starting artifact, already in normalised form
ARTIFACT_BUILTINS: Mapping[str, object]  # the sandbox's entire builtin set, key set pinned by test
MATH_MEMBERS: frozenset[str] = frozenset({"exp", "sqrt", "log", "isfinite", "fabs"})

def validate_artifact_source(source: str) -> None  # raises ArtifactError, message names the construct
def normalized_source(source: str) -> str          # validate, then ast.unparse
def artifact_updater(source: str) -> Callable[[], ForecastUpdater]
def diff_lines(parent: str, child: str) -> list[str]
def artifact_family(parent: str, child: str) -> str          # "params" | "logic"
def const_change_stats(parent: str, child: str) -> tuple[int, float]
```

`src/evolvets/improver/client.py` (new):

```python
@dataclass(frozen=True)
class LLMClientConfig:
    base_url: str = "https://open.bigmodel.cn/api/anthropic"
    model: str = "glm-5.3"
    api_key_env: str = "EVOLVETS_LLM_API_KEY"
    timeout_seconds: float = 300.0
    max_attempts: int = 6          # >= 1
    backoff_base_seconds: float = 2.0
    backoff_cap_seconds: float = 120.0
    max_tokens: int = 8192
    temperature: float = 1.0


class LLMClientError(RuntimeError):
    def __init__(self, message: str, call: LLMCall | None = None) -> None: ...


@dataclass(frozen=True)
class LLMAttempt:
    ok: bool
    status: int | None
    error: str | None


@dataclass(frozen=True)
class LLMCall:
    """The audit trail of one complete(): attempts, waits, tokens, chargeable wall time."""

    attempts: tuple[LLMAttempt, ...]
    waits_seconds: tuple[float, ...]
    input_tokens: int
    output_tokens: int
    wall_seconds: float  # monotonic elapsed minus every backoff wait


class HttpsLLMClient:
    def __init__(
        self,
        config: LLMClientConfig,
        *,
        post: Callable[..., tuple[int, str]] | None = None,  # tests inject; default urllib
        sleep: Callable[[float], None] | None = None,        # tests inject; default time.sleep
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None: ...
    def complete(self, system: str, user: str) -> LLMResponse: ...


def prompt_key(system: str, user: str) -> str  # canonical JSON of {"system", "user"}


class RecordedLLMClient:
    """Serves recorded responses by prompt; the replay path makes no network call."""
    def __init__(self, responses: Mapping[str, LLMResponse]) -> None: ...
    def complete(self, system: str, user: str) -> LLMResponse: ...
```

`src/evolvets/improver/observations.py` (new):

```python
class ImproverError(RuntimeError): ...

def observation_object(
    blobs: BlobStore, records: Sequence[Record], generation: int, max_history: int = 20
) -> dict[str, object]
```

`src/evolvets/improver/prompt.py` (new):

```python
IMPROVER_SYSTEM_PROMPT: str
def user_prompt(parent_source: str, observation: Mapping[str, object]) -> str
def prompt_object(config: LLMClientConfig, system: str, user: str) -> dict[str, object]
```

`src/evolvets/improver/propose.py` (new):

```python
PROPOSER_IDS: tuple[str, ...] = ("llm_edit", "random_param")


class ProposerError(RuntimeError): ...


@dataclass(frozen=True)
class ProposalDraft:
    parent_source: str
    child_source: str
    diagnosis: str
    proposer: dict[str, Any]  # exactly PROPOSER_KEYS


def propose_and_record(
    run: Run,                       # typed via a TYPE_CHECKING import; no runtime runner import
    *,
    proposer_id: str,
    parent_source: str,
    parent_record_id: str | None,   # the proposal whose child_artifact is parent_source; None at generation 0
    generation: int,
    parent_gain: float,
    history_records: Sequence[Record],
    rng: random.Random,
    client: LLMClient | None = None,        # required by "llm_edit"
    client_config: LLMClientConfig | None = None,
    max_repair_rounds: int = 1,
) -> Record: ...


def random_propose(parent_source: str, rng: random.Random) -> ProposalDraft: ...


def recorded_responses(blobs: BlobStore, records: Iterable[Record]) -> dict[str, LLMResponse]
```

### Exact behaviour

**The artifact validator** (`validate_artifact_source`) accepts exactly this grammar, checking by AST walk, rejecting with a message that names the rejected construct:

- Module body: an optional docstring, an optional `import math` (the only import allowed, and the only statement allowed before the functions), then exactly two top-level `def`s named `init` and `update`. No other top-level statement of any kind, no module-level assignment, no third function, no class.
- `init` takes exactly the parameters `n_experts, horizon, config, seed`; `update` takes exactly `state, feedback, context`. Annotations on parameters are allowed; variable annotation statements (`AnnAssign`) are not.
- Allowed statements inside functions: `def` (nested helpers), `for`, `if`, `return`, `assign`, `augmented assign`, `pass`, and expression statements that are exactly a string constant (inner docstrings).
- Allowed expressions: constants (`str`, `bool`, `int`, `float` — every numeric constant must be finite, so `1e400` is rejected), names, attributes (the attribute name must not start and end with `__`; if the value is the name `math`, the attribute must be in `MATH_MEMBERS`), tuples, lists, dicts, subscripts, slices, calls, binary `+ - * / % //`, unary `- + not`, comparisons, `and`/`or`, conditional expressions (`a if c else b`).
- Calls: a bare name must be in `ARTIFACT_BUILTINS`' key set (`abs`, `min`, `max`, `float`, `int`, `len`, `range`, `tuple`, `list`, `dict`, `sorted`, `enumerate`, `zip` — exactly this set, which `ARTIFACT_BUILTINS` maps to the real functions) or be the name of a function defined in the artifact; `math.<member>` with member in `MATH_MEMBERS` is allowed; every other call shape is rejected (this covers `open`, `eval`, `sum`, `print`, `__import__` and any call through an expression).
- Rejected everywhere: `while`, `try`, `raise`, `lambda`, comprehensions (list/set/dict/generator), the walrus operator, `break`/`continue`, `yield`/`await`, `async def`, `class`, `delete`, `global`/`nonlocal`, starred expressions, f-strings, `from ... import`, any import other than `math`, `**` (it is not in the binary-operator allowlist).

**The loader** (`artifact_updater`): validate, `compile` once, `exec` in a namespace whose globals are exactly `{"__builtins__": dict(ARTIFACT_BUILTINS), "math": math}` — no `__import__`, no `open`, no `eval`. It returns a zero-argument factory producing an adapter object whose `init`/`update` call the namespace's functions. `update` returns whatever the artifact returns as `(output, state)` and dispatches on shape: if the first element of `output` is itself a sequence, it is a weights row list and is passed through unchanged (the arena validates it, as for any weights-v1 output); otherwise `output` is a flat per-horizon forecast and is wrapped in `ForecastOutput(tuple(float(v) for v in output))`. Because the flat/nested distinction is the first element's type, the two are never confusable (`horizon >= 1`). A flat forecast replayed under `weights-v1` therefore fails as `invalid_weights`, exactly as a hand-written `ForecastOutput` does — which is how the weights-only condition of (f) keeps its object space honest. The functions are stateless (state travels through the state object), so one namespace serves every series.

**The reference artifact** (`REFERENCE_ARTIFACT_SOURCE`) is a module-level `init`/`update` translation of `BiasShare` with Fixed-Share inlined, written in the allowed grammar, stored in `normalized_source` form. Behavioural requirements, all matching `updater/baselines.py` and `updater/forecast.py` operation for operation:

- `init` validates `config`'s key set is exactly `{eta, alpha, bias_rate}`, `eta` a finite number `> 0`, `alpha` and `bias_rate` finite numbers in `[0, 1]` (bools rejected as everywhere), `n_experts >= 1`, `horizon >= 1`, `seed` an int whose value is ignored; the errors are `ValueError`s naming the field. It returns a dict state with the uniform weights `[[1.0 / n_experts] * n_experts for each horizon]`, a zero `bias` per horizon and an empty `applied` map.
- `update` keeps BiasShare's exact operation order: first the bias recursion over every `feedback` item (`applied.pop(step)`, a missing step raising `ValueError(f"no update call ever applied a bias at step {step}")`, `residual = combined_errors[h] + applied_row[h]`, `bias[h] += bias_rate * (residual - bias[h])`); then the Fixed-Share step for every feedback item and horizon column exactly as `_share_step` computes it (support set, lowest loss, `weights[e] * math.exp(-eta * (losses[e] - low))`, left-to-right total, `(1 - alpha) * (score / total) + alpha / n_experts`); then the combination `base = base + weights[e][h] * context.forecasts[e][h]` in the replay's loop order, `out[h] = base - bias[h]`; finally `applied[context.step] = tuple(bias)` and `return (tuple(out), state)`.

**Diffs, family, constants.** `diff_lines` is `difflib.unified_diff(parent.splitlines(), child.splitlines(), lineterm="")` minus the `---`/`+++` header lines, as a list; identical sources give `[]`. `artifact_family` parses both sources (both must validate first), replaces every numeric constant node with `0.0` in both ASTs via one `NodeTransformer` each, and returns `"params"` if the transformed `ast.dump`s are equal and the sources' texts differ or not, else `"logic"`. `const_change_stats` collects each source's numeric constants in document order (bools excluded); with equal counts it returns the count of positions that differ and the maximum of `abs(new - old) / max(abs(old), 1e-12)` clipped to `[0, 1]`, falling back to `(len(child_constants), 1.0)` when the counts differ or a parent constant is not finite; identical sources give `(0, 0.0)`.

**The HTTPS client.** `complete(system, user)`:

- reads the key from `os.environ[config.api_key_env]` at call time; a missing variable raises `LLMClientError` naming only the variable name;
- `POST`s `{base_url}/v1/messages` with headers `content-type: application/json`, `x-api-key: <key>`, `anthropic-version: 2023-06-01` and the UTF-8 body `json.dumps({...}, sort_keys=True, separators=(",", ":"))` of `{"max_tokens", "model", "system", "temperature", "messages": [{"role": "user", "content": user}]}`;
- through the injectable `post(url, key, version, body, timeout) -> (status, body_text)`, so tests script responses without network; the default builds an `urllib.request.Request` and opens it with `config.timeout_seconds`;
- retries — after waiting `min(cap, base * 2 ** (k - 1))` seconds for the k-th wait, through the injectable `sleep` — on HTTP 429, HTTP >= 500, any transport error, and on a 2xx whose body is not the expected JSON (not JSON, no `content` list, no text block, or a missing `usage`); up to `max_attempts` attempts in total, after which `LLMClientError` is raised carrying the `LLMCall` audit (`ok=False` attempts, the waits, the tokens of any attempt that reported usage);
- any other 4xx raises `LLMClientError` immediately with the status and at most the first 500 characters of the body — never the key, which is not in the body;
- a success returns `LLMResponse(text, input_tokens, output_tokens, stop_reason)`, `text` being the concatenation of the `text` of every `content` block of type `"text"` in order, and the audit carrying the summed usage and `wall_seconds` measured with the injectable `monotonic` around each attempt minus every recorded wait — the waits are recorded, never charged (owner decision #143.4).

**Observations.** `observation_object` walks `records` (the lineage's own records, both streams' order) and builds `{"generation": <int>, "parent_metrics": <the parent artifact's most recent dev evaluation's `metrics` dict, or `{}` at generation 0>, "history": [{"generation", "family", "gain", "accepted"} ...]}` where each history entry is one evaluated proposal of the lineage: `family` recomputed by `artifact_family` from the proposal's parent and child artifact blobs, `gain` the parent's `metrics["mase"]` minus the child's `metrics["mase"]` (positive is better), `accepted` from the `acceptance` record that follows, and the list capped at the last `max_history` entries in record order. It raises `ImproverError` on any `evaluation` record whose pool is not `"dev"` — a sealed or live score can never reach a prompt. A proposal whose evaluation failed or has not happened is skipped. At generation 0 the history is empty and `parent_metrics` is `{}`.

**The prompt.** `user_prompt` is the fixed instruction block, then the observation's canonical JSON, then the full parent source, each part clearly delimited. `IMPROVER_SYSTEM_PROMPT` states: the task (reduce the dev pooled MASE of this forecast-v2 updater), the reply contract (exactly one JSON object `{"diagnosis": <short str>, "source": <the complete new module source>}`, nothing else), the artifact rules (module-level `init`/`update`, the allowed arithmetic and calls, `import math` only, no `while`/`try`/comprehensions, state in the state object), and that anything violating the rules is discarded. `prompt_object` is `{"system", "user", "model", "temperature", "max_tokens"}`; its canonical bytes are the prompt blob.

**`propose_and_record`**, for `proposer_id = "llm_edit"`:

1. Build the observation from `history_records` and `generation`, the prompt from the parent source, and call `client.complete`. Parse: the substring from the first `{` to the last `}` must `json.loads` to a dict with exactly the keys `diagnosis` (a str) and `source` (a str), and `source` must pass `validate_artifact_source`.
2. Any parse or validation failure triggers a repair round (up to `max_repair_rounds`): the user message is extended with `"\n\nYour previous reply could not be used: <error>. Reply with only the JSON object described."` and the client is called again. Every call's attempts, waits, tokens and texts are recorded.
3. If every round fails, the draft is a **fallback no-op**: `child_source` is the parent source, the diff is empty, and the diagnosis is `f"proposer fallback after {n} call(s): {last error}"`. The cost is still charged — a failed call is part of the record (AGENTS.md 2.8). A valid reply whose source equals the parent is a genuine no-op proposal, not a fallback.
4. The child source is normalised (`normalized_source`), all blobs are written first through `run.put_blob` — parent source, child source, the joined diff lines, the observation object, the prompt object and the response audit object (`{"attempts", "waits_seconds", "input_tokens", "output_tokens", "stop_reason", "text"}`) — and then the `proposal` record is appended: `target="updater"`, the blob refs, `diagnosis`, `proposer` `{"id": "llm_edit", "version": "1", "model": config.model, "temperature": config.temperature}`, `improver_version=IMPROVER_VERSION`, `planted_defect=None`, `refers_to=[parent_record_id]` or `[]` at generation 0.
5. The record's budget is `Budget(llm_input_tokens=Σ, llm_output_tokens=Σ, wall_seconds=Σ call wall)` — the waits are excluded. The `proposal` payload and its blobs never contain the key.

For `proposer_id = "random_param"`: `random_propose` parses the parent, collects every numeric constant in document order, picks one index uniformly with `rng.choice`, multiplies it by `rng.uniform(0.5, 2.0)`, rounds to 12 significant digits, rebuilds the source with `ast.unparse` (constants live in the AST, so no text surgery), validates, and returns a draft with `proposer = {"id": "random_param", "version": "1", "model": None, "temperature": None}`. A parent with no numeric constant raises `ProposerError` (the reference artifact has several, so this is a caller error, not a runtime hazard). The record charges `wall_seconds` only, and `prompt_blob` and `response_blob` are `None` — `proposer.model = null` is exactly the owner's non-LLM track, on which PR B and PR C are tested.

Crash safety: blobs are written before the record, and `BlobStore.put` is atomic, so a crash mid-proposal leaves at most unreferenced blobs (content-addressed, harmless) and never a record pointing at a missing blob.

**Replay.** `recorded_responses` reads every `proposal` record of `records` with both prompt and response blobs, re-derives `prompt_key` from the prompt blob, and maps it to the recorded `LLMResponse`. `RecordedLLMClient.complete` returns the recorded response for `prompt_key(system, user)` — repair prompts are distinct keys, so a whole call sequence replays — and raises `LLMClientError` (message containing the prompt key's SHA-256) on a miss. A replayed proposal charges zero tokens and zero wall: the replay is an audit, not a run.

### Tests for PR A (all required)

In `tests/improver/test_artifacts.py` (a fake `post`/`sleep` client helper lives in `tests/improver/fake.py`):

1. **Reference artifact validity:** `validate_artifact_source(REFERENCE_ARTIFACT_SOURCE)` passes; `normalized_source` is a fixed point; the SHA-256 of the source is pinned, with a docstring saying TASK_017 pinned it and why.
2. **Equivalence with `bias_share`:** on the synthetic-dev (`per_family = 1`) fixture pool (the `tests/updater/test_baselines_replay.py` pattern), `artifact_updater(REFERENCE_ARTIFACT_SOURCE)` replayed with `interface="forecast-v2"` and `REFERENCE_ARTIFACT_CONFIG` gives `status == "ok"` and a metrics dict `==` to `bias_share`'s with `FORECAST_DEFAULT_CONFIGS["bias_share"]`, and every per-series `mase` equals. The float discipline made this bit-equality possible; say so in the docstring.
3. **Config parity:** the artifact's `init` raises `ValueError` on a missing key, an extra key, `eta = 0`, `alpha = -0.1` and `1.5`, `bias_rate = -0.1`, `1.5` and `True`, `eta = True`, `n_experts = 0`, `horizon = 0` and `seed = True`; two different int seeds give identical outputs.
4. **Update parity edges:** a direct `update` call with a `Matured` whose step was never applied raises the exact `ValueError`; two states driven interleaved give the same outputs as driven separately.
5. **AST rejects** (each an `ArtifactError` whose message names the construct): `import os`; `from math import exp`; `while`; `try`; `raise`; `lambda`; a list comprehension; the walrus operator; `break`; a class; `**`; a call to `sum`; a call to `open`; a call to `print`; an attribute `__dict__`; `math.pow`; the literal `1e400`; a module-level assignment; a third top-level `def`; a missing `update`; a missing `init`; an f-string; a call whose function is an expression.
6. **AST accepts:** `import math` with `math.exp`; a nested `def` called by name; `for`/`if`/`return`/assign/augmented assign; tuple/list/dict/subscript/slice; every name in `ARTIFACT_BUILTINS` called; `IfExp`; comparisons, `and`/`or`, unary `-`/`not`; `%` and `//`; string docstrings at module and function level.
7. **Sandbox:** the exec namespace's `__builtins__` key set is exactly `ARTIFACT_BUILTINS`'s key set (no `__import__`, `open`, `eval`, `exec`, `print`), `math` is present, and the namespace has no other globals.
8. **Normalisation:** two differently formatted but AST-equal programs normalise to the same source; `ast.dump` of the normalised source equals `ast.dump` of the input.
9. **Diffs:** a one-constant change gives the expected unified body lines; identical sources give `[]`.
10. **Family and constants:** a constants-only change gives `"params"` with hand-computed count and `max_rel_const_change`; a renamed local gives `"logic"`; identical sources give `("params", 0, 0.0)`; `[2.0] -> [4.0]` gives `1.0`; differing constant counts give `(n, 1.0)`.
11. **`feature_vector`:** hand-checked values for one full `ProposalFeatures`, plus the clips (`generation = 500` -> `1.0`, `changed_lines = 30` -> `1.0`, `parent_gain = -2.0` -> `-1.0`, `max_rel_const_change = 7.5` -> `1.0`).

In `tests/improver/test_client.py`:

12. **Success with a retry:** scripted `429` then `200` gives one wait of `backoff_base_seconds`, `wall_seconds` excludes it, both attempts are in the audit, and the response text/tokens parse.
13. **Exhaustion:** five `503` responses (config `max_attempts = 6`, then 5) exhaust and raise `LLMClientError` carrying the audit: attempts, waits `base * [1, 2, 4, 8]` (capped at `backoff_cap_seconds`), and the message naming the status.
14. **Non-retryable 4xx:** a `400` raises immediately with the status and the first 500 characters of the body, no waits, and — with a distinctive fake key in the environment — the key appears nowhere in the exception or audit.
15. **Malformed 2xx:** a body that is not JSON, one without `content`, and one whose content has no text block each count as a failed attempt and are retried; a following good `200` succeeds and the audit shows both attempts.
16. **Missing key:** with `api_key_env` unset in a scrubbed environment, `complete` raises `LLMClientError` naming the variable and nothing is posted.

In `tests/improver/test_propose.py`:

17. **`llm_edit` end to end with the fake client:** one `Record` of kind `proposal`; every blob reference resolves (`LedgerReader.verify(check_blobs=True)` on the run directory is ok); the payload fields are exactly as specified (target, blobs, proposer dict with the model and temperature, `improver_version`); `refers_to` is `[parent_record_id]` at generation 1 and `[]` at generation 0; the budget sums both calls' tokens; `validate_run` over the run's records reports no violations; the run's config blob contains the env-var *name* and no key material.
18. **Repair:** a first reply that is not JSON, then a good one, succeed; the response blob's audit holds both calls and the prompt blob holds the first prompt.
19. **Fallback:** two invalid replies give a no-op child (the child blob ref equals the parent's), an empty diff, the fallback diagnosis, and a budget still charged with both calls' tokens.
20. **Genuine no-op:** a valid reply whose source equals the parent is recorded as a normal proposal with an empty diff and the model's own diagnosis.
21. **`random_param`:** the same seed gives the same child source; two seeds differ; the child validates and differs from the parent; `proposer["model"]` and `["temperature"]` are `None`; the record charges zero tokens; a parent with no numeric constant raises `ProposerError`.
22. **Replay:** `recorded_responses` over the run of test 17 feeds a `RecordedLLMClient` that reproduces the same child source and diagnosis with zero charged tokens; an unknown prompt raises `LLMClientError` containing the prompt key's SHA-256.
23. **Observations:** hand-built records (two generations, one failed evaluation, one accepted, one rejected) give the exact object — gains as parent-minus-child `mase`, the cap at the last 20, `family` recomputed from blobs, `parent_metrics` from the latest dev evaluation; a non-dev pool evaluation record raises `ImproverError`; at generation 0 the history is empty and `parent_metrics` is `{}`.
24. **Prompt:** `user_prompt` contains the parent source verbatim and the observation's canonical JSON; `IMPROVER_SYSTEM_PROMPT` states the JSON reply contract and names the artifact rules (substring checks); `prompt_object` has exactly the five keys.

### Dependencies

None.

### Size

About 450 lines of implementation and 450 of tests — over the ~400-line rule, so split by the task's default:

- **PR A1** `feat/improver-artifacts`: `improver/interfaces.py`, `improver/artifacts.py`, the reference artifact, tests 1 to 11.
- **PR A2** `feat/improver-harness`: `improver/client.py`, `improver/observations.py`, `improver/prompt.py`, `improver/propose.py`, `tests/improver/fake.py`, tests 12 to 24.

Open A1 first; A2 depends on it. If A2 alone still exceeds ~400 lines, move tests 23 and 24 (`observations`, `prompt`) with their modules into a third PR `feat/improver-prompting`.

---

## PR B: `feat/selfmodel`

Item (d): per-proposal gain quantiles, logged and flushed before any evaluation.

### Files and interfaces

`src/evolvets/selfmodel/interfaces.py` (new):

```python
METRIC_DEV_MASE_GAIN = "dev_mase_gain"

#: The prior gain quantiles, from the headroom scan's per-edit magnitudes.
DEFAULT_PRIOR: Mapping[str, float] = MappingProxyType(
    {"q05": -0.02, "q25": -0.005, "q50": 0.002, "q75": 0.01, "q95": 0.04}
)


@dataclass(frozen=True)
class HistoryPoint:
    features: ProposalFeatures  # evolvets.improver.interfaces
    gain: float                 # realised dev MASE gain of that proposal


class SelfModel(Protocol):
    def identify(self) -> dict[str, str]: ...
    def predict(self, features: ProposalFeatures) -> Mapping[str, float]: ...
```

`src/evolvets/selfmodel/quantiles.py` (new):

```python
class ConformalRidgeSelfModel:
    """Ridge point forecast of the dev MASE gain, conformalised into quantiles."""

    def __init__(
        self,
        history: Sequence[HistoryPoint],
        *,
        prior: Mapping[str, float] = DEFAULT_PRIOR,
        prior_points: float = 8.0,
        ridge_lambda: float = 1.0,
    ) -> None: ...

    def identify(self) -> dict[str, str]  # {"id": "conformal_ridge", "version": "1"}
    def predict(self, features: ProposalFeatures) -> Mapping[str, float]


def append_prediction(
    run: Run,
    *,
    predictor: Mapping[str, str],
    proposal_record_id: str,
    quantiles: Mapping[str, float],
    metric: str = METRIC_DEV_MASE_GAIN,
    horizon_generations: int = 1,
    wall_seconds: float,
) -> Record: ...
```

### Exact behaviour

**Fit.** The design matrix is an intercept column plus the eight feature columns (`feature_vector`), rows in history order. The weights solve the ridge normal equations `(XᵀX + λI) w = Xᵀy` by Gaussian elimination with partial pivoting, ties to the lowest row — the `experts/ridge.py` pattern, no linear-algebra library. Every history gain must be finite; a non-finite gain raises `ValueError`. With no history, `μ̂(x) = 0` for every `x`.

**Conformal quantiles.** The residuals `r_i = y_i − μ̂(x_i)` are sorted ascending; the q-th offset is the order statistic at index `min(n − 1, max(0, ceil(q · n) − 1))` — no interpolation. The raw quantile for `q` is `μ̂(x) + offset_q`.

**Shrinkage to the prior.** With weight `w = n / (n + prior_points)`, the returned quantile is `w · raw_q + (1 − w) · prior_q`. With no history this is the prior exactly. The five values are then sorted ascending (they must be monotone for the payload), so the output is monotone even when raw quantiles cross.

**Determinism.** No randomness anywhere; the same history and features always give the same dict.

**`append_prediction`.** Validates that `quantiles` has exactly the `QUANTILE_KEYS` and is monotone (the payload re-checks), appends a `prediction` record with `refers_to=[proposal_record_id]`, `predictor`, `metric`, `horizon_generations` and the quantiles, and charges `Budget(wall_seconds=wall_seconds)`. Because `LedgerWriter.append` flushes and fsyncs before returning, the prediction is durable the moment this returns — before any evaluation of that proposal can be appended (AGENTS.md 2.3). `horizon_generations = 1` everywhere in Phase 1: the model predicts the immediate gain of one edit.

### Tests for PR B (all required)

In `tests/selfmodel/test_quantiles.py`:

1. **Identity and prior:** `identify()` is exact; `DEFAULT_PRIOR` is finite and strictly increasing.
2. **Empty history** returns `DEFAULT_PRIOR` exactly.
3. **Orthogonal hand computation:** a two-point history with feature vectors `+e₁` and `−e₁` (all other features 0, `is_llm` False) and gains `+1` and `-1`, `ridge_lambda = 1`: `w₁ = 2/3`, `μ̂(±e₁) = ±2/3`, residuals `±1/3`, offsets for `n = 2` at indexes `q05→0, q25→0, q50→0, q75→1, q95→1`, blend weight `2/10` against `DEFAULT_PRIOR` — every quantile equals the hand-computed value (`==`).
4. **Monotone output:** a history whose raw quantiles cross (constructed so `μ̂ + q05 offset > μ̂ + q25 offset` would hold unsorted, e.g. gains that make the prior blend invert) still returns five non-decreasing values.
5. **Offset indexing:** `n = 4` gains give the order statistics at indexes `q05→0, q25→0, q50→1, q75→2, q95→3`.
6. **Shrinkage:** with `prior_points = 0` the values are the raw conformal quantiles; with a large `prior_points` every value lies strictly between the raw value and the prior.
7. **Determinism:** two models built from the same history return equal dicts.
8. **Non-finite gain** in history raises `ValueError`.
9. **`append_prediction`:** the record has kind `prediction`, `refers_to == [proposal id]`, the exact payload fields, and budget `Budget(wall_seconds=0.5)`; `payload_from_dict` round-trips it; a non-monotone quantile dict raises `PayloadError`.
10. **Flushed before evaluation:** after `append_prediction`, the last line of the run's `main.jsonl` on disk (read directly) is that prediction; then a hand-built `evaluation` for the same proposal is appended and `validate_run` reports no violations (run started with `requires_predictions=True`).
11. **Ordering guard:** in a fresh run whose evaluation is appended with no prediction, `validate_run` reports exactly the prediction-before-evaluation violation for that evaluation.

### Dependencies

None. Depends on PR A's `improver/interfaces.py` (`ProposalFeatures`, `feature_vector`) being merged first.

### Size

About 150 lines of implementation and 200 of tests. No split.

---

## PR C: `feat/allocators`

Item (e): the allocator slot and every rival of (f)'s conditions.

### Files and interfaces

`src/evolvets/allocator/interfaces.py` (new):

```python
class AllocatorError(RuntimeError): ...


@dataclass(frozen=True)
class Candidate:
    record_id: str
    features: ProposalFeatures
    quantiles: Mapping[str, float]  # {} for conditions whose allocator must not see predictions


@dataclass(frozen=True)
class Decision:
    chosen: tuple[str, ...]  # record ids, in evaluation order
    scores: Mapping[str, float]  # one finite float per candidate it scored


class Allocator(Protocol):
    def identify(self) -> dict[str, str]: ...
    def choose(
        self,
        candidates: Sequence[Candidate],
        history: Sequence[HistoryPoint],  # evolvets.selfmodel.interfaces
        evaluations_left: int,
        rng: random.Random,
    ) -> Decision: ...
```

`src/evolvets/allocator/allocators.py` (new):

```python
ALLOCATOR_IDS: tuple[str, ...] = (
    "evaluate_all", "fixed", "random", "predictive", "shuffled", "bandit_ucb", "bo_gp",
)
FAMILY_ORDER: tuple[str, ...] = ("params", "logic")
ARM_UNSEEN_SCORE = 1e9


def make_allocator(
    allocator_id: str,
    *,
    k: int = 1,                      # per-generation evaluation slots, >= 1
    quantile: str = "q50",           # the predictive allocators' score key
    bandit_c: float = 1.0,           # UCB exploration rate, > 0
    gp_gamma: float = 1.0,           # RBF scale, > 0
    gp_noise: float = 0.01,          # observation noise variance, > 0
    gp_beta: float = 2.0,            # GP-UCB exploration weight, > 0
    gp_prior_std: float = 1.0,       # prior std with no history, > 0
    history_cap: int = 100,          # rivals fit at most this many latest points, >= 1
) -> Allocator: ...


def append_allocation(
    run: Run,
    *,
    allocator: Mapping[str, str],
    candidate_ids: Sequence[str],
    chosen_ids: Sequence[str],
    scores: Mapping[str, float],
    wall_seconds: float,
) -> Record: ...
```

### Exact behaviour

**Common to every allocator.** `identify()` returns `{"id": <id>, "version": "1"}`. `choose` raises `AllocatorError` on an empty candidate list or a negative `evaluations_left`, and never returns more than `min(k, evaluations_left, len(candidates))` chosen ids (`evaluate_all` caps at `len(candidates)` alone). No allocator imports, reads or could reach any pool, dataset or holdout: candidates and history are all it ever sees, and that is what keeps selection off the holdout. Only `random` and `shuffled` touch `rng`; the rest are deterministic functions of their arguments.

- **`evaluate_all`** (the Dream-RSI-style baseline): `chosen` is the first `evaluations_left` candidates in candidate order — no selection at all; `scores` is `1.0` for chosen and `0.0` for the rest.
- **`fixed`**: the first `min(k, evaluations_left)` candidates in order; same score shape. It ignores features, quantiles and history entirely.
- **`random`**: `rng.sample(range(n), size)` picks the set; `chosen` lists it in candidate order; scores as above.
- **`predictive`** (the self-model allocator): every candidate must carry all `QUANTILE_KEYS`, else `AllocatorError` — a condition without predictions must not silently use this allocator. The score of candidate `i` is its `quantiles[quantile]`; `chosen` is the top candidates by `(-score, index)` — ties keep candidate order; `scores` are the quantile values used.
- **`shuffled`** (the ablation): like `predictive`, but the score list is permuted across candidates with `rng.shuffle` first, and the permuted values are what is scored and recorded. Same allocator, same rule, destroyed information: the Phase 1 report's check that prediction *content* matters, not prediction-shaped noise.
- **`bandit_ucb`**: the arms are the two families of `FAMILY_ORDER`, the arm of a candidate read from `features.family`. From history: `n_arm`, `mean_arm` (left-to-right mean of gains), `N = Σn_arm`. An arm's score is `mean + bandit_c * sqrt(2 * log(N + 1) / n_arm)` (`math.log`, `math.sqrt`), and `ARM_UNSEEN_SCORE` when `n_arm == 0` — unexplored arms go first. Candidates are chosen arm by arm, best score first (ties by `FAMILY_ORDER`), within an arm in candidate order, until the cap; each candidate's recorded score is its arm's score.
- **`bo_gp`** (Bayesian optimisation over the feature vector): `x = feature_vector(features)`, `X, y` the last `history_cap` history points. With no history, `mean = 0.0` and `std = gp_prior_std` for every candidate. Otherwise the kernel is `K[i][j] = exp(-gp_gamma * ‖xi - xj‖²)` (squared distance in an explicit left-to-right loop), `A = K + gp_noise * I`; `α` and each candidate's `v` solve `A α = y` and `A v = k*` by Gaussian elimination with partial pivoting (ties to the lowest row); `mean* = Σ k*[i] α[i]`, `var* = 1.0 - Σ k*[i] v[i]`, `std* = sqrt(var* if var* > 0.0 else 0.0)` — the floor keeps the square root real near duplicate points. The acquisition score is `mean* + gp_beta * std*`; `chosen` is the top by `(-score, index)`; `scores` are the acquisition values. This is GP-UCB with a fixed beta, hand-rolled in the `experts/ridge.py` style — the equal-budget claim would not survive a numpy dependency, so there is none.

**`append_allocation`.** Appends an `allocation` record with `candidates`, `chosen` and `scores` as given (the payload rejects duplicates, non-subsets and score keys outside the candidates), `allocator` as given, `refers_to = list(candidate_ids)` in candidate order — the record links to every proposal it judged — and `Budget(wall_seconds=wall_seconds)`. Allocators select; they never charge tokens, and (f) charges every condition the same budget for the same candidates, which is what "equal budget" means here (AGENTS.md 2.5).

### Tests for PR C (all required)

In `tests/allocator/test_allocators.py`:

1. **Registry and constructors:** `ALLOCATOR_IDS` order is exact; `make_allocator` on an unknown id raises `AllocatorError`; `k = 0`, `k = True`, a `quantile` outside `QUANTILE_KEYS`, and a non-positive `bandit_c`, `gp_gamma`, `gp_noise`, `gp_beta` or `gp_prior_std` each raise `ValueError`; `history_cap = 0` raises.
2. **`evaluate_all`:** with 3 candidates and `evaluations_left = 2`, the first two are chosen; with `evaluations_left = 3` or more, all; scores are `1.0/0.0` accordingly.
3. **`fixed`:** `k = 1` takes the first; `k = 2` the first two; `k` beyond the candidate count takes all; the decision is identical with and without quantiles and history.
4. **`random`:** the same seed gives the same chosen set; the chosen order is candidate order; two pre-verified seeds give different sets; the cap is respected.
5. **`predictive`:** ranking by the quantile with ties keeping candidate order; `quantile="q75"` ranks by `q75`; a candidate with empty quantiles raises `AllocatorError`.
6. **`shuffled`:** with a pre-verified seed whose permutation moves the top score to a later candidate, the choice differs from `predictive` on the same input; the same seed reproduces it; the recorded scores are the permuted values.
7. **`bandit_ucb`:** with no history both arms are unseen and `params` candidates come first; a history with only strong `params` gains still explores the unseen `logic` arm first; once both arms have history, the higher-UCB arm wins and every score equals the hand formula recomputed in the test with `math.sqrt`/`math.log`.
8. **`bo_gp`:** with no history every score is `gp_beta * gp_prior_std` and the first `k` candidates are chosen; with one history point, the acquisitions of two candidates equal an in-test recomputation of the kernel-and-solve loops; a near-duplicate candidate yields a floored, real `std` (no NaN, no negative variance).
9. **`evaluations_left`:** `0` gives an empty `chosen` (and the allocator's scores) for every allocator; a negative value raises `AllocatorError`; `min(k, evaluations_left, n)` is never exceeded — parametrised across all seven.
10. **Empty candidates** raise `AllocatorError` — parametrised across all seven.
11. **Determinism:** `predictive`, `bandit_ucb` and `bo_gp` return equal decisions on repeated calls; `random` and `shuffled` return equal decisions for equal rng seeds and different ones for different seeds.
12. **`append_allocation`:** the payload round-trips through `payload_from_dict`; `refers_to` equals the candidate ids in order; score keys outside the candidates and a duplicated chosen id are rejected by `PayloadError`; the budget is `Budget(wall_seconds=...)`; `validate_run` over a run holding proposals, a prediction each and the allocation reports no violations.
13. **History cap:** a 150-point history whose last 100 points dominate gives the same decision as passing exactly those last 100 points — for both `bandit_ucb` and `bo_gp`.
14. **No predictions needed:** `evaluate_all`, `fixed`, `random`, `bandit_ucb` and `bo_gp` all decide over candidates whose quantiles are `{}`.

### Dependencies

None. Depends on PR A (`improver/interfaces.py`) and PR B (`selfmodel/interfaces.py`) being merged first.

### Size

About 280 lines of implementation and 320 of tests. If it exceeds ~400 lines:

- **PR C1** `feat/allocator-core`: `interfaces.py`, `evaluate_all`, `fixed`, `random`, `predictive`, `shuffled`, `append_allocation`, tests 1 to 6 and 9 to 14;
- **PR C2** `feat/allocator-rivals`: `bandit_ucb` and `bo_gp` with `make_allocator`'s new branches, tests 7, 8 and their halves of 13.

---

## Not in this task

- **(f) The lineage runner and run CLI** (`feat/phase1-runner`): the propose-predict-allocate-evaluate-accept loop, evaluation of child artifacts through `replay_dataset` (under a subprocess or signal timeout — the AST validator is a style and protocol guard, not a wall-clock one), equal budget accounting across conditions, the weights-only object-space condition, holdout scoring of accepted edits only (in a run separate from the lineage, so holdout records can never feed the improver's observations), the detached sp-1 launcher, and the pilot configuration (shift3, 1 seed, lineages one at a time).
- **(g) The Phase 1 report** (`exp/phase1-report`): gain per unit of compute, calibration of the self-model, safe-inheritance rate, Goodhart diagnostics.
- **(h) The meta level** (`feat/meta-replay`), entries 22 to 24, prompt caching, sharing one proposal stream across conditions, and any second model.
- Re-tuning baselines, new datasets, and anything under `arena/`, `ledger/`, `cache/` or `updater/`.

## Closing the task

PR C (or the last PR of its split) sets this file's first line to `Status: done` and sets roadmap entries 16, 17 and 18 to the merged PR numbers. The runner (f) is the next task; the roadmap notes it.
