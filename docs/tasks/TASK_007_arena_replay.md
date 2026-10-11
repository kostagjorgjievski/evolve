Status: done (PR A merged as #34; PR B split by the size rule: dev pools in #35, metrics and replay_dataset here)

# Task 007: Arena replay loop, horizon-aware label release and metrics

Read `AGENTS.md` and `docs/ROADMAP.md` first. This task covers roadmap entry 8 (`feat/arena-replay`). It is a stack of two PRs; open them in order and stop after each one for review.

## Why

An updater turns the five cached expert forecasts into one combined forecast at every origin, and adapts as errors come in. Research question 1 is measured by replaying updaters over series and scoring them. This task builds that instrument: the contract an updater implements (PR A), the replay loop that drives it and feeds it errors only once a forecast's full horizon has passed (PR A), and the dev pools and metrics that turn a replay into the numbers Gate 1 reports (PR B). Hand-designed updaters (entry 10) and the sealed process (entry 9) come later; this task only uses toy updaters written in the tests.

## Rules for both PRs

- Code lives in `src/evolvets/arena/` (strict mypy) plus one new file, `src/evolvets/updater/interfaces.py`. Add `"evolvets.updater.interfaces"` to the strict override list in `pyproject.toml`. Tests live in `tests/arena/`.
- **Standard library only.** No numpy.
- **Bit-exact determinism**, same rule as TASK_006: on floats use only `+ - * /`, comparisons and `abs`; **never call the builtin `sum()` on floats**, `math.fsum` or `statistics`; accumulate with explicit left-to-right `for` loops in the order this file states. Golden tests enforce this.
- `ARENA_VERSION = "1"` in `arena/__init__.py` (no imports in that file). This is the value runs put in `run_start.protocol_versions["arena"]`. The arena protocol covers `updater/interfaces.py`, `arena/labels.py`, `arena/replay.py`, `arena/dev.py`, `arena/metrics.py` and `arena/evaluate.py`: any change that alters what an updater receives, what it may return, which origins are scored or any metric value must bump `ARENA_VERSION`. Golden tests enforce this. The human tags `arena-v1` after this task and entry 9 have merged; until then the version stays `"1"`.
- **Origin convention** (same as TASK_004/TASK_006): origin `t` means `y_0 .. y_t` are observed, and the forecast for step `h = 1..H` targets `y_{t+h}`.
- **Label release** (AGENTS.md 2.2): the targets of origin `s` are revealed at origin `s + H` and not before, all `H` of them together. Every code path in this task that reads a value it uses as a target (feedback, scoring, the oracle comparison, the MASE scale) goes through `LabelRelease`. Nothing outside `arena/labels.py` indexes the target series directly.
- **The updater never sees a target, a value, an absolute origin, a series id, a timestamp, the cache, the oracle or a split boundary.** It receives the expert forecasts for the current origin and errors for matured origins (below), nothing else. Errors are `forecast - target`, so a target can be recovered from a matured error. That is allowed, because the target is in the past by then.
- **No ledger change.** Nothing here writes ledger records or touches `ledger/` or `PROTOCOL_KEYS`. Runs pin the real data through the cache: `cache_id` (recorded by the caller in the run config) hashes a manifest whose `source` holds `realdata_version` and `public_sha256`. So no `realdata` protocol key is needed, and issue #10 (tag `ledger-schema-v1`) is not blocked by this task.
- **Seeds:** the replay takes an explicit `seed` and derives one seed per series (PR A). No other randomness.
- Errors: `ArenaError(Exception)` for invalid arena inputs (bad pools, caches, arguments), `LookAheadError(ArenaError)` for a premature target request, `UpdaterFailure(Exception)` for a misbehaving updater, all in `arena/errors.py`. Bools are rejected wherever an int is expected.

### Import contracts (decision deferred here by TASK_005 and TASK_006)

Decision: **agent-editable code never imports `evolvets.cache`.** Cached forecasts at origins after `t` encode values after `t` (the naive forecast at `t + 1` is `y_{t+1}`), so direct cache access is a look-ahead path. Forecasts reach an updater only through the replay loop. The full contract set is:

```ini
[importlinter:contract:ground-truth-isolation]
name = Agent-editable packages never reach targets, oracles or raw forecasts
type = forbidden
source_modules =
    evolvets.updater
    evolvets.improver
    evolvets.selfmodel
    evolvets.allocator
forbidden_modules =
    evolvets.arena.synthetic
    evolvets.arena.realdata
    evolvets.cache

[importlinter:contract:updater-isolation]
name = Updaters see data only through the replay loop
type = forbidden
source_modules =
    evolvets.updater
forbidden_modules =
    evolvets.arena
    evolvets.experts
    evolvets.ledger
    evolvets.runner
```

The direction is arena → `evolvets.updater.interfaces`, never the reverse. `updater/interfaces.py` imports nothing from `evolvets`. Neither PR edits `.importlinter` or `.github/`: the developer agent may not (docs/agents/developer.md), and these contracts plus a CI violation probe for each are requested from a human in `fyi-human` issue #32. Entry 10 (the first PR to put updater code in `updater/`) must not merge before these contracts are in `.importlinter`. These contracts constrain Phase 0 only. Whether the improver, self-model or allocator may import the runner or the arena's replay is a Phase 1 decision.

Import contracts do not stop `open()` or `importlib`. Process isolation for untrusted updaters is entry 9 and Phase 1, not this task.

---

## PR A: `feat/arena-replay-core`

The updater contract, label release and the replay of one series. No pools, no metrics, no files written.

### Files and interfaces

`updater/interfaces.py`

```python
ConfigValue = float | int | str | bool
Weights = Sequence[Sequence[float]]  # [expert][h - 1]; each column h is a simplex over experts


@dataclass(frozen=True)
class Matured:
    """Errors for one origin whose full horizon has just passed."""

    step: int  # step index of the matured origin (see Context.step)
    errors: tuple[tuple[float, ...], ...]  # [expert][h - 1] = expert forecast - target
    combined_errors: tuple[float, ...]  # [h - 1] = this updater's combined forecast - target


@dataclass(frozen=True)
class Context:
    """Everything an updater may know at the current origin."""

    step: int  # 0 at the first replayed origin, +1 per origin
    season_index: int  # origin % period
    period: int
    horizon: int
    expert_names: tuple[str, ...]
    forecasts: tuple[tuple[float, ...], ...]  # [expert][h - 1] for targets origin+1 .. origin+H


class Updater(Protocol):
    """An online combiner of expert forecasts. A module defining init/update also satisfies it."""

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> object: ...

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Weights, object]: ...
```

`Matured` and `Context` have exactly these fields (a test checks the field sets).

`arena/errors.py`: `ArenaError`, `LookAheadError`, and

```python
class UpdaterFailure(Exception):
    kind: Literal["failed", "invalid_weights"]  # matches ledger EVALUATION_STATUSES
    message: str  # at most 1000 characters
    series_id: str
    step: int | None  # None when init failed
```

`arena/labels.py`

```python
class LabelRelease:
    """Holds a series' values and hands out targets only once they have matured."""

    def __init__(self, values: Sequence[float], horizon: int) -> None: ...
    def targets(self, origin: int, now: int) -> tuple[float, ...]: ...  # y_{origin+1..origin+H}
    def history(self, end: int, now: int) -> tuple[float, ...]: ...  # y_0 .. y_{end-1}
    def __len__(self) -> int: ...
```

`arena/replay.py`

```python
SEED_LIMIT = 2**63


def series_seed(seed: int, series_id: str) -> int: ...


@dataclass(frozen=True)
class SeriesTrace:
    series_id: str
    first_origin: int  # origin of step 0
    end_origin: int  # exclusive; last replayed origin is end_origin - 1
    horizon: int
    errors: tuple[tuple[float, ...], ...]  # [origin - first_origin][h - 1], combined - target


def replay_series(
    updater: Updater,
    config: Mapping[str, ConfigValue],
    seed: int,
    series_id: str,
    forecasts: SeriesForecasts,
    values: Sequence[float],
    period: int,
    horizon: int,
    end_origin: int,
) -> SeriesTrace: ...
```

### Exact behaviour: `LabelRelease`

- Constructor: `values` all finite `float`/`int` (not `bool`), else `ArenaError`; `horizon` int `>= 1`. It stores a private tuple copy.
- `targets(origin, now)`: ints (not bools), `origin >= 0`. If `origin + horizon > now`, raise `LookAheadError` naming both. If `origin + horizon > len - 1`, raise `ArenaError`. Otherwise return `values[origin + 1 : origin + horizon + 1]` as a tuple.
- `history(end, now)`: `0 <= end`; if `end > now + 1`, raise `LookAheadError`; if `end > len`, raise `ArenaError`; return `values[:end]`.
- No other public way to read the values.

### Exact behaviour: `series_seed`

`seed` is an int (not bool) in `[0, 2**63)`, else `ArenaError`. Returns `int.from_bytes(sha256(b"evolvets/replay/v1\x00" + str(seed).encode() + b"\x00" + series_id.encode("utf-8")).digest()[:8], "big") & (2**63 - 1)`.

### Exact behaviour: `replay_series`

Let `t0 = forecasts.first_origin`, `E = len(EXPERT_NAMES)`, `H = horizon`.

1. Validate (`ArenaError`): `horizon` int `>= 1` and equal to the length of each tuple in `forecasts.at(t0)`; `period` int `>= 2`; `t0 < end_origin` and `end_origin - 1 <= forecasts.last_origin`; `len(values) >= end_origin + H` (every replayed origin has all its targets); `seed` as in `series_seed`. Build `release = LabelRelease(values, H)`.
2. `state = updater.init(E, H, dict(config), series_seed(seed, series_id))`. Any `Exception` → `UpdaterFailure("failed", f"{type(exc).__name__}: {exc}"[:1000], series_id, None)`. `BaseException`s that are not `Exception` (e.g. `KeyboardInterrupt`) propagate.
3. For `t = t0 .. end_origin - 1`, with `k = t - t0`:
   - `feedback = (matured(t - H),)` if `t - H >= t0`, else `()`. `matured(s)` reads `release.targets(s, now=t)` and builds `Matured(step=s - t0, errors=…, combined_errors=…)` from the forecasts and the weights stored for `s`. So feedback for origin `s` arrives at exactly `s + H`, once.
   - `context = Context(step=k, season_index=t % period, period=period, horizon=H, expert_names=EXPERT_NAMES, forecasts=forecasts.at(t))`.
   - `result = updater.update(state, feedback, context)`. An `Exception` → `UpdaterFailure("failed", …, step=k)`. A result that is not a `tuple` of length 2 → `UpdaterFailure("failed", "update must return (weights, state)", …)`. Otherwise `weights, state = result`.
   - **Copy before validating.** Accept only exact container types: `type(weights) is list or type(weights) is tuple`, likewise each row, so a subclass with overridden `__iter__`/`__getitem__` cannot serve one contents to validation and another to the combination. Accept only exact `type(x) is int or type(x) is float` elements (this rejects `bool` outright). Build the copy `stored = tuple(tuple(float(x) for x in row) for row in weights)`; on exact `int`/`float` this conversion cannot lie. Any `Exception` raised while type-checking or copying is itself mapped to `UpdaterFailure("invalid_weights")`, never allowed to escape as an arbitrary exception.
   - Validate the copy (else `UpdaterFailure("invalid_weights", <what was wrong>, …)`): outer length `E`; each row length `H`; each element finite, `>= 0`; for each `h`, the sum over `e = 0..E-1` (in that order, starting from `0.0`) is within `1e-9` of `1.0`. The arena never renormalises. The stored weights are this validated copy, so what was validated is exactly what is used.
   - Compute the combined forecast from the stored copy: `c[h] = 0.0; for e in 0..E-1: c[h] = c[h] + w[e][h] * f[e][h]`.
4. Drain: for `t = end_origin .. end_origin - 1 + H`, mature the origins `s = t - H` that are `>= t0` and were not matured in step 3 (no `update` call). After the drain every replayed origin has been matured exactly once, and the updater never received feedback for the last `min(H, end_origin - t0)` origins.
5. `errors[s - t0][h - 1] = c_s[h] - target` from the maturing step, through `release.targets`. Return the `SeriesTrace`.
6. The forecasts passed in `Context` and `Matured` are fresh tuples. Mutating anything the updater received cannot change the arena's stored weights, forecasts or errors.

### Tests for PR A (all required; `tests/arena/test_labels.py`, `test_replay.py`)

Use a hand-built `SeriesForecasts` (constructed directly from an `array("d")`) with small `H` where useful; test updaters are defined in the test files.

1. `LabelRelease.targets`: returns the right slice when `origin + H == now` (the boundary) and when `now` is larger; raises `LookAheadError` at `now == origin + H - 1`; raises `ArenaError` when `origin + H > len - 1`; rejects `True`, `-1` and `"3"` for both arguments. `history`: returns `values[:end]` at `end == now + 1`, raises `LookAheadError` at `end == now + 2`, `ArenaError` past the end. The constructor rejects NaN, `inf`, `True` and horizon `0`.
2. Feedback timing (spy updater, 60 origins, `H = 5`): steps `0..4` receive `()`; every step `k >= 5` receives exactly one `Matured` with `step == k - 5`; `errors[e][h-1] == forecasts.at(t0 + k - 5)[e][h-1] - values[t0 + k - 5 + h]` exactly, and `combined_errors` equals the formula applied to the weights the spy returned at that step.
3. **No look-ahead through values:** for a cut `c` inside the replay, replacing every value after index `c` with different finite values (forecasts unchanged) leaves everything the spy recorded at origins `<= c` identical (`==` on the recorded `(feedback, context)` pairs). Repeat with forecasts built by `run_pool` from the altered values: again identical up to `c` (prefix property).
4. **No premature target request:** wrap `LabelRelease.targets` (monkeypatch) to record every `(origin, now)`. In a full replay including the drain, every call has `origin + H == now`, and every replayed origin appears exactly once.
5. The last `H` origins: the spy never receives a `Matured` for steps `>= end_origin - t0 - H`; the trace still has errors for all `end_origin - t0` origins.
6. `Context`: exact field set of `Context` and `Matured` (via `dataclasses.fields`); `season_index == origin % period`; `expert_names == EXPERT_NAMES`; `forecasts == forecasts.at(origin)`.
7. Combination: one-hot weights on expert `e` give trace errors equal to expert `e`'s errors; a fixed mixed weight matrix gives errors equal to the left-to-right formula exactly (`==`).
8. Isolation: a spy that mutates any list it can reach (it returns lists and then mutates them after returning) does not change the trace (compare with a well-behaved updater returning the same weights).
9. Invalid weights, one case each, all `UpdaterFailure` with `kind == "invalid_weights"` and the right `step`: wrong outer length, wrong inner length, a negative weight, NaN, `inf`, `True`, a string, `None` as the whole weights, a column summing to `1 + 1e-6`; a `list` subclass whose `__iter__` yields valid rows on the first pass and invalid ones on the second; a `float` subclass whose `__mul__` returns a wrong value; an `int` too large for `float()` to convert (the copy's `OverflowError` is mapped, not escaped). A column like `(0.1, 0.2, 0.7, 0.0, 0.0)`, whose float sum is not exactly 1, is accepted.
10. Updater errors: an exception in `init` → `kind == "failed"`, `step is None`, message starts with the exception type name; in `update` at step 7 → `step == 7`; a return value that is a list, or a tuple of length 3 → `failed`; a 5000-character exception message is truncated to 1000; `KeyboardInterrupt` propagates.
11. Seeds: `init` receives `series_seed(seed, series_id)`; the function matches the formula above for a pinned example; different ids give different seeds; `seed` of `True`, `-1` and `2**63` raises `ArenaError`.
12. Argument validation (`ArenaError`, `init` never called): `end_origin == t0`, `end_origin == last_origin + 2`, `len(values) == end_origin + H - 1`, `horizon` different from the forecasts', `period` of `1`.
13. Determinism: two replays with the same inputs give equal traces.
14. AST guards: `updater/interfaces.py` imports nothing from `evolvets`; `replay.py` and `labels.py` never use the builtin `sum`, `math.fsum` or `statistics`; in `replay.py` the name `values` appears only as an argument of the `LabelRelease(...)` call and in the length check of step 1.

### Dependencies

None.

### Size

About 250 lines of implementation. If it grows past ~400, move `LabelRelease` and its tests into a first PR (`feat/arena-labels`) and the replay into a second.

---

## PR B: `feat/arena-replay-metrics`

Dev pools, metrics, and the replay of a whole dataset into a result that later PRs write to the ledger.

### Files and interfaces

`arena/dev.py`

```python
@dataclass(frozen=True)
class DevSeries:
    series_id: str
    period: int
    values: tuple[float, ...]  # exactly values[: score_end + horizon]
    scale_end: int  # MASE scale uses y_0 .. y_{scale_end - 1}
    score_start: int  # first scored origin
    score_end: int  # exclusive; also the replay's end_origin
    change_points: tuple[int, ...]  # synthetic only, else ()
    oracle: (
        tuple[tuple[float, ...], ...] | None
    )  # true-best forecasts, [origin - score_start][h-1]; synthetic only


@dataclass(frozen=True)
class DevPool:
    dataset: str  # one of CACHE_DATASETS
    cache_root: Path
    cache_id: str
    manifest: CacheManifest
    series: tuple[DevSeries, ...]  # cache manifest order


def load_dev_pool(
    dataset: str, cache_root: Path, data_root: Path, manifest_dir: Path
) -> DevPool: ...
```

`arena/metrics.py`

```python
RECOVERY_WINDOW_PERIODS = 2
RECOVERY_TOLERANCE = 0.25


@dataclass(frozen=True)
class SeriesScore:
    series_id: str
    mase: float
    oracle_mase: float | None
    recovery: tuple[tuple[int, int | None], ...]  # (change point, origins to recover or None)


def mase_scale(release: LabelRelease, period: int, scale_end: int, now: int) -> float: ...
def origin_losses(errors: Sequence[Sequence[float]]) -> tuple[float, ...]: ...
def recovery_times(
    losses: Sequence[float],
    oracle_losses: Sequence[float],
    score_start: int,
    score_end: int,
    change_points: Sequence[int],
    period: int,
) -> tuple[tuple[int, int | None], ...]: ...
def aggregate(scores: Sequence[SeriesScore]) -> dict[str, float]: ...
```

`arena/evaluate.py`

```python
@dataclass(frozen=True)
class ReplayResult:
    dataset: str
    cache_id: str
    arena_version: str
    status: Literal["ok", "failed", "invalid_weights"]
    error: str | None  # None iff ok
    failed_series: str | None
    failed_step: int | None
    metrics: dict[str, float]  # {} unless ok
    series: tuple[SeriesScore, ...]  # () unless ok


UpdaterFactory = Callable[[], Updater]  # zero-argument; called once per series


def replay_dataset(
    updater_factory: UpdaterFactory,
    config: Mapping[str, ConfigValue],
    seed: int,
    pool: DevPool,
) -> ReplayResult: ...
def result_to_obj(
    result: ReplayResult,
) -> dict[str, object]: ...  # JSON-ready, canonical_bytes-able
```

`arena/versions.py`

```python
def protocol_versions() -> dict[str, str]: ...


# {"ledger": SCHEMA_VERSION, "arena": ARENA_VERSION, "generator": GENERATOR_VERSION,
#  "cache_manifest": CACHE_VERSION}
```

### Exact behaviour: `load_dev_pool`

- Reads the cache with `read_manifest` and `cache_id`. Any `CacheError`, and any `PrepareError` (raised by `load_public` or `read_split_manifest`: missing split manifest, missing or tampered public file, unknown dataset), is wrapped in `ArenaError`, so every failure to load a dev pool surfaces as one type. `H = manifest.horizon`.
- `etth1`, `ecl`: `load_public(dataset, data_root, manifest_dir)` (public values only; it never opens the sealed or raw file). For each cache series, in manifest order, the public series with the same id. Ids, order and count must match exactly. `sha256_hex(canonical_bytes(list(values)))` must equal the cache's `values_sha256`, `len(values)` its `length`, `24` its `period`. Then `scale_end = warmup.end`, `score_start = dev.start`, `score_end = dev.end`, `change_points = ()`, `oracle = None`.
- `synthetic-dev`: `per_family = int(manifest.source["per_family"])`, and `manifest.source["generator_version"]` must equal `GENERATOR_VERSION`. Tasks are `dev_tasks(family, per_family)` in `FAMILIES` order with ids `f"{family}/{i}"`, matched and hash-checked against the manifest as above. `scale_end = score_start = GrammarConfig().warmup` (300; no change point or burst before it), `score_end = length - H`, `change_points = world.change_points`, `oracle = tuple(true_best_forecast(world, t, H) for t in range(score_start, score_end))`.
- Every series must satisfy `first_origin <= score_start < score_end`, `period + 1 <= scale_end <= score_start`, and `score_end + H <= len(public values)`, else `ArenaError`. (The 520-row TASK_005 fixtures fail this: their dev segment starts at 171, before `first_origin(24) = 287`. Tests use bigger fixtures, see below.) `values` is truncated to `[: score_end + H]` after the hash check, so nothing past the last scored target is held.
- Any other dataset name → `ArenaError`.

### Exact behaviour: metrics

- `mase_scale(release, period, scale_end, now)`: `y = release.history(scale_end, now)`; the mean over `i = period .. scale_end - 1` (ascending) of `abs(y[i] - y[i - period])`. If the result is `0.0` or not finite, raise `ArenaError` (a pool problem, not an updater failure). `replay_dataset` calls it with `now = score_start`.
- `origin_losses(errors)`: per origin, the mean over `h` (ascending) of `abs(error)`.
- Series `mase`: the mean over scored origins `s = score_start .. score_end - 1` (ascending) of `origin_losses`, divided by the scale. Since every origin has `H` errors this equals the mean of all `|error|` over the scale. `oracle_mase` is the same for the oracle errors `oracle[s - score_start][h-1] - target`. Targets come from `release.targets(s, now=s + H)`.
- `recovery_times`: `w = RECOVERY_WINDOW_PERIODS * period`. `losses` and `oracle_losses` are indexed by `s - score_start`. Consider each change point `c` with `score_start <= c < score_end`, in ascending order; change points outside are skipped, not counted. Let `B` be the next change point after `c`, or `score_end` if there is none. The recovery time is the smallest `k >= 0` with `c + k + w <= B` such that the sum of `losses` over `[c + k, c + k + w)` is `<= (1 + RECOVERY_TOLERANCE)` times the sum of `oracle_losses` over the same window (both left-to-right). If no `k` qualifies, it is `None` (censored), including when `c + w > B`.
- `aggregate(scores)` (series order, left-to-right): `n_series`; `mase` = the mean of the series' `mase`. If any score has `oracle_mase` (synthetic): `oracle_mase` = the mean of those; `regret = mase - oracle_mase`; `recovery_n` = the number of recovery entries; `recovery_censored` = how many are `None`; `recovery_mean` = the mean of the non-`None` entries, a key present only if there is at least one. All values are `float`.

### Exact behaviour: `replay_dataset`

- **A fresh updater per series (cross-series isolation).** `replay_dataset` takes a zero-argument `updater_factory`, calls it once per series — in pool order, immediately before that series' replay — and passes the new instance only to that series' `replay_series`. The reason is AGENTS.md 2.2: the real pools are different columns or clients over one shared timeline (one CSV for every ETTh1 column, one window for every ECL client), and the replay is sequential, so an instance that survives from series 0 into series 1 holds series 0's matured feedback — values at calendar times up to `score_end - 1 + H` — while series 1 is still at its early origins: the future of a correlated sibling series. An updater that merely caches statistics on `self` hits this innocently. A fresh instance closes the path at the object level. The alternative, lockstep replay (advance every series through origin `t` before any series reaches `t + 1`), was considered and rejected: it must hold every series' forecasts in memory at once, against this task's one-series-at-a-time rule, and `Context` deliberately carries no series id, so a single instance fed interleaved contexts from all series could not attribute feedback to a series anyway — under this contract, cross-series state mixing is noise, not signal. What a fresh factory does not fix is module-level or process-global state shared behind the factory's back; that is out of scope here (see "Not in this task"), and entry 9 must isolate updaters per series, not just per evaluation. The factory discipline is part of the arena protocol and so of `ARENA_VERSION`.
- For each series in pool order: `updater = updater_factory()`, then `forecasts = load_series(pool.cache_root, pool.manifest, series_id)` (one series in memory at a time; `CacheError` → `ArenaError`), then `replay_series(updater, config, seed, series_id, forecasts, values, period, H, end_origin=score_end)`. The updater runs from the cache's `first_origin`, so origins before `score_start` are an unscored burn-in. Then score the series.
- An `UpdaterFailure` stops the replay: `status = failure.kind`, `error = failure.message`, `failed_series`, `failed_step`, empty `metrics` and `series`. `ArenaError` propagates (it is not the updater's fault).
- On success: `status = "ok"`, metrics from `aggregate`, `arena_version = ARENA_VERSION`.
- `result_to_obj` gives a dict with exactly the `ReplayResult` field names; `series` becomes a list of `{"series_id", "mase", "oracle_mase", "recovery": [[c, k_or_null], ...]}`. `canonical_bytes` must accept it. It is the planned content of an evaluation's `per_task_blob`, and `metrics` is the planned `metrics` field of `evaluation`/`baseline` records. `status` values are the ledger's `EVALUATION_STATUSES` (except `timeout`, which entry 9 adds).
- Budget: `replay_dataset` charges nothing itself. The caller (entries 10/11, Phase 1 runner) wraps it and charges `evaluations` and `wall_seconds` the same way for every condition (AGENTS.md 2.5).

### Tests for PR B (all required; `tests/arena/test_dev.py`, `test_metrics.py`, `test_evaluate.py`)

Real-data tests use the TASK_005 helpers with a longer ETTh1 fixture: `_build_etth1_fixture(fixtures, rows=1200)` gives warmup `[0, 139)`, dev `[307, 550)` and sealed from 718. Then `prepare`, then `build_cache` from `public_inputs`. Synthetic tests build a `synthetic-dev` cache with `per_family = 1`.

1. Real pool: ids in order; `scale_end == 139`, `score_start == 307`, `score_end == 550`; `values == public[:550 + 24]`; `oracle is None`. With the sealed processed file and the raw file deleted it still loads, and recording every file open (as in TASK_005 test 10) shows neither is opened.
2. Synthetic pool: 6 series `stationary/0 .. mixed/0`; `score_start == 300`; `score_end == 2000 - 24`; `change_points == world.change_points`; `oracle[0] == true_best_forecast(world, 300, 24)`.
3. Pool mismatches raise `ArenaError`: a cache built from a different fixture variant (`variant=1`, so hashes differ); a cache whose `source.generator_version` is edited (rewrite the manifest in `tmp_path`); a missing cache; the 520-row fixture (`score_start < first_origin`); an unknown dataset; a missing public file (deleted after `prepare`, so `load_public` raises `PrepareError` internally but `ArenaError` escapes).
4. `mase_scale` on a hand-written series with `period = 2` matches a hand computation; a series whose first `scale_end` values are constant raises `ArenaError`; `history` is called with `now = score_start` (spy).
5. `origin_losses` and series `mase` on a hand-built trace (`H = 2`, 4 scored origins) match values written out in the test.
6. `recovery_times`, hand-built losses with `period = 2` (`w = 4`): recovery at `k = 0`; at `k = 3`; censored because the loss never drops; censored because `c + w > B`; a change point before `score_start` skipped; two change points where the second bounds the first.
7. `aggregate`: real-only scores give exactly `{n_series, mase}`; synthetic scores give `oracle_mase`, `regret`, `recovery_n`, `recovery_censored`, and `recovery_mean` only when some recovery is not `None`; values match a hand computation.
8. Failures: a factory whose second returned instance raises during `update` gives `status == "failed"`, that `failed_series`, its step, `metrics == {}` and `series == ()`; a first-instance `init` failure likewise; invalid weights from any instance give `"invalid_weights"`. A corrupted cache series file gives `ArenaError`, not a failed result.
9. No look-ahead end to end: on the real pool, wrap `LabelRelease.targets` and `history` as in PR A test 4. Every `targets` call has `origin + H == now`, and `history` is called only with `end <= now + 1`. The largest value index any call touches is `score_end - 1 + H`.
10. Oracle sanity on the synthetic pool: a factory whose instances put all weight on `naive` gets `mase > oracle_mase` on every series, and `regret > 0`.
11. `protocol_versions()` has exactly the keys `PROTOCOL_KEYS` with the four version constants, and `RunStart(..., protocol_versions=protocol_versions(), ...)` validates.
12. `result_to_obj`: exact key set; `canonical_bytes` accepts it for an ok and a failed result.
13. Golden: an equal-weights updater (a factory defined in the test) on the `synthetic-dev` pool with `per_family = 1` and `seed = 0`. The SHA-256 of `canonical_bytes(result_to_obj(result))` is pinned in the test file. The docstring says changing it requires bumping `ARENA_VERSION` (or `EXPERTS_VERSION`/`GENERATOR_VERSION` if those changed).
14. Determinism: two `replay_dataset` calls give equal results. An AST guard: no module in `arena/metrics.py`, `arena/evaluate.py`, `arena/dev.py` uses builtin `sum`, `math.fsum` or `statistics`.
15. Fresh instance per series: a spy factory records every call and returns a new spy updater each time. On the real pool it is called exactly `len(pool.series)` times, in pool order (the k-th returned instance is the one whose `init` receives `series_seed(seed, pool.series[k].series_id)`), and the instances are pairwise distinct (`is not` on every pair). Each instance saw only its own series: its first recorded `Context` has `step == 0`, and it received exactly `score_end - first_origin` `update` calls. The residual — module or process state shared behind the factory's back — is not observable in-process and is recorded under "Not in this task" as an entry-9 requirement.

### Running it for real

Not required for merge, but the PR description must paste, for an equal-weights updater factory and for each single expert (one-hot weights), the `metrics` of `replay_dataset` on `etth1`, `ecl` and `synthetic-dev` (`--per-family 20` cache), with wall-clock times. These are sanity numbers, not results: they are not written to any ledger and not used for tuning. A script for this goes in the PR description, not in the package.

### Dependencies

None.

### Size

About 300 lines of implementation. If it grows past ~400, move `arena/dev.py` and its tests into a first PR (`feat/arena-dev-pools`) and keep metrics and `replay_dataset` in the second.

---

## Not in this task

- `.importlinter` and CI probe changes (human, issue #32).
- The sealed process, sealed caches, the sealed synthetic salt and the `timeout` status (entry 9).
- **Per-series process isolation for updaters (entry 9).** A fresh instance per series closes object-state leakage across series; module-level or process-global state shared behind the factory is not detectable in-process. Entry 9 must therefore isolate updaters per series, not just per evaluation.
- `updater/interfaces.py` lives in an agent-editable package, and the trusted arena imports it. That is acceptable for Phase 0, where only the developer agent writes updater code. Before Phase 1 lets the improver edit updater code, either the improver must be barred from editing `updater/interfaces.py`, or the arena must stop importing from `updater/` and the contract must move into `arena/`. That decision belongs to the Phase 1 specs; recorded here so it is not lost.
- Hand-designed updaters, tuning grids, `baseline` ledger records and the best-single-expert and switching oracles on real data (entry 10). The real-data oracles need targets, so they belong in `arena/`, not `updater/`.
- A replay CLI: there is no updater to run yet; entry 10 adds one.
- Issue #10 (tag `ledger-schema-v1`) is a human action and is not blocked or changed by this task.

## Closing the task

PR B (`feat/arena-replay-metrics`, or the last PR of its split) changes this file's first line to `Status: done`.
