Status: done (PR A merged as #145; PR B as #146 with its CLI split #147; PR C split by the size rule into C1 #152 and C2 #153)

# Task 016: forecast-output updaters (option B) and fresh-seed holdout replicates

Read `AGENTS.md` and `docs/ROADMAP.md` first. The owner asked for Phase 1 in issue #142, after picking the Phase 1 setting on #141. This task covers items (a) and (b) of #142. It is a stack of three small PRs: open them in order and stop after each one for review. One `.github/` workflow amendment is specified at the end. **The owner's delegate applies it** (AGENTS.md section 7), and the developer never writes it.

Items (c) to (h) of #142 are later tasks. Item (c), the improver, waits for the owner's answers on #143 (LLM provider, client, key handling, cost ceiling). Nothing in this task depends on those answers.

## Why

The owner's decisions on #141 and #142:

- **Setting.** Phase 1 runs on `synthetic-shift3` and `synthetic-shift2` with pool `simple+foundation`. These are the top two `simple+foundation` pools of the headroom scan (`docs/results/headroom-scan.md`): switch-8 headroom 0.0754 [0.0636, 0.0899] and 0.0580 [0.0524, 0.0642]. One evaluation takes about 163 s. `synthetic-shift1` and `synthetic-wide` are secondary checks. ETTh1 and ECL are not used for Phase 1 claims.
- **Option B.** Updaters may output forecasts, not only convex weights: for example a bias or level correction after a break, or a residual adjustment on top of the combination. This is a protocol change.
  - `ARENA_VERSION` is bumped and the updater interface is versioned.
  - The weights-only interface keeps working, so weights-only updaters remain a baseline.
  - Results from the two versions are never mixed in one table.
  - The look-ahead rules do not change: an updater still sees only `Context` and matured errors.
- **Honest verification without ceremony.** There are no sealed runs. Instead, held-out **fresh-seed** replicates of the same presets (a new `scan_world_seed` index range) score accepted edits. They are never used for selection.

This task gives Phase 1 its object space (PR A), a reference forecast-output updater that later becomes the starting point of a lineage (PR B), and the holdout datasets (PR C).

- **PR A** `protocol/forecast-updaters`: interface `forecast-v2` beside `weights-v1`, replay support, `ARENA_VERSION = "4"`. Labelled `protocol-change`.
- **PR B** `feat/forecast-updater-reference`: `bias_share`, a forecast-v2 updater (Fixed-Share plus an online per-horizon bias correction), and `arena replay` support for forecast-v2 updaters.
- **PR C** `feat/holdout-replicates`: datasets `synthetic-shift2-holdout` and `synthetic-shift3-holdout`, drawn from scan indexes 1000 and up. Every cache and arena tool accepts them, and the tune and probe commands refuse them.
- **W11** (owner-applied): the holdout datasets join `gpu-foundation-cache.yml`. The owner then dispatches the builds.

## Decisions already made (do not reopen)

- **Owner, #141 and #142:** the setting, option B, and holdout replicates that are never used for selection, all as above. There are no sealed runs, and results must be faithful and reproducible. Long runs on sp-1 use detached launches like `.github/scripts/headroom-detached.sh` (#138).
- **TASK_007/010:**
  - A fresh updater is used for each series.
  - `replay_dataset` is the only way to score an updater.
  - Every target goes through `LabelRelease`, and feedback for origin `s` arrives exactly once, at `s + H`.
  - Floats are accumulated in explicit left-to-right loops.
  - The hindsight oracles bound single-expert selection only.
- **TASK_015:**
  - `SCAN_PRESETS`, `scan_world_seed`, `preset_to_dict` and `preset_sha256` are unchanged, so every existing scan cache still loads.
  - `GENERATOR_VERSION` stays `"1"`.
  - `all` on the command line expands to `CACHE_DATASETS` only.
- **Ledger schema v1 is frozen and is not changed.**
  - `EVALUATION_STATUSES` stays `ok | failed | timeout | invalid_weights`, so a malformed forecast output is a `failed` result whose error starts with `invalid_forecast:` (PR A).
  - `PROTOCOL_KEYS` is unchanged. Arena version `"4"` goes in the existing `arena` key.

## Rules for all three PRs

- **The only protocol change is PR A.** PR A changes `updater/interfaces.py`, `arena/replay.py`, `arena/evaluate.py`, `arena/__init__.py` and, in one line only, `arena/sealed/server.py` (below). Nothing else in the arena protocol changes in this task. Nothing changes in `ledger/`, `PROTOCOL_KEYS`, `SCHEMA_VERSION`, `GENERATOR_VERSION`, `CACHE_VERSION`, any payload, label release, metrics or the oracles' definitions.
- **The sealed path keeps replaying `weights-v1`.** `arena/sealed/` has no forecast-v2 request and its call into `replay_dataset` is unchanged. Its one change rides PR A: the hand-built timeout result object in `_replay` gains `"updater_interface": "weights-v1"` — the only value it can have there — so it keeps `result_to_obj`'s exact key set, as its docstring promises.
- **Standard library only.** No new dependency.
- **Updater modules** use only `+ - * /`, comparisons and `abs`, accumulated in explicit left-to-right loops, as in `updater/baselines.py`. That means no `**`, no `sum()` on floats, no `math.fsum` and no `statistics`.
- **Seeds:** `bias_share` validates its seed and ignores it, like the baselines. The holdout worlds use `scan_world_seed` unchanged.
- **Agents never run anything on lab data.** Tests use fixtures in `tmp_path`.
- **Strict mypy.** `evolvets.arena.*` is already strict. `evolvets.updater.forecast` follows whatever `updater/` uses now. Do not weaken any override.
- **Import contracts.** The `.importlinter` contracts stay unchanged and must pass. `updater/forecast.py` imports only the standard library, `evolvets.updater.interfaces` and `evolvets.updater.baselines`.
- Do not edit `.importlinter`, `.github/`, `AGENTS.md` or `docs/agents/`.

---

## PR A: `protocol/forecast-updaters`

This PR changes the arena protocol (AGENTS.md 2.4):

- The branch is `protocol/forecast-updaters`.
- The PR is labelled `protocol-change`. If the label does not exist, create it with `gh label create protocol-change --description "Changes a frozen protocol component"`.
- The description explains the impact on existing results:
  - every weights-only score is unchanged;
  - sealed timeout result objects gain `"updater_interface": "weights-v1"` and keep `result_to_obj`'s key set; no stored blob changes interpretation;
  - arena-3 and arena-4 results are never mixed in one table, which the reports already enforce through the protocol-version check;
  - the headroom scan's numbers stay arena-3 numbers.

### Files and interfaces

`src/evolvets/updater/interfaces.py` (additions; the existing names are unchanged):

```python
UPDATER_INTERFACES: tuple[str, ...] = ("weights-v1", "forecast-v2")
FORECAST_ABS_LIMIT = 1e12  # bound on |forecast value|; keeps every metric finite


@dataclass(frozen=True)
class ForecastOutput:
    """A forecast-v2 updater's own point forecast for targets origin+1 .. origin+H."""

    values: tuple[float, ...]  # [h - 1]


Output = Weights | ForecastOutput


class ForecastUpdater(Protocol):
    """A forecast-v2 updater: like Updater, but update may return a ForecastOutput."""

    def init(
        self, n_experts: int, horizon: int, config: Mapping[str, ConfigValue], seed: int
    ) -> object: ...

    def update(
        self, state: object, feedback: tuple[Matured, ...], context: Context
    ) -> tuple[Output, object]: ...
```

- The module docstring gains one paragraph saying that `weights-v1` is the original contract and `forecast-v2` its extension. A forecast-v2 updater sees exactly what a weights-v1 updater sees.
- The `Matured.combined_errors` comment becomes `# [h - 1] = this updater's output forecast - target (the weighted combination under weights-v1)`.
- `Updater` is unchanged.

`src/evolvets/arena/replay.py`:

```python
def replay_series(
    updater: Updater | ForecastUpdater,
    config,
    seed,
    series_id,
    forecasts,
    values,
    period,
    horizon,
    end_origin,
    expert_names: Sequence[str] = EXPERT_NAMES,
    *,
    interface: str = "weights-v1",
) -> SeriesTrace: ...
```

`src/evolvets/arena/evaluate.py`:

```python
@dataclass(frozen=True)
class ReplayResult:
    ...  # every existing field, in the existing order, then:
    updater_interface: str  # one of UPDATER_INTERFACES


UpdaterFactory = Callable[[], Updater | ForecastUpdater]


def replay_dataset(
    updater_factory: UpdaterFactory,
    config: Mapping[str, ConfigValue],
    seed: int,
    pool: DevPool,
    *,
    interface: str = "weights-v1",
) -> ReplayResult: ...
```

- `result_to_obj` adds the key `"updater_interface"`, placed directly after `"arena_version"`. Key order does not affect `canonical_bytes`, but it keeps the code readable.

`src/evolvets/arena/__init__.py`: `ARENA_VERSION = "4"`. The docstring gains one sentence: "Version 4 added the forecast-v2 updater interface (`updater/interfaces.py`): a forecast-v2 updater may return its own point forecast instead of weights; weights-v1 replays are bit-identical to version 3 apart from the version string and the recorded interface."

`src/evolvets/arena/sealed/server.py` (one line only, in PR A): the hand-built timeout result object in `_replay` gains `"updater_interface": "weights-v1"`, placed directly after `"arena_version"` as in `result_to_obj`, keeping the docstring's promise of `result_to_obj`'s key set with `status = "timeout"`. `arena/sealed/` is already inside the arena protocol (`arena/__init__.py`), so this rides PR A's protocol change; the sealed server's replay call, and everything else in `arena/sealed/`, is unchanged.

### Exact behaviour

**Interface check.** `replay_series` and `replay_dataset` raise `ArenaError` when `interface` is not in `UPDATER_INTERFACES`. The check happens before anything else runs: no `init` call, no cache read. A non-str value is rejected the same way.

**`weights-v1`.** The code path is exactly today's. Anything other than valid weights fails as `invalid_weights` with today's messages. A `ForecastOutput` returned under `weights-v1` fails as `invalid_weights`: the existing type check rejects it, because it is not a list or a tuple.

**`forecast-v2`.** Let `output, state = pair` (the existing "update must return (weights, state)" checks still apply, with "weights" replaced by "an output" in the message only under forecast-v2).

- **Weights.** If `type(output) is not ForecastOutput`, `output` is validated as weights, exactly as under `weights-v1`, and combined as now. A forecast-v2 updater may return weights at some steps and forecasts at others.
- **A forecast.** If `type(output) is ForecastOutput` (a subclass is not accepted), the arena copies and validates `output.values` the way `_validated_weights` copies:
  - `values` must be an exact `list` or `tuple`;
  - every element must be an exact `int` or `float`, converted with `float()`;
  - the length must equal `horizon`;
  - every value must be finite with `abs(value) <= FORECAST_ABS_LIMIT`.

  Any violation raises `UpdaterFailure("failed", "invalid_forecast: <reason>", series_id, k)`. Exceptions raised while reading `output.values` are caught and reported the same way. The validated copy becomes this origin's combined row, without further arithmetic.
- **Unchanged under forecast-v2:**
  - `Context`;
  - when feedback arrives;
  - `Matured.errors`;
  - `Matured.combined_errors`, which is row minus targets, whatever produced the row;
  - the final drain;
  - `SeriesTrace`.

  No new information reaches the updater.
- **The status `ReplayResult` carries** comes from `UpdaterFailure.kind`, as now: `invalid_weights` for bad weights and `failed` for bad forecasts or exceptions.

**`replay_dataset`** passes `interface` through to every `replay_series` call. It stamps `updater_interface=interface` on every result: `ok`, `failed` and `invalid_weights` alike.

**Callers that are not touched.** `gate1 tune`, `gate1 probe`, `arena replay` and the sealed server keep calling `replay_dataset` without `interface`, so they replay `weights-v1` (the sealed server's only change is the timeout key above, not its replay call). `oracle_dataset` has no updater, and its result changes only in `arena_version`.

### Tests for PR A (all required)

In `tests/arena/test_replay.py` and `tests/arena/test_evaluate.py`, using updaters defined in the test files:

1. **Interface check:** `interface` values of `"weights-v2"`, `""`, `None` and `1` each raise `ArenaError` from both `replay_series` and `replay_dataset`, and a spy factory or `init` is never called.
2. **Weights-v1 unchanged:**
   - Under `weights-v1`, an updater returning `ForecastOutput(...)` gives `invalid_weights`.
   - Every existing test in `tests/arena/` and `tests/updater/` passes unchanged except the digest adjustments listed in test 9 (the repins and the stamp-back drops) and the version-string updates of test 8.
3. **Forecast output scores as given:** a forecast-v2 updater returns the same constant row `c` at every step on a hand-built series. The trace errors equal `c[h] - target` exactly, and the `combined_errors` it receives in `Matured` equal the same values.
4. **Mixed outputs:** an updater alternates weights (even steps) and forecasts (odd steps). Each trace row equals the weighted combination or the forecast, respectively.
5. **Equivalence:** a forecast-v2 updater that computes `sum_e w[e][h] * f[e][h]` itself, in the replay's loop order, from the same equal weights, gives a trace bit-identical to the weights-v1 equal-weights replay.
6. **Invalid forecasts** (each is `failed`, the error starts with `invalid_forecast:`, and `failed_step` is right):
   - the wrong length;
   - NaN, `inf` and `-inf`;
   - `1e13`;
   - an `int` element too large for a float (`10**400`): `float()` raises `OverflowError` inside the copy, caught and reported like any other read failure;
   - a `bool` element;
   - an element of a `float` subclass;
   - `values` given as a generator;
   - a `ForecastOutput` subclass;
   - `values` whose `__iter__` raises.
7. **No new information:** a recording forecast-v2 updater stores every `Context` and `Matured` it receives. On the same pool these equal, field by field, what a recording weights-v1 updater receives when both return the same combination (the forecast-v2 one through test 5's computation). Feedback for step `s` arrives exactly at step `s + H`.
8. **Result stamping:**
   - `replay_dataset` stamps `updater_interface` on `ok`, `failed` and `invalid_weights` results under both interfaces.
   - `result_to_obj` has the exact new key set.
   - `arena_version == "4"`, and `protocol_versions()["arena"] == "4"` (update `tests/arena/test_pool.py` lines 182–183, saying why in the PR).
9. **Version 4 changes no weights-only score.** This is the owner's "existing weights-only results replay unchanged" requirement.
   - Add `VERSION3_DIGESTS` to `tests/arena/test_oracles.py`, holding today's (version-3) digests:
     - the TASK_007 golden equal-weights replay (`test_evaluate.py` test 13);
     - the five baselines (`tests/updater/test_baselines_replay.py` `GOLDEN_DIGESTS`);
     - the oracle golden (`test_oracles.py` test 9).
   - For each replay, `result_to_obj` with `"updater_interface"` removed and `"arena_version"` set to `"3"` must hash to its version-3 digest. For the oracle, `oracle_to_obj` with `"arena_version"` set to `"3"` must hash to its version-3 digest.
   - Then repin the version-4 digests in those three places. Their docstrings say version 4 repinned them and this test shows why.
   - The existing `VERSION2_DIGESTS` test keeps passing: stamp `"2"`, and drop `"updater_interface"`, in the same way. Adjust it in the same commit and explain why in the PR.
   - The version-1 stamp-back in `tests/arena/test_pool.py` (`test_simple_pool_reproduces_version_1`) also drops `"updater_interface"`: add it to the `V2_EXTRA_KEYS` tuple of dropped keys — version 1's object has neither the pool fields nor the interface key — and adjust the test's docstring, which counts the dropped keys. `V1_GOLDEN_DIGEST` is unchanged. Explain why in the PR.
10. **The pooled spec:** a forecast-v2 replay on the `simple+foundation` fixture pool (fake foundation adapter, as in the existing pool tests) is `ok`, with `pool_spec` and `foundation_cache_ids` stamped as for weights-v1.
11. **Look-ahead guard still holds:** an existing look-ahead test (`LabelRelease` refusing a premature target) is re-run under `forecast-v2` through a forecast-v2 updater, and still raises at the same point. Parametrise the existing test; do not copy it.
12. **Sealed path:** `tests/arena/test_sealed_server.py` passes unchanged except for the version string, and a sealed replay's result object carries `"updater_interface": "weights-v1"`. In `test_timeout_end_to_end`, the timeout blob's `result` key set equals `result_to_obj`'s key set exactly, and it carries `"updater_interface": "weights-v1"`.
13. **Ledger and reports:** a `gate1 tune` fixture run now records `protocol_versions["arena"] == "4"`. `build_report_data` over one fixture run built under `"4"` and one hand-built run claiming `"3"` raises the existing protocol-mix `ReportError`.

### Dependencies

None.

### Size

About 120 lines of implementation and 250 of tests. If it exceeds ~400 lines:

- `protocol/forecast-updaters`: the interfaces, replay, evaluate, the version bump, and tests 1 to 9;
- `protocol/forecast-updaters-checks`: tests 10 to 13.

The second PR is tests only. Both PRs are labelled `protocol-change`.

### After PR A merges

The reviewer opens an `fyi-human` issue asking the owner to tag `arena-v4` on the merge commit. Issue #55 (tag `arena-v1`) is still open, so the issue links it.

---

## PR B: `feat/forecast-updater-reference`

### Files and interfaces

`src/evolvets/updater/forecast.py` (new):

```python
FORECAST_UPDATER_IDS: tuple[str, ...] = ("bias_share",)
FORECAST_DEFAULT_CONFIGS: Mapping[str, Mapping[str, ConfigValue]]
# {"bias_share": {"eta": 1.0, "alpha": 0.01, "bias_rate": 0.1}}, read-only like DEFAULT_CONFIGS


class BiasShare:
    """Fixed-Share combination minus an online per-horizon estimate of its own bias."""

    def init(self, n_experts, horizon, config, seed) -> object: ...
    def update(self, state, feedback, context) -> tuple[Output, object]: ...


def bias_share() -> ForecastUpdater: ...


FORECAST_UPDATERS: Mapping[str, Callable[[], ForecastUpdater]]  # MappingProxyType
```

`runner/cli.py`:

- `arena replay --updater` accepts `BASELINE_IDS + FORECAST_UPDATER_IDS`.
- A forecast updater id is replayed with `interface="forecast-v2"` and its `FORECAST_DEFAULT_CONFIGS` entry (or `--config`, as now).
- A baseline id keeps `weights-v1`.
- The printed JSON already contains `updater_interface` through `result_to_obj`.

### Exact behaviour

**Config.**

- `init` accepts exactly the keys `eta`, `alpha` and `bias_rate`, reusing the baselines' `_checked_init`, `_checked_eta` and `_checked_alpha`.
- `bias_rate` is a finite float in `[0.0, 1.0]`. Anything else, including a bool, raises `ValueError`.
- The seed is validated as the baselines validate it, then ignored.

**State.** All state is per series, held in the object `init` returns and never on the instance:

- a Fixed-Share state, from `FixedShare().init(n_experts, horizon, {"eta": ..., "alpha": ...}, seed)`;
- `bias: list[float]` of length `horizon`, all `0.0`;
- `applied: dict[int, tuple[float, ...]]`, the bias subtracted at each step that has not matured yet.

**`update(state, feedback, context)`:**

1. For each `matured` in `feedback`, in order:
   - `applied_row = applied.pop(matured.step)`. A missing step raises `ValueError`, which the replay reports as `failed`.
   - For each `h`: `residual = matured.combined_errors[h] + applied_row[h]`. This is the Fixed-Share combination's own error, since the output was combination minus bias.
   - `bias[h] = bias[h] + bias_rate * (residual - bias[h])`.
2. Call `FixedShare().update(fs_state, feedback, context)`. Fixed-Share reads only `matured.errors`, never `combined_errors`, so passing the same feedback is exact. The call gives `weights`.
3. For each `h`:
   - `base = 0.0`;
   - for each `e` in order: `base = base + weights[e][h] * context.forecasts[e][h]` (the replay's own loop);
   - `out[h] = base - bias[h]`.
4. `applied[context.step] = tuple(bias)`. Return `(ForecastOutput(tuple(out)), state)`.

**Edge cases.**

- `bias_rate = 0.0` keeps `bias` at `0.0`. Each output is `base - 0.0`, which equals the replay's weights-v1 combination of the same weights bit for bit, so the trace equals Fixed-Share's.
- `bias_rate = 1.0` sets the bias to the last residual.
- `applied` never holds more than `horizon + 1` entries.

### Tests for PR B (all required; `tests/updater/test_forecast.py`)

1. **Registry:**
   - `tuple(FORECAST_UPDATERS) == FORECAST_UPDATER_IDS`;
   - the mapping and the configs are read-only;
   - the ids are disjoint from `BASELINE_IDS` (`evolvets.updater.baselines`) and `LEARNED_IDS` (`evolvets.updater.learned`; the test imports it from there).
2. **Config validation:** a missing key, an extra key, `bias_rate` of `-0.1`, `1.5`, NaN or `True`, and a bad `eta` or `alpha` each raise `ValueError`.
3. **Reduces to Fixed-Share:** with `bias_rate = 0.0`, `replay_dataset(bias_share, cfg, 0, synthetic_pool, interface="forecast-v2")` has per-series MASE equal (`==`) to `replay_dataset(fixed_share, {"eta": 1.0, "alpha": 0.01}, 0, synthetic_pool)`.
4. **Bias recursion by hand:** a 1-expert, horizon-2 hand-built series with a constant forecast offset `+d`. After the first maturation the bias moves toward `d` by exactly `bias_rate * (d - 0)`, and the later outputs equal the hand-computed values.
5. **Removes a level shift:** on a hand-built 2-expert series whose target steps up by 10 at origin `T`, mean absolute error over the origins after `T + 50` is lower than Fixed-Share's at `bias_rate = 0.1`.
6. **Per-series state:** one instance serves two series' states side by side, and each gives the same outputs as on its own.
7. **Missing maturation:** a `Matured` with an unknown step makes `update` raise `ValueError`, and a replay that injects it reports `failed`. Use a direct call; the replay never sends an unknown step.
8. **Through the arena:** at its default config, `bias_share` replays the `synthetic-dev` (`per_family = 1`) fixture with `status == "ok"` and `updater_interface == "forecast-v2"`. The SHA-256 of `canonical_bytes(result_to_obj(result))` is pinned, with a docstring like `GOLDEN_DIGESTS`'s.
9. **Weights-v1 refusal:** replaying `bias_share` with the default `interface` gives `invalid_weights` at step 0.
10. **CLI:**
    - `arena replay synthetic-dev --updater bias_share` on a `per_family = 1` cache prints one JSON line with `"updater_interface":"forecast-v2"` and `"status":"ok"`.
    - `--updater fixed_share` still prints `"weights-v1"`.
11. **Import discipline:** `updater/forecast.py` contains no `**`, no `sum(` and no `math.fsum`, checked by a source scan like the one in `tests/updater/` for `learned.py`. The import-linter contracts pass.

### Dependencies

None.

### Size

About 120 lines of implementation and 200 of tests. There is no natural split. If it grows past ~400 lines, move CLI test 10 and the CLI change into `feat/forecast-updater-cli`.

---

## PR C: `feat/holdout-replicates`

### Files and interfaces

`src/evolvets/arena/synthetic/holdout.py` (new; imports only the standard library, `evolvets.arena.synthetic.*` and `evolvets.ledger.canonical`):

```python
HOLDOUT_INDEX_START = 1000  # first scan index of every holdout replicate
HOLDOUT_INDEX_STOP = 2000  # exclusive; the holdout range is [START, STOP)


@dataclass(frozen=True)
class HoldoutPreset:
    name: str  # the cache dataset name
    base: str  # the scan preset it replicates (a key of SCAN_PRESETS)
    index_start: int  # == HOLDOUT_INDEX_START


HOLDOUT_DATASETS: tuple[str, ...] = ("synthetic-shift2-holdout", "synthetic-shift3-holdout")
HOLDOUT_PRESETS: Mapping[str, HoldoutPreset]  # MappingProxyType, HOLDOUT_DATASETS order


def holdout_tasks(name: str, family: str, n: int) -> list[SyntheticTask]: ...
def holdout_series_id(family: str, i: int) -> str: ...  # f"{family}/{HOLDOUT_INDEX_START + i}"
```

`src/evolvets/cache/datasets.py`:

```python
DEV_DATASETS = (*CACHE_DATASETS, *SCAN_DATASETS, *HOLDOUT_DATASETS)
```

`CACHE_DATASETS` and `all` are unchanged.

`src/evolvets/arena/dev.py` gains `_holdout_series(manifest)`, and `load_dev_pool` dispatches holdout datasets to it.

### Exact behaviour

**Worlds.**

- `holdout_tasks(name, family, n)` returns `[make_task(family, scan_world_seed(p.base, family, p.index_start + i), SCAN_PRESETS[p.base].config) for i in range(n)]`, where `p = HOLDOUT_PRESETS[name]`.
- `family` must be in the base preset's families.
- `n` is an int in `[0, HOLDOUT_INDEX_STOP - HOLDOUT_INDEX_START]`, with bools rejected.
- An unknown name, a bad family or a bad `n` raises `ValueError`.
- These are the same preset and the same seed function as the dev scan datasets, at indexes the dev scan caches never use.

**Keeping the ranges disjoint.**

- `public_inputs` raises `CacheError` for a scan dataset (`SCAN_DATASETS`) with `per_family > HOLDOUT_INDEX_START`, and `_scan_series` raises `ArenaError` for one.
- Existing scan caches use 20 and 60, so they are unaffected.
- This is the only change to scan-dataset handling.

**`public_inputs` for a holdout dataset:**

- The series are `SeriesInput(holdout_series_id(family, i), task.world.y, task.world.spec.period)`, for each family in the base preset's order, then each `i`.
- The source is exactly `{"kind": "synthetic", "pool": "holdout", "preset": p.base, "preset_sha256": preset_sha256(SCAN_PRESETS[p.base]), "index_start": str(p.index_start), "generator_version": GENERATOR_VERSION, "per_family": str(count)}`.
- `per_family` is checked as for the scan datasets, and is also at most `HOLDOUT_INDEX_STOP - HOLDOUT_INDEX_START`.

**`_holdout_series(manifest)`** raises `ArenaError` unless all of these hold:

- `pool == "holdout"`;
- `preset` equals `HOLDOUT_PRESETS[manifest.dataset].base`;
- `preset_sha256` is current;
- `index_start == str(HOLDOUT_INDEX_START)`;
- `generator_version` is current;
- `per_family` is valid, and the series count matches.

It then builds the series with `checked_synthetic_series(manifest, expected)`, using the holdout series ids. The change points, oracle (true-best) and boundaries therefore come from the existing code.

**Never used for selection.** Holdouts score accepted edits only. Later tasks enforce this in the Phase 1 runner. This PR enforces the part that exists today:

- `gate1_tune._validate_spec` and the `gate1_probe` spec validation raise `ValueError("<dataset> is a holdout dataset: holdouts score accepted edits only and are never tuned on")` for any dataset in `HOLDOUT_DATASETS`.
- The CLI `choices` of `gate1 tune --dataset` and `gate1 probe --dataset` exclude `HOLDOUT_DATASETS`.
- `cache build|verify`, `foundation build|verify|compare` and `arena replay|oracle` accept them, through `DEV_DATASETS`.

**Leakage and the sealed side.**

- `pool_leakage_risk` needs no change: no model's `pretraining_overlap` names a holdout dataset.
- `arena/sealed/` is not touched.
- Holdout datasets are disjoint from `SEALED_DATASETS`.

**Reports.** `gate1_report_data._SYNTHETIC_DATASETS` is unchanged. No tune run can cover a holdout, so no report sees one.

### Tests for PR C (all required)

In `tests/arena/test_synthetic_holdout.py`:

1. **Registry:**
   - `tuple(HOLDOUT_PRESETS) == HOLDOUT_DATASETS`;
   - each `name` equals its key and passes the cache's dataset-name check;
   - each `base` is `"synthetic-shift2"` or `"synthetic-shift3"`, respectively;
   - each `index_start == HOLDOUT_INDEX_START`;
   - the mapping is read-only;
   - `HOLDOUT_DATASETS` is disjoint from `CACHE_DATASETS`, `SCAN_DATASETS` and `SEALED_DATASETS`.
2. **Worlds:**
   - `holdout_tasks(h, f, 3)[i]` equals `make_task(f, scan_world_seed(base, f, 1000 + i), base config)`, compared by `world.y` and `change_kinds`.
   - `holdout_tasks(h, f, 0) == []`.
   - `n = 1001`, `-1` and `True` each raise `ValueError`, and so do an unknown name and the family `stationary`.
3. **Disjoint from dev:** for each holdout and every base family, the seeds at holdout indexes `0..49` are disjoint from `scan_world_seed(base, f, i)` for `i < 1000`. The test computes all 1000 per family, which is cheap. The first holdout world differs from dev scan index 0 by `world.y`.
4. **Same distribution, by construction:** each holdout's base config is `SCAN_PRESETS[base].config` (identity), and its families equal the base's families.
5. **Scan unchanged:**
   - `SCAN_PRESETS`, `preset_sha256` of each scan preset and `GOLDEN_SHIFT1_SUDDEN_0` are unchanged;
   - every existing test in `tests/arena/test_synthetic_presets.py` passes unmodified;
   - a scan `per_family` of `1001` raises `CacheError` in `public_inputs`, and `ArenaError` from a hand-made manifest in `load_dev_pool`.

In `tests/cache/test_datasets.py` and `tests/arena/test_dev.py`:

6. **`public_inputs`:** for each holdout dataset with `per_family = 2`, the ids (`sudden/1000`, `sudden/1001`, ...), order, values, period and source keys are exactly as specified. A `per_family` of `None`, `0`, `True` or `1001` raises `CacheError`.
7. **Pool load:** a holdout cache built in `tmp_path` with `per_family = 1` gives one series per base family. Each series has `score_start == scale_end == 300`, change points equal to the world's, and an oracle equal to `true_best_forecast` at every scored origin. A `fixed_share` replay is `ok`, with an `oracle_mase` per series.
8. **Refusals** (`ArenaError` for each, from hand-made manifests as in the scan refusal tests):
   - `pool` is `"scan"`;
   - `preset` names the other base;
   - a `preset_sha256` mismatch;
   - `index_start` is `"0"`;
   - a `generator_version` mismatch;
   - a series-count mismatch.
9. **Foundation pool:** `build_foundation_cache` with the fake adapter on `synthetic-shift3-holdout` (`per_family = 1`), then `load_dev_pool(..., pool_spec="simple+foundation", foundation_models=["fake"])`, loads with `pool_leakage_risk == False`. A forecast-v2 `bias_share` replay on it is `ok` (PR B).

In `tests/runner/` and `tests/cache/test_cli.py`:

10. **Never tuned on:**
    - `TuneSpec` and the probe spec with a holdout dataset each raise the specified `ValueError`.
    - `gate1 tune --dataset synthetic-shift2-holdout` and `gate1 probe --dataset synthetic-shift3-holdout` are rejected by argparse.
    - `_gate1_axes` still expands `all` to exactly `CACHE_DATASETS`.
11. **CLI accepts holdouts:**
    - `cache build synthetic-shift2-holdout --per-family 1` prints its `OK` line.
    - `arena oracle synthetic-shift3-holdout` succeeds on that cache.
    - `arena replay synthetic-shift3-holdout --updater fixed_share` prints an `ok` result.

### Dependencies

None.

### Size

About 150 lines of implementation and 250 of tests. If it exceeds ~400 lines:

- `feat/holdout-presets`: `holdout.py` and tests 1 to 5;
- `feat/holdout-datasets`: datasets, pool loading, the refusals and the CLI, with tests 6 to 11.

### After PR C merges

The reviewer opens an `fyi-human` issue with these requests:

1. The owner's delegate applies **W11**.
2. The owner dispatches W11 for each of the 2 holdout datasets x 3 models (Chronos-2, TiRex, TimesFM 2.5), with `per_family = 20` (100 series each, like the dev shift caches). Two builds of one dataset never run at once.
3. The owner posts the six workflow run URLs and outcomes on that issue.

The same PR sets this file's first line to `Status: done` (see "Closing the task"). The builds are outside the developer's stack.

---

## Owner-applied workflow (`.github/`, applied by the owner's delegate, never by an agent)

**W11: amendment to `gpu-foundation-cache.yml`.**

- The `dataset` choices gain `synthetic-shift2-holdout` and `synthetic-shift3-holdout`.
- Their names start with `synthetic-`, so they already take the synthetic build path that #131 added (`cache build "$DATASET" --per-family "$PER_FAMILY"` and `--per-family` on `foundation build`).
- Nothing else changes, including `gpu-pick.sh`, `gpu-release.sh` and the OOM retries from #137.

**Cost estimate:** each holdout dataset has 100 series, like the shift presets, so each build costs about what the matching W8 build of `synthetic-shift2` or `synthetic-shift3` cost (`docs/results/headroom-scan.md`, provenance).

## Not in this task

- Items (c) to (h) of #142: the improver, self-model, allocator and baselines, run CLI and detached sp-1 launcher, report, and meta-level replay revision. The roadmap lists them as Phase 1 entries 16 to 21. Item (c) waits for #143.
- Re-tuning baselines under arena 4. Phase 1's weights-only conditions replay under arena 4 inside the Phase 1 runs, and their results are never tabled with the arena-3 headroom numbers.
- Holdouts for `synthetic-shift1`, `synthetic-wide` or `synthetic-dev`. These are secondary checks (#142), and a holdout can be added later in the same way.
- Probabilistic outputs and metrics (CRPS, calibration), real hard domains (#127) and larger expert pools.
- Forecast-v2 on the sealed path.

## Closing the task

PR C (or the last PR of its split) sets `Status: done` on this file's first line and sets roadmap entries 14 and 15 to the merged PR numbers. W11 and the builds are outside the developer's stack.
