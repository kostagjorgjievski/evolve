Status: active

# Task 010: Hand-designed updater baselines and the hindsight oracles

Read `AGENTS.md` and `docs/ROADMAP.md` first. This task covers roadmap entry 10 (`feat/updater-baselines`). It is a stack of three PRs. Open them in order and stop after each one for review.

## Why

Research question 1 asks whether learned adaptation can beat the best hand-designed online updaters. Gate 1 (entry 11) answers it by tuning those updaters on dev, evaluating them on sealed data, and measuring how far the best of them is from a floor: the true-best forecast on synthetic worlds (TASK_004, already in every synthetic `ReplayResult` as `oracle_mase`), and hindsight oracles over the expert pool on every dataset. This task builds both sides of that comparison:

- PR A: the five classical updaters (Equal, Follow-the-Leader, Hedge, Fixed-Share, AdaHedge). They are the first real code in `updater/`.
- PR B: the hindsight oracles (best fixed expert, best switching sequence with at most `m` switches, per-origin best expert), computed by the arena from targets.
- PR C: the sealed oracle (the sealed server computes the oracles on a sealed pool and records them), and a dev CLI that replays a named baseline or prints the oracles.

Tuning grids, `run_start` content, ledger writing on dev, sealed baseline runs and the Gate 1 report are entry 11. This task produces no result that is reported anywhere.

## Decisions already made (do not reopen)

- **TASK_007:** a fresh updater per series; `updater/interfaces.py` is the contract; the updater never sees a target, value, absolute origin, series id or timestamp. Agent-editable code never imports `evolvets.cache` or `evolvets.foundation`. The `.importlinter` contracts are in place (`sealed-isolation`, `ground-truth-isolation`, `updater-isolation`); this task does not change them.
- **TASK_007, "Not in this task":** the real-data oracles need targets, so they live in `arena/`, not `updater/`.
- **TASK_008:** the sealed server is the only process that reads sealed data; the caller gets only a receipt; every sealed run that started is in the ledger; a sealed evaluation charges `Budget(wall_seconds, evaluations=1)` and the caller never charges it again.
- **TASK_009 PR D:** pool specs `simple` and `simple+foundation`; `ARENA_VERSION = "2"`; `leakage_risk` is result-level, not inside `metrics` (metrics dicts hold floats only).

## Rules for all three PRs

- **Standard library only.** No new dependencies.
- **Bit-exact determinism**, as in TASK_007, with one named exception: on floats use `+ - * /`, comparisons, `abs`, `max`/`min` on two operands written out, and **in `updater/baselines.py` only, `math.exp` and `math.log`** (Hedge, Fixed-Share and AdaHedge need them; the generator already relies on `libm` the same way, TASK_004). Golden digests are pinned on the CI platform (Linux x86_64, CPython 3.11). **Never call the builtin `sum()` on floats, `math.fsum` or `statistics`**; accumulate with explicit left-to-right `for` loops in the order this file states. AST guards enforce this.
- **Tie-breaking is by pool order** (the order of `Context.expert_names`), and equality tests on floats are exact `==`. No randomness anywhere in this task: the baselines receive a seed through `init` and ignore it (AGENTS.md 2.6 is satisfied because nothing is stochastic).
- **Import rules.** `updater/baselines.py` imports only `math`, `types`, `dataclasses`, `collections.abc`, `typing` and `evolvets.updater.interfaces`. `arena/oracles.py` may import anything in `arena/` that `arena/evaluate.py` imports. `arena/sealed/client.py` keeps its TASK_008 rule: it imports nothing that can read data.
- **Strict mypy:** add `"evolvets.updater.baselines"` to the strict override list in `pyproject.toml` (PR A). `arena/` is already strict. Do not weaken any override.
- **`ARENA_VERSION` becomes `"3"` in PR B**, which adds `arena/oracles.py` to the arena protocol (list it in the `arena/__init__.py` docstring, and say in one sentence what version 3 added). PR C does not bump it again: the sealed oracle serves the same version-3 definitions. No existing updater score changes in this task; PR B proves it (test B10).
- **No ledger change.** No change to `ledger/`, `PROTOCOL_KEYS` or `SCHEMA_VERSION`. PR C writes `baseline` records with the existing schema.
- Do not edit `.importlinter`, `.github/`, `AGENTS.md` or `docs/agents/`.
- Tests live in `tests/updater/` (new; add `tests/updater/__init__.py`) and `tests/arena/`.

---

## PR A: `feat/updater-baselines`

The five baseline updaters. Nothing in `arena/` changes.

### Files and interfaces

`src/evolvets/updater/baselines.py`

```python
BASELINE_IDS = ("equal", "ftl", "hedge", "fixed_share", "adahedge")
DEFAULT_CONFIGS: Mapping[str, Mapping[str, ConfigValue]]
# {"equal": {}, "ftl": {}, "hedge": {"eta": 1.0},
#  "fixed_share": {"eta": 1.0, "alpha": 0.01}, "adahedge": {}}


class Equal: ...  # each satisfies evolvets.updater.interfaces.Updater


class FollowTheLeader: ...


class Hedge: ...


class FixedShare: ...


class AdaHedge: ...


def equal() -> Updater: ...  # zero-argument factories, one per id;
def ftl() -> Updater: ...  # "evolvets.updater.baselines:<id>" is a valid
def hedge() -> Updater: ...  # sealed updater_ref for each of them
def fixed_share() -> Updater: ...
def adahedge() -> Updater: ...


BASELINES: Mapping[str, Callable[[], Updater]]  # id -> factory, in BASELINE_IDS order
```

`DEFAULT_CONFIGS` and `BASELINES` are read-only (`types.MappingProxyType` over dicts whose inner configs are also proxies is fine; so is returning copies). The defaults are placeholders that make each baseline runnable; they are not tuned values. Tuning grids are entry 11.

### Exact behaviour: shared by all five

- **State.** Every per-series quantity lives in the state object returned by `init` and threaded through `update`. The instance itself holds nothing that changes after construction. (The arena already gives a fresh instance per series; this rule additionally makes two states from one instance independent, which Phase 1 will rely on when an improver edits these files.) The state may be mutated in place and returned.
- **`init(n_experts, horizon, config, seed)`** validates, raising `ValueError` (the arena maps it to `UpdaterFailure("failed")`):
  - `n_experts` and `horizon` are ints (not bools) `>= 1`;
  - `seed` is an int (not bool); its value is ignored;
  - `config`'s key set equals exactly the key set of `DEFAULT_CONFIGS[id]`: an unknown or missing key raises, so every run records its full configuration;
  - `eta` (Hedge, Fixed-Share): exact type `float` or `int` (not bool), finite, `> 0`;
  - `alpha` (Fixed-Share): exact type `float` or `int` (not bool), `0 <= alpha <= 1`.
- **`update(state, feedback, context)`:**
  1. Check that `context.forecasts` has `n_experts` rows of `horizon` floats and `context.horizon == horizon`; else `ValueError`.
  2. Process each `Matured` in `feedback` **in tuple order** (the arena sends at most one, but the updater must not assume a delay of exactly `H` or a count of one; unit tests feed it immediately). Each must have `n_experts` rows of `horizon` errors, all finite; else `ValueError`.
  3. Each horizon column `h = 0..H-1` is an **independent learner**: it has its own state and is updated only from column `h` of the errors.
  4. Return `(weights, state)` with `weights` a `tuple` of `n_experts` `tuple`s of `horizon` floats (`[expert][h]`), each column non-negative and summing to 1 within `1e-9` (the arena's check). Weights depend only on the state after step 2.
- **The shared loss.** For one matured origin and one column `h`: `a_e = abs(errors[e][h])` for `e = 0..E-1`; `M = a_0`, then for `e = 1..E-1`, `M = a_e if a_e > M else M`. If `M > 0`: `l_e = a_e / M`; else every `l_e = 0.0`. So each loss is in `[0, 1]`, the worst expert at that origin and step has loss `1`, and the loss is invariant to the series' scale. FTL, Hedge, Fixed-Share and AdaHedge all use this loss and nothing else from the feedback; `combined_errors` is ignored.
- **Normalising.** Wherever weights are formed from non-negative scores `u_e`: `U = 0.0; for e: U = U + u_e`; `w_e = u_e / U`.
- **Cumulative loss.** `L_e` starts at `0.0` and is updated `L_e = L_e + l_e` per matured origin, in feedback order. `Lmin = min_e L_e` by the same left-to-right scan as `M`.
- **Leaders.** `leaders = [e for e in 0..E-1 if L_e == Lmin]` (exact equality).

### Exact behaviour: each updater (per column)

- **Equal.** `w_e = 1.0 / E` for every `e` and every step. Feedback is validated and otherwise ignored.
- **FollowTheLeader.** `w_e = 1.0 / len(leaders)` for leaders, `0.0` otherwise. Before any feedback every expert is a leader, so the weights are uniform.
- **Hedge** (exponential weights on cumulative loss). `u_e = math.exp(-eta * (L_e - Lmin))`; normalise. The leader's score is exactly `1.0`, so `U >= 1` and nothing underflows to an all-zero column.
- **FixedShare** (Herbster and Warmuth, 1998). The column holds weights `p`, initially `1.0 / E` each. Per matured origin, with `S = [e for e if p_e > 0]` and `c = min(l_e for e in S)` (left-to-right scan over `S`): `v_e = p_e * math.exp(-eta * (l_e - c))` for `e` in `S`, `0.0` otherwise; `V = Σ v_e` (loop); `p_e = (1 - alpha) * (v_e / V) + alpha / E`. The output weights are `p`. (`V > 0` always: the expert in `S` with loss `c` keeps `v = p_e > 0`.) With `alpha = 0` this is Hedge with incremental updates; with `alpha = 1` it is Equal.
- **AdaHedge** (de Rooij, van Erven, Grünwald and Koolen, 2014, "Follow the leader if you can, hedge if you must", Algorithm 1). The column holds `L` and `Delta = 0.0`.
  - If `E == 1`: the weight is `1.0`; skip the rest.
  - Current weights: if `Delta == 0.0` (the learning rate is infinite), the FTL weights above. Otherwise `eta = math.log(E) / Delta` and the Hedge weights with that `eta`.
  - Per matured origin, with `w` the current weights computed **from the state just before this origin is processed**: `hmix = Σ w_e * l_e` (loop). If `Delta == 0.0`: `mix = min(l_e for e in leaders)` (scan in pool order). Else: `c = min_e l_e`; `mix = c - math.log(Σ_e w_e * math.exp(-eta * (l_e - c))) / eta`. Then `delta = hmix - mix`; if `delta < 0.0` set it to `0.0` (rounding); `Delta = Delta + delta`; then `L_e = L_e + l_e`.
  - With delayed feedback the weights used for `hmix` are the current ones, not the ones played at the matured origin. This is the standard reduction for delayed feedback and is written in the module docstring.
- All five have no other parameters and no other behaviour.

### Tests for PR A (all required; `tests/updater/test_baselines.py`, `tests/updater/test_baselines_replay.py`)

Unit tests drive `init`/`update` directly with hand-built `Context` and `Matured` objects, feeding the `Matured` for step `k - 1` at step `k` unless stated.

1. **Contract.** For every id: `BASELINES[id]()` returns a new object each call; `init(5, 3, DEFAULT_CONFIGS[id], 0)` then 20 `update` calls on random finite errors (fixed `random.Random(0)` in the test) return a `tuple` of 5 `tuple`s of 3 `float`s, every element `>= 0`, every column summing (left-to-right) within `1e-9` of 1. `BASELINE_IDS == tuple(BASELINES)`, and `importlib.import_module("evolvets.updater.baselines")` has an attribute named `id` that is `BASELINES[id]` for every id.
2. **Config validation** (each `ValueError` from `init`): an unknown key, a missing key (for Hedge and Fixed-Share), `eta` of `0`, `-1.0`, `nan`, `inf`, `True` and `"1"`; `alpha` of `-0.1`, `1.5`, `nan` and `True`; `n_experts` of `0` and `True`; `horizon` of `0`; `seed` of `True`. `alpha` of `0` and `1` and `eta` of `3` (int) are accepted.
3. **Shape validation** in `update` (`ValueError`): forecasts with the wrong number of rows, a row of the wrong length, a `Matured` with the wrong shape, a NaN error.
4. **The shared loss** on a hand-written column: errors `(2.0, -4.0, 0.0)` give losses `(0.5, 1.0, 0.0)`; all-zero errors give all-zero losses. Checked through FTL: after that feedback the leader is expert 2.
5. **Equal:** every weight is `1.0 / E` exactly at every step, for `E = 1, 3, 8`.
6. **FTL:** uniform before feedback; a hand-written three-step loss sequence picks the expected leader at each step; an exact tie between experts 0 and 2 gives `(0.5, 0.0, 0.5)`.
7. **Hedge:** after a hand-written sequence the weights equal, within `1e-12` relative, a reference written in the test as `exp(-eta * L_e) / Σ exp(-eta * L_j)` (computed in the naive form, on values small enough not to underflow). With `eta = 1e6` the weights equal FTL's when the leader is unique. A loss gap of `1000` with `eta = 10` does not produce NaN or an all-zero column.
8. **Fixed-Share:** with `alpha = 0` it matches Hedge within `1e-12` on the sequence of test 7; with `alpha = 1` it matches Equal exactly; for `alpha = 0.1` every weight is `>= 0.1 / E - 1e-15` at every step; after an expert is best for 50 steps and another for the next 50, Fixed-Share with `alpha = 0.05` puts more weight on the new best expert at step 100 than Hedge with the same `eta` (the property the switching comparator exists for).
9. **AdaHedge:** with identical losses for all experts the weights stay uniform and `Delta` stays `0`; with one expert at loss `0` and the rest at `1` (`E = 3`), the weight on that expert is non-decreasing from step to step and exceeds `0.99` by step 20; on a 2000-step sequence from `random.Random(1)` its weights match, within `1e-9` absolute, a reference implementation written in the test directly from the paper's pseudocode.
10. **Regret sanity** (immediate feedback, `E = 4`, `T = 2000`, losses from `random.Random(2)` plus one expert biased lower; the test computes the mixture loss `Σ_t Σ_e w_e(t) l_e(t)` from the returned weights for column 0 and the best expert's total): Hedge with `eta = sqrt(8 ln E / T)` has regret `<= sqrt(T ln E / 2)`; AdaHedge has regret `<= 2 * sqrt(T * ln E) + 16 / 3 * ln E + 2` (looser than Theorem 8 of the paper, which it implies).
11. **Columns are independent:** feedback whose column 0 favours expert 0 and column 1 favours expert 1 gives weights whose column 0 and column 1 lean accordingly, for FTL, Hedge, Fixed-Share and AdaHedge.
12. **State lives in the state:** for every id, interleaving `update` calls on two states from one instance (series A, series B, alternating) gives exactly the weights that two separate instances give.
13. **Through the arena** (`test_baselines_replay.py`): every baseline at its default config, replayed with `replay_dataset(BASELINES[id], DEFAULT_CONFIGS[id], 0, synthetic_pool)` on the `synthetic-dev` pool with `per_family = 1` (the TASK_007 fixture), has `status == "ok"`. The SHA-256 of `canonical_bytes(result_to_obj(result))` is pinned per id in the test file; the docstring says that changing one requires a reason in the PR (a baseline's behaviour changed) and that these digests are not `ARENA_VERSION`-governed but will change when PR B bumps the version (repin them there, test B10 shows why).
14. **Pooled replay:** Hedge and AdaHedge replay a `simple+foundation` pool with the fake foundation model (the TASK_009 test helpers) with `status == "ok"`, and `init` receives `n_experts == 6`.
15. **Sealed path:** `evaluate_sealed` with `updater_ref = "evolvets.updater.baselines:adahedge"` on the 1200-row real fixture (TASK_008 test setup) returns `status == "ok"`, and the sealed baseline's `metrics` equal the in-process `replay_dataset(adahedge, {}, 0, load_sealed_pool(...)).metrics` exactly. In a fresh `python -c` subprocess that imports `evolvets.updater.baselines`, the only `evolvets` modules in `sys.modules` are `evolvets`, `evolvets.updater`, `evolvets.updater.interfaces` and `evolvets.updater.baselines`.
16. **AST guards** on `updater/baselines.py`: imports are only those listed in "Import rules"; no builtin `sum`, `math.fsum`, `statistics`, `random`, `time`, `os` or `open`; the only `math` attributes used are `exp`, `log` and `isfinite`.

### Dependencies

None.

### Size

About 250 lines of implementation. If it grows past ~400, put Equal, FTL and Hedge (with tests 1 to 7, 11, 12, 16 and their part of 13) in `feat/updater-baselines-core`, and Fixed-Share and AdaHedge with the rest in `feat/updater-baselines-adaptive`.

---

## PR B: `feat/arena-oracles`

Hindsight oracles over the expert pool, computed by the arena from targets. Dev and in-process only; the sealed side is PR C.

### Files and interfaces

`src/evolvets/arena/oracles.py`

```python
ORACLE_SWITCHES = (0, 1, 2, 4, 8, 16, 32)


@dataclass(frozen=True)
class SeriesOracle:
    series_id: str
    expert_mase: tuple[float, ...]  # per pool expert, in pool order
    switch_mase: tuple[float, ...]  # per ORACLE_SWITCHES entry, same order
    select_mase: float  # best expert at every origin (unlimited switches)


@dataclass(frozen=True)
class OracleResult:
    dataset: str
    cache_id: str
    arena_version: str
    pool_spec: str
    foundation_cache_ids: tuple[str, ...]
    leakage_risk: bool
    expert_names: tuple[str, ...]
    switches: tuple[int, ...]  # == ORACLE_SWITCHES
    global_expert: str  # the single expert best over the whole pool
    metrics: dict[str, float]
    series: tuple[SeriesOracle, ...]  # pool order


def expert_origin_losses(
    pool: DevPool, series: DevSeries
) -> tuple[tuple[float, ...], ...]: ...  # [expert][origin - score_start]
def switching_total(losses: Sequence[Sequence[float]], max_switches: int) -> float: ...
def oracle_dataset(pool: DevPool) -> OracleResult: ...
def oracle_to_obj(result: OracleResult) -> dict[str, object]: ...  # canonical_bytes-able
```

### Exact behaviour

- **`expert_origin_losses`:** `release = LabelRelease(series.values, H)`; forecasts from `load_pool_forecasts(pool, series.series_id)`. For each scored origin `s = score_start .. score_end - 1` (ascending), `targets = release.targets(s, now=s + H)`, and for each expert `e` in pool order the origin loss is the left-to-right mean over `h` of `abs(forecast[e][h] - targets[h])` (the same arithmetic as `metrics.origin_losses`, so expert `e`'s losses equal `origin_losses` of a one-hot replay on `e`). Targets are read only through `LabelRelease`.
- **`switching_total(losses, m)`:** the minimum, over sequences `e_0 .. e_{n-1}` with at most `m` indices `k` where `e_k != e_{k-1}`, of `Σ_k losses[e_k][k]`. Exact dynamic programme: `D[j][e]` is the best total over sequences up to origin `k` that end at `e` and have at most `j` switches; at `k = 0`, `D[j][e] = losses[e][0]` for every `j`; for `k >= 1`, `best_prev[j] = min_e D[j][e]` (pool-order scan) and `D[j][e] = (D[j][e] if j == 0 else min(D[j][e], best_prev[j - 1])) + losses[e][k]`, computed for all `j` from the previous row's values (not in place). The result is `min_e D[m][e]`. Comparisons via `<`, and additions left to right in `k`, so for `m = 0` the result equals the loop sum of the best expert's losses exactly. Validation (`ArenaError`): `m` an int (not bool) `>= 0`; `losses` non-empty, all rows the same non-zero length, all values finite. When `m >= n - 1` the result equals the per-origin minimum's total; compute that case directly as `Σ_k min_e losses[e][k]` to keep the cost bounded.
- **`oracle_dataset`:** for each series in pool order: `scale = mase_scale(release, period, scale_end, now=score_start)`; `n = score_end - score_start`;
  - `expert_mase[e] = (loop sum of losses[e]) / n / scale`;
  - `switch_mase[i] = switching_total(losses, ORACLE_SWITCHES[i]) / n / scale`;
  - `select_mase = (Σ_k min_e losses[e][k]) / n / scale`.
- **Metrics** (all `float`, means over series are left-to-right in pool order):
  - `n_series`;
  - `best_expert_mase`: mean over series of `min_e expert_mase[e]` (the best fixed expert chosen per series);
  - `global_expert_mase`: `min_e` of the mean over series of `expert_mase[e]`; `global_expert` is that expert's name (first in pool order on a tie);
  - `switch_mase_m<m>` for each `m` in `ORACLE_SWITCHES` (e.g. `switch_mase_m8`): mean over series;
  - `select_mase`: mean over series;
  - `expert_mase/<name>` for every pool expert: mean over series.
- The oracle is in hindsight by design and is never given to an updater. It is an arena computation, like the true-best oracle.
- `OracleResult` carries `arena_version = ARENA_VERSION`, the pool's spec, foundation cache ids and `leakage_risk` (via `pool_leakage_risk`), exactly as `ReplayResult` does.
- `oracle_to_obj`: exactly the `OracleResult` field names; tuples become lists; `series` becomes a list of objects with exactly the `SeriesOracle` field names.
- A pool or cache problem raises `ArenaError`; there is no failed status (nothing untrusted runs).
- Bump `ARENA_VERSION` to `"3"` and extend the `arena/__init__.py` docstring. Repin the TASK_007 golden (`test_evaluate.py` test 13) and the PR A digests, after test B10 shows only the version string changed.

### Tests for PR B (all required; `tests/arena/test_oracles.py`)

1. `switching_total` on a hand-written `3 x 6` loss table: values for `m = 0, 1, 2, 5` match totals written out in the test (worked by hand in a comment); `m = 0` equals the best row's loop sum exactly (`==`); `m >= 5` equals the per-origin minimum's total.
2. `switching_total` against brute force: for 50 random tables (`random.Random(3)`, `E` in `2..4`, `n` in `1..7`), it equals, within `1e-12`, the minimum over all `E ** n` sequences with at most `m` switches, for every `m` in `0..n`.
3. Monotonicity on the synthetic pool: per series, `switch_mase` is non-increasing in `m`, `switch_mase[0] == min(expert_mase)` exactly, and `select_mase <= switch_mase[-1]`.
4. Validation (`ArenaError`): `m` of `-1`, `True`, `1.0`; an empty table; ragged rows; a NaN loss.
5. **Agreement with replay:** for each expert `e` of the `synthetic-dev` pool, a one-hot replay on `e` (`replay_dataset`) gives per-series `mase` equal (`==`) to `expert_mase[e]`.
6. **Metrics:** exact key set for a `simple` pool (`n_series`, `best_expert_mase`, `global_expert_mase`, the seven `switch_mase_m*`, `select_mase`, five `expert_mase/*`); with the fake foundation model the key set gains `expert_mase/fake`. `best_expert_mase <= global_expert_mase`. Each metric equals the left-to-right mean, over the `SeriesOracle` entries, of the corresponding per-series value, recomputed in the test.
7. **No look-ahead:** wrap `LabelRelease.targets` and `history` as in TASK_007 test 9. On the real 1200-row fixture's dev pool every `targets` call has `origin + H == now`, `history` is called only with `end <= now + 1`, and the largest value index any call touches is `score_end - 1 + H`.
8. **The oracles are comparators, not lower bounds:** a convex combination can beat every single expert at an origin when their forecasts straddle the target. On a hand-built two-expert series (forecasts always `target + 1` and `target - 1`; a hand-built `SeriesForecasts` as in TASK_007's replay tests), `replay_series` with equal weights gives all-zero errors, while every expert origin loss is `1`, so `switching_total` of the expert losses is positive for every `m`. The module docstring says the oracles bound single-expert selection only, so the Gate 1 report must not call them a floor for weighted combinations.
9. `oracle_to_obj`: exact key set; `canonical_bytes` accepts it; `leakage_risk` and `pool_spec` match the pool; a golden SHA-256 of `canonical_bytes(oracle_to_obj(oracle_dataset(synthetic_pool)))` is pinned, with a docstring saying a change requires bumping `ARENA_VERSION`.
10. **Version 3 changes no score:** for the TASK_007 golden replay, `result_to_obj(result)` with `"arena_version"` replaced by `"2"` hashes to the old digest `a3dcc509e2a085c7ff473cdfa02c9bc434b6aa419f1975fda03a6a31b7405fd5`. Then the test 13 digest is repinned for `"3"`.
11. AST guard: `arena/oracles.py` uses no builtin `sum`, `math.fsum` or `statistics` (extend the existing parametrised guard in `test_evaluate.py`).
12. Determinism: two `oracle_dataset` calls give equal results.

### Dependencies

None.

### Size

About 180 lines of implementation. If it grows past ~400, put `switching_total` with tests 1, 2 and 4 in `feat/arena-switching-oracle` and the rest in a second PR.

---

## PR C: `feat/arena-oracle-sealed-cli`

The sealed oracle, and a dev CLI that replays a baseline or prints the oracles. Nothing here writes to a ledger on dev; that is entry 11.

### Files and interfaces

`arena/sealed/client.py` gains:

```python
@dataclass(frozen=True)
class SealedOracleRequest:
    dataset: str  # one of SEALED_DATASETS
    pool_spec: str = "simple"
    foundation_models: tuple[str, ...] | None = None


def evaluate_sealed_oracle(
    request: SealedOracleRequest,
    *,
    ledger_root: Path,
    run_id: str,
    paths: SealedPaths,
) -> SealedReceipt: ...
```

`runner/cli.py` gains `evolvets arena replay` and `evolvets arena oracle`.

### Exact behaviour: the sealed oracle

- Caller side: same validation style as `evaluate_sealed` (dataset in `SEALED_DATASETS`, `pool_spec` a str, `foundation_models` `None` or a non-empty tuple of non-empty str, `run_id` non-empty; `SealedError`, nothing spawned). Same spawn, single reply, `join` and error mapping as `evaluate_sealed`. The client still imports nothing that can read data.
- Server side: the server's request JSON gains a `kind` field, `"updater"` or `"oracle"`, and `_decode` checks the exact key set for each kind. `evaluate_sealed` sends `"updater"` and is otherwise unchanged. For `"oracle"`:
  1. the same ledger preconditions and refusals, and the same sealed-stream lock, as for an updater request;
  2. `load_sealed_pool(dataset, ..., pool_spec=..., foundation_models=...)`; any failure before the record is written follows TASK_008 step 3 (log, no record, exit 2);
  3. start the clock; `oracle_dataset(pool)`; on an unexpected `Exception` here, write a `failed` baseline with `error = "sealed server error: <ExceptionTypeName>"` and `per_task_blob = None`, log the traceback, and reply normally (as TASK_008 step 5);
  4. on success, put the blob `{"result": oracle_to_obj(result)}` and append a sealed-stream `baseline` with `updater_id = "oracle"`, `hyperparams = {}`, `pool = "sealed"`, `status = "ok"`, `metrics = result.metrics`, `per_task_blob = ref`, `error = None`, and `Budget(wall_seconds=<elapsed>, evaluations=0)`. The oracle is part of the instrument, not an experimental condition, so it charges no evaluation; its wall-clock is still recorded;
  5. reply `{"record_id", "status"}`.
- No worker process is spawned for an oracle request (no untrusted code runs).

### Exact behaviour: the dev CLI

- `evolvets arena replay {etth1,ecl,synthetic-dev} --updater ID [--config JSON] [--seed 0] [--pool-spec simple] [--foundation-model NAME ...] [--cache-root cache_store] [--foundation-root foundation_cache] [--data-root data] [--manifest-dir data/manifests]`. `ID` is one of `BASELINE_IDS`; `--config` defaults to `DEFAULT_CONFIGS[ID]` and must be a JSON object. It loads the dev pool with `load_dev_pool`, runs `replay_dataset(BASELINES[ID], config, seed, pool)` and prints `canonical_json` of `{"result": result_to_obj(result) without "series", "wall_seconds": ...}` on one line. The default foundation root is the one `evolvets foundation build` uses; read it from the existing CLI rather than repeating the literal.
- `evolvets arena oracle {etth1,ecl,synthetic-dev} [same pool options]` prints `oracle_to_obj` without `series`, the same way.
- Both are dev only: neither accepts a sealed dataset name, a sealed path or a ledger path. An `ArenaError` prints its message and exits 1, like the existing commands; an updater failure is printed as the result (status not `ok`) and exits 0.
- The runner may import `evolvets.updater.baselines`; no import contract forbids it.

### Tests for PR C (all required; `tests/arena/test_sealed_oracle.py`, `tests/arena/test_arena_cli.py`)

1. **Same numbers as in-process:** on the real 1200-row fixture and on the synthetic sealed pool (`per_family = 1`), `evaluate_sealed_oracle` returns `ok`, and the sealed baseline's `metrics` equal `oracle_dataset(load_sealed_pool(...)).metrics` exactly; the blob's `result` equals `oracle_to_obj` of it.
2. **The record:** one new sealed record, kind `baseline`, `updater_id == "oracle"`, `hyperparams == {}`, `pool == "sealed"`, `budget.evaluations == 0`, `wall_seconds > 0`; `verify(check_blobs=True)` is ok; `validate_run` returns `[]`. An updater request and an oracle request in one run give `seq` 0 and 1.
3. **Refusals** as TASK_008 test 10 (no main stream, wrong `protocol_versions`, `run_end` present, sealed stream locked), each raising `SealedError` and writing no record. Caller-side validation (unknown dataset, empty `foundation_models`) spawns nothing.
4. **Server failure:** a missing sealed cache raises `SealedError` naming `server.log`, writes no record, and logs an `ArenaError`. A sealed real fixture whose warmup values are constant (or any equivalent pool the developer can build that loads but makes `oracle_dataset` raise) loads (pool loading does not compute the scale) but makes `mase_scale` raise inside `oracle_dataset`; that request writes one `failed` record with `error` starting `"sealed server error: ArenaError"` and `per_task_blob is None`, and replies normally.
5. **No caller access:** recording file opens in the test process during `evaluate_sealed_oracle`, nothing under the sealed data, the salt or the sealed cache root is opened. The caller's raw reply has exactly the keys `record_id` and `status`.
6. **The updater request is unchanged:** the existing `tests/arena/test_sealed_server.py` passes without edits other than expected `kind` plumbing, and a request JSON with a missing or unknown `kind` makes the server exit 2.
7. **Pooled:** an oracle request with `pool_spec = "simple+foundation"` and the fake model on a fake-built sealed foundation cache gives `expert_mase/fake` in the metrics.
8. **CLI:** `arena replay synthetic-dev --updater hedge` on a fixture cache prints one JSON line whose `result.metrics` equals an in-process replay's and whose `result` has no `series` key; `--config '{"eta": 0}'` prints a result with `status == "failed"` and exits 0; `--updater nope` and `--config '[1]'` exit non-zero; `arena oracle synthetic-dev` prints metrics equal to `oracle_dataset`'s; `arena replay sealed` (or any sealed name not in the dev list) is rejected by argparse.
9. **Import isolation** (extends TASK_008 test 12): a fresh subprocess importing `evolvets.arena.sealed.client` still loads none of `evolvets.arena.sealed.pools`, `evolvets.arena.oracles`, `evolvets.arena.realdata.prepare` or `evolvets.cache.store`.

### Running it for real

Not required for merge. If the developer has the dev caches, the PR description may paste `arena oracle` output for `etth1`, `ecl` and `synthetic-dev` (`simple` pool) with wall-clock times, as sanity numbers. Do not paste baseline replay numbers on dev and do not try configs to see which looks good: tuning is entry 11, where every tried configuration is recorded in the ledger (AGENTS.md 2.8).

### Dependencies

None.

### Size

About 200 lines of implementation. If it grows past ~400, put the sealed oracle (tests 1 to 7, 9) in `feat/arena-oracle-sealed` and the CLI (test 8) in `feat/arena-cli`.

---

## Not in this task

- **Entry 11 (`exp/gate1`):** tuning grids and their budget accounting, the run config and `run_start` (it must record the pool spec, the simple cache id and the foundation cache ids, review note 2 of #75), dev `baseline` records, sealed baseline and oracle runs, `SERIES_TIMEOUT_SECONDS` review, where the sealed data and salt live when Gate 1 runs (TASK_008), and the report, regenerable from the ledger, with `leakage_risk` rows split out (review note 1 of #75). Its spec is written after this task. Gate 1's decision itself is a human gate.
- **Entry 13 (`feat/live-commit`):** daily public timestamped commitments. It publishes outside the repository on a schedule, so the planner will likely need the owner's decisions on where commitments are published and on any scheduled workflow (a `.github/` change) before it can be specified.
- **Other baselines** (online gradient descent on weights, per-season weights, OGD-based stacking, Bayesian model averaging). Not in roadmap entry 10. Gate 1 can add them only through a roadmap change.
- **A best fixed convex combination oracle.** Its hindsight optimum is a convex program; it is not needed to measure switching headroom and is left out.
- **Tagging.** Issue #55 asks for `arena-v1` on the version-1 commit. After this task the arena is at version 3; whether to tag a later version before Gate 1 is the owner's call. The reviewer mentions this on issue #55 when PR B merges.
- `updater/interfaces.py` and now `updater/baselines.py` live in an agent-editable package. Before Phase 1 lets an improver edit `updater/`, the Phase 1 specs must decide whether baselines are frozen copies outside the editable area (TASK_007 "Not in this task").

## Closing the task

PR C (`feat/arena-oracle-sealed-cli`, or the last PR of its split) changes this file's first line to `Status: done` and sets roadmap entry 10 to its merged PR numbers.
