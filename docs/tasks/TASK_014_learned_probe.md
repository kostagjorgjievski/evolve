Status: done (PR A merged as #112; PR B as #113, with refactor #114; PR C as #115; PR D split by the size rule into data #117 and rendering #119; W6 #120 and W7 #121/#124 applied by the owner's delegate; PR E as #128 adds the results note and closes the task)

# Task 014: learned-updater probe on dev (research question 1)

Read `AGENTS.md` and `docs/ROADMAP.md` first. The owner asked for this task in issue #110. It is a stack of five small PRs. Open them in order and stop after each one for review. Two `.github/` workflows are specified here; **the owner's delegate applies them** (AGENTS.md section 7), and the developer never writes them.

(The number is 014, as #110 asks. The number 013 was used by the closed planning PR #107, so skipping it keeps references unambiguous.)

## Why

The first complete Gate 1 dev run (`gate1-tune-20261013T151300Z-b36261f7`; results note `docs/results/gate1-dev-first-run.md`, PR #108) shows the following. With the foundation models in the pool, the best tuned hand-designed updater is essentially tied with the best single foundation model:

| dataset | best tuned updater | best single foundation model |
| --- | --- | --- |
| ETTh1 | 0.8133 | 0.8154 |
| ECL | 0.6837 | 0.6889 |
| synthetic | 0.7771 | 0.7766 |

There is still room above it. A switching oracle with at most 8 switches is 0.011 to 0.030 MASE better, and on synthetic the true-best forecast is 0.055 better.

Research question 1 asks whether learned adaptation can beat the best hand-designed updaters. This task tests that directly on the same dev pools, with no sealed ceremony (owner, #109 and #110). It adds two learned updaters, replays their small grids through the existing arena in recorded runs, and reports them with the existing report generator against the best tuned baseline of the existing tune run, using paired bootstrap intervals.

- **PR A** `feat/updater-ridge-stack`: `ridge_stack`, a constrained least-squares stacker with exponential forgetting (an online, shrunk "optimal combination" of the pool).
- **PR B** `feat/updater-loss-gate`: `loss_gate`, an online ridge model that predicts each expert's next loss from features of its recent errors, its season and the pool's disagreement, then gates on the predictions.
- **PR C** `feat/gate1-probe-run`: `evolvets gate1 probe`. This is one recorded run that replays the learned updaters' grids over dev pools a completed tune run covers.
- **PR D** `feat/gate1-probe-report`: a "Learned-updater probe" section in `evolvets gate1 report`, with a mechanical verdict for each pool and learned updater.
- **PR E** `docs/gate1-probe-results`: `docs/results/gate1-learned-probe.md`, written after the owner's run.
- **W6** (owner-applied) runs the probe on the lab node. **W7** (owner-applied) adds probe runs to the dev report workflow.

The finding is one of these two. The report computes each verdict mechanically:

- a learned updater beats the best tuned baseline outside the 95 % interval on some pool, in which case Phase 1 is justified in the owner's framing;
- none does, in which case the generator is hardened before Phase 1.

**Which branch to take is the Gate 1 decision. It belongs to the owner and is not planned here.**

## Decisions already made (do not reopen)

- **Owner, #109 and #110 (2026-10-14):** results must be faithful and reproducible. There is no sealed run and no tune redo for protocol reasons. The first dev tune run stays the reference. Its tarball commit `575906467740659c72c91aa7ca8962cc9d771b6c` maps to `main` at `7a56750ff68e8aee9a6a3c040695a842839c2507`, as stated in the results note. TASK_012 is superseded (#109).
- **TASK_007/010:**
  - A fresh updater is used for each series.
  - `replay_dataset` is the only way to score an updater.
  - An updater sees only `Context` and `Matured`, and returns convex weights: each horizon column is non-negative and sums to 1 within `1e-9`.
  - The hindsight oracles bound single-expert selection only.
- **TASK_011:**
  - `select_configs` is the selection rule: lowest dev MASE, and the earlier record wins an exact tie.
  - The report never mixes protocol versions, tables only `completed` runs, and lists every other run.
  - `leakage_risk` pools get their own section.
  - Floats are accumulated in explicit left-to-right loops.
  - The bootstrap is `_gap_rows`: 2000 resamples, one `random.Random(seed)` per table, interval indexes 50 and 1949.
- **The updater interface is not changed.** A learner that outputs forecasts directly (bias correction, non-convex stacking) would need an arena protocol change. That is out of this task (see "Not in this task").

## Rules for all five PRs

- **No ledger or arena change.**
  - No change to `ledger/`, `arena/`, `updater/interfaces.py`, `updater/baselines.py`, `PROTOCOL_KEYS`, `SCHEMA_VERSION`, `ARENA_VERSION` or any payload.
  - Learned-updater replays are dev `baseline` records (`updater_id` = the learned id), the kind the tune run already uses for fixed comparators.
- **Standard library only.** No numpy, scikit-learn, torch or any new dependency, so there is no gradient-boosted model and no neural combiner in this task.
  - **Updater modules** use only `+ - * /`, comparisons, `abs`, `math.exp` and `math.log`, in explicit left-to-right loops. There is no `**`, no `sum()` on floats, no `math.fsum` and no `statistics`, as in `updater/baselines.py`.
- **Seeds.**
  - Both learned updaters are deterministic. `init` validates `seed` as an int (bools rejected) and otherwise ignores it, as the baselines do.
  - The run seed is recorded in `run_start` and its config blob (AGENTS.md 2.6).
  - The only other randomness is the report's bootstrap, seeded from its CLI.
- **Equal accounting.** Each learned replay charges `Budget(wall_seconds=<measured>, evaluations=1)`, exactly like a tune replay (AGENTS.md 2.5). The report shows each learned updater's configurations, evaluations and wall seconds next to the baselines' cost.
- **Agents never run the probe on real dev data.** Tests use fixtures in `tmp_path`. The real run is W6 on the lab node, dispatched by the owner.
- **Strict mypy.** Add `evolvets.updater.learned`, `evolvets.runner.gate1_probe` and any new runner module to the strict override list in `pyproject.toml`. Do not weaken any override.
- **Import contracts.** `updater/learned.py` imports only the standard library, `evolvets.updater.interfaces` and `evolvets.updater.baselines`. It may import that module's private helpers, such as `_checked_init`, `_validated_forecasts`, `_validated_errors`, `_loss_column`, `_hedge_column` and `_uniform_weights`. The import-linter contracts stay unchanged and must pass.
- Do not edit `.importlinter`, `.github/`, `AGENTS.md` or `docs/agents/`.
- **Cost evidence.** The PR A and PR B descriptions each report the measured time of one `replay_dataset` of the new updater (default config) and of `fixed_share` (default config) on the `synthetic-dev` `per_family = 1` fixture, as a ratio. PR C's description uses those ratios and the wall seconds in the first run's report to estimate the wall-clock of each W6 piece.

---

## PR A: `feat/updater-ridge-stack`

### Files and interfaces

`src/evolvets/updater/learned.py` (new):

```python
LEARNED_IDS: tuple[str, ...] = ("ridge_stack",)  # PR B appends "loss_gate"
LEARNED_DEFAULT_CONFIGS: Mapping[
    str, Mapping[str, ConfigValue]
]  # MappingProxyType, as DEFAULT_CONFIGS
# ridge_stack: {"half_life": 336.0, "shrink": 1.0, "refit_every": 24}
LEARNED: Mapping[str, Callable[[], Updater]]  # id -> zero-argument factory, LEARNED_IDS order


def project_to_simplex(values: Sequence[float]) -> list[float]: ...
def solve_linear(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float] | None: ...


class RidgeStack:  # init / update per the Updater protocol
    ...


def ridge_stack() -> Updater: ...
```

`"evolvets.updater.learned:<id>"` names each learned updater in records (`updater_ref`).

### Exact behaviour: helpers

- **`solve_linear(matrix, rhs)`** works on an `n x n` system with `n >= 1`. It uses Gaussian elimination with partial pivoting, generalising `experts/ridge.py:_solve`:
  - The pivot is the largest absolute value at or below the diagonal; a tie goes to the lowest row index.
  - It returns `None` when a pivot is `0.0`, or when any entry of the solution is NaN or infinite.
  - It never mutates its arguments.
  - A non-square matrix, or `len(rhs) != n`, raises `ValueError`.
- **`project_to_simplex(values)`** is the Euclidean projection onto the probability simplex (Duchi et al., 2008). The input is non-empty and every entry finite, else `ValueError`.
  1. Order the indexes by `(-values[i], i)`.
  2. Accumulate the sorted values left to right as `cs_k`.
  3. `rho` is the last 1-based position `k` with `v_(k) - (cs_k - 1) / k > 0`.
  4. `theta = (cs_rho - 1) / rho`.
  5. `out_e = max(values[e] - theta, 0.0)`.
  6. Finally divide each `out_e` by the loop-sum of `out`, so that every column the updater returns sums to 1 well within the arena's `1e-9`.

### Exact behaviour: `ridge_stack`

**Config.** The config keys are exactly `half_life` (finite, `> 0`, in matured origins), `shrink` (finite, `> 0`) and `refit_every` (int `>= 1`, bools rejected). Anything else raises `ValueError` in `init`. The forgetting factor is `gamma = math.exp(-math.log(2.0) / half_life)`.

**State.** Each horizon column `h` has its own:
- `S[h]`, an `n x n` discounted sum of `e e^T`;
- `m[h]`, the discounted count;
- `weights[h]`, starting uniform.

The state also holds `absorbed` (matured origins seen) and `fitted` (bool). Everything per-series lives in the state object `init` returns, never on the instance.

**`update(state, feedback, context)`** runs these steps in order:

1. Validate the state type, the forecasts and every `Matured`, with the baselines' validators.
2. **Absorb** each `Matured` in feedback order. For each column `h`, let `e` be the column of raw expert errors, `errors[e][h]`.
   - Set `S[h][i][j] = gamma * S[h][i][j] + e_i * e_j`, looping `i` outer and `j` inner.
   - Set `m[h] = gamma * m[h] + 1.0`.
   - Increment `absorbed` by 1.
3. **Refit** if `absorbed >= 1` and either `not fitted` or `context.step % refit_every == 0`. For each column:
   - `Sbar = S[h] / m[h]`, entry by entry.
   - `trace` is the loop-sum of `Sbar[i][i] / n`. If `trace == 0.0`, the column's weights become uniform.
   - Otherwise `A = Sbar + shrink * trace * I`. Solve `A z = 1`, a vector of ones.
     - If `solve_linear` returns `None`, or the loop-sum of `z` is not finite and `> 0`, the column **keeps its previous weights**.
     - Otherwise the column's weights become `project_to_simplex(z_e / sum(z))`.
   - Set `fitted = True`. This is the minimiser of `w^T Sbar w + shrink * trace * |w - u|^2` subject to `sum(w) = 1`, which is `A^{-1} 1` normalised, followed by the projection that enforces `w >= 0`.
4. **Return** the column weights as `[e][h]` rows (`_rows_from_columns`), and the state.

Before the first refit, every column is uniform.

### Tests for PR A (all required; `tests/updater/test_learned.py`)

Replays use `replay_series` on a hand-built forecasts object (as `tests/updater/test_baselines.py` does) unless a test says otherwise.

1. **Registry:** `LEARNED_IDS == ("ridge_stack",)`; `LEARNED["ridge_stack"]()` is a fresh instance on each call; `LEARNED_DEFAULT_CONFIGS["ridge_stack"]` is accepted by `init` and is read-only.
2. **Init validation:** each of these raises `ValueError`:
   - a missing key or an extra key;
   - `half_life` of `0`, `-1`, `nan`, `inf` or `True`;
   - `shrink` of `0` or `nan`;
   - `refit_every` of `0`, `1.5` or `True`;
   - `seed=True`;
   - `n_experts=0` or `horizon=0`.
3. **Uniform first:** before any feedback, and with feedback but `absorbed == 0`, every column is exactly `1/n`.
4. **Valid weights over a long replay:** 400 origins, `n = 4`, `horizon = 3`, with forecasts and targets from `random.Random(0)` (including one expert that is ten times worse). `replay_series` succeeds, which means every column passed the arena's simplex check. A wrapper that records the returned weights shows every entry `>= 0`.
5. **Closed form:** two experts, `horizon = 1`, `refit_every = 1`, three matured origins with given errors. The weights after the third equal a hand-computed `project_to_simplex(A^{-1} 1 / 1^T A^{-1} 1)`, within `1e-12`.
6. **Behaviour:**
   - Constant errors `(+1, -1)` give weights `(0.5, 0.5)` within `1e-12`.
   - Constant errors `(1, 3)` with `shrink = 1e-6` give `w_0 >= 0.999`. The projection clipped the negative weight.
7. **Refit schedule:** with `refit_every = 24`, after the first fit the returned weights at every step whose `step % 24 != 0` equal the previous step's weights exactly. At the next multiple of 24 they change, given changing errors.
8. **Fallbacks:**
   - All-zero errors give uniform columns.
   - Errors of `1e200` overflow `S` to `inf`. The column keeps its previous weights, and the replay does not fail.
   - `solve_linear` on a singular matrix returns `None`.
9. **`project_to_simplex`:**
   - A point already on the simplex is returned unchanged within `1e-15`.
   - `(2.0, 0.0)` gives `(1.0, 0.0)`.
   - `(0.4, 0.4, 0.4)` gives thirds.
   - Negative entries are clipped.
   - The result is the same whatever the input order, up to the matching permutation.
   - Empty or non-finite input raises `ValueError`.
10. **No look-ahead:** two replays whose target series differ only at indexes `> t0 + k` return identical weights at every step `<= k` (recorded through a wrapper) for `k = 50`. This catches any path from future targets to weights.
11. **Determinism and isolation:**
    - Two replays of the same inputs give bit-identical traces.
    - Interleaving `update` calls of two states from one instance gives each the same weights as running it alone.
12. **Arena replay:** `replay_dataset(LEARNED["ridge_stack"], LEARNED_DEFAULT_CONFIGS["ridge_stack"], 0, pool)` on the `synthetic-dev` `per_family = 1` fixture returns `status == "ok"`, and the same again with the pooled fake-foundation fixture.

### Dependencies

None.

### Size

About 200 lines of implementation. Split only if it exceeds ~400 lines with tests: put the helpers (`solve_linear`, `project_to_simplex`, tests 8 and 9) first in `feat/updater-learned-helpers`.

---

## PR B: `feat/updater-loss-gate`

### Files and interfaces

In `src/evolvets/updater/learned.py`:

```python
LEARNED_IDS = ("ridge_stack", "loss_gate")
# LEARNED_DEFAULT_CONFIGS gains loss_gate:
#     {"beta": 16.0, "half_life": 336.0, "ridge": 0.01, "refit_every": 24}
FAST_HALF_LIFE = 4.0  # matured origins
SLOW_HALF_LIFE = 64.0  # matured origins


class LossGate:  # init / update per the Updater protocol
    ...


def loss_gate() -> Updater: ...
```

### Exact behaviour: `loss_gate`

**Config.** The config keys are exactly `beta` (finite, `> 0`), `half_life` (finite, `> 0`), `ridge` (finite, `> 0`) and `refit_every` (int `>= 1`). `gamma` is computed as in `ridge_stack`. The EWMA rates are `a_fast = 1 - math.exp(-math.log(2.0) / FAST_HALF_LIFE)` and likewise `a_slow`.

**Loss.** The loss is the baselines' shared loss: `l_e = _loss_column(errors, h, n)[e]`, in `[0, 1]`.

**State.** Each column `h` has:
- for each expert, `fast[h][e]` and `slow[h][e]` (EWMAs of `l_e`, unset until the first matured loss, which sets both);
- `season_last[h][e]`, a dict from season index to the last matured loss of an origin with that season index;
- a 5x5 `G[h]` and a 5-vector `r[h]` (discounted sums), with discounted count `m[h]`;
- `beta_hat[h]` (5 floats or `None`).

The state also holds `pending`, a dict from step to `(season_index, features)`, where `features[h][e]` is a 5-tuple. It also holds `absorbed`, `fitted` and the config.

**Features at the current origin.** For each column `h` and expert `e`, computed after this call's feedback is absorbed:
- `x = (1.0, fast, slow, season, dev)`;
- `season` is `season_last[h][e][context.season_index]` if present, else `slow`;
- `dev`: let `mu` be the loop-mean of the column's forecasts and `d_e = abs(f_e - mu)`. Then `dev_e = d_e / max_e d_e` when that maximum is `> 0`, else `0.0`.

Before the first matured loss, `fast`, `slow` and `season` are all `0.0`.

**`update(state, feedback, context)`** runs these steps in order:

1. Validate, as in PR A.
2. **Absorb** each `Matured` in feedback order. Pop `pending[matured.step]`; if it is missing, raise `ValueError` (the arena records the replay as failed). For each column and each expert in pool order, with the stored `x` and `l = l_e`:
   - `G[h] = gamma * G[h] + x x^T` and `r[h] = gamma * r[h] + x * l`, entry by entry; after all experts of the origin, `m[h] = gamma * m[h] + n`.
   - Then update `fast`, `slow` (`v = v + a * (l - v)`, or set to `l` the first time) and `season_last[h][e][stored season index] = l`.
   - Increment `absorbed` by 1.
3. **Refit**, on the same schedule as `ridge_stack`. For each column, `A = G[h] / m[h] + ridge * diag(0, 1, 1, 1, 1)` (the intercept is not shrunk) and `b = r[h] / m[h]`. `beta_hat[h] = solve_linear(A, b)`. On `None`, keep the previous `beta_hat[h]`.
4. **Compute features**, store `pending[context.step] = (context.season_index, features)`, and compute the weights:
   - if `beta_hat[h]` is `None`, the column is uniform;
   - otherwise `p_e = beta_hat[h] . x_e`, as a left-to-right dot product. If any `p_e` is not finite, the column is uniform. Otherwise the column is `_hedge_column(p, beta)`, which is `exp(-beta * (p_e - min p))` normalised.
5. Return the `[e][h]` rows and the state. `pending` never holds more than `horizon` entries.

### Tests for PR B (all required; appended to `tests/updater/test_learned.py`)

1. **Registry:** `LEARNED_IDS == ("ridge_stack", "loss_gate")`; the default config is accepted. PR A's tests 2, 3, 4, 10, 11 and 12 are repeated for `loss_gate`, parametrised over `LEARNED_IDS`, with `loss_gate`'s own invalid values for each key.
2. **Unknown step:** a `Matured` whose `step` was never returned by an `update` call on this state raises `ValueError`. Inside `replay_series`, a wrapper that rewrites `step` makes the replay `failed`.
3. **Persistent best:** three experts, where expert 1's error is always `0.1` and the others' `1.0`. With `beta = 64`, after 200 origins expert 1's weight is `>= 0.9` in every column.
4. **Switch:** expert 0 is best for origins `0..499`, then expert 1 is best. With `half_life = 48`, expert 1's weight at origin 999 is `> 0.5` in every column.
5. **Seasonal:** `period = 4`, `horizon = 1`. Expert 0 is best only at origins with season index 0 and expert 1 at the others. After 400 origins, expert 0's mean weight over season-0 origins exceeds its mean weight over the other origins.
6. **Training rows use origin-time features:** with `refit_every = 1` and a hand-built two-step sequence, `G` and `r` after maturity equal the values computed from the features stored at the origin, not from features recomputed at the maturity step. The forecasts at the maturity step are chosen so the two differ.
7. **Pending bound:** after any number of updates in a replay, `len(pending) <= horizon`.
8. **Fallbacks:** `beta_hat` stays `None` before the first matured origin, so the column is uniform; a singular `A` keeps the previous `beta_hat`.

### Dependencies

None.

### Size

About 200 lines of implementation. If it exceeds ~400 lines with tests, put the parametrisation of PR A's tests (test 1) in a follow-up `fix/learned-tests-parametrise` PR after this one.

---

## PR C: `feat/gate1-probe-run`

One recorded run that replays every grid point of every learned updater over dev pools that a completed tune run covers. It is pinned to that tune run.

### Files and interfaces

`src/evolvets/runner/gate1_probe.py` (new):

```python
PROBE_CONDITION = "gate1-dev-probe"
PROBE_GRIDS: Mapping[str, tuple[Mapping[str, ConfigValue], ...]]
# ridge_stack: {"half_life": hl, "shrink": s, "refit_every": 24}
#     for hl in (48.0, 336.0, 2016.0) for s in (0.1, 1.0)          -> 6, half_life outer
# loss_gate:   {"beta": b, "half_life": hl, "ridge": 0.01, "refit_every": 24}
#     for b in (4.0, 16.0, 64.0) for hl in (336.0, 2016.0)         -> 6, beta outer


class ProbeError(Exception): ...


@dataclass(frozen=True)
class ProbeSpec:
    tune_run_id: str
    datasets: tuple[str, ...]  # each in CACHE_DATASETS, no duplicates, in run order
    pool_specs: tuple[str, ...]  # each in POOL_SPECS, no duplicates, in run order
    foundation_models: tuple[str, ...] | None
    seed: int
    max_wall_seconds: float


def probe(
    spec: ProbeSpec,
    paths: DevPaths,
    *,
    ledger_root: Path,
    run_id: str,
    git_commit: str,
    grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]] = PROBE_GRIDS,
) -> str: ...  # the run_end status
```

Small refactors are allowed, provided every existing test in `tests/runner/` passes unchanged:

- `select_configs(records, read_blob, *, updater_ids: Sequence[str] = BASELINE_IDS)`. Only the filter changes.
- `gate1_tune._record_replay` gains keyword-only `factories: Mapping[str, Callable[[], Updater]] = BASELINES` and `ref_prefix: str = "evolvets.updater.baselines:"`.
- `gate1_probe` may import `DevPaths`, `TUNE_CONDITION`, `_load_pools`, `_TimedFactory` and `_record_replay` from `gate1_tune`.

`runner/cli.py` gains `evolvets gate1 probe`.

### Exact behaviour: `probe`

1. **Validate** the spec as `tune` does, and the grids, with `ValueError` before anything is read:
   - grids keys must be in `LEARNED_IDS`;
   - each grid is non-empty;
   - each point is a mapping of str to float, int or str.
2. **Read the tune run.** Every failure here raises `ProbeError`, and nothing is written:
   - `LedgerReader(ledger_root, tune_run_id).verify(check_blobs=True)` must be ok.
   - Its first main record is a `run_start` with `condition == TUNE_CONDITION` and a config with `experiment == "gate1"`.
   - It has a `run_end` with status `completed`.
   - Its `protocol_versions` equal the current `protocol_versions()`, since results from different versions are never mixed (AGENTS.md 2.4).
   - Every requested `(dataset, pool spec)` is in its config's `datasets` x `pool_specs`.
3. **Load every pool first**, as `tune` does. An `ArenaError` propagates and nothing is written.
4. **Pin to the tune run.**
   - For every pool, `pool.cache_id` must equal the tune config's `simple_cache_ids[dataset]`.
   - The pool's foundation cache ids must equal the tune config's `foundation_cache_ids[dataset][pool_spec]`.
   - Otherwise `ProbeError` before start, and nothing is written.
5. **Config blob**, exactly these keys:
   - `"experiment": "gate1"`, `"stage": "dev-probe"`;
   - `"tune_run_id"`, and `"tune_run_end_record_id"` (the tune run's `run_end` record id);
   - `"datasets"`, `"pool_specs"`, `"simple_cache_ids"`, `"foundation_cache_ids"`, `"foundation_models"`, all as `tune` builds them;
   - `"grids"`, `"selection_metric": "mase"`, `"seed"`, `"max_wall_seconds"`;
   - `"updater_refs"`: `{id: "evolvets.updater.learned:<id>"}`.
6. **Start** `Run.start(..., condition=PROBE_CONDITION, seed=spec.seed, budget_limits=Budget(wall_seconds=max_wall_seconds, evaluations=<planned replays>), git_commit=git_commit)`.
7. **Loop.** For each dataset, then each pool spec, then each learned id in `LEARNED_IDS` order, then each config in grid order:
   - check the budget first, exactly as `tune` does;
   - then record one dev `baseline` replay through `_record_replay` with the learned factories and ref prefix. The blob shape is the same (`updater_ref`, `series_wall_seconds`, `result`).
   - **There are no oracle records:** the oracles come from the tune run. A failed updater is a result: record it and continue.
8. **End** with `completed` if every planned replay ran, else `budget_exhausted`. The summary holds `"replays"`, `"ok"`, `"failed"`, `"invalid_weights"`, and `"selected": select_configs(<this run's records>, read_blob, updater_ids=LEARNED_IDS)`. Any other exception ends the run `aborted` through the `Run` context and propagates.

### Exact behaviour: the CLI

`evolvets gate1 probe --tune-run ID --dataset {etth1,ecl,synthetic-dev,all} [--dataset ...] --pool-spec {simple,simple+foundation} [--pool-spec ...] [--foundation-model NAME ...] [--seed 0] [--max-wall-seconds 172800] [--ledger-root ledgers] [--run-id ID] [--cache-root cache_store] [--foundation-root ...] [--data-root data] [--manifest-dir data/manifests]`

- Expansion, defaults and output are as for `gate1 tune`.
- `--run-id` defaults to `new_run_id("gate1-probe")`.
- The git commit comes from `current_git_commit(Path.cwd())`.
- `ValueError`, `ProbeError` or `ArenaError` before start: print `FAIL <message>`, exit 1, and write nothing.
- Success: `OK <run_id> <status>`, then the canonical summary. Exit 0 for `completed`, 2 for `budget_exhausted`.
- An exception after start: `FAIL <run_id> aborted: <message>`, exit 1.

Several probe runs over disjoint pools of one tune run are allowed. This is how W6 runs the pieces in parallel.

### Tests for PR C (all required; `tests/runner/test_gate1_probe.py`)

The fixtures are those of `tests/runner/test_gate1_tune.py`. Each test first makes a tune run with small grids in `tmp_path`, then probes with small learned grids unless the test is about `PROBE_GRIDS`.

1. **Grids:**
   - `PROBE_GRIDS` keys are `LEARNED_IDS` in order, with sizes 6 and 6 and the outer orders stated.
   - There are no duplicates.
   - Every point passes its updater's `init(n_experts=8, horizon=24, config, seed=0)`.
   - Each `LEARNED_DEFAULT_CONFIGS` entry is in its grid.
2. **Record sequence:** main holds `run_start`, then one `baseline` per planned replay in the stated order, then `run_end` `completed`. There is no oracle record, every record has `pool == "dev"`, and there is no sealed stream.
3. **Same numbers as in-process:** each record's `metrics` and blob `result` equal `replay_dataset(LEARNED[id], config, seed, pool)` and `result_to_obj` of it, exactly. `updater_ref` and `hyperparams` are as specified, `evaluations == 1` and `wall_seconds > 0`.
4. **Config blob:** exactly the keys listed. `tune_run_id` and `tune_run_end_record_id` match the tune run, and `condition == "gate1-dev-probe"`.
5. **Refusals before start**, where each one leaves no directory for the probe run id:
   - the tune run is missing;
   - its chain is tampered (one byte of `main.jsonl` changed);
   - it is `aborted`, `budget_exhausted`, or has no `run_end`;
   - its condition is not `gate1-dev-tune` (pass a probe run as the tune run);
   - its `protocol_versions` differ (the tune run is made with `protocol_versions` monkeypatched in `evolvets.runner.runs`);
   - a requested pool is not in the tune run's config;
   - the simple cache id differs (the tune run is made over a cache built with a different `per_family`);
   - a grid key is not a learned id.
6. **Failure is a result:** a grid point with `half_life = 0.0` gives a `failed` record, and the run still ends `completed`, with the failure counted in the summary.
7. **Budget:** `max_wall_seconds` small enough that the second replay is not started gives `budget_exhausted`, and CLI exit code 2.
8. **Selection:**
   - `run_end.summary["selected"] == select_configs(records, read_blob, updater_ids=LEARNED_IDS)`.
   - Over the tune run's records, `select_configs(..., updater_ids=LEARNED_IDS)` is `{}`.
   - Over a probe run's records, the default `select_configs` is `{}`.
9. **Disjoint pieces:** two probe runs of one tune run, one over `simple` and one over `simple+foundation` (pooled fixture), both end `completed` with distinct run ids.
10. **CLI:**
    - the `OK` line and summary;
    - `FAIL` and exit 1 with nothing written for a missing tune run;
    - `FAIL` for a dirty tree (`current_git_commit` monkeypatched to raise `RunError`).
11. **Tune unchanged:** every existing test in `tests/runner/test_gate1_tune.py` and `test_gate1_grids.py` passes without modification.

### Dependencies

None.

### Size

About 180 lines of implementation. If it exceeds ~400 lines with tests, put the two refactors with test 8 first in `feat/gate1-probe-refactor`.

### After PR C merges

The reviewer opens an `fyi-human` issue with three requests:

1. The owner's delegate applies **W6** (below) in a `chore(ci)` PR.
2. The owner dispatches it with `tune_run_id = gate1-tune-20261013T151300Z-b36261f7`.
3. The owner posts the six probe run ids and the workflow run URL on that issue.

PR D does not wait for the run.

---

## PR D: `feat/gate1-probe-report`

### Files and interfaces

`src/evolvets/runner/gate1_probe_data.py` (new; it may import the private helpers of `gate1_report_data`):

```python
@dataclass(frozen=True)
class ProbeRunRow:
    run_id: str
    status: str  # run_end status or "incomplete"
    git_commit: str
    tune_run_id: str
    records: int
    violations: tuple[str, ...]


@dataclass(frozen=True)
class GainRow:
    updater: str
    mean: float  # loop-mean of d_s = best_baseline_mase_s - learned_mase_s
    lower: float | None
    upper: float | None  # None with a single series
    verdict: str  # "beats" | "worse" | "inside" | "n/a"


@dataclass(frozen=True)
class ProbeSection:
    dataset: str
    pool_spec: str
    leakage_risk: bool
    best_baseline: str | None  # None when no baseline has an ok config on this pool
    best_baseline_mase: float | None
    learned: tuple[TunedRow, ...]  # LEARNED_IDS order, as _tuned_rows builds them
    gains: tuple[
        GainRow, ...
    ]  # one per learned updater with a selection; () if best_baseline is None


@dataclass(frozen=True)
class ProbeData:
    run_rows: tuple[ProbeRunRow, ...]
    sections: tuple[ProbeSection, ...]  # leakage_risk False, in tune config order
    leakage_sections: tuple[ProbeSection, ...]


def build_probe_data(
    ledger_root: Path,
    tune_run_ids: Sequence[str],
    probe_run_ids: Sequence[str],
    *,
    bootstrap_seed: int,
) -> ProbeData: ...
```

`gate1_report.build_report(ledger_root, run_ids, *, bootstrap_seed, probe_run_ids: Sequence[str] = ())`.

`evolvets gate1 report` gains a repeatable `--probe-run ID`.

### Exact behaviour

**Unchanged without probe runs.** With `probe_run_ids == ()`, `build_report` returns exactly what it returns on `main`, and every existing report test passes unchanged.

**`build_report_data` gains one refusal.** A run with `condition == "gate1-dev-probe"` passed as `--run` raises `ReportError`.

**`build_probe_data`.** Everything not listed below as a refusal or an untabled case is computed from the ledger alone.

Each of the following is a `ReportError`:
- an empty `probe_run_ids`, or a duplicate in it;
- a probe run id that is also in `tune_run_ids`;
- a probe run whose chain or blobs fail `verify`;
- a sealed-stream record;
- a first main record that is not a `run_start` with condition `gate1-dev-probe`;
- protocol versions that differ from the tune runs';
- a config `tune_run_id` not in `tune_run_ids`;
- a config `tune_run_end_record_id` not equal to that tune run's `run_end` record id;
- a named tune run that is not `completed`;
- a result whose dataset, pool spec, cache id or foundation cache ids disagree with the probe config or the tune run's config;
- an ok result whose series ids differ from the tune run's oracle series ids for that pool;
- a `run_end` `selected` that `select_configs(..., updater_ids=LEARNED_IDS)` does not reproduce;
- two complete probe runs covering one `(dataset, pool spec)`;
- a learned record whose `updater_id` is not in `LEARNED_IDS`.

Incomplete probe runs are listed in `run_rows` and never tabled.

**For each covered pool:**
- The best tuned baseline is the one the main report uses: `_best_baseline` over the tune run's selection.
- Its per-series MASE comes from the selected record's blob.
- For each learned updater with an ok selection, `d_s = best_mase_s - learned_mase_s`, and `GainRow` comes from `_gap_rows([(id, d) ...], bootstrap_seed)`, one table per pool in `LEARNED_IDS` order.
- The verdict is:
  - `beats` if `lower > 0`;
  - `worse` if `upper < 0`;
  - `inside` otherwise;
  - `n/a` with a single series.
- `learned` rows use `_tuned_rows` semantics over the probe records, including the cost columns.
- Sections are ordered by the tune runs' config order, and split by `leakage_risk`.

**Rendering.** This applies only when probe runs are given. After `## Per-series wall-clock` and before `## Notes`, the report adds:

```
## Learned-updater probe

### Probe runs
| run id | condition | status | git commit | tune run | records | validate_run |

### Probe verdicts
| dataset | pool | leakage risk | <one column per LEARNED_IDS entry> |
(cells: verdict, or "none" without an ok selection; every section, non-leakage first)

### <dataset>, pool <pool_spec>[ (leakage risk)]
Best tuned baseline: <id> (dev MASE <x>)   | or: No tuned baseline has an ok configuration on this pool.
| updater | selected config | dev MASE | configurations | failed | evaluations | wall seconds |
| updater | mean gain over <id> | 95% interval | verdict |
```

The Notes section then gets these extra fixed lines after the existing ones:

- Gain is the best tuned baseline's per-series dev MASE minus the learned updater's; positive favours the learned updater. `beats` means the whole 95% paired bootstrap interval is above zero.
- Both the baselines and the learned updaters were selected on these same dev series; the configurations column shows how many grid points each had.
- Each verdict is one of several comparisons (pools x learned updaters); no multiple-comparison correction is applied.
- The learned updaters return convex weights over the pool (the arena's updater interface); learners that output forecasts directly are not tested here.

**Formatting.** Every float is `f"{x:.4f}"` and every config is `canonical_json`. The generating line becomes `evolvets gate1 report --run ... --probe-run ... --bootstrap-seed N` when probe runs are given.

**CLI.** `--check` and `--out` behave as before.

### Tests for PR D (all required; `tests/runner/test_gate1_probe_data.py`, plus new tests in `tests/runner/test_gate1_report.py`)

The fixtures are a tune run plus probe runs in `tmp_path` (PR C's `probe`), and hand-built ledgers for refusals.

1. **Unchanged:**
   - `build_report(root, [tune], bootstrap_seed=0)` is byte-identical to `build_report(root, [tune], bootstrap_seed=0, probe_run_ids=())`.
   - Neither contains `probe`.
   - All existing tests in `test_gate1_report.py` and `test_gate1_report_data.py` pass unchanged.
2. **Numbers:**
   - `learned` selections equal the probe `run_end` `selected`.
   - Each `GainRow.mean` equals the loop-mean of `d_s` computed by hand from the two selected blobs.
   - The bounds equal `_gap_rows` on the same input and seed.
   - The same seed gives identical data. A different seed gives equal means.
3. **Verdicts:** hand-built differences that are all positive give `beats`, all negative give `worse`, mixed give `inside`, and a single series gives `n/a` with `None` bounds.
4. **Refusals:** each refusal listed under `build_probe_data` has a test, plus the new `build_report_data` refusal of a probe run passed as `--run`.
5. **Incomplete probe run:** listed with status `aborted` (or `incomplete`) in `run_rows`, and no section uses it.
6. **No best baseline:** a tune pool where every baseline failed gives `best_baseline None`, `gains == ()`, and the sentence in the rendering.
7. **All-failed learned updater:** its `TunedRow` has `config None`, and it has no `GainRow`. The verdict cell is `none`.
8. **Leakage order:** the `etth1_env` fixture of `tests/runner/test_gate1_report_data.py`, probed over `simple` and `simple+foundation`, gives the pooled section after the non-leakage ones, with ` (leakage risk)` in the heading. The verdict table lists it last with `leakage risk` `yes`.
9. **Rendering:**
   - The section order, headings and table headers are as specified.
   - The notes lines are present only with probe runs.
   - The generating line includes the `--probe-run` ids.
   - Two renders of the same ledger are byte-identical.
10. **CLI:**
    - `--probe-run` is repeatable.
    - A refusal prints `FAIL` and leaves `--out` unwritten.
    - `--check` passes on the written file and fails on a one-character change.

### Dependencies

None.

### Size

About 300 lines of implementation. If it exceeds ~400 lines with tests, split:

- `feat/gate1-probe-report-data`: `gate1_probe_data.py`, the `build_report_data` refusal, and tests 2 to 7;
- then `feat/gate1-probe-report`: the rendering, CLI, and tests 1, 8 (rendering part), 9 and 10.

### After PR D merges

The reviewer opens an `fyi-human` issue with three requests:

1. The owner's delegate applies **W7** (below).
2. Once W6 has finished, the owner dispatches W7 with the tune run id and every completed probe run id.
3. The owner posts the W7 workflow run URL and the artifact's SHA-256 on that issue.

---

## PR E: `docs/gate1-probe-results`

PR E starts only after the owner has posted the W6 and W7 workflow run URLs, as above. Until then, the developer has nothing to do on this task.

### Files

`docs/results/gate1-learned-probe.md`, structured like `docs/results/gate1-dev-first-run.md`:

1. **What this is:** two or three sentences on what was replayed, against what, and on which pools.
2. **Verdicts:** the report's `### Probe verdicts` table, copied verbatim. Then exactly one sentence in the owner's decision rule:
   - either "A learned updater beats the best tuned baseline outside the 95 % interval on: <dataset/pool list, with the updater>."
   - or "No learned updater beats the best tuned baseline outside the 95 % interval on any pool."

   No other interpretation, and no statement about Phase 1 or the generator. That is the Gate 1 decision.
3. **Caveats**, all of these:
   - dev numbers, with every selection made on the same series;
   - ETTh1 and ECL foundation results are leakage-risk;
   - several comparisons, uncorrected;
   - only convex-weight learners were tried (two of them, with 6 grid points each);
   - the baselines' grid-edge note from the first run (Hedge `eta` at the lowest grid value on ETTh1 and ECL).
4. **Provenance:**
   - the tune run id and every probe run id W6 started, with each one's status, including any aborted piece;
   - each run's recorded git commit, and its mapping to the real `main` SHA if the tarball fallback was used;
   - the commands, the seeds, and the W6 and W7 workflow run URLs;
   - the report artifact's SHA-256.
5. **The full report**, verbatim, byte-identical to the W7 artifact. The developer downloads it with `gh run download` and checks `sha256sum` before pasting.

The same PR changes this file's first line to `Status: done` and sets roadmap entry 11b's status to the merged PR numbers.

### Checks

These are documentation-only checks, stated in the PR description:

- the artifact's SHA-256 matches the posted one;
- the embedded report equals the artifact;
- every probe run id from W6's log is listed.

### After PR E merges

The reviewer opens a `needs-human` issue titled `Gate 1 decision after the learned-updater probe`. It holds:

- the verdict table;
- the dev MASE of the best tuned baseline and of each learned updater per pool;
- the switch-8 oracle and true-best gaps from the first run;
- a link to the results note.

It lays out the owner's two options from #110. No agent plans past it.

---

## Owner-applied workflows (`.github/`, applied by the owner's delegate, never by an agent)

These follow the rules of `gate1-tune.yml`:

- `workflow_dispatch` only;
- `if: github.actor == github.repository_owner`;
- `runs-on: [self-hosted, gpu]`;
- `CUDA_VISIBLE_DEVICES: ""` (CPU work, which never touches a GPU);
- the same paths and `gpu-env.sh dev "$VENV"`;
- `concurrency` with `cancel-in-progress: false`.

**Checkout.** Try git first, with three attempts. If the tarball fallback is used, the job prints `::warning` with `github.sha`, so the results note can map the recorded commit to the real one, as #108 does.

**W6 `gate1-probe.yml`: the probe run.**

- **Inputs:**
  - `tune_run_id` (required);
  - `max_wall_seconds` (default `172800`, per piece);
  - `seed` (default `0`);
  - `pieces`, which defaults to `etth1:simple etth1:simple+foundation ecl:simple ecl:simple+foundation synthetic-dev:simple synthetic-dev:simple+foundation`.
- **Settings:** `timeout-minutes: 3600`, concurrency group `gate1-probe`.

Steps:

1. Checkout, paths and environment, as `gate1-tune.yml`. `git status --porcelain` must be empty.
2. For each piece `d:p`, start `evolvets gate1 probe --tune-run <id> --dataset d --pool-spec p --seed <seed> --max-wall-seconds <n> --ledger-root "$LEDGER_ROOT" --cache-root "$CACHE_ROOT" --foundation-root "$FM_ROOT" --data-root "$DATA_ROOT" --manifest-dir data/manifests` in the background, with its output in its own log file. Then `wait` for each piece and collect its exit code.
3. Print each piece's `OK` or `FAIL` line, and one line `PROBE_RUN_IDS=<space-separated ids>` covering every run that started.
4. The job succeeds only if every exit code is 0 or 2.

A piece that ended `aborted` may be re-dispatched alone through `pieces`. A piece that completed is never re-run. If one is re-run anyway, the results note lists both runs and the report tables neither until the owner says which one counts. The report refuses two complete runs over one pool.

**W7 amendment to `gate1-dev-report.yml`.**

- New input `probe_run_ids` (space-separated, default empty). Each id is passed as `--probe-run`.
- The "check its commit" step also prints each probe run's recorded commit and its GitHub lookup status.
- Everything else is unchanged.

---

## Not in this task

- **The Gate 1 decision**, and anything in Phase 1 or any generator hardening.
- **A gradient-boosted or neural combiner.** Either needs a dependency (scikit-learn, LightGBM, torch) or a pure-Python implementation too slow for these pools. If the owner wants one, it is a separate request with a dependency decision (AGENTS.md section 7).
- **Learners that output forecasts rather than weights** (bias correction, unconstrained stacking). They need an `updater/interfaces.py` change and an `ARENA_VERSION` bump, which is a `protocol/` PR and an owner decision.
- **A wider baseline grid or a tune redo** (owner, #105: a research choice for later).
- **Sealed evaluation of the learned updaters** (TASK_012 is superseded).

## Closing the task

PR E sets `Status: done` and the roadmap status, as above. The run itself, W6 and W7 are outside the developer's stack.
