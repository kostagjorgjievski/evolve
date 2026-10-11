Status: active

# Task 004: Synthetic world generator with true-best forecast

Read `AGENTS.md` and `docs/ROADMAP.md` first. This task covers roadmap entry 5 (`feat/generator`). It is a stack of two PRs; open them in order and stop after each one for review.

## Why

Research question 1 asks how far hand-designed updaters are from the best possible forecast. On real data we only have a switching oracle over experts. On synthetic worlds we know the data-generating process, so we can compute the **true-best forecast**: the lowest expected error any forecaster could achieve. The gap between a tuned baseline and that floor is the headroom Gate 1 measures.

Each world plants known **mismatches**: changes (level shifts, slope changes, seasonality changes, noise changes, recurring regimes, noise-only bursts) that simple experts and updaters handle badly. Their positions and kinds are recorded so that PR 8 can measure recovery time and Phase 2 can use them as planted defects.

## Rules for both PRs

- Code lives in the package `src/evolvets/arena/synthetic/` (strict mypy, as for all of `arena/`). Tests live in `tests/arena/`.
- **Standard library only** (`random`, `math`, `hashlib`, `dataclasses`). No numpy. Series are `tuple[float, ...]`. Later PRs may convert them.
- **Randomness**: only `random.Random(seed)` with an `int` seed, and only its `random()` method. Normals come from our own Box-Muller (below), never `random.gauss` or `random.normalvariate`, whose output Python does not guarantee across versions. No module-level RNG, no default seeds.
- `GENERATOR_VERSION = "1"` in `arena/synthetic/__init__.py`. This is the value runs put in `run_start.protocol_versions["generator"]`. Any change that alters the output for an existing (spec, seed) or (family, seed) must bump it; the golden-fingerprint tests enforce this.
- The true-best forecast knows the future mean path. It is an **oracle for scoring only**. It must never be passed to an expert or an updater. In this task nothing outside `arena/synthetic/` and `tests/` calls it.
- All dataclasses are `frozen=True` and validate in `__post_init__`, raising `WorldSpecError(ValueError)` (PR A) or `GrammarError(ValueError)` (PR B). Every float field must be finite.

---

## PR A: `feat/generator-core`

Defines a world exactly, simulates it, and computes the true-best forecast. No randomness in choosing the world; that is PR B.

### Files and interfaces

`arena/synthetic/spec.py`

```python
@dataclass(frozen=True)
class Regime:
    start: int                 # first index where this regime is active
    level: float
    slope: float               # per step, measured from `start`
    season_amp: float          # >= 0
    season_phase: float        # radians, 0 <= phase < 2*pi
    ar_phi: float              # -1 < phi < 1
    sigma: float               # innovation std, > 0
    transition: Literal["sudden", "gradual"]
    ramp: int                  # 0 if sudden, >= 2 if gradual
    recurs_from: int | None    # index of an earlier regime this one repeats, or None

@dataclass(frozen=True)
class NoiseBurst:
    start: int
    length: int                # >= 1
    sigma_mult: float          # > 1

@dataclass(frozen=True)
class WorldSpec:
    length: int                # >= 2
    period: int                # seasonal period, >= 2
    regimes: tuple[Regime, ...]
    bursts: tuple[NoiseBurst, ...]

def spec_to_dict(spec: WorldSpec) -> dict[str, Any]: ...   # JSON-ready, for hashing and the ledger
```

`WorldSpec` validation (each is a separate error case):

1. `regimes` non-empty; `regimes[0].start == 0`, `transition == "sudden"`, `ramp == 0`, `recurs_from is None`.
2. Starts strictly increasing and all `< length`.
3. For a gradual regime `r`, `start_r + ramp_r <= start_{r+1}` (or `<= length` for the last regime). A ramp may not overlap the next change.
4. `recurs_from`, if set, is a smaller index, and the regime's `level`, `slope`, `season_amp`, `season_phase`, `ar_phi`, `sigma` equal the referenced regime's exactly.
5. Bursts sorted by `start`, non-overlapping, each inside `[0, length)` (`start + length <= length` of the world).
6. Booleans are not accepted where ints are expected (`True` is not a valid `start`).

`arena/synthetic/simulate.py`

```python
def standard_normals(seed: int, n: int) -> tuple[float, ...]: ...
def simulate_from_innovations(spec: WorldSpec, z: Sequence[float]) -> World: ...
def simulate(spec: WorldSpec, seed: int) -> World: ...   # = simulate_from_innovations(spec, standard_normals(seed, spec.length))

@dataclass(frozen=True)
class World:
    spec: WorldSpec
    seed: int | None           # None when built from explicit innovations
    y: tuple[float, ...]       # observed series
    mean: tuple[float, ...]    # mu_t
    noise: tuple[float, ...]   # e_t
    sigma: tuple[float, ...]   # sigma_t actually used
    phi: tuple[float, ...]     # phi_t actually used
    change_points: tuple[int, ...]           # starts of regimes 1..R-1
    burst_spans: tuple[tuple[int, int], ...] # (start, end exclusive)

def world_fingerprint(world: World) -> str: ...
```

### Exact behaviour

- `standard_normals(seed, n)`: `seed` is an int in `[0, 2**63)`, else `ValueError`. `rng = random.Random(seed)`; for each value draw `u1 = rng.random()`, `u2 = rng.random()` and return `sqrt(-2 * log(1 - u1)) * cos(2 * pi * u2)`. (`1 - u1` is in `(0, 1]`, so no `log(0)`.) One normal per pair; the sine partner is discarded.
- `simulate_from_innovations`: `len(z)` must equal `spec.length` and every `z` finite, else `ValueError`. Let `r(t)` be the last regime with `start <= t`, and `m_r(t) = level_r + slope_r * (t - start_r) + season_amp_r * sin(2*pi*t/period + season_phase_r)`.
  - Mean: `mu_t = m_{r(t)}(t)`, except during a gradual ramp (`start_r <= t < start_r + ramp_r`), where `alpha = (t - start_r + 1) / (ramp_r + 1)` and `mu_t = (1 - alpha) * m_{r-1}(t) + alpha * m_r(t)`. So `alpha` is strictly between 0 and 1 during the ramp, and `mu_t = m_r(t)` from `start_r + ramp_r` on.
  - Noise parameters switch at `start_r` with no ramp: `phi_t = ar_phi_{r(t)}`; `sigma_t = sigma_{r(t)} * sigma_mult` inside a burst, else `sigma_{r(t)}`.
  - `e_0 = sigma_0 / sqrt(1 - phi_0**2) * z_0` (stationary start); `e_t = phi_t * e_{t-1} + sigma_t * z_t`; `y_t = mu_t + e_t`.
- `world_fingerprint`: SHA-256 hex of `canonical_bytes` (from `ledger/canonical.py`) of `{"generator_version", "spec": spec_to_dict(spec), "seed", "y": [format(v, ".9e") for v in y]}`. Formatting to 10 significant digits keeps the fingerprint stable against last-bit `libm` differences.

`arena/synthetic/oracle.py`

```python
def true_best_forecast(world: World, origin: int, horizon: int) -> tuple[float, ...]: ...
def true_best_error_variance(world: World, origin: int, horizon: int) -> tuple[float, ...]: ...
```

- Origin `t` means `y_0 .. y_t` have been observed. Valid iff `horizon >= 1`, `origin >= 0` and `origin + horizon <= length - 1`; otherwise `ValueError`. Bools are rejected.
- Forecast for `h = 1..H`: `mu_{t+h} + (prod_{k=1..h} phi_{t+k}) * e_t`. This is the conditional mean given the full spec (including future change points and bursts) and the past. Innovations are Gaussian, so it is also the conditional median: it minimises both expected squared and expected absolute error. No forecaster can beat it in expectation; that is what "true-best" means here.
- Error variance for `h`: `sum_{j=1..h} sigma_{t+j}**2 * prod_{k=j+1..h} phi_{t+k}**2`.
- The forecast reads only `world.mean`, `world.phi` and `world.noise[origin]`. It never reads `y` or `noise` after `origin`.

### Tests for PR A (all required; `tests/arena/test_synthetic_core.py`, `test_synthetic_oracle.py`)

1. Each validation rule 1–6 above has a failing case that raises `WorldSpecError`, plus one valid multi-regime spec with a gradual ramp, a recurring regime and two bursts that constructs. Also: NaN or infinite `level`, `sigma <= 0`, `|ar_phi| >= 1`, `season_amp < 0`, `period < 2`, `length < 2`, `sigma_mult <= 1` each raise.
2. `standard_normals`: same seed gives identical output; different seeds differ; negative seed, seed `>= 2**63` and a `bool` seed raise; `n = 0` returns `()`. Over `n = 200_000` with a fixed seed, sample mean is within 0.01 of 0 and sample variance within 0.02 of 1.
3. `simulate_from_innovations` rejects a wrong-length `z` and a `z` containing NaN.
4. Mean path, sudden change: at `start_r - 1` the mean equals `m_{r-1}`, at `start_r` it equals `m_r` (exact formula check at both indices).
5. Mean path, gradual change with `ramp = 4`: `alpha` at the five indices `start_r .. start_r + 4` is `1/5, 2/5, 3/5, 4/5` and then the mean equals `m_r` exactly.
6. Noise statistics: one stationary regime, `phi = 0.6`, `sigma = 0.5`, `length = 200_000`, fixed seed: sample variance of `noise` within 5% of `sigma**2 / (1 - phi**2)`, lag-1 autocorrelation within 0.02 of `phi`.
7. Bursts change only noise: the same spec with and without a burst, same seed, gives identical `mean`; identical `noise` before the burst start; `sigma` multiplied by exactly `sigma_mult` inside the burst and unchanged outside it.
8. Determinism: `simulate(spec, s)` twice gives equal `World`s and equal fingerprints; a different seed gives a different fingerprint.
9. Golden fingerprint: a fixed hand-written spec (in the test file) with seed `12345` has a pinned fingerprint string. The test's docstring says that changing it requires bumping `GENERATOR_VERSION`.
10. `true_best_forecast` matches the closed form on a two-regime world, including an origin just before a sudden change, where the forecast for steps after the change uses the new regime's mean.
11. No look-ahead in the oracle beyond the spec: build a world from `z`, then rebuild with `z[t+1:]` replaced by other values; `true_best_forecast(world, t, H)` is unchanged.
12. Origin boundaries: `origin = length - 1 - H` works; `origin = length - H`, `origin = -1`, `horizon = 0` and `bool` arguments raise.
13. Calibration: on a world with two regimes and one burst, `length = 200_000`, `H = 24`, the empirical mean squared error of the oracle over all valid origins, per `h in {1, 6, 24}`, is within 5% of the mean of `true_best_error_variance` over the same origins.
14. Ceiling: on the same world, the oracle's MAE at every `h` is lower than that of the naive forecast (`y_t` for all `h`) and of the mean-only forecast (`mu_{t+h}`).

### Size

About 250 lines of implementation. If it grows past ~400, split `oracle.py` and its tests into a follow-up PR.

---

## PR B: `feat/generator-grammar`

Samples worlds from named families and derives seeds per pool.

### Files and interfaces

`arena/synthetic/grammar.py`

```python
FAMILIES = ("stationary", "sudden", "gradual", "recurring", "false_drift", "mixed")
CHANGE_KINDS = ("level_shift", "slope_change", "season_change", "noise_change")

@dataclass(frozen=True)
class GrammarConfig:
    length: int = 2000
    period: int = 24
    warmup: int = 300            # no change point or burst before this index
    min_segment: int = 200       # minimum gap between consecutive change points
    max_changes: int = 3         # changes drawn uniformly from 1..max_changes
    level_range: tuple[float, float] = (-5.0, 5.0)
    slope_range: tuple[float, float] = (-0.002, 0.002)
    new_slope_range: tuple[float, float] = (-0.01, 0.01)
    season_amp_range: tuple[float, float] = (0.0, 3.0)
    phi_range: tuple[float, float] = (0.0, 0.9)
    sigma_range: tuple[float, float] = (0.3, 1.0)
    shift_range: tuple[float, float] = (2.0, 5.0)   # level shift size, in stationary noise stds
    ramp_range: tuple[int, int] = (20, 150)
    max_bursts: int = 3
    burst_length_range: tuple[int, int] = (10, 50)
    burst_mult_range: tuple[float, float] = (2.0, 4.0)

@dataclass(frozen=True)
class SyntheticTask:
    family: str
    world: World
    change_kinds: tuple[str, ...]   # aligned with world.change_points; "recurrence" for recurring changes

def sample_spec(family: str, rng: random.Random, config: GrammarConfig) -> tuple[WorldSpec, tuple[str, ...]]: ...
def make_task(family: str, seed: int, config: GrammarConfig = GrammarConfig()) -> SyntheticTask: ...
def config_to_dict(config: GrammarConfig) -> dict[str, Any]: ...
```

`arena/synthetic/seeds.py`

```python
def world_seed(pool: Literal["dev", "sealed"], family: str, index: int, salt: bytes = b"") -> int: ...
def dev_tasks(family: str, n: int, config: GrammarConfig = GrammarConfig()) -> list[SyntheticTask]: ...
```

### Exact behaviour

- **`GrammarConfig` validation**: every range has `lo <= hi`; `ramp_range[0] >= 2`; `ramp_range[1] < min_segment`; `max_changes >= 1`; `(length - warmup) // max(2, max_changes, max_bursts) >= min_segment` (so slotting below always fits; the 2 covers the recurring family's minimum); `burst_length_range[1] < min_segment`; `burst_mult_range[0] > 1`; `sigma_range[0] > 0`; `phi_range` inside `(-1, 1)`; `season_amp_range[0] >= 0`. Each violation raises `GrammarError`.
- **Uniform draws**: a float in `[lo, hi]` is `lo + (hi - lo) * rng.random()`; an int in `[lo, hi]` is `lo + int(rng.random() * (hi - lo + 1))`, capped at `hi`. Nothing else consumes the RNG. The order of draws is the order written in the implementation and is pinned by the golden test.
- **Base regime** (`start = 0`, sudden): level, slope, season amplitude, phase in `[0, 2*pi)`, phi, sigma, each drawn from its range.
- **Placing `k` events** (change points or bursts): split `[warmup, length)` into `k` equal slots of width `w = (length - warmup) // k`. Event `i` is at `warmup + i*w + randint(0, w - min_segment)`. This guarantees gaps of at least `min_segment` and that every ramp (`< min_segment`) and burst (`< min_segment`) ends before the next event.
- **Change kinds** (new regime copies the previous one, then changes only the named parameters):
  - `level_shift`: `level += sign * shift * sigma / sqrt(1 - phi**2)`, `sign` is +1 or -1 with probability 1/2 (`rng.random() < 0.5`), `shift` from `shift_range`.
  - `slope_change`: `slope` redrawn from `new_slope_range`.
  - `season_change`: `season_amp` and `season_phase` redrawn.
  - `noise_change`: `ar_phi` and `sigma` redrawn.
- **Families**:
  - `stationary`: one regime, no bursts, `change_kinds = ()`.
  - `sudden`: `k` from `1..max_changes`; each change has a kind drawn uniformly from `CHANGE_KINDS`; all transitions sudden.
  - `gradual`: as `sudden`, but every transition is gradual with `ramp` from `ramp_range`.
  - `recurring`: draw base regime A and a second regime B (B is A after one drawn change kind). `k` from `2..max(2, max_changes)`. Regimes alternate A, B, A, B, ...; every regime after the first two has `recurs_from` set to 0 or 1 and copies its parameters exactly. All transitions sudden. `change_kinds[0]` is B's kind; the rest are `"recurrence"`.
  - `false_drift`: one regime; `1..max_bursts` bursts placed by the slotting rule, each with length and multiplier from their ranges. `change_kinds = ()`.
  - `mixed`: as `sudden`, but each transition is independently sudden or gradual (probability 1/2), plus `0..max_bursts` bursts. Burst slots are computed independently of change slots, so a burst may coincide with a change. That is intended.
- `make_task(family, seed, config)`: unknown family raises `GrammarError`. Uses two independent sub-seeds: `spec_seed` and `noise_seed`, each the first 8 bytes (big-endian, top bit cleared) of `SHA-256(b"evolvets/generator/v1/" + label + b"/" + str(seed).encode())` with label `b"spec"` or `b"noise"`. `sample_spec(family, random.Random(spec_seed), config)`, then `simulate(spec, noise_seed)`. Same seed gives the same task.
- **Pools and seed secrecy**. Dev and sealed tasks come from disjoint, domain-separated seed spaces. `world_seed(pool, family, index, salt)` is the first 8 bytes (top bit cleared) of `SHA-256(b"evolvets/generator/v1\x00" + pool + b"\x00" + family + b"\x00" + str(index) + b"\x00" + salt)`.
  - `pool == "dev"`: `salt` must be empty, else `ValueError`. Dev tasks are public and reproducible from the repo.
  - `pool == "sealed"`: `salt` must be at least 32 bytes, else `ValueError`. Without the salt, sealed tasks cannot be regenerated from the repository, even though the grammar is public. This task does not create, store or read any salt. Where the sealed salt lives, and how it is committed to and later revealed so sealed results stay reproducible from the ledger, is decided in the PR 9 (`feat/arena-sealed`) spec.
  - Unknown pool or family, negative or `bool` index raise `ValueError`.
- `dev_tasks(family, n, config)` returns `[make_task(family, world_seed("dev", family, i), config) for i in range(n)]`. There is deliberately no sealed counterpart in this task.

### Tests for PR B (all required; `tests/arena/test_synthetic_grammar.py`, `test_synthetic_seeds.py`)

1. For every family and 200 seeds each, `make_task` succeeds (so every sampled spec passed `WorldSpec` validation) and `len(change_kinds) == len(world.change_points)`.
2. Family structure, checked over 50 seeds per family: `stationary` has one regime and no bursts; `sudden` has 1..3 changes, all sudden; `gradual` has all transitions gradual with ramps in `ramp_range`; `recurring` alternates A/B with `recurs_from` pointing at 0 or 1 and matching parameters exactly, kinds `[<kind>, "recurrence", ...]`; `false_drift` has one regime and 1..3 bursts; `mixed` has 1..3 changes and 0..3 bursts.
3. Placement: over the same tasks, every change point and burst start is `>= warmup`; consecutive change points are at least `min_segment` apart; consecutive bursts likewise.
4. Change kinds change only their parameters: for each change in `sudden` tasks, comparing regime `r` with `r - 1`, exactly the parameters named by its kind differ (with `level_shift` magnitude within `shift_range` in stationary-noise units).
5. `make_task` is deterministic, and different seeds give different fingerprints.
6. Golden fingerprints: `make_task(f, world_seed("dev", f, 0))` for every family has a pinned fingerprint string in the test file. The docstring says changing them requires bumping `GENERATOR_VERSION`.
7. `GrammarConfig` validation: each rule above has a failing case; the default config constructs.
8. A non-default config works: `length = 800, warmup = 100, min_segment = 150, max_changes = 4, max_bursts = 4` constructs (`700 // 4 = 175 >= 150`) and produces valid tasks for every family. The same config with `max_changes = 5` raises (`700 // 5 = 140 < 150`).
9. `world_seed`: deterministic; dev with a non-empty salt raises; sealed with no salt or a 31-byte salt raises; for the same family and index, dev and sealed (32-byte salt) seeds differ, and two different sealed salts give different seeds; different families give different seeds; over indices `0..9999` the dev seeds are all distinct and none equals a sealed seed for a fixed salt; unknown pool, unknown family, negative index and `True` as index raise.
10. `dev_tasks("sudden", 3)` equals the three tasks built by hand from `world_seed("dev", "sudden", i)`.
11. Headroom sanity: over 20 dev `sudden` tasks with `H = 24`, the true-best forecast's mean absolute error over all valid origins is lower than the naive forecast's at `h = 1` and `h = 24`.

### Size

About 200 lines of implementation. If it grows past ~400, split `seeds.py` and its tests into a follow-up PR.

---

## Not in this task

- **Import contract for the oracle.** `true_best_forecast` reads the future mean path, so agent-editable packages must not import `evolvets.arena.synthetic` once updaters exist (PR 10). That needs a new contract in `.importlinter`, which a human must review, so it is not part of this task. The planner raises it again when specifying PR 8 or PR 10.
- Sealed-salt storage and reveal (PR 9), experts and their mismatch with these worlds (PR 7), recovery-time metrics (PR 8).

## Closing the task

PR B (`feat/generator-grammar`) changes this file's first line to `Status: done`.
