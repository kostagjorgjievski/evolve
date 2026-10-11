Status: active

# Task 006: Five simple experts and the prediction cache

Read `AGENTS.md` and `docs/ROADMAP.md` first. This task covers roadmap entry 7 (`feat/experts-cache`). It is a stack of two PRs; open them in order and stop after each one for review.

## Why

An updater (PR 10) combines the forecasts of a fixed pool of experts. Every updater in every condition must see exactly the same expert forecasts, and computing them once is far cheaper than recomputing them for each evaluation. This task defines five simple, deterministic, look-ahead-safe experts (PR A) and a content-addressed cache of their forecasts over every origin of every public series (PR B). PR 8's replay loop reads forecasts only from this cache.

## Rules for both PRs

- **Standard library only** (`math`, `array`, `collections`, `dataclasses`, `hashlib`, `json`, `os`, `sys`). No numpy, no statsforecast, no LightGBM. TASK_001 (superseded) named AutoETS via `statsforecast` and LightGBM; both are heavy native dependencies that need human approval (AGENTS.md section 7), so this task uses pure-Python equivalents: a damped Holt-Winters with fixed parameters in place of AutoETS, and a ridge autoregression in place of gradient-boosted lags. Adding heavier experts later is a separate, human-approved roadmap entry (compare entry 12).
- **Bit-exact determinism.** Same inputs give byte-identical cache files on every platform and Python 3.11+. Use only `+ - * /`, comparisons, `abs` and `math.sqrt` on floats. **Never call the builtin `sum()` on floats** (its algorithm changed in Python 3.12, so results differ between 3.11 and 3.12) and never use `statistics` or `math.fsum`: accumulate with an explicit left-to-right `for` loop. Golden-hash tests enforce this.
- **No randomness.** Nothing in this task draws random numbers; there are no seeds to plumb.
- **Origin convention** (same as `arena/synthetic/oracle.py`): origin `t` means `y_0 .. y_t` have been observed; the forecast for step `h = 1..H` targets `y_{t+h}`.
- `EXPERTS_VERSION = "1"` in `experts/__init__.py`; `CACHE_VERSION = "1"` in `cache/__init__.py`. Any change that alters any forecast value must bump `EXPERTS_VERSION`; any change to the cache file layout or manifest must bump `CACHE_VERSION`. Golden tests enforce both. `CACHE_VERSION` is the value runs will put in `run_start.protocol_versions["cache_manifest"]` (PR 8); the specific cache a run used is identified by the cache id below.
- **No ledger change.** Nothing here writes ledger records or touches `ledger/`, `PROTOCOL_KEYS` or anything frozen. Issue #10 (tag `ledger-schema-v1`) is unaffected by this task.
- **Strict mypy on `experts/` too:** add `"evolvets.experts.*"` to the strict override list in `pyproject.toml` (`cache/` is already strict).
- **Experts see values only.** `experts/` imports nothing from `evolvets` except its own modules. It never receives a `World`, a mean path, a split or a timestamp: just floats, one at a time.
- Errors: `ExpertError(ValueError)` (PR A); `CacheError(Exception)` (PR B). Bools are rejected wherever an int is expected.

---

## PR A: `feat/experts-core`

The expert interface, the five experts, and a function that runs them over one series in memory. No files written.

### Files and interfaces

`experts/interfaces.py`

```python
class Expert(Protocol):
    """A streaming univariate forecaster. It only knows the values it has been shown."""

    name: str
    min_history: int  # forecast() is valid once this many values have been observed

    def observe(self, y: float) -> None: ...
    def forecast(self, horizon: int) -> tuple[float, ...]: ...
    def params(self) -> dict[str, float | int]: ...  # JSON-ready, for the cache manifest
```

`experts/simple.py`

```python
# name "naive"
class Naive: ...


# name "seasonal_naive"; __init__(self, period: int)
class SeasonalNaive: ...


# name "ses"; __init__(self, alpha: float = 0.3)
class SimpleExpSmoothing: ...


# name "holt_winters"; __init__(self, period: int, alpha: float = 0.2, beta: float = 0.01,
#                               gamma: float = 0.1, phi: float = 0.98)
class DampedHoltWinters: ...
```

`experts/ridge.py`

```python
# name "ar_ridge"; __init__(self, period: int, window: int = 240, refit_every: int = 24,
#                           ridge: float = 1.0)
class RidgeAR: ...
```

`experts/pool.py`

```python
EXPERT_NAMES = ("naive", "seasonal_naive", "ses", "holt_winters", "ar_ridge")  # cache order


def make_pool(period: int) -> tuple[Expert, ...]: ...  # fresh, in EXPERT_NAMES order, defaults
def first_origin(period: int) -> int: ...  # max(min_history) - 1 over make_pool(period)
def run_pool(
    values: Sequence[float], period: int, horizon: int
) -> tuple[tuple[tuple[float, ...], ...], ...]: ...
```

### Exact behaviour: common

- `observe(y)`: `y` must be a finite `float` or `int` (not `bool`), else `ExpertError`. Experts store what they need and nothing else.
- `forecast(horizon)`: `horizon` is an int `>= 1`, else `ExpertError`. Before `min_history` values have been observed, raises `ExpertError`. Returns exactly `horizon` floats. If any is non-finite, raises `ExpertError` (never returned silently). Calling `forecast` does not change the expert's state: two calls in a row return equal tuples, and the next `observe` behaves as if `forecast` had not been called.
- Constructors validate: `period` int `>= 2`; `alpha`, `beta`, `gamma` in `(0, 1)`; `phi` in `(0, 1]`; `window` int `>= 2`; `refit_every` int `>= 1`; `ridge` `> 0` and finite. Violations raise `ExpertError`.
- `mean(v_a .. v_b)` below always means: left-to-right loop sum, then divide by the count.

### Exact behaviour: the five experts (`m` = period, `t` = index of the last observed value)

1. **`naive`**, `min_history = 1`: `y_t` for every `h`.
2. **`seasonal_naive`**, `min_history = m`: `ŷ_{t+h} = y_{t+h - m*k}` with `k = (h + m - 1) // m` (the most recent value in the same phase). Keeps only the last `m` values (`collections.deque(maxlen=m)`).
3. **`ses`**, `min_history = 1`: on the first value `L = y_0`; afterwards `L = alpha*y + (1 - alpha)*L`. Forecast `L` for every `h`.
4. **`holt_winters`** (additive season, damped trend), `min_history = 2m`. Buffer the first `2m` values; when the `2m`-th value (`y_{2m-1}`) arrives, initialise:
   - `a0 = mean(y_0 .. y_{m-1})`, `a1 = mean(y_m .. y_{2m-1})`;
   - `B = (a1 - a0) / m`;
   - `L = a1 + B * (m - 1) / 2` (the second block's mean sits at its centre, `(m - 1) / 2` steps before `2m - 1`), computed as `a1 + B * ((m - 1) / 2)`;
   - `S[j] = ((y_j - a0) + (y_{m+j} - a1)) / 2` for `j = 0 .. m-1` (`S` is indexed by time index mod `m`).
   For every later value `y_t` (`t >= 2m`): `s = S[t % m]`; `L_prev = L`; `L = alpha*(y_t - s) + (1 - alpha)*(L_prev + phi*B)`; `B = beta*(L - L_prev) + (1 - beta)*phi*B`; `S[t % m] = gamma*(y_t - L) + (1 - gamma)*s`.
   Forecast: `d = 0.0; p = 1.0`; for `h = 1..H`: `p = p*phi; d = d + p`; `ŷ_{t+h} = L + d*B + S[(t + h) % m]`.
5. **`ar_ridge`** (recursive one-step ridge autoregression, refit periodically). Lags `(1, 2, 3, m, 2m)`. The feature vector for target index `s` is `x_s = (1, y_{s-1}, y_{s-2}, y_{s-3}, y_{s-m}, y_{s-2m})`, here on normalised values. `min_history = window + 2m`, so the first valid origin is `t0 = window + 2m - 1`.
   - **Refit** at origin `t` when no model exists yet or `(t - t0) % refit_every == 0`. The schedule depends only on `t`, never on the series length.
   - Fit data: the `window + 2m` most recent values `y_{t-window-2m+1} .. y_t`. `mu = mean(those)`, `sd = sqrt(mean((v - mu)*(v - mu)))` (population). If `sd == 0.0`, use `sd = 1.0`. `z_i = (y_i - mu) / sd`.
   - Training pairs: targets `s = t-window+1 .. t` (ascending), features `x_s` from `z`. Minimise `sum_s (z_s - w·x_s)^2 + ridge * sum_{i>=1} w_i^2` (intercept not penalised). Build `A = XᵀX + ridge*diag(0,1,1,1,1,1)` and `b = Xᵀz` by looping over `s` in ascending order and adding each row's products into the running entries. Solve `A w = b` by Gaussian elimination with partial pivoting (pivot = largest absolute value in the column at or below the diagonal; ties go to the lowest row index), then back substitution. `A` is positive definite, so this always succeeds.
   - Store `(mu, sd, w)` until the next refit. Between refits, the model is unchanged but new observations are still used as lag inputs.
   - Forecast: normalise the last `2m` observed values with the stored `mu, sd`; for `h = 1..H` compute `ẑ_{t+h} = w·x_{t+h}`, using predicted `ẑ` for indices `> t`; `ŷ_{t+h} = mu + sd*ẑ_{t+h}`. Dot products are left-to-right in feature order.
   - Keep only the last `window + 2m` values (`deque`).
   - The forecast at origin `t` must not depend on which earlier origins `forecast` was called at: a `RidgeAR` that observed `y_0 .. y_t` and forecast at every origin from `t0` to `t` gives the same result at `t` as one that forecast only at `t`. The simplest way is to refit inside `observe` when the new `t` is on the schedule.

With the default `period = 24`: `min_history` is `1, 24, 1, 48, 288`, so `first_origin(24) = 287`. This fits before the synthetic grammar's earliest change point (`warmup = 300`) and inside the real-data warmup segments (ETTh1 `[0, 3383)`, ECL `[0, 5160)`).

### Exact behaviour: `run_pool`

`run_pool(values, period, horizon)` builds `make_pool(period)`, feeds `values` in order, and after observing `y_t` for every `t` from `first_origin(period)` to `len(values) - 1` (inclusive) collects `tuple(e.forecast(horizon) for e in pool)`. It returns one entry per origin, in origin order. Origins whose targets lie beyond the end of `values` are included: a forecast never needs its target. If `len(values) <= first_origin(period)`, raises `ExpertError`. Each value goes to every expert before any expert forecasts at that origin; no expert ever sees `y_{t+1}` before forecasting at `t`.

### Tests for PR A (all required; `tests/experts/test_simple.py`, `test_ridge.py`, `test_pool.py`)

1. `naive`, `seasonal_naive` (`m = 3`, `H = 7`, so `k` reaches 3) and `ses` on hand-written series of 6–10 values: exact forecasts, checked against values written out in the test.
2. `holt_winters` with `m = 2` on 7 hand-written values: the initial `L`, `B`, `S` after 4 values and the forecast after 4, 5 and 7 values equal values computed in the test with the formulas written out step by step.
3. `ar_ridge` with `period = 2`, `window = 6`, `refit_every = 3`: forecast at `t0` and `t0 + 1` equals a reference computed in the test by building `X`, `A`, `b` explicitly and solving with an independent 6×6 solver written in the test (Gauss-Jordan is fine); agreement within `1e-12` relative. Between refits (`t0 + 1`, `t0 + 2`) the coefficients are those of the `t0` fit; at `t0 + 3` they change.
4. `ar_ridge` refit schedule independence: for a 400-value series, the forecast at origin `t` is identical (`==`) whether `forecast` was called at every origin from `t0` or only at `t`, for every `t` in `t0 .. t0 + 60`.
5. `ar_ridge` on a constant series: `sd = 0` path, forecasts equal the constant exactly.
6. `ar_ridge` recovers a noiseless linear recurrence: with `period = 24` and `ridge = 1e-6`, on `y_s = sin(2*pi*s/24)` (600 values; it satisfies `y_s = 2*cos(2*pi/24)*y_{s-1} - y_{s-2}` exactly), one-step forecasts at the last 50 origins are within `1e-4` of the true next value.
7. Before `min_history`: each expert's `forecast` raises `ExpertError` after `min_history - 1` values and succeeds after `min_history`.
8. Input validation: `observe` rejects NaN, `inf`, `True` and `"1"`; `forecast` rejects `0`, `-1`, `True`; each constructor bound above has a failing case.
9. `forecast` is pure: two calls return equal tuples; a sequence observe/forecast/observe/forecast gives the same results as observe/observe/forecast, for every expert.
10. Non-finite output is refused: an expert driven to overflow (`holt_winters` fed `1e308`-scale values) raises `ExpertError` instead of returning `inf` or NaN.
11. **No look-ahead** (`run_pool`): for a 500-value series and every cut `c` in `{287, 288, 300, 400}`, replacing every value after index `c` with other values leaves the entries for origins `<= c` identical (`==`); and `run_pool(values[:c+1])` equals the first `c - 286` entries of `run_pool(values)` (prefix property, which PR 9 relies on).
12. `run_pool` shape and errors: entry count `len(values) - first_origin`, each entry has 5 tuples of `horizon` floats in `EXPERT_NAMES` order; `len(values) == first_origin(24)` raises; `first_origin(24) == 287`.
13. Golden: the SHA-256 of `canonical_bytes` (from `ledger/canonical.py`) of `run_pool(y, 24, 24)` as nested lists, where `y` is the `y` of `make_task("mixed", world_seed("dev", "mixed", 0))`, is pinned in the test file. The docstring says changing it requires bumping `EXPERTS_VERSION`.
14. No float `sum()`: a test parses every module in `src/evolvets/experts/` with `ast` and fails if the builtin `sum`, `math.fsum` or the `statistics` module is used, or if any `evolvets` import outside `evolvets.experts` appears.

### Dependencies

None.

### Size

About 300 lines of implementation. If it grows past ~400, move `ridge.py` and its tests into a follow-up PR (`feat/experts-ridge`) and keep `make_pool` in this PR with four experts until then.

---

## PR B: `feat/experts-cache`

Run the pool over every public series once and store the forecasts in a content-addressed cache that PR 8 reads.

### Files and interfaces

`cache/interfaces.py`

```python
@dataclass(frozen=True)
class SeriesInput:
    series_id: str
    values: tuple[float, ...]  # finite
    period: int  # >= 2


@dataclass(frozen=True)
class CachedSeries:
    series_id: str
    length: int
    period: int
    first_origin: int
    n_origins: int  # length - first_origin
    file: str  # "series/<index:04d>.f64"
    sha256: str  # of the file bytes
    values_sha256: str  # sha256_hex(canonical_bytes(list(values)))


@dataclass(frozen=True)
class CacheManifest:
    cache_version: str
    experts_version: str
    dataset: str
    source: dict[str, str]
    horizon: int
    experts: tuple[tuple[str, dict[str, float | int]], ...]  # (name, params) in pool order
    series: tuple[CachedSeries, ...]
```

`cache/store.py`

```python
DEFAULT_HORIZON = 24


def cache_dir(cache_root: Path, dataset: str) -> Path: ...  # <cache_root>/<dataset>
def build_cache(
    dataset: str,
    source: Mapping[str, str],
    series: Sequence[SeriesInput],
    cache_root: Path,
    horizon: int = DEFAULT_HORIZON,
) -> CacheManifest: ...
def read_manifest(cache_root: Path, dataset: str) -> CacheManifest: ...
def cache_id(cache_root: Path, dataset: str) -> str: ...  # SHA-256 of manifest.json bytes
def load_series(cache_root: Path, manifest: CacheManifest, series_id: str) -> SeriesForecasts: ...
def verify_cache(cache_root: Path, dataset: str) -> CacheManifest: ...


class SeriesForecasts:
    series_id: str
    first_origin: int
    last_origin: int  # length - 1

    def at(self, origin: int) -> tuple[tuple[float, ...], ...]: ...  # [expert][h-1]
    def expert_forecast(self, origin: int, expert: str) -> tuple[float, ...]: ...
```

`cache/datasets.py`

```python
CACHE_DATASETS = ("etth1", "ecl", "synthetic-dev")


def public_inputs(
    dataset: str, data_root: Path, manifest_dir: Path, per_family: int | None = None
) -> tuple[dict[str, str], tuple[SeriesInput, ...]]: ...  # (source, series)
```

CLI: `evolvets cache build DATASET` and `evolvets cache verify DATASET`, `DATASET` in `{etth1, ecl, synthetic-dev, all}`, options `--cache-root` (default `cache_store`), `--data-root` (default `data`), `--manifest-dir` (default `data/manifests`), `--horizon` (default 24, build only), `--per-family` (int `>= 1`; required when building `synthetic-dev` or `all`). Output per dataset in `CACHE_DATASETS` order: `OK <dataset>: <n> series, <total origins> origins, cache <cache_id>` or `FAIL <dataset>: <reason>`; exit 1 if any failed, else 0. Errors are caught per dataset, as in `evolvets data`.

### Exact behaviour: layout

- `<cache_root>/<dataset>/manifest.json` and `<cache_root>/<dataset>/series/<index:04d>.f64`, `index` = position in `series`. Dataset names match `^[a-z0-9][a-z0-9_-]*$`; series ids are unique and non-empty, else `CacheError`.
- A series file is the forecasts of `run_pool(values, period, horizon)` as IEEE-754 float64, **little-endian** (`array("d")`, `byteswap()` when `sys.byteorder == "big"`), row-major `[origin][expert][h]`: the value for origin `first_origin + i`, expert index `e`, step `h` is at element `(i * 5 + e) * horizon + (h - 1)`. File size is exactly `n_origins * 5 * horizon * 8` bytes.
- `manifest.json` is `canonical_bytes` of `{"cache_version", "experts_version", "dataset", "source", "horizon", "experts": [{"name", "params"}], "series": [{CachedSeries fields}]}`. `cache_id` is its SHA-256; it identifies the cache a run used (PR 8 records it in the run config).
- `source` is an opaque `str → str` map from the caller, copied into the manifest. `values_sha256` lets PR 8 check that a cache matches the series it replays.

### Exact behaviour: build

- Validates inputs (`horizon` int `>= 1`; each series long enough for at least one origin; values finite), then computes everything **into a fresh temporary directory** `<cache_root>/.<dataset>.tmp-<pid>-<random hex>` (the random suffix is from `os.urandom`, not a seeded RNG: it is a file name, not data). Series files are written one at a time and fsynced; the manifest last. Peak memory stays around one series' forecasts (ECL: about 14 MB per series as packed floats), not the whole dataset: write each series from an `array("d")`, not from nested tuples of the whole dataset.
- If `<cache_root>/<dataset>` does not exist: `os.rename(tmp, final)` and return. If the rename fails because the target appeared in the meantime, continue as below.
- If it exists: compare the new `manifest.json` bytes with the existing one. Equal: delete the temporary directory and return the existing manifest (idempotent, nothing rewritten). Different: delete the temporary directory, raise `CacheError` naming both cache ids, and leave the existing cache untouched. Never overwrite or delete an existing cache directory; rebuilding with new experts means the user removes the old cache by hand.
- On any exception during the build, the temporary directory is removed and the error re-raised; a hard kill leaves only a `.<dataset>.tmp-*` directory, which readers ignore.

### Exact behaviour: read and verify

- `read_manifest`: the file must parse as JSON with exactly the keys above (and exactly the `CachedSeries` keys per series), `cache_version == CACHE_VERSION`, `experts_version == EXPERTS_VERSION`, `experts` names equal to `EXPERT_NAMES` in order, `n_origins == length - first_origin`, `first_origin == first_origin(period)`; otherwise `CacheError`. Unknown versions are refused, not reinterpreted.
- `load_series`: unknown `series_id` raises `CacheError`. Reads the file, checks size and SHA-256 against the manifest (`CacheError` on mismatch or missing file), and returns a `SeriesForecasts`. `at(origin)` and `expert_forecast(origin, expert)` raise `CacheError` for an origin outside `[first_origin, last_origin]`, a `bool` origin, or an unknown expert name.
- `verify_cache`: `read_manifest` plus a size and hash check of every series file. No recomputation.

### Exact behaviour: datasets

- `etth1`, `ecl`: `load_public` (TASK_005), so only `values[:sealed.start]`; `period = 24`; series in manifest order. `source = {"kind": "realdata", "dataset": <name>, "realdata_version": <from split manifest>, "public_sha256": <from split manifest>}`. The sealed processed file and the raw files are never opened.
- `synthetic-dev`: for each family in `FAMILIES` order and `i in range(per_family)`, the task `dev_tasks(family, per_family)[i]`; `series_id = f"{family}/{i}"`; `values = task.world.y`; `period = task.world.spec.period`. Only `world.y` and `spec.period` are read: never `mean`, `noise`, `sigma`, `phi`, the oracle or change points. `source = {"kind": "synthetic", "pool": "dev", "generator_version": GENERATOR_VERSION, "per_family": str(per_family)}`. `per_family` missing or `< 1` raises `CacheError`.
- **No sealed caches in this task.** Sealed real data is served by PR 9's separate process, and sealed synthetic tasks need a salt PR 9 defines. The prefix property (PR A test 11) guarantees that a cache PR 9 builds over full series agrees with this one on every shared origin. Note in `cache/datasets.py`'s docstring that `cache_store/` is readable by any process, which is why only public data may go into it.

### Budget accounting

Building the cache is shared infrastructure, run once before any experiment and identical for every condition (AGENTS.md 2.5). It is not charged to any condition. Runs record which cache they used via `cache_id` (PR 8).

### Tests for PR B (all required; `tests/cache/test_store.py`, `test_datasets.py`, `test_cli.py`)

1. Round trip: `build_cache` on two synthetic-looking fixture series (lengths 300 and 350, `period = 24`, `horizon = 5`); `load_series(...).at(t)` equals `run_pool(...)` entry `t - first_origin` exactly for every origin; `expert_forecast` agrees with `at`.
2. File layout: the raw bytes of a series file, decoded by the test with `struct.unpack("<d", ...)` at the offset formula above, give the expected values for three hand-picked `(origin, expert, h)` triples, including the last origin; file size is exactly `n_origins * 5 * horizon * 8`.
3. Manifest: exact key sets; `cache_id` equals the SHA-256 of the file bytes; `values_sha256` equals `sha256_hex(canonical_bytes(list(values)))`; `source` copied verbatim.
4. Idempotent rebuild: building the same inputs twice leaves the manifest and every series file byte-identical (compare bytes and `st_mtime_ns`), and no temporary directory remains.
5. Conflicting rebuild: changing one value, the horizon or the source map raises `CacheError`; the existing directory's files are byte-identical afterwards; no temporary directory remains.
6. Crash safety: with `os.rename` monkeypatched to raise on the final rename, `build_cache` raises, no `<cache_root>/<dataset>` exists, and no temporary directory remains; a rerun succeeds. A leftover `.<dataset>.tmp-*` directory created by hand does not affect `read_manifest`, `verify_cache` or a new build.
7. Corruption: one byte flipped in a series file makes `load_series` and `verify_cache` raise; a truncated file likewise; a missing file likewise; a manifest with an extra key, a missing key, `cache_version` `"2"`, a reordered `experts` list or `n_origins` off by one each make `read_manifest` raise.
8. Input validation: duplicate series ids, an empty id, a dataset name with `/` or uppercase, `horizon` of `0` or `True`, a series of length `first_origin(24)`, and a NaN value each raise `CacheError` and create no cache directory.
9. Origin bounds: `at(first_origin - 1)`, `at(last_origin + 1)`, `at(True)` and `expert_forecast(t, "lightgbm")` raise `CacheError`.
10. No look-ahead through the cache: two series identical up to index `c = 320` and different afterwards give byte-identical file prefixes up to and including origin `c`.
11. Golden: the `cache_id` for the fixture in test 1 is pinned in the test file; the docstring says changing it requires bumping `CACHE_VERSION` or `EXPERTS_VERSION`.
12. Real-data inputs never touch sealed or raw files: on the TASK_005 fixture setup (reuse its helpers), `public_inputs("etth1", ...)` returns `values[:sealed.start]` per series and the expected `source`; it still works with the sealed processed file and the raw file deleted; recording every file open (as in TASK_005 test 10) shows neither is opened.
13. Synthetic inputs: `public_inputs("synthetic-dev", ..., per_family=2)` returns 12 series with ids `stationary/0 ... mixed/1` in `FAMILIES` order, values equal to `dev_tasks(f, 2)[i].world.y`; `per_family` of `None` and `0` raise. A guard test: run it with a `World` whose `mean`, `noise`, `sigma` and `phi` are replaced by objects that raise on access (monkeypatch `dev_tasks` to return such tasks) and it still succeeds.
14. CLI: on fixtures, `cache build etth1` then `cache verify etth1` exit 0 with one `OK` line; `cache build synthetic-dev --per-family 1` exits 0; `cache build synthetic-dev` without `--per-family` prints `FAIL` and exits 1; after flipping a byte in a series file, `cache verify etth1` exits 1 with a `FAIL` line; an unknown dataset is rejected by argparse (exit 2).

### Running it for real

Not required for merge, but the PR description must paste the output of `uv run evolvets cache build all --per-family 20` and `uv run evolvets cache verify all` with wall-clock times, and the total size of `cache_store/` (`du -sh`). Expected size with `H = 24` (960 bytes per origin): about 65 MB for ETTh1 (7 series, public length 9,639), 275 MB for ECL (20 series, public length 14,526) and 200 MB for `synthetic-dev` with `--per-family 20` (120 series of length 2,000). If a full build takes more than 30 minutes, say so in the PR; do not change the experts to make it faster without a spec change. Nothing under `cache_store/` is committed.

### Dependencies

None.

### Size

About 300 lines of implementation. If it grows past ~400, move `cache/datasets.py`, the CLI and their tests into a follow-up PR (`feat/experts-cache-cli`).

---

## Not in this task

- **Import contracts.** The `.importlinter` contract banning agent-editable packages (`updater`, `improver`, `selfmodel`, `allocator`) from importing `evolvets.arena.synthetic` (the oracle) and `evolvets.arena.realdata.prepare` stays deferred to the PR 8 / PR 10 specs, as TASK_005 recorded: no agent-editable code exists yet, and a new contract needs human review. Whether agent-editable code may import `evolvets.cache` directly, or only receive forecasts through the arena's replay loop, is decided in the PR 8 spec.
- **Sealed caches**, served from the sealed process (PR 9).
- Replay, label release and metrics (PR 8); updater baselines and the switching oracle (PR 10).
- Foundation-model experts (roadmap entry 12, needs human approval).
- Issue #10 (tag `ledger-schema-v1`) is a human action and is not blocked or changed by this task.

## Closing the task

PR B (`feat/experts-cache`, or its follow-up if split) changes this file's first line to `Status: done`.
