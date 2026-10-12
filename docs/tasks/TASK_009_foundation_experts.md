Status: active

# Task 009: Foundation-model experts and their forecast caches

Read `AGENTS.md` and `docs/ROADMAP.md` first. This task covers roadmap entry 12 (`feat/foundation-experts`). It is a stack of four PRs; open them in order and stop after each one for review.

## Why

Today the expert pool is five simple statistical forecasters. Adaptation (switching and weighting experts online) is worth most when the pool contains strong, different experts. Phase 3 also claims state of the art on GIFT-Eval, which is impossible without pretrained time-series foundation models in the pool. This task adds five of them as **cached experts**. Their forecasts are computed once, on the lab GPU, for every origin of every series. They are stored next to the simple-expert cache with the same origins, so the replay loop treats them exactly like the simple experts.

## Decisions already made (do not reopen)

- **Moirai 2.0 is dropped as well, after Toto (planner decision, 2026-10-12, issue #66).** Every `uni2ts` release (including the only line with Moirai 2.0 support) requires `torch>=2.1,<2.5`, which cannot coexist with the `torch==2.6.*` pin. Its license (`cc-by-nc-4.0`, non-commercial) was a second reason not to build around it. The pins stay as they are. The foundation pool is **three models**: Chronos-2, TiRex, TimesFM 2.5. The `moirai2` placeholder stays in the registry unpinned and is excluded from every built set and from the pool, exactly like `toto`. There is no C3 and no C4. If upstream relaxes the torch bound, Moirai 2.0 can return as a separate roadmap entry; the verified facts from issue #66 (pin commit `30f43ff08c8494f4943ae1521e9d4e94a0fbb389`, `max_seq_len` 512, native quantiles, and an overlap entry that must be re-derived because the card says "non-leaking") are kept there.
- **Toto is dropped from this task (planner decision, 2026-10-12, issue #64).** Neither Toto generation installs under the pins: every `toto-ts` release pins `torch==2.7.0`, and `toto-2` needs Python >= 3.12. The task's own fallback applies (leave it out), and the pins stay as they are. The foundation pool is **four models**: Chronos-2, TiRex, Moirai 2.0, TimesFM 2.5. The `toto` placeholder stays in the registry with `hf_revision = "unpinned"` and is **excluded from every built set and from the `simple+foundation` pool**: a model is in the pool only if its `hf_revision` is a pinned commit hash. There is no C3. Toto 2.0 (native quantiles, Apache-2.0) is a possible later addition if the project moves to Python 3.12; that is a human decision and a separate roadmap entry.

- **Approval (owner, 2026-10-12):** heavy dependencies and the GPU are approved for this task: PyTorch, numpy, and the five model libraries named below, each in its own optional dependency group. This approval covers only what this file names.
- **Models** (one adapter each, in this order): Chronos-2 (Amazon), TiRex (NX-AI), TimesFM 2.5 (Google). (Toto and Moirai 2.0 were planned and dropped; see the first two decisions.) Before PR C, verify on each official model card or repository the Hugging Face model id, the pip package name, the license, and which pretraining corpora are published. Record them in the registry (PR A). Never guess a value: write `"unknown"` and say so in the PR.
- **Torch is pinned to `torch==2.6.*`.** Its PyPI Linux wheel bundles CUDA 12.4, which matches the lab GPU driver (550). The GPU smoke test (#50) verified exactly this. If a model library cannot run on torch 2.6, leave that model out of PR C and open a `needs-human` issue. Do not raise the pin, vendor the library, or install it from a git URL.
- **Cached, not live.** Foundation models never run inside the replay loop, an updater, or any agent-editable package. They run only in the cache builder, as an offline GPU job. The cache is the reproducible artefact: the same reasoning as TASK_006, with "bit-exact recomputation" replaced by "built once, content-hashed, verified by hash" (below).
- **Univariate, point and quantiles.** Each series is forecast on its own past values only: no covariates, no cross-series context, no timestamps. Each forecast stores nine quantiles (0.1 … 0.9). The point forecast used by updaters is the 0.5 quantile.

## Rules for all four PRs

- **New package `src/evolvets/foundation/`** (instrument code, not agent-editable; strict mypy on everything except the files under `foundation/models/`, which wrap untyped libraries). The package skeleton and its import contracts already exist (added by the human in the PR that added this task): agent-editable packages and `evolvets.updater` may not import `evolvets.foundation`, and CI probes this. `foundation/` may import `evolvets.cache`, `evolvets.experts.pool` (for `first_origin`), `evolvets.arena.realdata` and `evolvets.arena.synthetic` (through `cache.datasets`), and `evolvets.ledger.canonical`. It never imports `evolvets.arena.sealed` except in PR D's sealed builder, which lives in `arena/sealed/`, not in `foundation/`.
- **Heavy imports only inside `foundation/models/`**, and only inside functions (lazy), so `import evolvets.foundation` and the whole test suite work with no heavy dependency installed. CI installs only the `dev` group. Tests that need a real model library use `pytest.importorskip("<lib>")` and carry the marker `@pytest.mark.foundation`. This does not weaken CI: those tests run in the GPU workflow, where the libraries are installed. Every other test must run in plain CI.
- **Dependency groups** in `pyproject.toml`, one per model, each listing that library and `torch==2.6.*`: `fm-chronos`, `fm-tirex`, `fm-toto`, `fm-moirai`, `fm-timesfm`. If two groups cannot resolve together, declare them in `[tool.uv] conflicts`. Never add any of these to `[project] dependencies` or to `dev`. `uv.lock` is updated and committed. A model's build runs as `uv run --group fm-<model> evolvets foundation build ...`, one model per process.
- **Origin convention** (same as TASK_004/006/007): origin `t` means `y_0 .. y_t` are observed; the forecast for step `h = 1..H` targets `y_{t+h}`. The context passed to a model at origin `t` is exactly `values[max(0, t + 1 - L) : t + 1]`, where `L` is that model's `max_context` from the registry. **No other value of the series ever reaches the model.** This is the look-ahead rule for this task, and tests enforce it.
- **Same origins as the simple cache.** For a series of period `p`, the foundation cache covers origins `first_origin(p) .. length - 1`, exactly like the simple cache, so the two can be joined row for row. `horizon` defaults to the simple cache's `DEFAULT_HORIZON` (24).
- `FOUNDATION_CACHE_VERSION = "1"` in `foundation/__init__.py`. Bump it on any change to the file layout, manifest fields or quantile levels. Golden tests enforce this. A change to a model id, revision or adapter does **not** bump it: those are recorded in each manifest, and a cache built from them gets a different cache id.
- **Budgets (AGENTS.md 2.5):** foundation forecasts are computed once and shared by every condition, so no condition is charged for them. The manifest records `build_seconds` and the GPU name, so the shared cost is reported.
- **No ledger change.** Nothing here writes ledger records or touches `ledger/` or `PROTOCOL_KEYS`.
- Errors: `FoundationError(Exception)` in `foundation/errors.py`. Bools are rejected wherever an int is expected.

---

## PR A: `feat/foundation-core`

The adapter contract, the model registry, the batching builder that turns a series into contexts and forecasts, and a fake adapter. Nothing is written to disk and no heavy dependency is added in this PR.

### Files and interfaces

`foundation/interfaces.py`

```python
QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class Adapter(Protocol):
    """A batch forecaster: many variable-length contexts in, quantile forecasts out."""

    name: str  # registry key, e.g. "chronos2"

    def forecast(
        self, contexts: Sequence[Sequence[float]], horizon: int
    ) -> list[list[list[float]]]: ...  # [context][quantile][h], len(QUANTILES) quantiles
```

`foundation/registry.py`

```python
@dataclass(frozen=True)
class ModelSpec:
    name: str  # "chronos2", "tirex", "toto", "moirai2", "timesfm25"
    hf_model_id: str  # verified from the model card, or "unknown" until PR C
    hf_revision: str  # commit hash pinned in PR C; "unpinned" until then
    pip_package: str
    dependency_group: str  # "fm-chronos", ...
    max_context: int  # L
    license: str  # SPDX id or the license's name, as on the model card
    pretraining_overlap: dict[str, str]  # dataset -> "yes" | "no" | "unknown", keys "etth1", "ecl"
    overlap_source: str  # URL of the page the overlap entries were read from


MODELS: dict[str, ModelSpec]  # in the order listed under "Models" above


def get_spec(name: str) -> ModelSpec: ...  # FoundationError for an unknown name
```

`foundation/build.py`

```python
def origin_contexts(
    values: Sequence[float], first: int, max_context: int
) -> Iterator[tuple[int, tuple[float, ...]]]:
    """Yield (t, values[max(0, t + 1 - max_context) : t + 1]) for t = first .. len(values) - 1."""


def forecast_series(
    adapter: Adapter,
    values: Sequence[float],
    first: int,
    max_context: int,
    horizon: int,
    batch_size: int,
) -> list[list[list[float]]]:
    """Forecasts for every origin, [origin][quantile][h], in origin order."""
```

`foundation/fake.py`: `FakeAdapter(name="fake")`. A deterministic adapter for tests: for each context it returns quantile `q` at step `h` as `last + (q - 0.5) * spread + h * slope`, where `last` is the context's last value, `spread` is the context's max minus min, and `slope` is `(last - first) / len(context)`. It records every context it receives, in order, in `self.seen`.

### Exact behaviour

- `forecast_series` sends contexts to the adapter in batches of `batch_size` (the last batch may be shorter), in origin order. It validates the adapter's output: the right number of contexts, `len(QUANTILES)` quantile rows, `horizon` steps, every value a finite float. Otherwise it raises `FoundationError` naming the adapter, the origin and the problem. Bad output is never patched or skipped.
- **Quantile crossing:** if a model returns quantiles that are not non-decreasing at some step, `forecast_series` sorts the quantile values at that step (monotone rearrangement) and counts how many steps it sorted. The count is returned alongside the forecasts (change the return type to a small dataclass `SeriesForecast(values, sorted_steps)`) so PR B can record it in the manifest.
- Input validation: `values` non-empty and finite, `0 <= first < len(values)`, `max_context >= 1`, `horizon >= 1`, `batch_size >= 1`. Otherwise `FoundationError`.

### Tests (all required; `tests/foundation/test_build.py`, `test_registry.py`)

1. `origin_contexts` yields exactly `values[max(0, t + 1 - L) : t + 1]` for every `t`, for `L` smaller than, equal to and larger than the series length.
2. **Look-ahead:** for a series and an origin `t`, changing every value after `t` leaves the forecasts for origins `<= t` unchanged (`FakeAdapter`), and no context in `adapter.seen` for origin `t` contains an index `> t`.
3. Batch-size invariance with `FakeAdapter`: `batch_size` 1, 7 and larger than the number of origins give identical outputs.
4. Output shape is `[n_origins][9][horizon]`, with `n_origins = len(values) - first`.
5. Each kind of malformed adapter output raises `FoundationError`: wrong context count, wrong quantile count, wrong horizon, NaN, inf, a non-float.
6. Crossed quantiles are sorted, and `sorted_steps` counts them exactly.
7. Each invalid argument raises `FoundationError`, including a bool where an int is expected.
8. The registry has the five model names in order. Every `dependency_group` is one of the five group names. `pretraining_overlap` keys are exactly `{"etth1", "ecl"}`, with values in `{"yes", "no", "unknown"}`. `get_spec` of an unknown name raises.
9. `import evolvets.foundation` and `import evolvets.foundation.registry` succeed with no `torch` installed. Check this by asserting `"torch" not in sys.modules` after importing them in a subprocess.

Dependencies: none. Size: ~350 lines including tests.

---

## PR B: `feat/foundation-cache`

The on-disk cache for one (dataset, model) pair: build, resume, verify, load, and the CLI. It is tested end to end with `FakeAdapter`; no real model is used yet.

### Layout

`<root>/<dataset>/<model>/manifest.json` plus `<root>/<dataset>/<model>/series/<index:04d>.f64`. Each series file holds little-endian float64 values, row-major `[origin][quantile][h]`. The default root is `foundation_cache/`, which is added to `.gitignore`. Like `cache_store/`, this root holds public data only. Sealed foundation caches are PR D's job and go under the sealed cache root.

### Manifest fields (all required)

`foundation_cache_version`, `dataset`, `source` (copied from `cache.datasets.public_inputs`), `horizon`, `quantiles`, `model` (every `ModelSpec` field), `library_versions` (`{package: version}` for the model library, torch and numpy, read with `importlib.metadata`; the fake adapter records `{}`), `device` (e.g. `"cuda:NVIDIA GeForce RTX 3080 Ti"` or `"cpu"`), `dtype`, `batch_size`, `build_seconds`, `simple_cache_id` (the `cache_id` of the simple cache for the same dataset, which must exist and match origin for origin), and `series`: a list of `{series_id, length, period, first_origin, n_origins, file, sha256, values_sha256, sorted_steps}`.

### Interfaces

`foundation/store.py`

```python
def build_foundation_cache(
    dataset: str,
    adapter: Adapter,
    spec: ModelSpec,
    root: Path,
    cache_root: Path,
    data_root: Path,
    manifest_dir: Path,
    batch_size: int,
    per_family: int | None = None,
) -> FoundationManifest: ...
def read_foundation_manifest(root: Path, dataset: str, model: str) -> FoundationManifest: ...
def foundation_cache_id(
    root: Path, dataset: str, model: str
) -> str: ...  # sha256 of manifest bytes
def verify_foundation_cache(root: Path, dataset: str, model: str) -> FoundationManifest: ...
def load_foundation_series(
    root: Path, manifest: FoundationManifest, series_id: str
) -> FoundationSeries: ...
def compare_rebuild(
    root: Path, dataset: str, model: str, adapter: Adapter, sample: int, seed: int
) -> RebuildReport: ...
```

`FoundationManifest` and `FoundationSeries` are frozen dataclasses in `foundation/interfaces.py`. `FoundationSeries.point(origin_index)` returns the 0.5-quantile row.

### Exact behaviour

- **Inputs** come from `cache.datasets.public_inputs` (the same public values as the simple cache), never from raw or sealed files. Every series' `values_sha256` and origin range must equal the simple cache's entry for the same `series_id`. Otherwise raise `FoundationError` before calling the adapter.
- **Never overwrite** (same as TASK_006): build into `<root>/<dataset>/.<model>.partial-<params_hash>/`, write the manifest last, then rename it into place. If the final directory already exists, the build refuses with `FoundationError` naming the existing cache id. Foundation outputs are not bit-reproducible on a GPU, so "rebuild and compare equal" does not apply.
- **Resume:** `params_hash` hashes the dataset, the model spec, the horizon, the batch size and `source`. A series file already present and complete (right size) in the matching partial directory is kept, and the build continues from the next series. A truncated file (from a crash mid-write) is rewritten. Each file is written to `<name>.tmp`, fsynced and renamed.
- **Verify** checks every file's size and SHA-256 against the manifest, without recomputing anything. A run's reproducibility unit is the cache id plus the cache files. The owner backs them up; this task does not upload them anywhere.
- `compare_rebuild` recomputes a seeded random sample of `sample` origins (spread over series) and reports the maximum absolute and relative difference of the 0.5 quantile against the stored values. It is a diagnostic: it never edits the cache.
- **CLI** (`runner/cli.py`): `evolvets foundation build --dataset D --model M [--batch-size N] [--device auto|cpu|cuda] [--per-family N] [--root foundation_cache]`, `evolvets foundation verify --dataset D --model M`, and `evolvets foundation compare --dataset D --model M --sample N --seed S`. `--model fake` selects `FakeAdapter`, for tests and CI only. `--device auto` uses CUDA when available. In PR B, any model other than `fake` exits with a clear error saying it arrives in PR C.

### Tests (all required; `tests/foundation/test_store.py`, `test_cli.py`)

1. Build with `FakeAdapter` on a small synthetic dataset, then verify, then load. The loaded values equal `forecast_series` output, and `point()` equals quantile index 4.
2. Origins match the simple cache row for row. A series whose values differ from the simple cache's is refused before any adapter call.
3. A build refuses to replace an existing cache, and the existing files are byte-identical afterwards.
4. Resume: simulate a crash after two series (an adapter that raises on the third). Re-running reuses the first two files without calling the adapter for them (count calls) and gives the same manifest as an uninterrupted build, apart from `build_seconds`.
5. A truncated series file in the partial directory is rewritten.
6. Verify catches a flipped byte, a truncated file and a missing file. Each raises `FoundationError` naming the file.
7. The manifest has every required field. An unknown `foundation_cache_version` is refused on read.
8. A golden test pins `FOUNDATION_CACHE_VERSION` and the manifest layout: the manifest hash of a fixed tiny fake build, with `build_seconds` and `library_versions` fixed by the test.
9. `compare_rebuild` with `FakeAdapter` reports zero difference. It reports a nonzero difference after one stored value is changed in a copy of the cache.
10. The CLI runs `build`, `verify` and `compare` end to end with `--model fake` and exits nonzero with a clear message for a real model name.

Dependencies: none. Size: ~400 lines. If it grows, split the CLI and `compare_rebuild` into `feat/foundation-cache-cli`.

---

## PR C: three real adapters, one model per PR (C1, C2, C5; there is no C3 or C4)

**Why split (decided after two developer runs hit the 60-minute limit on the unsplit version):** each run must finish in well under an hour. C1 builds the shared machinery and the first model. C2 to C5 add one model each and follow C1's pattern exactly. Open them in this order and stop after each one for review: `feat/foundation-models-chronos2` (C1), `feat/foundation-models-tirex` (C2), `feat/foundation-models-toto` (C3), `feat/foundation-models-moirai2` (C4), `feat/foundation-models-timesfm25` (C5).

**Time box and fallbacks (all of C1 to C5):**

- Spend at most ~15 minutes of a run looking up a model's facts. Use the model's official Hugging Face card and repository. For `hf_revision`, read the current commit hash of the model repo (`https://huggingface.co/api/models/<id>` field `sha`) and pin that hash.
- A field you cannot verify gets the literal `"unknown"` (for `license`, `pretraining_overlap` values and `overlap_source`). Say so in the PR description. Never guess, and never block the PR on it: the reviewer decides.
- If the model's library cannot be installed with `torch==2.6.*`, or its `smoke` check fails on CPU after one honest attempt at fixing it, do not keep trying. Leave the model out of the registry's built set, open a `needs-human` issue naming the exact error, and stop.
- Do not spend a run reading other models' libraries. Only the model of the current PR matters.

### Shared design (introduced in C1, reused by C2 to C5)

`foundation/models/<name>.py`, one per model. Each defines `make_adapter(spec: ModelSpec, device: str) -> Adapter`. Heavy imports happen inside `make_adapter`. Each adapter:

- loads the model at the pinned `hf_revision`, in eval mode with no gradients, on `device`, using the dtype the model card recommends (recorded in the manifest);
- left-pads or otherwise handles variable-length contexts as that library documents. Contexts in one batch must not affect each other (no cross-series attention or context);
- returns the nine quantiles in `QUANTILES` order. If the library returns other quantile levels, it interpolates linearly between the nearest levels it has; if it returns samples, it takes empirical quantiles. Both are recorded in the adapter's docstring and in the manifest (`model.quantile_method`).

The registry entry for the model gets the verified `hf_model_id`, a pinned `hf_revision` (a commit hash, not a branch), `license`, `pretraining_overlap` with `overlap_source`, and `max_context` (the largest context the model card supports, capped at 4096).

`foundation/models/__init__.py` holds `load_adapter(name: str, device: str) -> Adapter`. It looks the name up in the registry, imports `foundation.models.<name>` lazily, and raises `FoundationError` naming the dependency group when the library is missing.

### C1: `feat/foundation-models-chronos2` (Chronos-2, plus everything shared)

Adds: `load_adapter`; the `fm-chronos` dependency group (and `uv.lock`); the `chronos2` adapter and its registry entry; `evolvets foundation build --model chronos2` resolving through `load_adapter` (replacing PR B's "arrives in PR C" error for models that exist); `evolvets foundation smoke --model <name> [--device auto]`; and the `foundation` pytest marker registered in `pyproject.toml`.

- Model weights come from Hugging Face. The endpoint is whatever `HF_ENDPOINT` is set to in the environment (on the lab runner it points at a mirror). Never hard-code an endpoint or token.
- `smoke`: forecasts a fixed sine plus trend of length 512 at 20 origins with batch sizes 1 and 16. It checks shapes and finiteness, that batch size changes the 0.5 quantile by at most 1e-3 relative, and that the forecast's MAE beats the naive forecast's MAE on that series. It prints a one-line JSON summary.
- Registry entries for the other four models are added by their own PRs, not here. The registry's five model names stay in the order fixed by PR A (a model not yet added still has its PR A placeholder entry, with `hf_revision = "unpinned"`).

Tests for C1 (all required; `tests/foundation/test_models.py`):

1. For `chronos2`: `hf_revision` is a 40-character hex commit hash, and `license` and `overlap_source` are non-empty (the literal `"unknown"` is allowed for the last two only, if it is stated in the PR).
2. With `pytest.importorskip("chronos")` and `@pytest.mark.foundation`: `smoke` passes on CPU with a context of 64. It runs in the GPU workflow, not in CI.
3. Without the library installed, `build --model chronos2` exits nonzero and names the missing group `fm-chronos`. This runs in CI.
4. `import evolvets.foundation.models.chronos2` does not import torch (subprocess check, as in PR A test 9).
5. `load_adapter` raises `FoundationError` for an unknown name, and for a known name whose library is missing.

Size: ~300 lines. Dependencies: the `fm-chronos` group only.

### C2 and C5: one model each

C2 (`tirex`, group `fm-tirex`) is merged. C5 (`feat/foundation-models-timesfm25`: `timesfm25`, group `fm-timesfm`; its library needs `torch>=2.0` and Python >= 3.10, which fit our pins) adds: the dependency group (and `uv.lock`), `foundation/models/timesfm25.py`, the filled-in registry entry, and the same four tests as C1 tests 1 to 4 for that model. **C5 also carries two closing deliverables**: the registry comment saying that `toto` and `moirai2` are intentionally unbuilt (see the first two decisions), and a test that the built set (models with a pinned `hf_revision`) is exactly the models whose adapters exist (`chronos2`, `tirex`, `timesfm25`). If TimesFM's library needs a different torch pin or Python version, stop and open a `needs-human` issue instead of changing either. Size: ~150 lines.

**After C1 merges and Chronos-2 passes the GPU smoke test**, the human adds the GPU workflow `gpu-foundation-cache.yml` (see below) so a first cache can be built while C2 to C5 are written. C5 changes this file's first-line status only if PR D is already done; otherwise the last PR of the whole task does.

**GPU workflow** (human-applied): `gpu-foundation-cache.yml` runs on `[self-hosted, gpu]`, takes inputs `model`, `dataset` and `gpus`, and runs `gpu-pick.sh`, `smoke`, `build`, `verify` and `gpu-release.sh`. The cache is written to the runner's persistent data volume. The developer then opens `chore/foundation-manifests`, which commits each built manifest (hashes only, no forecasts) under `data/manifests/foundation/<dataset>/<model>.json`.

---

## PR D: `feat/foundation-pool`

Let the arena replay a pool of the five simple experts plus the cached foundation experts, on dev and on sealed data. Start this PR only after TASK_008 is done.

### Behaviour

- A **pool spec** names the experts a replay uses: `"simple"` (the five, as now) or `"simple+foundation"` (the five, then the pinned foundation models in registry order). It is part of the run config and is recorded in the run's `run_start` config, alongside the simple and foundation cache ids.
- `arena/dev.py` loads the foundation caches for the dataset, checks that each manifest's `simple_cache_id` equals the simple cache in use, and joins the forecasts row for row. Updaters receive the point forecast (0.5 quantile). `Context.expert_names` lists every expert in the pool.
- **Sealed:** add `evolvets sealed build-foundation-cache --model M`. It builds a sealed pool's foundation cache under the sealed cache root, from inside the arena's sealed code path, with the same adapter and builder. The sealed server loads it the same way as dev. The sealed CLI refuses any root under `cache_store/` or `foundation_cache/`.
- Bump `ARENA_VERSION` to `"2"`. Replays with pool spec `"simple"` must give results identical to version 1 (a golden test pins this). A run records the pool spec, so Gate 1 can report headroom with both pools.
- **Leakage flag:** any result on a real dataset from a pool containing a model whose `pretraining_overlap` for that dataset is `"yes"` or `"unknown"` carries `leakage_risk: true` in its metrics. The Gate 1 report (entry 11) shows these in separate rows. Synthetic results never carry the flag.

### Tests (all required)

1. With pool spec `"simple"`, the replay's output is identical to version 1's on the existing golden fixtures.
2. `"simple+foundation"` with fake-built foundation caches: `expert_names` lists 5 + N names, and the forecasts reaching the updater equal the cached 0.5 quantiles.
3. A foundation cache whose `simple_cache_id` does not match is refused.
4. The look-ahead test from TASK_007 passes unchanged with the larger pool.
5. The sealed foundation build refuses public roots and writes only under the sealed cache root. The sealed server replays a fake-built sealed foundation cache end to end.
6. `leakage_risk` is set exactly when this section says.

Dependencies: none new. Size: ~400 lines. Split the sealed part into `feat/foundation-pool-sealed` if needed.

PR D (or the last PR of its split) changes this file's first line to `Status: done` and sets roadmap entry 12 to its merged PR numbers.
