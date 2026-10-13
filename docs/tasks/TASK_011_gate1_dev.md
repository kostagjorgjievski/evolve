Status: done (PR A merged as #85; PR B split into grids #86 and tune #87; PR C split into data #91 and report #92; side PRs #89 and #90)

# Task 011: Gate 1 on dev: recorded runs, tuning grids and the report generator

Read `AGENTS.md` and `docs/ROADMAP.md` first. This task covers the **dev half** of roadmap entry 11 (`exp/gate1`). It is a stack of three PRs. Open them in order and stop after each one for review.

## Why

Gate 1 asks whether there is headroom over the best tuned hand-designed updater. Answering it needs: every tuning replay recorded in the ledger (AGENTS.md 2.8), a `run_start` that pins the pool and its caches (review note 2 of issue #75), tuned configurations chosen by a fixed rule on dev, and a report generated only from ledger records (review note 1 of #75: `leakage_risk` rows split out). Entries 7 to 10 built the instrument but never write a run; nothing in the repository yet writes `run_start` or `run_end`.

- PR A: the run lifecycle. A `Run` writes `run_start` (config blob, protocol versions, clean git commit), appends records while tracking spent budget, and always ends with a `run_end`, including on errors.
- PR B: the tuning grids and `evolvets gate1 tune`, which replays every grid configuration of every baseline, plus the hindsight oracles, over dev pools inside one recorded run and selects the tuned configurations.
- PR C: `evolvets gate1 report`, a deterministic Markdown report regenerated from named ledger runs, with leakage-risk rows in their own section and per-series wall-clock for the `SERIES_TIMEOUT_SECONDS` review.

The sealed half of entry 11 is **not** in this task (see "Not in this task"): it needs the owner's decision on where the sealed data and salt live when Gate 1 runs (TASK_008, "Not in this task"). Gate 1's decision itself is a human gate.

## Decisions already made (do not reopen)

- **TASK_007/010:** a fresh updater per series; `replay_dataset` is the only way to score an updater; the oracles are arena computations (`oracle_dataset`) and bound single-expert selection only. They are **not** a floor for weighted combinations, and no report may present them as one.
- **TASK_008:** sealed evaluations go through `arena/sealed/client.py` only; the sealed server refuses a run whose first main record is not a `run_start` with the current `protocol_versions()`, or whose main stream already has a `run_end`. `SERIES_TIMEOUT_SECONDS = 600.0`, recorded in every sealed blob; entry 11 may revisit it before any sealed result exists.
- **TASK_009 PR D / issue #75:** `leakage_risk` is result-level (on `ReplayResult`, `OracleResult` and their blobs), never inside the float-only `metrics` dicts. Entry 11's `run_start` must record the pool spec, the simple cache id and the foundation cache ids.
- **TASK_010 PR C:** the sealed oracle is a `baseline` record with `updater_id = "oracle"`, `hyperparams = {}`, blob `{"result": oracle_to_obj(...)}` and `Budget(wall_seconds=..., evaluations=0)`: the oracle is part of the instrument and charges no evaluation. Dev oracle records in this task use the same shape.

## Rules for all three PRs

- **No ledger change.** No change to `ledger/`, `PROTOCOL_KEYS`, `SCHEMA_VERSION` or any payload. The run config lives in `run_start.config_blob`.
- **No arena change.** No change to `arena/` or `updater/`, and no `ARENA_VERSION` bump. This task only calls existing arena functions.
- **Standard library only.** No new dependencies.
- **Sealed blindness.** The new modules import from `evolvets.arena.sealed` at most `client` (for `SERIES_TIMEOUT_SECONDS` only, PR C). Nothing in this task opens a sealed file, the salt or the sealed cache root.
- **Strict mypy:** add every new `evolvets.runner.*` module to the strict override list in `pyproject.toml`. Do not weaken any override.
- **Determinism.** Selection, report numbers and bootstrap intervals are bit-exact: accumulate floats with explicit left-to-right loops (no builtin `sum()` on floats, no `math.fsum`, no `statistics`), as in TASK_010. The only randomness is the report's bootstrap, seeded from its CLI (AGENTS.md 2.6).
- **Agents never run `gate1 tune` on real dev data.** A run on an ephemeral CI machine would vanish with its ledger, against AGENTS.md 2.8. Tests use fixtures in `tmp_path`. The real dev run is done by the owner (see "After PR B merges").
- Do not edit `.importlinter`, `.github/`, `AGENTS.md` or `docs/agents/`.
- Tests live in `tests/runner/` (new; add `tests/runner/__init__.py`).

---

## PR A: `feat/runner-runs`

The run lifecycle every later experiment uses. No tuning, no CLI.

### Files and interfaces

`src/evolvets/runner/runs.py`

```python
class RunError(Exception): ...


def current_git_commit(
    repo: Path,
) -> str: ...  # 40 lowercase hex; raises RunError if dirty or not a repo


def new_run_id(prefix: str) -> str: ...  # f"{prefix}-{UTC %Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"


class Run:
    @classmethod
    def start(
        cls,
        ledger_root: Path,
        run_id: str,
        *,
        condition: str,
        seed: int,
        config: Mapping[str, object],
        budget_limits: Budget,
        git_commit: str,
        requires_predictions: bool = False,
    ) -> "Run": ...

    @property
    def run_id(self) -> str: ...
    @property
    def run_start_id(self) -> str: ...
    @property
    def limits(self) -> Budget: ...
    @property
    def spent(self) -> Budget: ...  # sum of the budgets of every record this Run appended

    def put_blob(
        self, obj: object
    ) -> str: ...  # BlobStore(<ledger_root>/<run_id>/blobs).put(canonical_bytes(obj))
    def append(
        self,
        kind: str,
        payload: Payload,
        *,
        refers_to: Sequence[str] = (),
        budget: Budget | None = None,
    ) -> Record: ...
    def end(
        self, status: str, summary: Mapping[str, object]
    ) -> Record: ...  # writes run_end, closes the writer

    def __enter__(self) -> "Run": ...
    def __exit__(self, exc_type, exc, tb) -> None: ...
```

### Exact behaviour

- `current_git_commit(repo)`: runs `git -C <repo> rev-parse HEAD` and `git -C <repo> status --porcelain --untracked-files=no` with `subprocess.run(..., check=False, capture_output=True, text=True)`. A non-zero exit of either (not a repository, no commit, no `git`) raises `RunError`. Any output from `status` (a modified or staged tracked file) raises `RunError("working tree has uncommitted changes; commit before starting a run")`. Untracked files are allowed (ledgers, caches and data are git-ignored or untracked). Returns the stripped `rev-parse` output after checking it is 40 lowercase hex. A run never records a dirty commit: every number must be reproducible from the ledger plus a commit.
- `new_run_id(prefix)`: `prefix` must match `^[a-z0-9][a-z0-9-]*$`, else `RunError`. Two calls never return the same id (the uuid part).
- `Run.start`, in this order. Nothing is written before step 3:
  1. validate: `run_id` matches `^[A-Za-z0-9][A-Za-z0-9._-]*$` (no path separators); `canonical_bytes(config)` succeeds (its errors, e.g. NaN or a non-JSON value, become `RunError`); `config` is a dict with str keys.
  2. refuse with `RunError` if `<ledger_root>/<run_id>/main.jsonl` or `<ledger_root>/<run_id>/sealed.jsonl` exists. A run never appends to another run's ledger.
  3. open `LedgerWriter(ledger_root, run_id, "main")` (it holds the main lock for the run's lifetime), put the config blob, and append `run_start` with `RunStart(condition, seed, requires_predictions, config_blob, git_commit, protocol_versions(), budget_limits)` and an empty budget. `protocol_versions` comes from `evolvets.arena.versions`.
  4. If step 3 raises after the writer opened, close the writer and re-raise. A config blob with no `run_start` is harmless (blobs are content-addressed and unreferenced).
- `append` delegates to the writer and adds the record's budget to `spent`, field by field, in append order. It refuses `run_start` and `run_end` kinds (`RunError`; use `start`/`end`), and anything after `end` (`RunError`).
- `end(status, summary)` appends `RunEnd(status, dict(summary))` with an empty budget, then closes the writer. A second `end` raises `RunError`.
- Context manager. `__exit__` with an exception, including `KeyboardInterrupt` (any `BaseException`), and no `end` yet: append `run_end` with status `"aborted"` and summary `{"error": f"{type(exc).__name__}: {exc}"}`, close, and let the exception propagate. Normal exit without `end`: append `run_end` `"aborted"` with `{"error": "run exited without run_end"}`. After `end`: do nothing. If writing the aborting `run_end` itself fails, close the writer and propagate the original exception. Never swallow it.
- A process killed with `SIGKILL` leaves a run with no `run_end` and a stale lock. That is an honest record; `ledger unlock` exists for it. PR C reports such runs as incomplete.

### Tests for PR A (all required; `tests/runner/test_runs.py`)

1. **run_start:** after `Run.start` in `tmp_path`, main has exactly one record at seq 0, kind `run_start`. Its `config_blob` resolves to bytes equal to `canonical_bytes(config)`; `protocol_versions == protocol_versions()`; `condition`, `seed`, `git_commit`, `budget_limits` are as given; `requires_predictions is False`.
2. **Verify:** a started, appended-to (one dev `baseline` record) and ended run passes `evolvets ledger verify` (exit 0), and `validate_run` returns `[]`.
3. **No reuse:** `Run.start` on a run id whose `main.jsonl` exists raises `RunError`, and every file under the run directory is byte-identical afterwards. Same for an existing `sealed.jsonl` with no `main.jsonl`.
4. **Bad config:** a config containing `float("nan")`, and one containing a `set`, each raise `RunError`, and no run directory is created. A `run_id` containing `/` or `..` raises `RunError` the same way.
5. **Abort paths:** an exception raised inside `with Run.start(...)` gives a final `run_end` with status `aborted` whose `summary["error"]` starts with the exception type name, and the exception propagates. Same for `KeyboardInterrupt`. Normal exit without `end` gives `aborted` with `"run exited without run_end"`. After `end("completed", ...)`, leaving the block adds nothing. A second `end` raises `RunError`. `append` after `end` raises `RunError`.
6. **Spent:** after appending records with budgets `(wall 1.5, evaluations 1)` and `(wall 2.0, evaluations 0)`, `spent == Budget(wall_seconds=3.5, evaluations=1)`, equal to the field-wise sum over the records read back.
7. **Kinds:** `append("run_start", ...)` and `append("run_end", ...)` raise `RunError` and write nothing.
8. **Lock:** while a `Run` is open, `LedgerWriter(root, run_id, "main")` raises `LedgerLockedError`; after `end` it opens.
9. **Git:** in a `tmp_path` repository (`git init`, configure a user, commit one file), `current_git_commit` returns the 40-hex `HEAD`. With the tracked file modified, or staged, it raises `RunError`. With only an untracked file added, it succeeds. In a plain directory, it raises `RunError`.
10. **Run ids:** `new_run_id("gate1-tune")` matches `^gate1-tune-\d{8}T\d{6}Z-[0-9a-f]{8}$`; two calls differ; `new_run_id("Bad/Prefix")` raises `RunError`.

### Dependencies

None.

### Size

About 150 lines of implementation. No split expected.

---

## PR B: `feat/gate1-tune`

Tuning grids, the recorded tuning run over dev pools, and the selection rule.

### Files and interfaces

`src/evolvets/runner/gate1_tune.py`

```python
ETA_GRID: tuple[float, ...] = (
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
)  # 2**-6 .. 2**4
ALPHA_GRID: tuple[float, ...] = (0.0001, 0.001, 0.003, 0.01, 0.03, 0.1)
GATE1_GRIDS: Mapping[str, tuple[Mapping[str, ConfigValue], ...]]
# equal: ({},); ftl: ({},); adahedge: ({},)
# hedge: ({"eta": e} for e in ETA_GRID)                                  -> 11
# fixed_share: ({"eta": e, "alpha": a} for e in ETA_GRID for a in ALPHA_GRID)  -> 66 (eta outer)
SELECTION_METRIC = "mase"
TUNE_CONDITION = "gate1-dev-tune"


@dataclass(frozen=True)
class TuneSpec:
    datasets: tuple[str, ...]  # each in CACHE_DATASETS, no duplicates, in run order
    pool_specs: tuple[str, ...]  # each in POOL_SPECS, no duplicates, in run order
    foundation_models: tuple[str, ...] | None  # None: the registry's built models
    seed: int
    max_wall_seconds: float  # > 0, finite


@dataclass(frozen=True)
class DevPaths:
    cache_root: Path
    foundation_root: Path
    data_root: Path
    manifest_dir: Path


def planned_replays(
    spec: TuneSpec, grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]]
) -> int: ...


def tune(
    spec: TuneSpec,
    paths: DevPaths,
    *,
    ledger_root: Path,
    run_id: str,
    git_commit: str,
    grids: Mapping[str, Sequence[Mapping[str, ConfigValue]]] = GATE1_GRIDS,
) -> str: ...  # returns the run_end status


Selection = dict[str, dict[str, dict[str, dict[str, object] | None]]]
# dataset -> pool_spec -> baseline id -> {"config": {...}, "mase": float, "record_id": str} | None


def select_configs(records: Sequence[Record], read_blob: Callable[[str], object]) -> Selection: ...
```

`runner/cli.py` gains `evolvets gate1 tune`.

### Exact behaviour: the grids

- `GATE1_GRIDS` keys are exactly `BASELINE_IDS`, in that order. Every entry is accepted by its baseline's `init` (no grid point is a known failure). Each `DEFAULT_CONFIGS` entry appears in its baseline's grid. No grid has duplicates. The grids are fixed in code before any real dev run, so the commit in `run_start` pins them.
- Budget accounting for tuning: each replay charges `Budget(wall_seconds=<elapsed>, evaluations=1)`, where `<elapsed>` is `time.monotonic()` taken before and after `replay_dataset`. Each oracle charges `Budget(wall_seconds=<elapsed>, evaluations=0)`. The report (PR C) shows each baseline's tuning cost (configurations tried, evaluations, wall seconds), so a parameter-free baseline's single evaluation and Fixed-Share's 66 are both visible.

### Exact behaviour: `tune`

1. **Validate** `spec` (`ValueError` on any problem, before anything is loaded): non-empty `datasets` and `pool_specs` with no duplicates and valid names; `seed` an int in `[0, 2**63)`; `max_wall_seconds` finite and `> 0`. The grids are non-empty mappings from baseline ids in `BASELINE_IDS` to non-empty sequences.
2. **Load every pool first:** for each dataset in order, and each pool spec in order, `load_dev_pool(dataset, cache_root, data_root, manifest_dir, pool_spec=..., foundation_root=(foundation_root if pool_spec == "simple+foundation" else None), foundation_models=spec.foundation_models)`. An `ArenaError` propagates and **no run is started**: a run that could not load its pools never began, and nothing is written under `ledger_root`.
3. **Config** (the `run_start` config blob), exactly these keys:
   - `"experiment": "gate1"`, `"stage": "dev-tune"`;
   - `"datasets"`, `"pool_specs"`: lists, in run order;
   - `"simple_cache_ids"`: `{dataset: pool.cache_id}` (from the `simple` pool, or the first pool spec loaded, since every pool spec of a dataset shares the simple cache; check that they agree, else `ArenaError` before start);
   - `"foundation_cache_ids"`: `{dataset: {pool_spec: [cache ids in pool order]}}` (`[]` for `simple`);
   - `"foundation_models"`: `{dataset: {pool_spec: [model names in pool order]}}`;
   - `"grids"`: `{baseline_id: [config dicts in grid order]}`;
   - `"selection_metric": "mase"`, `"seed": spec.seed`, `"max_wall_seconds"`;
   - `"updater_refs"`: `{baseline_id: "evolvets.updater.baselines:<id>"}`.
4. **Start** `Run.start(ledger_root, run_id, condition=TUNE_CONDITION, seed=spec.seed, config=..., budget_limits=Budget(wall_seconds=spec.max_wall_seconds, evaluations=planned_replays(spec, grids)), git_commit=git_commit)`. `planned_replays` is `len(datasets) * len(pool_specs) * <total grid points>`.
5. **Loop**, inside the `Run` context: for each dataset, for each pool spec, in order:
   - the **oracle**: `oracle_dataset(pool)`; blob `{"result": oracle_to_obj(result)}`; a dev `baseline` record with `updater_id = "oracle"`, `hyperparams = {}`, `pool = "dev"`, `status = "ok"`, `metrics = result.metrics`, `error = None`, budget as above;
   - then for each baseline id in `BASELINE_IDS` order, for each config in grid order: replay `replay_dataset(factory, config, spec.seed, pool)`, where `factory` wraps `BASELINES[id]` and records `time.monotonic()` at each call. Blob: `{"updater_ref": ..., "series_wall_seconds": [...], "result": result_to_obj(result)}`. `series_wall_seconds[i]` is the time from factory call `i` to call `i + 1`, and the last entry runs to the end of `replay_dataset`; it has one entry per factory call, including the failing series of a failed result. Record: dev `baseline`, `updater_id = id`, `hyperparams = dict(config)`, `pool = "dev"`, `status = result.status`, `metrics = result.metrics`, `error = result.error`, budget as above. An updater failure is a result: record it and continue.
   - **Budget check** before every oracle and every replay: if `run.spent.wall_seconds >= spec.max_wall_seconds`, stop the loop. The record whose charge crosses the limit is recorded as measured (never clipped), so `ledger verify` reports the overrun truthfully.
6. **End:** `run.end(status, summary)` with status `"completed"` if every planned oracle and replay ran, else `"budget_exhausted"`. Summary keys: `"replays"`, `"oracles"`, `"ok"`, `"failed"`, `"invalid_weights"` (ints), and `"selected": select_configs(<this run's records>)`.
7. Any other exception inside the loop (an `ArenaError` from a tampered cache, `KeyboardInterrupt`) ends the run `aborted` through the `Run` context and propagates.

### Exact behaviour: `select_configs`

- Input: one run's records (both streams; only `main`-stream dev `baseline` records with `updater_id` in `BASELINE_IDS` are considered) and a blob reader returning the parsed JSON of a blob ref. Dataset and pool spec come from each record's blob `result` (`dataset`, `pool_spec`); a failed result's blob carries them too.
- For each `(dataset, pool_spec, baseline id)` seen: among records with `status == "ok"`, the lowest `metrics["mase"]` wins; on an exact tie the **earlier record** (lower `seq`, so earlier grid point) wins. If none is `ok`, the entry is `None`. Comparisons are exact `<`.
- Output is ordered by first appearance (dicts preserve insertion order), so canonical JSON of it is deterministic.

### Exact behaviour: the CLI

`evolvets gate1 tune --dataset {etth1,ecl,synthetic-dev,all} [--dataset ...] --pool-spec {simple,simple+foundation} [--pool-spec ...] [--foundation-model NAME ...] [--seed 0] [--max-wall-seconds 172800] [--ledger-root ledgers] [--run-id ID] [--cache-root cache_store] [--foundation-root <as arena commands>] [--data-root data] [--manifest-dir data/manifests]`.

- `all` expands to `CACHE_DATASETS` in order; repeated `--dataset`/`--pool-spec` keep first-seen order and drop duplicates.
- `--run-id` defaults to `new_run_id("gate1-tune")`. The git commit is `current_git_commit(Path.cwd())`; a `RunError` prints `FAIL <message>` and exits 1 before anything is loaded or written.
- On success, print `OK <run_id> <status>` and then `canonical_json(summary)` on a second line; exit 0 for `completed`, 2 for `budget_exhausted`. An `ArenaError` before start prints `FAIL <message>`, exits 1, and writes nothing. After start, an exception prints `FAIL <run_id> aborted: <message>` and exits 1 (the ledger holds the aborted run).

### Tests for PR B (all required; `tests/runner/test_gate1_tune.py`)

Fixtures: the `synthetic-dev` `per_family = 1` cache (as `synthetic_env` in `tests/arena/test_arena_cli.py`), and the pooled environment with the fake foundation model (as `pooled_env` in `tests/updater/test_baselines_replay.py`). Tests pass a small `grids` argument unless they test `GATE1_GRIDS` itself; `git_commit` is passed as a fixed 40-hex string, and the CLI tests monkeypatch `current_git_commit`.

1. **Grids:** sizes 1, 1, 11, 66, 1 (total 80) in `BASELINE_IDS` order; Fixed-Share's order is eta-outer; no duplicates; each `DEFAULT_CONFIGS` entry is in its grid; every grid point passes the baseline's `init(n_experts=5, horizon=24, config, seed=0)` without raising.
2. **Record sequence:** a tune over `synthetic-dev` × `simple` with grids `{"equal": [{}], "hedge": [{"eta": 0.5}, {"eta": 1.0}]}` writes, in main: `run_start`, one oracle `baseline`, three updater `baseline`s in grid order, `run_end` `completed`. All are `pool == "dev"`, and the sealed stream does not exist.
3. **Same numbers as in-process:** each updater record's `metrics`, and its blob's `result`, equal `replay_dataset(BASELINES[id], config, seed, pool)` and `result_to_obj` of it, exactly; the oracle record's equal `oracle_dataset(pool)`'s. `hyperparams == config`, `updater_ref` as specified. Budgets: `evaluations == 1` (`0` for the oracle), `wall_seconds > 0`.
4. **Series wall-clock:** `len(series_wall_seconds) == number of series`, all entries `>= 0`, and their loop-sum is `<=` the record's `budget.wall_seconds`. For a result that failed at series `k` (0-based), the length is `k + 1`.
5. **run_start config:** exactly the keys listed; `simple_cache_ids["synthetic-dev"] == pool.cache_id`; `grids` equal to the passed grids as lists; `budget_limits.evaluations == planned_replays(...)`; `condition == "gate1-dev-tune"`; `seed` matches.
6. **Pooled:** with `pool_specs = ("simple", "simple+foundation")` and the fake model, `foundation_cache_ids[dataset]["simple+foundation"]` and `foundation_models` name the fake cache and model, `[]` for `simple`; each pooled blob's `result.foundation_cache_ids` matches the config, and `result.leakage_risk` equals `pool_leakage_risk` of the pool.
7. **Failure is a result:** a grid point `{"eta": 0.0}` for Hedge gives a record with `status == "failed"`, a non-empty `error`, `metrics == {}`, and the run continues to `completed`; the summary counts it under `"failed"`.
8. **Selection:** on hand-built record lists (a `LedgerWriter` in `tmp_path` plus blobs), the lowest `mase` wins; an exact tie goes to the lower `seq`; all-failed gives `None`; sealed-stream records and `oracle` records are ignored. On the run from test 2, `run_end.summary["selected"] == select_configs(records, read_blob)`.
9. **Budget exhausted:** with `max_wall_seconds = 1e-9`, the run writes the oracle record (spent is 0 before it) and then stops. `run_end.status == "budget_exhausted"`, and `summary["replays"] == 0`. `evolvets ledger verify` reports the wall-seconds overrun, and the record is not clipped.
10. **Aborted:** monkeypatching `oracle_dataset` to raise `ArenaError` on its second call, in a run over two pool specs, leaves `run_end` `aborted` with the error type in its summary. The records before it are intact, and `tune` re-raises.
11. **Nothing before start:** a missing cache for the second dataset raises `ArenaError`, and `ledger_root` is empty (no run directory).
12. **Sealed blindness:** recording file opens during a tune run with the existing `_record_opens` helper (imported from `tests.arena.test_realdata_prepare`, as `tests/arena/test_sealed_oracle.py` does; it covers `builtins.open`, `io.open`, `os.open` and `Path.open`): no opened path has a path component `sealed`, a component `synthetic.salt`, or a component ending `.sealed.json` — path components, not substrings, so the test's own `tmp_path`, which contains the test name, cannot trip the check. AST guard: `runner/gate1_tune.py` has no import from `evolvets.arena.sealed`.
13. **Determinism:** two tune runs with the same inputs (different run ids) give equal `metrics`, equal blob `result` objects and equal `selected` summaries.
14. **Seed:** monkeypatching `replay_dataset` with a spy shows every call receives `spec.seed`. `seed = -1` raises `ValueError` before anything is written.
15. **CLI:** `gate1 tune --dataset synthetic-dev --pool-spec simple` on the fixture (with `GATE1_GRIDS` monkeypatched to the small grid) prints `OK <run_id> completed` and a JSON summary, exits 0, and `ledger verify` passes on the run. With `current_git_commit` raising `RunError`, it exits 1, prints `FAIL`, and writes nothing. `--dataset nope` is rejected by argparse.

### Dependencies

None.

### Size

About 300 lines of implementation. If it grows past ~400, put the grids, `select_configs` and tests 1 and 8 in `feat/gate1-grids` and the tuning run and CLI in `feat/gate1-tune`.

### After PR B merges

The reviewer opens an `fyi-human` issue: the owner may run `evolvets gate1 tune --dataset all --pool-spec simple --pool-spec simple+foundation` on the machine holding the dev caches and foundation caches, from a clean checkout of `main`, keep the ledger directory, and report the run id. Agents do not run it (see the rules above). Every started run stays in the ledger, including aborted ones.

---

## PR C: `feat/gate1-report`

A deterministic Markdown report of named dev runs, regenerated from the ledger alone.

### Files and interfaces

`src/evolvets/runner/gate1_report.py`

```python
class ReportError(Exception): ...


BOOTSTRAP_RESAMPLES = 2000


def build_report(ledger_root: Path, run_ids: Sequence[str], *, bootstrap_seed: int) -> str: ...
```

`runner/cli.py` gains `evolvets gate1 report --ledger-root DIR --run ID [--run ID ...] [--bootstrap-seed 0] (--out PATH | --check PATH)`.

### Exact behaviour: inputs and refusals

- For each run id, in the order given: `LedgerReader.verify(check_blobs=True)`. A failed chain or a missing blob raises `ReportError` naming the run and the position. `validate_run` violations do not refuse; they are listed in the Runs section.
- Every run must start with a `run_start` whose config blob has `"experiment": "gate1"`, else `ReportError`. Runs with different `protocol_versions` raise `ReportError` naming both (AGENTS.md 2.4: results from different versions are never mixed). Duplicate run ids raise `ReportError`.
- **Complete runs only in the tables:** results come only from runs whose `run_end` status is `completed`. Other runs (no `run_end`, `aborted`, `budget_exhausted`) appear in the Runs section with their status and record counts and are never silently dropped.
- **Consistency checks** (`ReportError` on any mismatch): each blob's `result.arena_version` equals the run's `protocol_versions["arena"]`; `result.dataset`, `result.pool_spec`, `result.cache_id` and `result.foundation_cache_ids` agree with the run's config blob — failed and `invalid_weights` results carry all of these in their blobs (`result_to_obj` stamps them whatever the status), so they are checked too; for each `(dataset, pool_spec)` the oracle's per-series ids equal the per-series ids of every baseline result with `status == "ok"`, in order — a failed result has `series == ()`, no ids to compare, so this check skips it and only the checks above apply to it; `select_configs` recomputed from the records equals `run_end.summary["selected"]`.
- Two complete runs covering the same `(dataset, pool_spec)` raise `ReportError`: the caller names the runs, and the report never picks one silently.
- Only `dev` records are read; a sealed-stream record raises `ReportError("sealed results are reported by the sealed Gate 1 task")`. The sealed sections are added by that task.

### Exact behaviour: content

All numbers are formatted with `f"{x:.4f}"`, configs with `canonical_json`. The output depends only on the ledger and `bootstrap_seed`: no current time, hostname or path outside the run ids. Sections, in this order:

1. `# Gate 1 report (dev)`, then one line naming the generating command with its arguments (run ids and seed only).
2. `## Runs`: one row per named run: run id, condition, `run_end` status (or `incomplete`), git commit, record count, `validate_run` violations (or `none`).
3. `## Protocol versions`: the four keys and values.
4. For each `(dataset, pool_spec)` with `leakage_risk == false`, in config order, `## <dataset>, pool <pool_spec>`:
   - **Tuned baselines** table: baseline, selected config, dev MASE, configurations tried, failed (status not `ok`), evaluations charged, wall seconds (loop-sums of the records' budgets). A baseline whose selection is `None` (no `ok` record: every grid point failed or was `invalid_weights`) renders `none` in the selected-config and dev-MASE columns; its tried, failed, evaluations and wall-seconds columns are still filled in from its records;
   - **Oracles** table: true-best forecast (`oracle_mase` from any baseline result's metrics; synthetic only), best fixed expert over the pool (`global_expert_mase`, with the expert's name), best expert per series (`best_expert_mase`), switching oracles `switch_mase_m<m>` for every `m` in `ORACLE_SWITCHES`, per-origin best (`select_mase`);
   - **Gaps** table: the **best tuned baseline** is the one with the lowest selected dev MASE among the baselines whose selection is not `None`; the MASE comparison is exact `<`, as in `select_configs` (an exact tie keeps the earlier baseline in `BASELINE_IDS` order). If every selection is `None` there is no best tuned baseline and no gaps to compute: the Gaps table is replaced by the fixed sentence `No tuned baseline has an ok configuration on this pool.` For each oracle row, the mean over series of `d_s = baseline_mase_s - oracle_s` with a 95% paired bootstrap interval. Per-series oracle values: `expert_mase[global expert index]`, `min(expert_mase)` (left-to-right, first minimum), `switch_mase[index]`, `select_mase`, and on synthetic pools `oracle_mase` from the baseline's own series scores;
   - on `synthetic-dev` only, the same Gaps rows per family (series id prefix before `/`), in `FAMILIES` order.
5. `## Leakage-risk results`: the same subsections for every `(dataset, pool_spec)` with `leakage_risk == true`, introduced by one sentence: a pooled foundation model may have seen this dataset in pretraining. If there are none, the sentence `No leakage-risk results.`.
6. `## Per-series wall-clock`: per `(dataset, pool_spec)`, the largest `series_wall_seconds` entry over all its replays, and `SERIES_TIMEOUT_SECONDS` (imported from `evolvets.arena.sealed.client`), as the evidence for the timeout review.
7. `## Notes`: fixed sentences: these are dev numbers, and the selected configuration was chosen on these same series, so the dev gaps favour the baselines' fit; the expert-pool oracles bound single-expert selection only and are not a floor for weighted combinations; the Gate 1 decision is the owner's.

**Bootstrap:** `rng = random.Random(bootstrap_seed)`, a new one for each Gaps table (so the order of tables does not change their numbers). For `b` in `range(BOOTSTRAP_RESAMPLES)`, draw `n` indices with `rng.randrange(n)` and take the loop-mean of `d` at those indices. Sort the `B` means; the interval is `[means[50], means[1949]]`. With one series, print the mean and `n/a` for the interval.

### Tests for PR C (all required; `tests/runner/test_gate1_report.py`)

Fixtures: a completed tune run on the synthetic fixture (PR B, small grid), a tune run on the ETTh1 fixture with pool specs `simple` and `simple+foundation` joining the `leaky` model (as `real_env` in `tests/arena/test_pool.py`; its ETTh1 overlap is `"yes"`, so the pooled results carry `leakage_risk` true and the `simple` ones false), and hand-built ledgers (a `LedgerWriter` plus `BlobStore` in `tmp_path`) for the refusal cases.

1. **Regenerable:** `build_report` twice gives identical strings. Copying the ledger directory elsewhere and building from the copy gives the identical string.
2. **Numbers trace to records:** the tuned-baseline row of `hedge` shows the selected config and the record's `metrics["mase"]` formatted `.4f`; the oracle rows equal the oracle record's metrics; the configurations/evaluations/wall columns equal loop-sums over the records.
3. **Gaps and bootstrap:** the Gaps mean equals the loop-mean of the per-series differences computed independently in the test. The interval equals an independent recomputation with `random.Random(seed)` and the stated index rule. Constant differences give an interval equal to the mean twice. A different `bootstrap_seed` changes the interval but not the mean.
4. **Synthetic rows:** the true-best row and per-family Gaps rows appear for `synthetic-dev` and not for a real dataset.
5. **Leakage split:** the pooled ETTh1 results appear only under `## Leakage-risk results`. With no flagged results, that section holds `No leakage-risk results.`.
6. **Failed grid points:** a completed tune run whose grid includes a failing point (Hedge `{"eta": 0.0}`, as PR B test 7) next to an `ok` point builds a report without `ReportError`: the failed record still passes the `arena_version`/`dataset`/`pool_spec`/`cache_id`/`foundation_cache_ids` checks, the series-id check skips it (its blob has `series == []`), and the Tuned baselines row counts it in the `failed` column. A completed run whose whole grid fails (for example `{"hedge": [{"eta": 0.0}]}`) renders Hedge's `None` row (`none` in the config and MASE columns, tried/failed/evaluations/wall still filled in) and — no baseline having an `ok` selection — the fixed no-gaps sentence instead of the Gaps table.
7. **Incomplete runs:** an aborted run, a `budget_exhausted` run and a run without `run_end`, named next to a completed one, each appear in Runs with their status, and none of their numbers appear in any table.
8. **Refusals,** each a `ReportError` and no output file: protocol versions differ between two runs; a tampered record (chain failure); a deleted blob; a blob whose `cache_id` differs from the config; mismatched series ids between the oracle and an `ok` baseline result (a failed result has no ids to compare and is never a mismatch); `selected` in `run_end` differing from the recomputation; two complete runs covering the same `(dataset, pool_spec)`; a sealed-stream record; a config without `"experiment": "gate1"`; a duplicate run id.
9. **Violations listed:** the budget-exhausted run from PR B test 9 shows its wall-seconds overrun in the Runs section.
10. **Wall-clock section:** it shows the maximum `series_wall_seconds` per `(dataset, pool_spec)` and `600.0000`.
11. **Notes:** the report contains the sentence that the oracles are not a floor for weighted combinations, and the dev-selection sentence.
12. **CLI:** `--out` writes the report and exits 0. `--check` against the same file exits 0; against a file with one changed character, it exits 1 and prints the first differing line number. Without `--run`, argparse rejects the call. A `ReportError` prints `FAIL <message>`, exits 1, and leaves `--out` unwritten.
13. **Import guard:** `runner/gate1_report.py` imports from `evolvets.arena.sealed` only `client`, and a fresh subprocess importing `evolvets.runner.gate1_report` does not load `evolvets.arena.sealed.server` or `evolvets.arena.sealed.pools`.

### Dependencies

None.

### Size and split (mandatory)

Two developer runs on the unsplit PR C hit the 60-minute limit with no output (issue discussion in the planner log, 2026-10-13). PR C is therefore split in two PRs, opened in order, each stopping for review:

- **PR C1, `feat/gate1-report-data`:** `src/evolvets/runner/gate1_report_data.py`. It does everything in "Exact behaviour: inputs and refusals" and computes every number in "Exact behaviour: content" as plain dataclasses (`ReportData`: run rows, protocol versions, per `(dataset, pool_spec)` the tuned-baseline rows, oracle rows, gap rows with bootstrap intervals, per-family gap rows, leakage flag, max `series_wall_seconds`). It renders no Markdown and touches no CLI. Tests: 2 (as dataclass fields), 3, 6, 7, 8 (except the CLI/`--out` parts), 9, and the import guard of test 13.
- **PR C2, `feat/gate1-report`:** `src/evolvets/runner/gate1_report.py` (`build_report`, which calls C1 and renders the sections and Notes sentences) and the `evolvets gate1 report` CLI. Tests: 1, 4, 5, 10, 11, 12, and the subprocess part of 13.

Time box for each PR: if you are 40 minutes in and CI is not green locally, open the PR with what works and list what is missing in the description; do not keep going silently. Write small files first, commit often, and run only the tests for the file you are editing until the end.

---

## Not in this task

- **The sealed half of entry 11:** sealed baseline runs with the dev-selected configurations, the sealed oracles, the final `SERIES_TIMEOUT_SECONDS` value, sealed sections of the report, and committing a report to the repository. It needs the owner's decision on **where the sealed data and salt live when Gate 1 runs** (the owner's machine, or a runner with a secret, which is a `.github/` change). It also needs the salt created (`evolvets sealed init-salt`, TASK_008 "After PR B merges"; `data/manifests/synthetic-sealed.yaml` is not committed yet) and the sealed foundation caches built. When this task closes, the planner escalates with a `needs-human` issue before specifying it.
- **The Gate 1 decision** (harden the generator, or proceed to Phase 1) is a human gate after the sealed report.
- **Tagging:** issue #55 (arena protocol tag, now at version 3) stays the owner's call; this task changes no frozen component.
- **Phase 1** decisions (whether `updater/baselines.py` becomes a frozen copy outside the editable area; OS-level isolation of sealed workers) stay with the Phase 1 specs.
- **Entry 13 (`feat/live-commit`)** still needs the owner's decisions on where commitments are published and on any scheduled workflow.

## Closing the task

PR C (or the last PR of its split) changes this file's first line to `Status: done` and sets roadmap entry 11's status to `TASK_011 (dev, PRs ...); sealed half spec pending`.
