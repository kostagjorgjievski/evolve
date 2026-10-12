Status: active

# Task 008: Baseline status in the ledger, sealed pools and the sealed server process

Read `AGENTS.md` and `docs/ROADMAP.md` first. This task covers roadmap entries 4c (`feat/ledger-baseline-status`, the owner's decision in issue #38) and 9 (`feat/arena-sealed`). It is a stack of three PRs. Open them in order and stop after each one for review.

## Why

Gate 1 compares hand-designed updaters on the sealed segment. Sealed values must never be visible to code an agent may edit (AGENTS.md 2.1), and every sealed run, including failed and timed-out ones, must be in the ledger (2.8). This task builds the three pieces that make that possible:

- PR A lets a `baseline` record say how the run ended (issue #38, option A).
- PR B loads the sealed pools (the sealed segment of the real data, and salted synthetic worlds) and builds their forecast caches outside `cache_store/`.
- PR C adds the sealed server. It is a separate process that alone reads sealed data. It replays an updater that runs in a fresh worker process per series, and writes the score straight to the run's sealed ledger stream. The caller gets back only a receipt.

Hand-designed updaters (entry 10) and the Gate 1 experiment (entry 11) come later. This task uses toy updaters defined in the tests.

## Decisions already made (do not reopen)

- **Issue #38, option A** (owner): `Baseline` gains `status`, `error` and `per_task_blob`, with `Evaluation`'s rules, and `metrics == {}` unless ok. `schema_version` stays `"1.0"`. The owner tags `ledger-schema-v1` on PR A's merge commit (issue #10), not before.
- **Issue #38 defaults for entry 9** (accepted by the owner as written): sealed pools and burn-in as on dev; the sealed synthetic salt created once by a human with `evolvets sealed init-salt`, with only its hash committed; the process layout (caller → spawned server → one spawned worker per series, JSON only, nothing unpickled from a worker); no OS-level sandbox in Phase 0; the live pool stays a stub.
- **TASK_007:** a fresh updater per series. The arena protocol and `ARENA_VERSION = "1"` cover what an updater receives and how it is scored. Agent-editable code never imports `evolvets.cache`. The import contracts from #32 are in `.importlinter` (#39).

## Rules for all three PRs

- Standard library only. No new dependencies.
- Strict mypy already covers `ledger/` and `arena/`. New modules under `arena/sealed/` are covered by the existing `evolvets.arena.*` override, so check that and do not weaken it.
- Bit-exact determinism, as in TASK_007. No new float arithmetic is needed: sealed scoring reuses `replay_dataset` and the TASK_007 metrics unchanged. Never call the builtin `sum()`, `math.fsum` or `statistics` on floats.
- `ARENA_VERSION` stays `"1"`. This task adds no new scoring. PR C extends the arena protocol's file list in the `arena/__init__.py` docstring to include `arena/sealed/`. The human tags `arena-v1` after PR C merges (TASK_007).
- **Nothing outside `arena/sealed/`** opens `<dataset>.sealed.json`, the salt file or `data/sealed_cache/`. The one exception is the `evolvets sealed` CLI commands (PR B), which are one-shot trusted processes started by a human.
- **Paths (all git-ignored through `data/*`):** sealed salt `data/sealed/synthetic.salt`; sealed caches `data/sealed_cache/<dataset>/`; server error log `data/sealed/server.log`. This task commits no data files. `data/manifests/synthetic-sealed.yaml` is committed later by the human (see "After PR B merges").
- Do not edit `.importlinter`, `.github/`, `AGENTS.md` or `docs/agents/`.

---

## PR A: `feat/ledger-baseline-status`

The owner's decision on issue #38: a `baseline` record can say that the run failed, timed out or returned invalid weights, and why.

### Schema (`ledger/payloads.py`)

`Baseline` becomes, with fields in this order:

```python
@dataclasses.dataclass(frozen=True)
class Baseline:
    """A hand-designed updater's score in one pool, or why it has none."""

    updater_id: str
    hyperparams: dict[str, Any]
    pool: str
    status: str  # one of EVALUATION_STATUSES
    metrics: dict[str, float]  # {} unless status == "ok"
    per_task_blob: str | None
    error: str | None  # None iff status == "ok"
```

Validation, in `__post_init__`, in addition to the existing checks (non-empty `updater_id`, str-keyed `hyperparams`, `pool` in `EVALUATION_POOLS`, `_require_float_dict` on `metrics`, `_check_finite` on the whole payload):

1. `status` in `EVALUATION_STATUSES` (`_require_literal`).
2. `per_task_blob` is `None` or a blob ref (`_require_nullable_blob_ref`). It is allowed on every status: a failure can carry a detail blob.
3. `status == "ok"` → `error is None`. Otherwise `error` is a non-empty `str`. Use the same messages as `Evaluation`.
4. `status != "ok"` → `metrics == {}`, else `PayloadError("metrics must be empty when status is not 'ok'")`. `status == "ok"` with `metrics == {}` is accepted, as for `Evaluation`.

`Evaluation`, `REQUIRED_REFERS_TO`, `validate_run`, `PROTOCOL_KEYS` and `SCHEMA_VERSION` do not change. There is no `dataset` field: as for `Evaluation`, the dataset, cache id and per-series detail live in `per_task_blob` (PR C writes it). Update the `baseline` fixture in `tests/ledger/conftest.py` to the new fields.

### Tests for PR A (all required; `tests/ledger/test_payloads_kinds.py`, plus `test_store.py`/`test_validate.py` where noted)

1. An ok baseline with metrics, a `per_task_blob` and `error=None` constructs, and `payload_from_dict("baseline", payload_to_dict(b)) == b`. The same holds with `per_task_blob=None`.
2. Each of the four statuses is accepted when its `error` and `metrics` follow rules 3 and 4. `status` of `"done"`, `""`, `None` and `"OK"` raises `PayloadError`.
3. `error` rule, both directions: ok with `error="x"` raises; non-ok with `error=None`, `""` or `3` raises; non-ok with `error="ValueError: boom"` is accepted.
4. `metrics` rule, both directions: `status="failed"` with `{"mase": 1.0}` raises; `status="timeout"` with `{}` is accepted; ok with `{}` is accepted.
5. `per_task_blob` of `"sha256:xyz"`, `"abc"` and `5` raises. A valid ref on a `failed` baseline is accepted.
6. `payload_from_dict("baseline", d)` with any one of `status`, `per_task_blob` or `error` removed raises `missing fields`. An extra key raises `unknown fields`.
7. NaN in `hyperparams` or `metrics` still raises (the existing `_check_finite`).
8. The existing test that a `pool="sealed"` baseline is rejected in the main stream still passes with the new fixture. A `pool="sealed"`, `status="timeout"` baseline in the sealed stream is accepted.
9. Store round trip (`test_store.py`): a main stream with a valid `run_start`, then a sealed stream with one `timeout` baseline whose `per_task_blob` was put in the run's `BlobStore`. `LedgerReader.records()` returns an equal payload. `verify(check_blobs=True)` is ok, and fails after the blob file is deleted. `validate_run` on these records returns `[]`.

### Size

About 30 lines of implementation plus tests.

### After merge

The reviewer comments on issue #10: the schema is final for v1, and the human should tag `ledger-schema-v1` on PR A's merge commit. From then on, changes to `ledger/` need a `protocol/` PR.

---

## PR B: `feat/arena-sealed-pools`

Loading the sealed pools and building their forecast caches. No server process yet, and nothing is written to a ledger.

### Files and interfaces

`src/evolvets/arena/sealed.py` becomes the package `src/evolvets/arena/sealed/`. The module name `evolvets.arena.sealed` stays, so the existing `sealed-isolation` import contract (forbidden imports include submodules) keeps covering everything in it. `arena/sealed/__init__.py` holds only a docstring and **no imports**: PR C's worker imports a submodule of this package, and importing the package must load nothing sealed.

`arena/sealed/names.py` (no imports; the caller-side client of PR C reads it)

```python
SEALED_DATASETS = ("etth1", "ecl", "synthetic-sealed")
```

`arena/sealed/pools.py`

```python
SALT_BYTES = 32
SYNTHETIC_MANIFEST_VERSION = "1"


@dataclass(frozen=True)
class SyntheticSealedManifest:
    generator_version: str
    salt_sha256: str
    per_family: int


def salt_path(data_root: Path) -> Path: ...  # <data_root>/sealed/synthetic.salt
def synthetic_manifest_path(manifest_dir: Path) -> Path: ...  # <manifest_dir>/synthetic-sealed.yaml
def init_salt(data_root: Path, manifest_dir: Path, per_family: int) -> str: ...  # returns salt sha256
def read_synthetic_manifest(manifest_dir: Path) -> SyntheticSealedManifest: ...
def sealed_inputs(
    dataset: str, data_root: Path, manifest_dir: Path, horizon: int
) -> tuple[dict[str, str], tuple[SeriesInput, ...]]: ...
def build_sealed_cache(
    dataset: str, data_root: Path, manifest_dir: Path, sealed_cache_root: Path,
    horizon: int = DEFAULT_HORIZON,
) -> CacheManifest: ...
def load_sealed_pool(
    dataset: str, data_root: Path, manifest_dir: Path, sealed_cache_root: Path
) -> DevPool: ...
```

`load_sealed_pool` returns the TASK_007 `DevPool`/`DevSeries` types, so `replay_dataset` and its scoring run on sealed pools unchanged. The type names are historical. Do not rename them in this task. To share the checks, make `dev.py`'s `_match_values` and `_checked_series` public (`match_values`, `checked_series`) with no change in behaviour. You may also factor the synthetic-series builder so that it takes the task list. The TASK_007 golden test (`test_evaluate.py` test 13) must pass unchanged, and that is the proof.

### Exact behaviour: the salt (`init_salt`, `read_synthetic_manifest`)

- `per_family` is an int `>= 1` (bools rejected), else `ArenaError`.
- The manifest is JSON in a `.yaml` file, written with `realdata.manifest.manifest_text` and `write_atomic`. It has exactly these keys: `manifest_version` (`"1"`), `pool` (`"synthetic-sealed"`), `generator_version` (`GENERATOR_VERSION`), `salt_sha256` (64 lowercase hex), `salt_bytes` (`32`) and `per_family` (int).
- `init_salt` cases:
  - neither salt nor manifest exists: create `<data_root>/sealed/` and write `secrets.token_bytes(32)` to the salt file. Open it with `O_CREAT | O_EXCL | O_WRONLY`, mode `0o600`, then fsync. Then write the manifest and return the hash;
  - the salt exists and the manifest does not (a crash between the two writes): read the salt, check that it is 32 bytes, write the manifest from it, and return the hash;
  - both exist and the manifest matches the salt and `per_family`: no-op, return the hash;
  - any other case raises `ArenaError` and changes nothing. That covers: the manifest exists but the salt is missing (the message says to restore the salt from backup and never regenerate it); the hashes differ; `per_family` differs; the salt is not 32 bytes.
- `init_salt` never overwrites or deletes a salt or a manifest.
- `read_synthetic_manifest` raises `ArenaError` when the manifest is missing, unreadable, not an object, has the wrong key set or values of the wrong type, has `generator_version != GENERATOR_VERSION` (unknown versions are refused, not reinterpreted) or a bad hex hash.
- Reading the salt (a private helper used by `sealed_inputs`): the file must be exactly 32 bytes and its SHA-256 must equal `salt_sha256`, else `ArenaError`. The salt bytes never appear in an exception message, a return value other than the task list, or a log.
- Agents never run `init_salt` against the real `data/` directory. Tests use `tmp_path`.

### Exact behaviour: `sealed_inputs` (what the sealed caches are built from)

- `etth1`, `ecl`:
  - `read_split_manifest`, then read `<data_root>/processed/<dataset>.sealed.json`. Its file SHA-256 must equal the manifest's `sealed_sha256`.
  - Same structural checks as `load_public` (keys, dataset, `realdata_version`, series ids and starts in manifest order, numeric non-bool values). Each series' length must equal its split's `length`.
  - **Prefix check:** `load_public(...)` must give, for every series, values equal to `sealed_values[: sealed.start]` (`==` on tuples). This ties the sealed file to the public data the dev pool and caches use.
  - Each `SeriesInput` holds `values[: sealed.end + horizon]` and `period = 24`. Nothing from the reserve segment or past `sealed.end + horizon - 1` enters a sealed cache.
  - `source = {"kind": "realdata-sealed", "dataset", "realdata_version", "sealed_sha256", "public_sha256"}`.
- `synthetic-sealed`:
  - `read_synthetic_manifest` and the salt as above. The tasks are `make_task(family, world_seed("sealed", family, i, salt))` for `family` in `FAMILIES` and `i in range(per_family)`, with ids `f"{family}/{i}"`, values `world.y` and `period = world.spec.period`.
  - `source = {"kind": "synthetic", "pool": "sealed", "generator_version", "per_family": str(n), "salt_sha256"}`.
- Any other dataset raises `ArenaError`. A `PrepareError` from the real-data layer is wrapped in `ArenaError`.

### Exact behaviour: `build_sealed_cache` and `load_sealed_pool`

- `build_sealed_cache` calls `cache.store.build_cache(dataset, source, series, sealed_cache_root, horizon)`, so it gets that function's guarantees: built in a temp directory, never overwritten, and equal content makes a rebuild a no-op. A `CacheError` is wrapped in `ArenaError`. The sealed cache root is never the public cache: if the last component of `sealed_cache_root` is `cache_store`, raise `ArenaError`.
- `load_sealed_pool`:
  - `read_manifest` and `cache_id` on the sealed cache, then `sealed_inputs` with the cache's horizon. Ids, order and count must match the cache. Each `SeriesInput.values` must pass `match_values` against the cache entry, and the cache's `source` must equal the recomputed `source`.
  - Real data: `scale_end = warmup.end`, `score_start = sealed.start`, `score_end = sealed.end`, `change_points = ()`, `oracle = None`. Origins from `first_origin` through the dev segment are an unscored burn-in. The MASE scale comes from warmup, as on dev.
  - Synthetic: exactly as `synthetic-dev`. `scale_end = score_start = GrammarConfig().warmup`, `score_end = length - H`, `change_points = world.change_points`, and `oracle` from `true_best_forecast`.
  - All of `checked_series`' boundary checks apply. `DevPool.dataset` is the sealed dataset name, and `cache_root` is the sealed cache root.
  - Every failure raises `ArenaError`.

### CLI (`runner/cli.py`)

- `evolvets sealed init-salt --per-family N [--data-root data] [--manifest-dir data/manifests]`. `--per-family` is required. It prints the salt's SHA-256 and the manifest path, never the salt.
- `evolvets sealed build-cache {etth1,ecl,synthetic-sealed,all} [--data-root data] [--manifest-dir data/manifests] [--sealed-cache-root data/sealed_cache] [--horizon 24]`. It prints, per dataset, the sealed cache id and series count, never values or metrics.
- Exit codes and error printing follow the existing `cache` commands. An `ArenaError` prints its message and exits 1.

### Tests for PR B (all required; `tests/arena/test_sealed_pools.py`, CLI cases in `tests/cache/test_cli.py` or a new `tests/arena/test_sealed_cli.py`)

Real-data tests use the TASK_007 fixture: `_build_etth1_fixture(rows=1200)`, giving warmup `[0, 139)`, dev `[307, 550)`, sealed `[718, 926)` and reserve from 1094. Synthetic tests use `per_family = 1`.

1. Salt creation: `init_salt` in `tmp_path` writes a 32-byte file with mode `0o600` and a manifest with exactly the keys above; it returns the salt's SHA-256. A second call with the same `per_family` is a no-op (salt bytes and manifest bytes unchanged).
2. Salt refusals, each leaving every file byte-identical: a different `per_family`; the manifest present and the salt deleted; the salt replaced by different 32 bytes; a 31-byte salt. Recovery: with the salt present and the manifest deleted, `init_salt` rewrites the same manifest.
3. `read_synthetic_manifest` rejects a missing file, an extra key, `per_family` of `True`, an edited `generator_version` and a 63-character hash.
4. Real `sealed_inputs`: ids in order; each series has length `926 + 24`; `values == sealed_file_values[:950]`; the source's keys and values are as specified. No value at index `>= 950` is reachable from the result.
5. Real tampering, each `ArenaError`: one sealed value changed without updating the manifest (hash mismatch); one value before index 718 changed with the manifest's `sealed_sha256` rewritten to match (prefix mismatch with public); the sealed file deleted; a series id swapped.
6. Synthetic `sealed_inputs`: 6 series `stationary/0 .. mixed/0`; `values == make_task(f, world_seed("sealed", f, 0, salt)).world.y`; every series differs from the dev task with the same id; a different salt (new `tmp_path`) gives different values.
7. Synthetic refusals: a salt whose hash does not match the manifest; a missing salt; a manifest with a different `generator_version`. The salt bytes do not appear in any of the exception messages (`salt.hex()` and `repr(salt)` checked).
8. `build_sealed_cache` then `load_sealed_pool` on real data: `scale_end == 139`, `score_start == 718`, `score_end == 926`, `values == sealed[:950]`, `oracle is None`. A rebuild is a no-op with the same cache id. A sealed cache root whose last component is `cache_store` raises.
9. Synthetic pool: `score_start == 300`, `score_end == 2000 - 24`, `change_points` and `oracle[0]` as for the world built from the salted seed.
10. Pool mismatches, each `ArenaError`: a sealed cache built from different data (rebuild the fixture with `variant=1` in a separate `tmp_path` and point at its cache); a cache whose `source` was edited; a missing sealed cache; an unknown dataset.
11. The dev path stays blind: with the sealed cache built, `load_dev_pool("etth1", ...)` and `public_inputs` still open neither the sealed file, the salt nor anything under the sealed cache root (record opens, as in TASK_007 test 1).
12. Determinism: `replay_dataset` with a test equal-weights factory on the real sealed pool, twice, gives equal results. `metrics["mase"]` is finite and positive.
13. AST guards: `arena/sealed/__init__.py` contains no `import` or `from ... import` statement. No module outside `arena/sealed/` and `runner/cli.py` has a non-docstring string constant containing `"synthetic.salt"` or `"sealed_cache"`.
14. CLI: `sealed init-salt` without `--per-family` exits non-zero; with it, it prints a 64-hex hash and not the salt (check that `salt.hex()` is not in the output). `sealed build-cache etth1` prints the cache id. On a tampered sealed file it exits 1 with the `ArenaError` message.

### Size

About 250 lines of implementation. If it grows past ~400, put the salt and the synthetic pool in a first PR (`feat/arena-sealed-salt`) and the real pool, caches and CLI in a second.

### After PR B merges

The reviewer opens an `fyi-human` issue (it does not pause anything) asking the owner to:

1. run `evolvets sealed init-salt --per-family N` once, on the machine that will hold the sealed data (the planner suggests `N = 20`, matching the dev sanity runs);
2. back up `data/sealed/synthetic.salt` outside the repository;
3. commit `data/manifests/synthetic-sealed.yaml` in a `chore/` PR;
4. reveal the salt only after the Gate 1 report is published.

PR C does not wait for this, because its tests use `tmp_path`.

---

## PR C: `feat/arena-sealed-server`

The sealed server process, the per-series worker process, and the caller's receipt API.

### Process layout

```
caller process                 sealed server process                  worker process (one per series)
evaluate_sealed(request) --spawn--> reads sealed pool + sealed cache
                                     replay_dataset(factory, ...) --spawn--> imports updater_ref only
                                     Context/Matured as JSON  ------------->  updater.init / update
                                     <------------ weights as JSON
                                     writes blob + baseline to sealed stream
        <-- receipt JSON {record_id, status}
```

- Every process is started with `multiprocessing.get_context("spawn")`, never `fork`. Pipes carry only bytes, through `Connection.send_bytes` and `Connection.recv_bytes(maxlength)`. **`Connection.recv()` is never called in either direction, and `pickle` is never imported.** (Spawning pickles the start arguments from parent to child. That is the trusted direction and is allowed.)
- Sealed values, targets, the salt, sealed forecasts outside the current `Context`, sealed metrics and failure details never reach the caller process or a worker process.

### Files and interfaces

`arena/sealed/client.py` (imported by callers: entries 10 and 11, and the Phase 1 runner)

```python
SERIES_TIMEOUT_SECONDS = 600.0


@dataclass(frozen=True)
class SealedRequest:
    dataset: str  # one of SEALED_DATASETS
    updater_id: str  # non-empty; recorded as Baseline.updater_id
    updater_ref: str  # "package.module:attr"; attr is a zero-argument factory returning an Updater
    config: dict[str, ConfigValue]  # recorded as Baseline.hyperparams
    seed: int


@dataclass(frozen=True)
class SealedPaths:
    data_root: Path
    manifest_dir: Path
    sealed_cache_root: Path


@dataclass(frozen=True)
class SealedReceipt:
    record_id: str  # the sealed-stream baseline record
    status: str  # one of EVALUATION_STATUSES


class SealedError(Exception):
    """The server refused the request or failed before writing a record."""


def evaluate_sealed(
    request: SealedRequest,
    *,
    ledger_root: Path,
    run_id: str,
    paths: SealedPaths,
    series_timeout_seconds: float = SERIES_TIMEOUT_SECONDS,
) -> SealedReceipt: ...
```

`client.py` imports nothing that reads data: it takes `SEALED_DATASETS` from `arena/sealed/names.py`, never from `pools.py`, and never imports `server.py`, `remote.py`, `realdata` or `cache`. The spawned server imports those itself. A spawn target is pickled by reference, so the caller would import whatever module defines it. Make the target a small function in `client.py` that imports `server` inside its body and calls `server._serve(...)`; that import then happens only in the child. Test 12 checks this.

`arena/sealed/server.py`: `_serve(conn, request_json, ledger_root, run_id, paths_json, timeout)`, the server's entry point. `arena/sealed/remote.py`: the server-side `RemoteUpdater` proxy and its factory. `arena/sealed/worker.py`: `_worker_main(conn, updater_ref)`. `worker.py` imports from `evolvets` only `evolvets.updater.interfaces`; an AST guard checks this.

### Exact behaviour: `evaluate_sealed` (caller side)

1. Validate the request in the caller, raising `SealedError` and spawning nothing:
   - `dataset` in `SEALED_DATASETS`;
   - `updater_id` is a non-empty str;
   - `updater_ref` matches `^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$`;
   - `config` has str keys and values of exact type `bool`, `int`, `str` or finite `float`;
   - `seed` is an int (not bool) in `[0, 2**63)`;
   - `series_timeout_seconds` is a finite float or int `> 0`;
   - `run_id` is non-empty.
2. Spawn the server as a non-daemon process, because it must spawn workers. Pass it one end of a `Pipe(duplex=True)`, the request as canonical JSON, and the paths and the timeout. Wait for exactly one reply: `recv_bytes(maxlength=4096)`. Then `join()` the server.
3. The reply is a JSON object, either `{"record_id": <uuid>, "status": <status>}` or `{"refused": <reason>}`. A refusal raises `SealedError(reason)`. EOF, a malformed reply or a server exit code other than 0 raises `SealedError("sealed server failed (exit code N); see <data_root>/sealed/server.log")`. The caller never sees a traceback or an exception message from the server.

### Exact behaviour: the server

1. **Ledger preconditions**, checked before any sealed file is opened. A failure replies `{"refused": reason}` and writes nothing:
   - `LedgerReader(ledger_root, run_id).records(["main"])` reads cleanly;
   - its first record is a `run_start` whose `protocol_versions == protocol_versions()` (current code);
   - main contains no `run_end`.

   The reason strings name only which check failed. They are not sealed information.
2. Open `LedgerWriter(ledger_root, run_id, "sealed")` **before** evaluating. A second concurrent request on the same run then fails fast with `refused: sealed stream is locked` and evaluates nothing. The writer is closed in a `finally`.
3. Load the pool with `load_sealed_pool`. On `ArenaError`, or any other unexpected `Exception` before the record is written:
   - append the traceback, a UTC timestamp, `run_id` and `updater_id` to `<data_root>/sealed/server.log`;
   - write no record;
   - exit with code 2, so the caller raises `SealedError`.

   No sealed run has started at that point. For failures after the evaluation has started, see step 5.
4. Start the clock (`time.monotonic()`). Run `replay_dataset(factory, request.config, request.seed, pool)`, where `factory` is the `RemoteUpdater` factory below. It produces one of:
   - an ok or failed or invalid-weights `ReplayResult`;
   - a `SeriesTimeout` (below), which becomes status `timeout` with `error = f"series {series_id} exceeded {limit}s at step {step}"` (`step` is `None` if the timeout came during `init`), `failed_series`, `failed_step`, empty metrics and series.
5. Write, in this order:
   1. the per-task blob, with `BlobStore(ledger_root / run_id / "blobs").put(canonical_bytes(obj))`. `obj` has exactly the keys `updater_ref`, `series_timeout_seconds` and `result`. `result` is `result_to_obj(result)`, or for a timeout the same key set with `status = "timeout"`;
   2. a `baseline` record in the sealed stream: `updater_id`, `hyperparams = dict(config)`, `pool = "sealed"`, `status`, `metrics`, `per_task_blob = ref`, `error`. The budget is `Budget(wall_seconds=<monotonic elapsed since step 4>, evaluations=1)`. This is the only budget charge for a sealed evaluation, and the caller must not charge it again.

   If an unexpected exception is raised during step 4 (not an `UpdaterFailure`, which `replay_dataset` already turns into a result, and not a `SeriesTimeout`), the server still writes a baseline with `status = "failed"`, `error = "sealed server error: <ExceptionTypeName>"` and `per_task_blob = None`, logs the traceback as in step 3, and replies normally. A sealed run that started is always in the ledger (AGENTS.md 2.8), except when the server process itself dies.
6. Reply `{"record_id": record.record_id, "status": status}`, close the writer and exit 0. Kill and join any remaining worker in a `finally`.

### Exact behaviour: `RemoteUpdater` and the worker

- **Factory.** `replay_dataset` calls the factory once per series, in pool order. Each call first shuts down the previous worker, if any: it sends `{"op": "close"}`, joins with a 5 s limit, then calls `kill()` and `join()`. Then it spawns a fresh worker with `_worker_main(child_conn, updater_ref)` and returns a new `RemoteUpdater` bound to it. The series' **deadline** is the spawn time plus `series_timeout_seconds`, so it covers worker start-up, `init` and every `update`.
- **Messages from server to worker** (JSON via `json.dumps(..., allow_nan=False)`; floats round-trip exactly through `repr`):
  - `{"op": "init", "n_experts", "horizon", "config", "seed"}`;
  - `{"op": "update", "feedback": [{"step", "errors", "combined_errors"}, ...], "context": {"step", "season_index", "period", "horizon", "expert_names", "forecasts"}}`;
  - `{"op": "close"}`.

  These are exactly the fields of `Context` and `Matured`. No series id, origin, value or path is ever sent.
- **Worker.** At start-up it redirects file descriptors 1 and 2 to `os.devnull` (with `os.dup2`), so nothing an updater prints reaches the caller's console or CI logs. Then it resolves `updater_ref` with `importlib.import_module` and `getattr` and calls the factory once.
  - On each message it rebuilds `Matured`/`Context` with tuples and calls `init` or `update`, keeping the state locally.
  - It replies `{"ok": true}` after `init`, `{"weights": <weights as returned, converted to JSON>}` after `update`, or `{"error": f"{type(exc).__name__}: {exc}"[:1000]}` on any `Exception`. That includes a failed import or factory call, which is reported on the first (`init`) message.
  - Weights that `json.dumps(allow_nan=True)` cannot encode produce an `error` reply. NaN and infinity are encoded, and the arena rejects them as invalid weights, the same as on dev.
  - It exits on `close` or on EOF.
- **`RemoteUpdater.init`/`update`** (server side) send the message, then wait with `conn.poll(deadline - now)`.
  - Not ready by the deadline: raise `SeriesTimeout(series_id, step)`. This is a **`BaseException` subclass**, so `replay_series` (which wraps only `Exception`) and `replay_dataset` let it through unchanged. `replay.py` and `evaluate.py` are not modified.
  - Then `recv_bytes(maxlength=MAX_REPLY_BYTES)` with `MAX_REPLY_BYTES = 1 << 22`. EOF, `OSError` (including a message that is too long), invalid JSON, or a reply that is not an object with exactly one of the keys above: raise `WorkerError(Exception)` with a message that names the cause and, after EOF, the worker's exit code. `replay_series` maps it to `UpdaterFailure("failed")`.
  - `{"error": msg}`: raise `WorkerError(msg)`.
  - `{"weights": w}`: return `(w, None)`. `replay_series`' existing validation then applies, so bad contents become `invalid_weights` exactly as on dev.
- Because each series gets a fresh process, module-level and process-global updater state cannot carry over between series. This closes the gap TASK_007 left to entry 9.

### Tests for PR C (all required; `tests/arena/test_sealed_server.py`, toy updaters in `tests/arena/sealed_toys.py`)

Use the 1200-row real fixture and, where stated, the synthetic sealed pool with `per_family = 1`. Each test builds a ledger with a valid `run_start` (`protocol_versions()`) in main. Keep the module under about 90 s: use the real fixture unless a test needs synthetic.

1. **Same numbers as in-process.** For an equal-weights toy and a one-hot-on-`naive` toy, on both the real and the synthetic sealed pools, `evaluate_sealed` returns `status == "ok"`. The sealed-stream baseline's `metrics` equal, exactly (`==`), `replay_dataset(factory, config, seed, load_sealed_pool(...)).metrics` computed in the test process. The blob's `result` equals `result_to_obj` of that result.
2. **The record:** exactly one new sealed-stream record, kind `baseline`, with `pool == "sealed"`, `hyperparams == config`, `updater_id` as requested and `budget.evaluations == 1`, plus `wall_seconds > 0`. `LedgerReader.verify(check_blobs=True)` is ok and `validate_run` returns `[]`. Two sequential requests in one run give two records with `seq` 0 and 1.
3. **Receipt only:** `SealedReceipt` has exactly the fields `record_id` and `status` (`dataclasses.fields`). The raw reply bytes the caller receives (spy on the caller's `recv_bytes`) parse to an object with exactly those two keys.
4. **The caller never touches sealed data:** record every file open in the test process during `evaluate_sealed` (TASK_005 helper). Nothing under `processed/*.sealed.json`, `<data_root>/sealed/` or the sealed cache root is opened.
5. **What the worker sees:** a spy toy appends every `(feedback, context)` it receives, as JSON, to the file named in `config["spy_path"]`. The sealed run's recording equals the recording the same spy class makes during an in-process `replay_dataset` on the sealed pool (series by series). So the worker received exactly the `Context`/`Matured` stream, and nothing else.
6. **A fresh process per series:** a toy writes `os.getpid()` and a module-level counter, incremented on every `init`, to its spy file. There is one distinct pid per series, and the counter is 1 in every series.
7. **Timeout:** a toy that sleeps 30 s at step 3 of the first series, with `series_timeout_seconds = 2`, gives `status == "timeout"`. The record has `error` starting `"series etth1/HUFL exceeded"`, `metrics == {}`, and a blob whose `result.failed_step == 3`. The call returns within 15 s. That worker's pid (from its spy file) is no longer alive afterwards. A toy that sleeps in its factory gives `timeout` with `failed_step is None`.
8. **Failures:**
   - `init` raising gives `failed`, `failed_step is None` in the blob, and an error containing the exception type name;
   - `update` raising at step 7 of series 2 gives `failed` with that series and step;
   - non-simplex weights give `invalid_weights`;
   - NaN weights give `invalid_weights`;
   - weights that are a set (not JSON-encodable) give `failed`;
   - a toy calling `os._exit(3)` at step 5 gives `failed` with `"exit code 3"` in the error;
   - a toy returning a weights list of about 8 MB gives `failed` (reply too long);
   - an `updater_ref` naming a missing module gives `failed`, `step None`.

   Each case writes exactly one record, and `validate_run` returns `[]`.
9. **No stray output:** a toy that prints a marker to stdout and stderr on every call. With `capfd`, the marker does not appear in the test process's captured output.
10. **Refusals** (each raises `SealedError` and writes no sealed record; except in the locked case, no `sealed.lock` is left behind): no main stream; main whose first record is not `run_start`; a `run_start` with `protocol_versions["arena"] = "0"`; main already ending in `run_end`; the sealed stream locked by another `LedgerWriter` held open by the test. Caller-side validation (no spawn, check with a spy on `get_context`): an unknown dataset, an `updater_ref` without `:`, a config value of `float("nan")`, `seed = -1`, timeout `0`.
11. **Server failures:**
    - a missing sealed cache raises `SealedError` mentioning `server.log`, writes no record, and leaves a log entry with `ArenaError`. The exception message in the caller contains no path under the sealed cache root;
    - a toy that kills its worker's parent (the server, `os.kill(os.getppid(), SIGKILL)`) at step 5 raises `SealedError` in the caller. Afterwards the sealed stream verifies (no record, or only complete ones), the stale `sealed.lock` can be broken with `LedgerWriter.break_stale_lock`, and the orphaned worker exits within 10 s (it sees EOF).
12. **Import isolation:**
    - in a fresh `python -c` subprocess that imports `evolvets.arena.sealed.client`, none of `evolvets.arena.sealed.pools`, `evolvets.arena.realdata.prepare` or `evolvets.cache.store` is in `sys.modules`;
    - in a fresh subprocess that imports `evolvets.arena.sealed.worker`, the only `evolvets` modules loaded are `evolvets`, `evolvets.arena`, `evolvets.arena.sealed`, `evolvets.arena.sealed.worker`, `evolvets.updater` and `evolvets.updater.interfaces`.
13. **AST guards:** no module under `arena/sealed/` imports `pickle` or calls an attribute named `recv` (only `recv_bytes`). Every `get_context` call passes the literal `"spawn"`. `worker.py` imports from `evolvets` only `evolvets.updater.interfaces`.
14. **Determinism:** two `evaluate_sealed` calls with the same request in two runs give equal `metrics` and equal blob bytes.

### Dependencies

None.

### Size

About 350 lines of implementation. If it grows past ~400, put `remote.py` and `worker.py` with tests 5 to 9 in a first PR (`feat/arena-sealed-worker`, tested by driving `replay_dataset` with the remote factory in-process on the sealed pool), and `server.py`, `client.py` and the remaining tests in a second.

### After PR C merges

The reviewer opens an `fyi-human` issue asking the owner to tag `arena-v1` on PR C's merge commit (TASK_007). From then on, changes to the arena protocol files need a `protocol/` PR.

---

## Not in this task

- **OS-level isolation.** All processes run as the same user, so a worker could still `open()` the sealed files or the salt, and the server's blobs live in the run directory. That is acceptable while only developer-written baselines run (issue #38, default 4). The Phase 1 specs must decide on OS-level isolation (separate user, container or seccomp) before the improver writes updater code.
- **Timing side channel.** The caller can measure how long a sealed evaluation took. Not a concern for developer-written baselines; Phase 1 decides.
- **The live pool** stays the stub in `arena/live.py` until entry 13.
- Hand-designed updaters, tuning, oracles on real data and a CLI that runs sealed evaluations (entries 10 and 11). The value of `SERIES_TIMEOUT_SECONDS` may be revisited in entry 11 before any sealed result exists. It is recorded in every blob, and every condition uses the same value.
- Where the sealed salt and sealed data live when Gate 1 runs (on the owner's machine or in CI with a secret) is decided in entry 11's spec.
- `updater/interfaces.py` remains in an agent-editable package (TASK_007, "Not in this task"). Still a Phase 1 decision.

## Closing the task

PR C (`feat/arena-sealed-server`, or the last PR of its split) changes this file's first line to `Status: done` and sets roadmap entry 9 to its merged PR numbers.
