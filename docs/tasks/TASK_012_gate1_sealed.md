Status: active

# Task 012: Gate 1 on sealed: the sealed run, sealed report sections and the lab workflows

Read `AGENTS.md` and `docs/ROADMAP.md` first. This task covers the **sealed half** of roadmap entry 11 (`exp/gate1`). It is a stack of three PRs. Open them in order and stop after each one for review. It also specifies three `.github/` workflows that **the owner's delegate applies** (AGENTS.md section 7); the developer never writes them.

## Why

TASK_011 built the dev half: recorded tuning runs, the selection rule and a dev report. Gate 1 needs the dev-selected configuration of every baseline evaluated **once** on the sealed pools, the sealed hindsight oracles, and a report that shows those numbers, regenerated from the ledger alone. Everything for a single sealed evaluation already exists (TASK_008, TASK_009 PR D, TASK_010 PR C): `evaluate_sealed`, `evaluate_sealed_oracle`, sealed pools with `pool_spec`, `sealed init-salt`, `sealed build-cache` and `sealed build-foundation-cache`. What is missing:

- PR A: `evolvets gate1 sealed`, one recorded run that takes a completed dev tune run and evaluates the sealed oracles and every dev-selected configuration through the sealed client.
- PR B: the sealed report data: checks and numbers for sealed runs, as dataclasses.
- PR C: rendering the sealed sections in `evolvets gate1 report`.
- Workflows W1 to W3 (owner-applied): create the salt, build the sealed caches, and run the sealed evaluation plus the report on the lab node.

## Decisions already made (do not reopen)

- **Owner's decisions on #94 (2026-10-14):**
  1. Sealed data and the salt live on the lab node's persistent volume under `/data/evolve/sealed/`. The salt file is mode 600. It is **never** a GitHub secret. The owner-dispatched workflow W1 creates it. Sealed workflows are `workflow_dispatch` only, gated by `github.actor == github.repository_owner`, and run on `[self-hosted, gpu]`. They never have a PR, issue or comment trigger, and they never print the salt or any sealed value. The owner backs up the salt once (`docker cp`). A lost salt means a new sealed pool and a new protocol version, never a silent regeneration.
  2. `SERIES_TIMEOUT_SECONDS` stays `600.0` in code. It is revisited from the dev report's per-series wall-clock section before any sealed result exists (see "Before the first sealed run"). If it changes, that is a protocol-version bump inside this task, before the first sealed run.
  3. The sealed foundation caches are built on the lab node by a sealed variant of the cache workflow (W2), with the same owner-only rules. Their data path is on the volume, never in the repository.
  4. `--per-family 20` for the salt.
  5. Entry 13 (live commit) is not planned.
- **TASK_008:** the caller learns only a receipt (`record_id`, `status`). Sealed scores are written to the sealed stream by the server, which is the only budget charge for a sealed evaluation. The caller must not charge it again. The server refuses a run whose main stream already has a `run_end`. The salt path is `<data_root>/sealed/synthetic.salt` (fixed in code).
- **TASK_010 PR C:** a sealed oracle is a sealed `baseline` record with `updater_id = "oracle"` that charges `evaluations = 0`.
- **TASK_011:** `select_configs` is the selection rule. The report never mixes protocol versions, tables only `completed` runs, and lists every other run. `leakage_risk` results get their own section. The oracles bound single-expert selection only; they are not a floor for weighted combinations.

## Rules for all three PRs

- **No ledger, arena or updater change.** No change to `ledger/`, `arena/`, `updater/`, `PROTOCOL_KEYS`, `SCHEMA_VERSION`, `ARENA_VERSION` or any payload. The exception is the conditional timeout PR under "Before the first sealed run", which is a separate `protocol/` PR and needs the owner's approval.
- **Caller blindness.** `runner/` modules import from `evolvets.arena.sealed` only `client` and `names`. The sealed run's process never opens a sealed data file, the salt, anything under the sealed cache root, or the server log. It also never reads the run's **sealed ledger stream**: it learns only receipts. The report (PR B) does read sealed-stream records and their blobs. That is the report's purpose, and it runs as a separate CLI command.
- **Standard library only.** No new dependencies.
- **Strict mypy:** add every new `evolvets.runner.*` module to the strict override list in `pyproject.toml`. Do not weaken any override.
- **Determinism:** loop-sums and loop-means only, as in TASK_011 (no builtin `sum()` on floats, no `math.fsum`, no `statistics`). The bootstrap follows TASK_011 PR C exactly.
- **Agents never run `gate1 sealed`, `sealed init-salt` or a sealed cache build against real data.** Tests use `tmp_path` fixtures. The real runs happen only through W1 to W3, dispatched by the owner.
- Do not edit `.importlinter`, `.github/`, `AGENTS.md` or `docs/agents/`.
- Tests live in `tests/runner/`. Reuse the sealed fixtures of `tests/arena/test_sealed_server.py`, `test_sealed_oracle.py` and `test_sealed_foundation.py` (the real 1200-row fixture, synthetic `per_family = 1`, the fake foundation model) and the tune fixtures of `tests/runner/test_gate1_tune.py`. Move shared fixture builders into `tests/runner/conftest.py` if needed. Copy nothing into `src/`.
- **Time box (each PR):** one developer run is 60 minutes. If you are 40 minutes in and CI is not green locally, open the PR with what works and list what is missing in the description. Write small files first, commit often, and run only the tests of the file you are editing until the end. The sealed tests spawn processes, so use the smallest grids (one or two baselines, one config each) and the synthetic `per_family = 1` fixture unless a test needs real data.

---

## PR A: `feat/gate1-sealed-run`

The recorded sealed run and its CLI.

### Files and interfaces

`src/evolvets/runner/gate1_sealed.py`

```python
SEALED_CONDITION = "gate1-sealed"
SEALED_DATASET_OF: Mapping[str, str] = {  # dev dataset -> sealed dataset
    "etth1": "etth1",
    "ecl": "ecl",
    "synthetic-dev": "synthetic-sealed",
}


@dataclass(frozen=True)
class SealedPlanItem:
    dataset: str  # sealed dataset name
    pool_spec: str
    updater_id: str  # a BASELINE_IDS entry, or "oracle"
    config: dict[str, ConfigValue]  # {} for the oracle
    foundation_models: tuple[str, ...] | None  # None for pool_spec "simple"


@dataclass(frozen=True)
class SealedPlan:
    tune_run_id: str
    tune_run_end_record_id: str
    tune_git_commit: str
    seed: int  # the tune run's seed
    items: tuple[SealedPlanItem, ...]  # all oracles first, then baselines (order below)
    skipped: tuple[
        tuple[str, str, str], ...
    ]  # (sealed dataset, pool_spec, baseline id) with selection None
    previous_sealed_runs: tuple[str, ...]


def plan_from_tune_run(ledger_root: Path, tune_run_id: str) -> SealedPlan: ...


def sealed_run(
    plan: SealedPlan,
    paths: SealedPaths,
    *,
    ledger_root: Path,
    run_id: str,
    git_commit: str,
    max_wall_seconds: float,
) -> str: ...  # returns the run_end status
```

`runner/cli.py` gains `evolvets gate1 sealed`.

### Exact behaviour: `plan_from_tune_run`

Raise `RunError` on every refusal, before anything is written:

1. `LedgerReader(ledger_root, tune_run_id)`: `verify(check_blobs=True)` must pass. A tune run's **main** stream only is read; if the tune run has a sealed stream, refuse.
2. The first record is a `run_start` with `condition == TUNE_CONDITION`. Its config blob has `"experiment": "gate1"` and `"stage": "dev-tune"`. `protocol_versions` equals the current `protocol_versions()`, because a sealed run under other versions could never share a report with its tune run.
3. The last record is a `run_end` with status `completed`. `budget_exhausted`, `aborted` and missing `run_end` are refused, since the selection would be incomplete.
4. `select_configs(<tune records>, read_blob)` equals `run_end.summary["selected"]`.
5. **Items**, in this order:
   - first, one oracle item per `(dataset, pool_spec)`, with dataset in the tune config's `datasets` order and pool spec in its `pool_specs` order, the dataset mapped through `SEALED_DATASET_OF`. A dev dataset missing from the mapping is refused;
   - then, for each `(dataset, pool_spec)` in the same order and each baseline id in `BASELINE_IDS` order, one item with the selected `config` when the selection is not `None`. A `None` selection goes to `skipped`.
   - `foundation_models` is `tuple(config["foundation_models"][dev dataset][pool_spec])` for `simple+foundation` and `None` for `simple`. The sealed pool then joins exactly the models the dev pool joined. The tuple cannot be empty: `load_foundation_caches` refuses an empty foundation pool, and the tune run loads every pool before `Run.start`, so a completed tune run never records an empty list for `simple+foundation`. (A hand-built `()` would still be caught: `_validate_oracle` refuses it as a `SealedError`.)
6. **Previous sealed runs:** scan every run directory directly under `ledger_root` that has a `main.jsonl`. Read only the **main** stream (`LedgerReader(...).records(["main"])`); a run that fails to read is skipped. Keep runs whose `run_start.condition == SEALED_CONDITION` and whose config `tune_run_id == tune_run_id`. If any of them ends `completed`, refuse with `RunError("tune run <id> already has a completed sealed run <id>")`, because a configuration is evaluated on sealed once. The others (aborted, `budget_exhausted`, no `run_end`) go to `previous_sealed_runs`, sorted by run id.

### Exact behaviour: `sealed_run`

1. **Validate** (`ValueError`, nothing written): `max_wall_seconds` is finite and `> 0`, and `plan.items` is non-empty.
2. **Config** (the `run_start` config blob), exactly these keys:
   - `"experiment": "gate1"`, `"stage": "sealed"`;
   - `"tune_run_id"`, `"tune_run_end_record_id"`, `"tune_git_commit"`;
   - `"datasets"` (sealed names, run order) and `"dev_datasets"` (the matching dev names);
   - `"pool_specs"`;
   - `"foundation_models"`: `{sealed dataset: {pool_spec: [names]}}`, with `[]` for `simple`;
   - `"selected"`: `{sealed dataset: {pool_spec: {baseline id: config dict | None}}}`, in `BASELINE_IDS` order;
   - `"updater_refs"`: `{baseline id: "evolvets.updater.baselines:<id>"}`;
   - `"seed"`, `"max_wall_seconds"`, `"series_timeout_seconds": SERIES_TIMEOUT_SECONDS`;
   - `"previous_sealed_runs"`: a list.
3. **Start** `Run.start(..., condition=SEALED_CONDITION, seed=plan.seed, budget_limits=Budget(wall_seconds=max_wall_seconds, evaluations=<number of non-oracle items>), git_commit=git_commit)`. The main stream holds only `run_start` and `run_end`. Every evaluation record is the server's, in the sealed stream. If every baseline selection is `None`, `evaluations` is `0`: `Budget` allows that, and the run is still valid — an oracles-only run.
4. **Loop** over `plan.items` in order, inside the `Run` context. Take `<start> = time.monotonic()` right after `Run.start` returns. The first item always runs; **after each item** (equivalently, before each item after the first), stop — without starting the next item — when `time.monotonic() - <start> >= max_wall_seconds`. The elapsed time is never checked before the first item: unlike TASK_011's tune loop, which checks `run.spent.wall_seconds` (still `0.0` at the first check, since nothing has been charged), the sealed run's main stream charges nothing — every evaluation record is the server's, in the sealed stream — so the loop must time itself against the monotonic clock, and any monotonic clock already reads above `1e-9` at the start. The caller cannot read the sealed budgets, and its own elapsed time bounds their sum from above.
   - For an oracle item, call `evaluate_sealed_oracle(SealedOracleRequest(dataset, pool_spec, foundation_models), ledger_root=..., run_id=..., paths=...)`.
   - For a baseline item, call `evaluate_sealed(SealedRequest(dataset, updater_id, "evolvets.updater.baselines:<id>", dict(config), plan.seed, pool_spec, foundation_models), ...)` with the default `series_timeout_seconds`.
   - Keep each receipt as `{"dataset", "pool_spec", "updater_id", "record_id", "status"}`. An updater `failed`, `timeout` or `invalid_weights` receipt is a result: continue.
5. **End:** call `run.end(status, summary)` with status `completed` if every item ran, else `budget_exhausted`. The summary has exactly the keys `"receipts"` (list, in order), `"skipped"` (list of `{"dataset", "pool_spec", "updater_id"}`), `"oracles"`, `"evaluations"` (ints) and `"statuses"` (`{status: count}` over the baseline receipts, keys sorted). It contains no metric.
6. A `SealedError` (refusal or server failure: a missing sealed cache, an unjoinable foundation cache) ends the run `aborted` through the `Run` context and propagates. Records already in the sealed stream stay; the report lists the run without tabling it. Any other exception, including `KeyboardInterrupt`, does the same.

### Exact behaviour: the CLI

`evolvets gate1 sealed --tune-run ID [--max-wall-seconds 172800] [--ledger-root ledgers] [--run-id ID] [--data-root data] [--manifest-dir data/manifests] [--sealed-cache-root data/sealed_cache]`

- There is no `--seed`: the seed is the tune run's.
- `--run-id` defaults to `new_run_id("gate1-sealed")`. The git commit is `current_git_commit(Path.cwd())`. A `RunError` from it or from `plan_from_tune_run` prints `FAIL <message>`, exits 1 and writes nothing.
- On success, print `OK <run_id> <status>` and then `canonical_json(summary)` on a second line. The summary holds receipts only, never a metric. Exit 0 for `completed`, 2 for `budget_exhausted`. After start, an exception prints `FAIL <run_id> aborted: <message>` and exits 1.

### Tests for PR A (all required; `tests/runner/test_gate1_sealed.py`)

Fixtures: a dev environment and a sealed environment for the same dataset in `tmp_path`. Use the synthetic dev cache with `per_family = 1` plus a sealed salt with `per_family = 1`, the sealed simple cache, and the fake foundation model built for both dev and sealed. Run a completed tune with a small grid, for example `{"equal": [{}], "hedge": [{"eta": 0.5}, {"eta": 1.0}]}`, through `tune(...)`. Pass `git_commit` as a fixed 40-hex string. The CLI tests monkeypatch `current_git_commit`.

1. **Plan:** for a tune over `synthetic-dev` × (`simple`, `simple+foundation`), the plan has two oracle items first, then the selected baselines per pool in `BASELINE_IDS` order with each item's `config` equal to the tune's selection. Datasets are mapped to `synthetic-sealed`. `foundation_models` is `("fake",)` for the pooled items and `None` for `simple`. A baseline with no `ok` grid point (Hedge `{"eta": 0.0}` alone) goes to `skipped`, not `items`.
2. **Plan refusals,** each a `RunError` that writes nothing: a missing tune run; a tampered tune record (chain failure); a `budget_exhausted` tune run; an aborted one; one with no `run_end`; a run whose condition is not `gate1-dev-tune`; a `run_end.summary["selected"]` that differs from the recomputation (hand-built ledger); a tune run with different `protocol_versions` (hand-built); a tune run that has a sealed stream.
3. **Record sequence:** after `sealed_run`, main has exactly `run_start` and `run_end` `completed`. The sealed stream has one oracle `baseline` per pool, then one `baseline` per plan item, in plan order. Their `updater_id` and `hyperparams` match the items, and `evolvets ledger verify` passes on the run.
4. **Same numbers as the client:** each sealed record's `metrics` equal those from calling `evaluate_sealed` (or `evaluate_sealed_oracle`) directly for the same request in a second, hand-started run. This shows that the run adds nothing to the evaluation.
5. **Config:** the `run_start` config has exactly the listed keys. `tune_run_end_record_id` is the tune run's `run_end` record id. `selected` matches the tune selection's configs, with `None` kept. `series_timeout_seconds == 600.0`. `budget_limits.evaluations` equals the number of baseline items, and `condition == "gate1-sealed"`.
6. **Summary:** `receipts` lists every record id of the sealed stream in order, with matching statuses. `skipped` lists the `None` selections. The summary contains no key `metrics` or `mase` at any depth.
7. **Caller blindness:** record file opens in the test process during `sealed_run` with `_record_opens` (as TASK_011 PR B test 12). No opened path lies under the sealed data root's `sealed/` directory, under the sealed cache root, or is a `*.sealed.json` or `sealed.jsonl` file. Use path components, not substrings. AST guard: `runner/gate1_sealed.py` imports from `evolvets.arena.sealed` only `client` and `names`.
8. **Once per tune run:** a second `plan_from_tune_run` after a completed sealed run raises `RunError` naming it. After an aborted sealed run (test 10), the plan succeeds and `previous_sealed_runs` names the aborted run, which then appears in the next run's config.
9. **Budget:** with `max_wall_seconds = 1e-9`, the first item still runs — the elapsed time is checked after each item, never before the first — and the loop then stops. `run_end.status == "budget_exhausted"`, and the receipts hold one entry. With `max_wall_seconds` large enough for every item (for example `1e9`), the same plan ends `completed`, with a receipt for every item.
10. **Aborted:** a sealed cache root missing the pooled foundation cache makes the first `simple+foundation` request raise `SealedError`. The run ends `aborted` with `SealedError` in its summary, `sealed_run` re-raises, and the records written before it are intact.
11. **Updater failure is a result:** a plan with a hand-built item `hedge` with `{"eta": 0.0}` (the worker cannot be monkeypatched, because it is a separate process) gives a `failed` receipt. The run continues to `completed`, and `statuses` counts it.
12. **CLI:** `gate1 sealed --tune-run <id>` on the fixture prints `OK <run_id> completed` and a JSON summary, and exits 0. With `current_git_commit` raising `RunError`, it exits 1, prints `FAIL`, and writes nothing. A second call (a completed sealed run exists) exits 1 with `FAIL` and writes nothing.

### Dependencies

None.

### Size

About 220 lines of implementation. If it grows past ~400 with tests, or the time box hits, put `plan_from_tune_run` with tests 1, 2 and 8 (plan part) in `feat/gate1-sealed-plan`, and `sealed_run`, the CLI and the rest in `feat/gate1-sealed-run`.

### After PR A merges

The reviewer opens an `fyi-human` issue asking the owner's delegate to apply workflows **W1** and **W2** (below) in a `chore(ci)` PR, and the owner to then:

1. dispatch W1 once, back up the salt file (`docker cp <container>:/data/evolve/sealed/data/sealed/synthetic.salt <somewhere off the node>`), and commit the printed `data/manifests/synthetic-sealed.yaml` in a `chore/sealed-manifest` PR (TASK_008 step 3);
2. after that PR merges, dispatch W2 for `simple` on each of `etth1`, `ecl` and `synthetic-sealed`, then for each foundation model the dev tune run joined, on each dataset.

W3 is applied after PR C merges.

---

## PR B: `feat/gate1-sealed-report-data`

The checks and numbers of sealed runs, as dataclasses. No Markdown and no CLI change.

### Files and interfaces

`src/evolvets/runner/gate1_sealed_data.py` holds the sealed logic. It may import the private helpers of `gate1_report_data` (the `_obj_*` readers, `_loop_mean`, `_gap_rows`, `_labeled_differences`, `_best_baseline`, `_oracle_numbers`); do not duplicate the bootstrap. `gate1_report_data.py` changes only where stated.

```python
@dataclass(frozen=True)
class SealedRow:
    baseline: str
    config: dict[str, object] | None  # the dev-selected config; None when skipped
    dev_mase: float | None
    status: str  # an EVALUATION_STATUSES value, or "skipped"
    sealed_mase: float | None  # None unless status == "ok"
    wall_seconds: float | None  # the sealed record's budget; None when skipped


@dataclass(frozen=True)
class SealedPoolSection:
    dataset: str  # sealed name
    dev_dataset: str
    pool_spec: str
    leakage_risk: bool
    cache_id: str
    foundation_cache_ids: tuple[str, ...]
    rows: tuple[SealedRow, ...]  # BASELINE_IDS order
    oracles: OracleNumbers
    dev_best: str | None  # chosen on dev MASE (TASK_011 rule)
    gaps: tuple[GapRow, ...] | None  # vs dev_best; None if dev_best is None or not ok on sealed
    sealed_best: str | None  # lowest sealed MASE among ok rows; ties to BASELINE_IDS order
    hindsight_gaps: (
        tuple[GapRow, ...] | None
    )  # vs sealed_best; None if it equals dev_best or is None
    family_gaps: tuple[FamilyGaps, ...]  # synthetic-sealed only, vs dev_best
```

`ReportData` gains `sealed_sections: tuple[SealedPoolSection, ...] = ()`, `sealed_leakage_sections: tuple[SealedPoolSection, ...] = ()` and `previous_sealed_runs: tuple[str, ...] = ()`. A report over dev runs only gives exactly the same `ReportData` as before; the existing tests pass unchanged.

### Exact behaviour

- **Which reader:** a run whose config has `"stage": "sealed"` is read by the sealed path. Every other run keeps the dev path unchanged, including its refusal of a sealed-stream record. A sealed run must have `run_start.condition == "gate1-sealed"`.
- **Pairing,** each failure a `ReportError` naming the runs:
  - the sealed run's `tune_run_id` must be among the named runs and must be `completed`;
  - its `run_end` record id must equal the sealed config's `tune_run_end_record_id`;
  - the sealed config's `selected` must equal the tune selection's configs, mapped through `SEALED_DATASET_OF` (import it from `gate1_sealed`);
  - two named sealed runs for one tune run are refused.
  - Protocol versions are already checked across all named runs.
- **Complete runs only:** as in TASK_011. An incomplete sealed run appears in Runs and nowhere else.
- **Receipts are the index.** Every sealed-stream record of a `completed` sealed run must appear exactly once in `run_end.summary["receipts"]`, with the same status, and every receipt must name a record. The main stream must hold only `run_start` and `run_end`. A record's `(dataset, pool_spec, updater_id)` comes from its receipt, so a `failed` record with `per_task_blob = None` (a server error) still has a place. Every planned item has a receipt, and every `skipped` entry has a `None` selection.
- **Consistency checks** (`ReportError` on any mismatch):
  - each blob's `result.arena_version` equals the run's `protocol_versions["arena"]`;
  - `result.dataset` and `result.pool_spec` equal the receipt's;
  - `cache_id` and `foundation_cache_ids` are identical across every blob of one `(dataset, pool_spec)`, the oracle's included;
  - an updater blob's `series_timeout_seconds` equals the config's;
  - `hyperparams` equals the selected config;
  - the oracle's per-series ids equal those of every `ok` baseline result, in order.
  - A `timeout` blob has `series == []` and is skipped by the series-id check, as a failed one is.
- **Numbers:**
  - `dev_mase` comes from the tune selection.
  - `sealed_mase` is the sealed record's `metrics["mase"]`.
  - `dev_best` is TASK_011's `_best_baseline` over the tune selection: dev MASE, exact `<`, ties to `BASELINE_IDS` order. The headline gaps never pick a baseline by its sealed score.
  - Gaps against `dev_best` are computed exactly as in TASK_011 PR C, with a new `random.Random(bootstrap_seed)` per Gaps table, and are `None` if `dev_best` is `None` or its sealed status is not `ok`.
  - `sealed_best` and `hindsight_gaps` give a second view chosen in hindsight. It is conservative for the headroom question and is labelled as hindsight in PR C.
  - Per-family gaps are computed for `synthetic-sealed` only, in `FAMILIES` order (series id prefix before `/`), against `dev_best`.
  - The true-best row appears on synthetic pools only.
- **Leakage split:** `leakage_risk` comes from the blobs and must be the same across one pool's blobs; flagged sections go to `sealed_leakage_sections`.
- `previous_sealed_runs` is the union of the named sealed runs' config lists, sorted.

### Tests for PR B (all required; `tests/runner/test_gate1_sealed_data.py`)

Fixtures: the PR A fixture, which gives a completed tune run plus a completed sealed run over `synthetic-dev`/`synthetic-sealed` × (`simple`, `simple+foundation`), plus one real-data pair on the ETTh1 fixture with the `leaky` model (as `real_env` in `tests/arena/test_sealed_foundation.py`). Use hand-built sealed runs (`LedgerWriter` and `BlobStore` in `tmp_path`, copying a real run's records and editing one thing) for the refusals.

1. **Unchanged dev path:** `build_report_data` over the tune run alone equals the result before this PR (compare with a stored `repr` or field-by-field), and the sealed fields are empty.
2. **Numbers trace to records:** each `SealedRow.sealed_mase` equals its sealed record's `metrics["mase"]`, and `dev_mase` equals the tune selection's. `wall_seconds` equals the record's budget. The oracle numbers equal the sealed oracle record's metrics.
3. **Gaps:** the Gaps mean against `dev_best` equals an independent loop-mean of per-series differences computed in the test from the sealed blobs. The interval equals an independent recomputation with `random.Random(seed)`. `dev_best` is the dev-best baseline even when another baseline is better on sealed: build that case by hand. In that case `hindsight_gaps` is set and labelled to `sealed_best`. When the two agree, `hindsight_gaps is None`.
4. **Dev-best not ok on sealed:** a hand-built sealed run where `dev_best`'s record is `timeout` gives `gaps is None`, and its row shows `status == "timeout"` with `sealed_mase is None`.
5. **Skipped:** a tune selection with a `None` baseline gives a `skipped` row with `config is None`.
6. **Per-family and true-best:** present on `synthetic-sealed`, absent on `etth1`.
7. **Leakage split:** the pooled ETTh1 sealed section with `leaky` lands in `sealed_leakage_sections`, and the `simple` one does not.
8. **Incomplete sealed runs:** an aborted sealed run and a `budget_exhausted` one, named with a completed pair, appear in Runs only. `previous_sealed_runs` from the completed run's config is carried through.
9. **Refusals,** each a `ReportError`:
   - a sealed run named without its tune run;
   - a tune `run_end` record id mismatch;
   - `selected` differing from the tune selection;
   - a sealed record missing from the receipts;
   - a receipt naming no record;
   - a receipt status differing from its record;
   - a main-stream record other than `run_start`/`run_end` in a sealed run;
   - a blob `cache_id` differing within one pool;
   - a blob `series_timeout_seconds` differing from the config;
   - `hyperparams` differing from the selected config;
   - mismatched series ids between the oracle and an `ok` result;
   - two sealed runs for one tune run;
   - a sealed-stream record in a dev-tune run (the existing refusal, still raised).
10. **Server-error record:** a hand-built `failed` sealed record with `per_task_blob = None` and a matching receipt builds without error, and its row shows `failed`.
11. **Import guard:** `runner/gate1_sealed_data.py` imports from `evolvets.arena.sealed` at most `client` and `names`. A fresh subprocess importing it loads neither `evolvets.arena.sealed.server` nor `evolvets.arena.sealed.pools`.

### Dependencies

None.

### Size

About 300 lines of implementation. If the time box hits, put the pairing, receipt and consistency checks with tests 9 and 10 in `feat/gate1-sealed-checks`, and the numbers with the rest in `feat/gate1-sealed-report-data`.

---

## PR C: `feat/gate1-sealed-report`

Rendering the sealed sections. `build_report` and the `gate1 report` CLI keep their signatures and arguments.

### Exact behaviour

- If no sealed run is named, the output is **byte-identical** to before this PR.
- With a sealed run named:
  - the title is `# Gate 1 report` instead of `# Gate 1 report (dev)`;
  - after `## Leakage-risk results` and before `## Per-series wall-clock`, insert:
    1. `## Sealed results`, opened by fixed sentences: each baseline was evaluated on the sealed pools exactly once, with the configuration it was tuned to on dev; selection used dev numbers only; and the sealed values were served by the sealed server, which the run's process never read. Then, if `previous_sealed_runs` is non-empty, the sentence `Earlier sealed attempts for this tune run (not tabled): <ids>.`
    2. One `### Sealed: <dataset>, pool <pool_spec>` subsection per non-leakage section, in config order, containing:
       - a **Sealed baselines** table: baseline, dev-selected config (`none` if skipped), dev MASE, sealed status, sealed MASE (`n/a` unless ok), wall seconds (`n/a` if skipped);
       - the **Oracles** table, with the same renderer as dev;
       - **Gaps vs the dev-chosen baseline (`<id>`)**, with the same renderer as dev. If `gaps is None`, show the fixed sentence `The dev-chosen baseline has no ok sealed result; no gaps.`, or the existing no-ok-baseline sentence when `dev_best is None`;
       - if `hindsight_gaps` is set, **Gaps vs the sealed-best baseline (`<id>`, chosen in hindsight)**;
       - per-family Gaps on `synthetic-sealed`;
       - one line with the sealed `cache_id` and `foundation_cache_ids`.
    3. `## Sealed leakage-risk results`, with the same subsections and the same one-sentence introduction as the dev leakage section, or `No sealed leakage-risk results.`
  - `## Notes` gains fixed sentences: the hindsight gaps pick a baseline by its sealed score and so favour the baselines; a sealed series that exceeded `SERIES_TIMEOUT_SECONDS` is a `timeout` result, not a missing one; and the sealed synthetic salt is revealed only after this report is published.
- All numbers use `f"{x:.4f}"`, as before. The output depends only on the ledger and `bootstrap_seed`.

### Tests for PR C (all required; `tests/runner/test_gate1_report.py`, new tests only)

1. **Dev-only unchanged:** `build_report` over a dev run is byte-identical to the expected output the existing tests check. Pin it with a golden string built before the change, and say so in the PR.
2. **Regenerable:** with a tune and sealed pair, two builds are identical, and a build from a copied ledger directory is identical.
3. **Sections and order:** the title, then `## Sealed results` after `## Leakage-risk results` and before `## Per-series wall-clock`. Sealed subsections follow config order.
4. **Numbers:** the Sealed baselines row of a baseline shows its sealed MASE `.4f` and dev MASE. A skipped baseline shows `none` and `n/a`. A `timeout` row shows `timeout` and `n/a`.
5. **Gaps headings:** the dev-chosen heading names `dev_best`. The hindsight table appears only when `sealed_best != dev_best`. The no-gaps sentence appears in the dev-best-not-ok case.
6. **Leakage:** the pooled ETTh1 `leaky` section appears only under `## Sealed leakage-risk results`.
7. **Notes and attempts:** the hindsight, timeout and salt sentences appear. The earlier-attempts sentence appears exactly when `previous_sealed_runs` is non-empty.
8. **CLI:** `gate1 report --run <tune> --run <sealed> --out PATH` writes the report and exits 0, and `--check` against it exits 0. Naming the sealed run without its tune run prints `FAIL`, exits 1, and leaves `--out` unwritten.

### Dependencies

None.

### Size

About 150 lines of implementation. No split expected.

### After PR C merges

The reviewer opens an `fyi-human` issue asking the owner's delegate to apply workflow **W3** in a `chore(ci)` PR, and pointing at "Before the first sealed run" below.

---

## Owner-applied workflows (`.github/`, applied by the owner's delegate, never by an agent)

The developer does not write these files. The planner specifies them, and the owner's delegate writes and applies them (AGENTS.md section 7). Rules common to all three:

- Trigger: `on: workflow_dispatch` only. No `pull_request`, `push`, `issues`, `issue_comment` or `schedule`. Job condition: `if: github.actor == github.repository_owner`. Runner: `runs-on: [self-hosted, gpu]`.
- **Clean checkout:** use the "Get code" step of `gate1-tune.yml`, but **without the tarball fallback**. If git cannot fetch `github.sha`, the job fails. A tarball commit has a different hash from `github.sha`, so a run recorded from one would pin a commit that does not exist in the repository. (Known issue for the owner: check that the dev tune run's `git_commit` equals a real `main` commit. If the tarball fallback was used, it does not, and the dev tune run should be redone from a git checkout before the sealed run.)
- **Paths:** `base="${EVOLVETS_DATA:-/data}/evolve"`, `SEALED_BASE="$base/sealed"`, `SEALED_DATA="$SEALED_BASE/data"` and `SEALED_CACHE="$SEALED_BASE/cache"`. The salt is then `$SEALED_DATA/sealed/synthetic.salt`, under `/data/evolve/sealed/`, created mode 600 by `init_salt`. Every sealed step runs with `umask 077`, and the first step runs `install -d -m 700 "$SEALED_BASE" "$SEALED_DATA" "$SEALED_CACHE"`. The real sealed data is fetched and prepared into `$SEALED_DATA` (`data verify` against the committed manifests; `data fetch` plus `data prepare` only if verify fails). Nothing sealed is ever written under the checkout, `cache_store/` or `foundation_cache/`.
- **No sealed output:** never `set -x`, never `cat`, `ls -l` or `upload-artifact` anything under `$SEALED_BASE` or `$base/ledgers`, and never print a metric. Allowed output: the CLI's own `OK`/`FAIL` lines (hashes, cache ids, series counts, run ids, receipt statuses) and `sha256sum` of the report file.
- `concurrency` with `cancel-in-progress: false`.

**W1 `sealed-salt.yml` (create the salt once):**

1. Clean checkout.
2. Paths as above.
3. Python environment `gpu-env.sh dev "$base/venv-dev"` with `CUDA_VISIBLE_DEVICES: ""`. CPU only.
4. `evolvets sealed init-salt --per-family 20 --data-root "$SEALED_DATA" --manifest-dir data/manifests`. `20` is fixed in the file, not an input.
5. `stat -c '%a' "$SEALED_DATA/sealed/synthetic.salt"` must print `600`, else fail.
6. Print `data/manifests/synthetic-sealed.yaml`. It holds only the salt's SHA-256, `per_family` and versions, so it may be printed and uploaded as an artifact, and the owner commits it.

Re-dispatching is safe: `init_salt` is a no-op on a matching salt, rewrites a missing manifest from the existing salt, and refuses every other case without changing a file. Once the manifest is committed, a missing salt makes W1 fail with the "restore from backup" message. It never regenerates the salt.

**W2 `sealed-cache.yml` (sealed caches):** inputs `dataset` (choice `etth1`, `ecl`, `synthetic-sealed`), `model` (choice `simple`, `chronos2`, `tirex`, `timesfm25`), `batch_size` (default `32`) and `gpus` (default `1`). Concurrency group `sealed-cache-${{ inputs.dataset }}`.

1. Clean checkout, then paths.
2. For a real dataset, verify the data into `$SEALED_DATA` (and fetch/prepare if needed).
3. `model == simple`: environment `dev`, `CUDA_VISIBLE_DEVICES: ""`, then `evolvets sealed build-cache <dataset> --data-root "$SEALED_DATA" --manifest-dir data/manifests --sealed-cache-root "$SEALED_CACHE"`.
4. Otherwise: `gpu-pick.sh <gpus>`, the model's `fm-*` environment (as `gpu-foundation-cache.yml`), `foundation smoke --device cuda`, then `evolvets sealed build-foundation-cache <dataset> --model <model> --device cuda --batch-size <n> --data-root "$SEALED_DATA" --manifest-dir data/manifests --sealed-cache-root "$SEALED_CACHE"`, and `gpu-release.sh` in an `if: always()` step. No manifest upload.

Builds are no-ops when the cache already exists with equal content.

**W3 `gate1-sealed.yml` (the sealed run and the report):** inputs `tune_run_id` (required), `max_wall_seconds` (default `172800`) and `bootstrap_seed` (default `0`). Concurrency group `gate1-sealed`. `timeout-minutes: 3600`, `CUDA_VISIBLE_DEVICES: ""` (CPU work, as `gate1-tune.yml`).

1. Clean checkout, then paths.
2. Environment `dev`.
3. `evolvets gate1 sealed --tune-run <id> --max-wall-seconds <n> --ledger-root "$base/ledgers" --data-root "$SEALED_DATA" --manifest-dir data/manifests --sealed-cache-root "$SEALED_CACHE"`. Print its two output lines (receipts only). Exit codes 0 and 2 continue; 1 fails the job.
4. Take the run id from the `OK` line. Run `evolvets gate1 report --ledger-root "$base/ledgers" --run <tune_run_id> --run <sealed_run_id> --bootstrap-seed <seed> --out "$base/reports/gate1-<sealed_run_id>.md"` after `install -d -m 700 "$base/reports"`.
5. Print only `sha256sum` of the report. The report is not uploaded. The owner retrieves it with `docker cp`.

---

## Before the first sealed run (W3 dispatch)

1. The dev tune run (issue #88) has ended `completed`, and its `git_commit` is a real commit on `main` (see the tarball note above).
2. **Timeout review** (owner decision 2 on #94): the owner builds the dev report from the tune run and reads `## Per-series wall-clock`. A sealed series replays more origins than its dev counterpart (burn-in through dev plus the sealed segment), and every step crosses a process boundary as JSON, so the margin must be wide.
   - If the largest per-series wall-clock over all dev pools is **at most 60 seconds** (one tenth of `SERIES_TIMEOUT_SECONDS`), the value stays `600.0`. The owner or delegate posts the number on the `fyi-human` issue opened after PR C, and nothing changes.
   - Otherwise, the reviewer (on reading that number) opens a `needs-human` issue with the numbers and a proposed value: 10 times the largest, rounded up to a multiple of 100. If the owner approves, a separate PR `protocol/series-timeout`, labelled `protocol-change`, changes the constant and bumps `ARENA_VERSION`, and the TASK_011 tests for `600.0000` are updated in that PR with the reason. Because the report never mixes protocol versions, **the dev tune run must then be redone** under the new version before the sealed run. The owner weighs that cost.
3. W1 has run, the salt is backed up, `data/manifests/synthetic-sealed.yaml` is committed, and W2 has built every sealed simple cache and every sealed foundation cache the tune run joined.

## After the sealed run

The owner commits the report as `docs/results/gate1.md` in an `exp/gate1-report` PR, naming the two run ids and the report's SHA-256 in the description. Only then is the salt revealed (TASK_008), which the owner does. The **Gate 1 decision** (harden the generator, or proceed to Phase 1) is a research gate. The planner escalates it with a `needs-human` issue holding the headline numbers once the report is committed, and does not plan past it.

## Not in this task

- **The Gate 1 decision** and anything in Phase 1.
- **Entry 13 (`feat/live-commit`):** not planned (owner decision 5 on #94).
- **Tagging:** issue #55 (arena protocol tag) stays the owner's call.
- **OS-level isolation and the timing side channel** (TASK_008, "Not in this task") stay Phase 1 decisions. Only developer-written baselines run on sealed here.

## Closing the task

PR C (or the last PR of its split) changes this file's first line to `Status: done` and sets roadmap entry 11's status to `TASK_011 (dev, PRs ...); TASK_012 (sealed, PRs ...)`. The real sealed run, the committed report and the Gate 1 decision follow outside the developer's stack, as described above.
