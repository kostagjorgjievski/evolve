Status: active

# Task 013: Gate 1 redo: a wider eta grid, parallel tune runs, and the sealed report over several tune runs

Read `AGENTS.md`, `docs/ROADMAP.md` and `docs/tasks/TASK_012_gate1_sealed.md` first. This task **supersedes the rest of TASK_012** (issue #105). TASK_012 PR A is merged as #100. TASK_012 PRs B and C have not started, and they are carried here as PRs C and D, with the amendments stated below. Everything else in TASK_012 stays in force unless this file changes it: the owner's decisions on #94, its rules for all PRs, workflows W1 to W3, "Before the first sealed run" and "After the sealed run".

This task is a stack of four PRs. Open them in order and stop after each one for review. PRs A and B are new. PRs C and D are TASK_012 PRs B and C, which have already been specified and reviewed, and are taken over with amendments only. The task also specifies two workflow changes (W4, W5) and an amendment to W3. **The owner's delegate applies these**; the developer never writes them (AGENTS.md section 7).

## Why

The first dev tune run, `gate1-tune-20261013T151300Z-b36261f7` (488 records, `completed`, about 16 h on one core), cannot be used for Gate 1 (issue #105):

1. **Its `git_commit` (`575906467740659c72c91aa7ca8962cc9d771b6c`) is not a commit of this repository.** The lab node could not fetch over git, and `gate1-tune.yml` fell back to a tarball plus a local commit. TASK_012 requires a real commit, so the redo must run from a real git checkout (W4).
2. **Selected configs sit on the edge of the grid.** Hedge picked `eta = 2**-6` (the smallest value) on `etth1` and `ecl`, and Fixed-Share picked `eta = 2**-5` (the second smallest). A baseline tuned at the grid boundary is under-tuned, which would overstate headroom for learned adaptation. PR A extends `ETA_GRID` down to `2**-10`. The dev report also flags any selection that still lands on a grid edge.
3. **Runtime.** `fixed_share` is 66 of the 80 grid points and about 70 % of the wall-clock. The redo is split into one tune run per `(dataset, pool spec)`, each its own recorded run, and the pieces run in parallel on the lab node (W4). The dev report (PR A) and the sealed plan (PR B) then have to accept several tune runs that together cover each `(dataset, pool spec)` exactly once.

The first run stays in the ledger untouched (AGENTS.md 2.7, 2.8). It is never named in a Gate 1 report together with the redo runs, because the report refuses two complete runs over one pool. "After the redo" below says how it is accounted for.

## What changes in TASK_012

- **TASK_012 PR A (merged, #100):** PR B here changes `SealedPlan`, `plan_from_tune_run`, the sealed run's `run_start` config keys (`tune_run_id` → `tune_run_ids`, `tune_run_end_record_id` → `tune_run_end_record_ids`), the "once per tune run" rule (now once per pool) and the `gate1 sealed` CLI (`--tune-run` repeatable). No sealed run exists in any real ledger yet, because W3 has never been applied, so no recorded run carries the old keys.
- **TASK_012 PR B → PR C here:** pairing with several tune runs; every section names its tune run.
- **TASK_012 PR C → PR D here:** the dev-only baseline for "byte-identical" is the output after PR A of this task. The cache-id line names the tune run.
- **W3:** takes several tune run ids, and checks that each tune run's commit is fetchable from this repository.
- **"Before the first sealed run," item 1** now reads: the redo tune runs (W4) have all ended `completed`, and each one's `git_commit` is a real commit of this repository. Item 2's timeout review reads the dev report built from the redo runs (W5).

## Decisions already made (do not reopen)

- Everything under "Decisions already made" in TASK_012, including the owner's decisions on #94.
- **Owner's direction on #105:**
  - the redo runs from a real git checkout (git only, with retries, no tarball fallback);
  - `ETA_GRID` is extended downward to `2**-10`, as a small change recorded here;
  - the redo is split into one recorded tune run per `(dataset, pool spec)`, run in parallel;
  - `gate1 report` and the sealed plan accept several tune runs that together cover each `(dataset, pool spec)` exactly once.
- **The grid is experiment config, not protocol.** The grids are recorded in each tune run's `run_start` config (TASK_011 PR B) and are not in `protocol_versions()`. Changing them bumps no version. Runs under different grids are never tabled together: PR A refuses that.
- **Only `ETA_GRID` changes.** `ALPHA_GRID` and every other grid stay as they are. If a redo selection still sits on a grid edge, nobody extends the grid again without the owner's decision (see "After the redo").

## Rules for all four PRs

- TASK_012's "Rules for all three PRs" apply unchanged: no change to ledger, arena or updater; caller blindness; standard library only; strict mypy for new `evolvets.runner.*` modules; loop-sums only; agents never run `gate1 tune`, `gate1 sealed` or a sealed build on real data; no edits to `.importlinter`, `.github/`, `AGENTS.md` or `docs/agents/`; tests in `tests/runner/` with `tmp_path` fixtures; the 60-minute time box.
- **Existing tests:** a test that pins behaviour this task changes on purpose is updated in the same PR, in its own commit, and the PR description names it with the reason. Examples: the 80-point grid size, and `tune_run_id` in the sealed config. No other existing test is edited.

---

## PR A: `exp/gate1-eta-grid`

The wider grid, consistency checks across several tune runs in the dev report, and a grid-edge flag. This is the only PR the redo needs: W4 can be dispatched once it merges, while PRs B to D proceed.

### Files and interfaces

`src/evolvets/runner/gate1_grids.py`

```python
#: Hedge's learning-rate grid, ``2**-10 .. 2**4``.
ETA_GRID: tuple[float, ...] = (
    0.0009765625,
    0.001953125,
    0.00390625,
    0.0078125,
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
)  # 15 values; ALPHA_GRID unchanged, so fixed_share has 90 points and the grids 108

#: Config keys that every tune run tabled together must share.
TUNE_SET_KEYS: tuple[str, ...] = ("grids", "selection_metric", "seed", "updater_refs")


@dataclass(frozen=True)
class TuneRunInfo:
    run_id: str
    git_commit: str
    config: Mapping[str, object]  # the run_start config blob


def tune_set_mismatch(runs: Sequence[TuneRunInfo]) -> str | None: ...


def grid_edges(
    grid: Sequence[Mapping[str, object]], config: Mapping[str, object]
) -> tuple[tuple[str, str], ...]: ...  # ((param, "lowest" | "highest"), ...)
```

`TunedRow` (in `gate1_report_data.py`) gains a last field, `edges: tuple[tuple[str, str], ...] = ()`.

### Exact behaviour

- **Grid:** `ETA_GRID` holds exactly `2.0**k` for `k = -10 .. 4`, in ascending order (the literals above are exact binary fractions). `GATE1_GRIDS` is otherwise built as before, so Hedge has 15 points and Fixed-Share has 90 (eta outer, alpha inner, as now). Nothing else in `gate1_tune.py` changes. A run records the new grids in its `run_start` config automatically.
- **`tune_set_mismatch`** returns `None` when the runs can be tabled together. Otherwise it returns a message naming the two runs and what differs. It checks, in this order, against the first run:
  1. `git_commit` is equal;
  2. for each key of `TUNE_SET_KEYS` in order, `config[key]` is equal (`==` on the parsed JSON). A key missing from any run, the first one included, is a mismatch that names that run and the key;
  3. no `(dataset, pool spec)` from `config["datasets"] × config["pool_specs"]` appears in two runs. The message names the pool and both runs.

  An empty sequence returns `None`. One run is checked for missing keys only (check 2), since there is nothing to compare it with. The function reads nothing from disk.
- **Dev report:** `build_report_data` collects a `TuneRunInfo` for every named run that is `completed` and has `stage == "dev-tune"`, in the order named. Once every named run has been read and the existing per-run checks (protocol versions included) have passed, and before any section is built, it calls `tune_set_mismatch` on them. A mismatch raises `ReportError(message)`. Incomplete runs are not part of the set: they are listed in Runs as now, whatever their grids or seed. The existing "two complete runs cover" refusal stays, unchanged and reachable: check 3 catches the same case first, so the existing test still gets a `ReportError`. Keep both messages naming the pool. Sections keep their current order: runs in the order named, and within a run, the config's order.
- **`grid_edges`:** for each parameter name in `config`, in sorted name order:
  - take the values that parameter has across the grid's points;
  - skip the parameter unless every value is an `int` or a `float` (not `bool`) and there are at least two distinct values;
  - report `"lowest"` if `config[param]` equals the minimum, or `"highest"` if it equals the maximum.

  A config of `{}` gives `()`. Each `TunedRow.edges` is `grid_edges(run config["grids"][baseline], row.config)` when `row.config` is not `None`, else `()`.
- **Rendering (`gate1_report.py`):** after the Tuned baselines table and its blank line, if any row of the section has non-empty `edges`, add one line and then a blank line. The line is exactly:

  `Selected at a grid edge (the grid may be too narrow here): <items>.`

  `<items>` joins one entry per such row, in `BASELINE_IDS` order, with `", "`. Each entry is `<baseline> (<param> <lowest|highest>[, <param> <lowest|highest>])`, for example `hedge (eta lowest), fixed_share (eta lowest, alpha highest)`. Without any edge, the output is byte-identical to before this PR.

### Tests for PR A (all required)

`tests/runner/test_gate1_grids.py`:

1. **Grid sizes:** update the existing size test, in its own commit: sizes `1, 1, 15, 90, 1` in `BASELINE_IDS` order, 108 points, no duplicates, and every config initialises its updater.
2. **Exact eta values:** `ETA_GRID == tuple(2.0**k for k in range(-10, 5))`, strictly ascending. `ALPHA_GRID` equals `(0.0001, 0.001, 0.003, 0.01, 0.03, 0.1)`, so the test shows it is unchanged.
3. **`tune_set_mismatch`:** two runs that agree on every checked field over disjoint pools give `None`. Each of these gives a message naming both run ids and the differing field or pool: a different `git_commit`; for each key in `TUNE_SET_KEYS`, a different value; that key missing from one run; and one pool in both runs. An empty sequence and one run both give `None`.
4. **`grid_edges`:** in Hedge's 15-point grid, `eta = 2**-10` gives `(("eta", "lowest"),)`, `16.0` gives `(("eta", "highest"),)` and `0.25` gives `()`. In the Fixed-Share grid, `{"eta": 2**-10, "alpha": 0.1}` gives `(("alpha", "highest"), ("eta", "lowest"))` (sorted names). A one-point grid (`[{}]` or `[{"eta": 1.0}]`) gives `()`. A string-valued parameter is skipped, and so is a bool-valued one.

`tests/runner/test_gate1_report_data.py`:

5. **Several tune runs:** two completed tune runs from the synthetic fixture, one over `simple` and one over `simple+foundation`, with the same seed, grids and `git_commit`. `build_report_data` gives two sections in the order named. Each equals the section the same pool gets in a report over a single tune run covering both pools: build that run too and compare field by field.
6. **Mismatch refusals:** the same pair with a different seed (pass `seed=8` to one) gives a `ReportError`. So do a different grid (another `grids=`) and a different `git_commit`. An incomplete run (aborted) with a different seed, named next to a completed run, is listed in Runs and refuses nothing.
7. **Edges:** with `grids={"hedge": [{"eta": 0.5}, {"eta": 1.0}]}`, the Hedge row's `edges` is non-empty. It is `(("eta", "lowest"),)` or `(("eta", "highest"),)`, matching the selected eta. With `{"equal": [{}]}`, every row's `edges` is `()`.

`tests/runner/test_gate1_report.py`:

8. **Edge line:** the two-point Hedge report contains the edge line once per section, worded exactly, naming `hedge` and the side that matches its selection. The `{"equal": [{}]}` report contains no line starting with `Selected at a grid edge`.

### Dependencies

None.

### Size

About 90 lines of implementation and 200 of tests. If the time box hits, open the grid change with tests 1 and 2 alone. It unblocks the redo. Put the rest in `exp/gate1-tune-set`, and say so in the description.

### After PR A merges

The reviewer opens an `fyi-human` issue asking the owner's delegate to apply **W4** and **W5** (below) in a `chore(ci)` PR, and the owner to then dispatch W4 once from the merged `main`. The issue also says that the workflow must check out `github.sha` with git (no tarball) and must fail rather than fall back.

---

## PR B: `feat/gate1-sealed-multi-tune`

The sealed plan and the sealed run over several tune runs. This changes TASK_012 PR A's merged code.

### Files and interfaces

`src/evolvets/runner/gate1_sealed.py`:

```python
@dataclass(frozen=True)
class SealedPlan:
    tune_run_ids: tuple[str, ...]  # sorted ascending
    tune_run_end_record_ids: tuple[str, ...]  # same order as tune_run_ids
    tune_git_commit: str  # shared by every tune run
    seed: int  # shared by every tune run
    items: tuple[SealedPlanItem, ...]
    skipped: tuple[tuple[str, str, str], ...]
    previous_sealed_runs: tuple[str, ...]


def plan_from_tune_runs(ledger_root: Path, tune_run_ids: Sequence[str]) -> SealedPlan: ...


def plan_from_tune_run(ledger_root: Path, tune_run_id: str) -> SealedPlan:
    """``plan_from_tune_runs(ledger_root, [tune_run_id])``."""
```

`SealedPlanItem` and `sealed_run`'s signature are unchanged.

### Exact behaviour: `plan_from_tune_runs`

Raise `RunError` on every refusal, before anything is written:

1. `tune_run_ids` is non-empty and has no duplicates.
2. **Each run** passes TASK_012 PR A's checks 1 to 4 unchanged: it verifies, has no sealed stream, has the right condition, stage and current protocol versions, ends `completed`, and its selection recomputes. The first failing run is named. Runs are checked in sorted id order.
3. **The set:** `tune_set_mismatch` (PR A) over the runs, in sorted id order, must return `None`. Otherwise the refusal carries its message. The plan's `seed` is the shared `run_start.seed`, which must also equal `config["seed"]`. `tune_git_commit` is the shared commit.
4. **Rectangular coverage:** let `D` be the union of the runs' datasets and `P` the union of their pool specs. Every `(d, p)` in `D × P` must be covered by exactly one run. Otherwise refuse, naming the first missing pool in the order below. A dev dataset missing from `SEALED_DATASET_OF` is refused, as now.
5. **Order:** datasets in `CACHE_DATASETS` order and pool specs in `POOL_SPECS` order, whatever the run order. Items are all oracles first, then baselines per pool in `BASELINE_IDS` order, as in TASK_012. Each pool's `config` and `foundation_models` come from the run that covers it. For one tune run whose config lists datasets and pool specs in that canonical order, the items are exactly what TASK_012 PR A produced.
6. **Once per pool** (replaces "once per tune run"): scan the ledger as TASK_012 PR A step 6 does. A sealed run is related if its config's `tune_run_ids` is a list that shares at least one id with `tune_run_ids`. If a related run ended `completed`, refuse with `RunError("tune run <id> already has a completed sealed run <id>")`, naming the first shared id. The other related runs go to `previous_sealed_runs`, sorted. A sealed run whose config has no `tune_run_ids` list is not related. No real ledger holds such a run (see "What changes in TASK_012").

### Exact behaviour: `sealed_run` and the CLI

- **Config keys:** TASK_012 PR A's list, with `"tune_run_id"` replaced by `"tune_run_ids"` (a list, in plan order) and `"tune_run_end_record_id"` replaced by `"tune_run_end_record_ids"` (a list, same order). Everything else is unchanged, including `"tune_git_commit"`, `"seed"` and the `datasets`/`pool_specs` axes in plan order.
- **CLI:** `evolvets gate1 sealed --tune-run ID [--tune-run ID ...]`, with `action="append"`, `required=True`. It calls `plan_from_tune_runs`. Output and exit codes are unchanged.

### Tests for PR B (all required; `tests/runner/test_gate1_sealed.py`)

Update the existing tests that read `plan.tune_run_id`, `tune_run_end_record_id` or the config key set, in their own commit (see "Existing tests"). Then add:

1. **Two tune runs, one plan:** tune `synthetic-dev`/`simple` and `synthetic-dev`/`simple+foundation` as two runs, with the same seed, grid and `git_commit`. The plan's `items` and `skipped` equal those of the existing single run over both pools. Its `tune_run_ids` is the two ids sorted, and `tune_run_end_record_ids` matches them.
2. **Order-independent:** `plan_from_tune_runs(root, [b, a])` equals `plan_from_tune_runs(root, [a, b])`.
3. **Set refusals,** each a `RunError` that writes nothing:
   - an empty list;
   - a duplicate id;
   - two runs covering the same pool;
   - a different seed;
   - a different grid;
   - a different `git_commit`;
   - one good run plus one aborted run, with the refusal naming the aborted run;
   - non-rectangular coverage: `synthetic-dev` over both pools plus `etth1` over `simple` only, from the ETTh1 dev fixture. The refusal names `etth1/simple+foundation`. A hand-built ledger is acceptable if the ETTh1 fixture is too slow.
4. **Once per pool:** after a completed sealed run over `[a, b]`, `plan_from_tune_runs(root, [a])` and `plan_from_tune_run(root, b)` both refuse, naming the sealed run. After an aborted sealed run over `[a, b]`, a plan over `[a, b]` succeeds and lists it in `previous_sealed_runs`.
5. **Config:** a sealed run over `[a, b]` has exactly the amended key set. `tune_run_ids == sorted([a, b])`, `tune_run_end_record_ids` are their `run_end` record ids, and there is no key `tune_run_id`.
6. **CLI:** `gate1 sealed --tune-run a --tune-run b` prints `OK <run_id> completed` and exits 0. With a pair of different seeds, it prints `FAIL`, exits 1 and writes nothing.

### Dependencies

None.

### Size

About 100 changed lines of implementation and 200 of tests. If the time box hits, put steps 1 to 5 of the plan with tests 1 to 3 in `feat/gate1-sealed-multi-plan`, and the once-per-pool rule, config, CLI and tests 4 to 6 in `feat/gate1-sealed-multi-tune`.

---

## PR C: `feat/gate1-sealed-report-data`

TASK_012 PR B, exactly as specified there, with these amendments:

- `SealedPoolSection` gains `tune_run_id: str`, placed after `pool_spec`: the dev run that tuned this pool.
- **Pairing** (replaces TASK_012 PR B's pairing list). Each failure is a `ReportError` naming the runs:
  - every id in the sealed config's `tune_run_ids` must be among the named runs and must be `completed`;
  - `tune_run_end_record_ids[i]` must equal the `run_end` record id of `tune_run_ids[i]`;
  - the sealed config's `selected` must equal the union of those tune runs' selections, mapped through `SEALED_DATASET_OF`;
  - when a sealed run is named, every named `completed` dev-tune run must be one of its tune runs;
  - two named sealed runs that share any tune run are refused;
  - the tune-set check of PR A already covers the named tune runs.
- `dev_mase` and `dev_best` for a pool come from the selection of the tune run that covers it.

### Tests for PR C (all required; `tests/runner/test_gate1_sealed_data.py`)

TASK_012 PR B's tests 1 to 11. In test 9, "a sealed run named without its tune run" becomes "named without one of its tune runs". Add:

12. **Two tune runs:** a completed sealed run over the two tune runs of PR B test 1 builds. Each section's `tune_run_id` is the run that covers it, and its `dev_mase` values equal that run's selection.
13. **Refusals,** each a `ReportError`:
    - a named completed dev-tune run that is not in the sealed run's `tune_run_ids`;
    - `tune_run_end_record_ids` with two entries swapped (hand-built);
    - two named sealed runs that share one tune run.

### Size

As TASK_012 PR B, with the same split.

---

## PR D: `feat/gate1-sealed-report`

TASK_012 PR C, exactly as specified there, with these amendments:

- **Byte-identical dev-only output** means identical to the output on `main` after PR A of this task, which includes the grid-edge line. The golden string of TASK_012 PR C's test 1 is built from that.
- The one line per sealed subsection with the cache ids also names the tune run: `Sealed cache <cache_id>; foundation caches <ids or none>; tuned in run <tune_run_id>.`
- TASK_012 PR C's test 8 (CLI) also runs with two tune runs: `gate1 report --run <a> --run <b> --run <sealed> --out PATH` writes the report and exits 0.
- **Closing the task:** this PR (or the last PR of its split) changes this file's first line to `Status: done`. It also sets roadmap entry 11's status to `TASK_011 (dev, PRs ...); TASK_012 (sealed, PR A #100); TASK_013 (redo and sealed report, PRs ...)`.

### Size

As TASK_012 PR C. No split expected.

### After PR D merges

As after TASK_012 PR C: the reviewer opens an `fyi-human` issue asking the owner's delegate to apply **W3 as amended below**, and pointing at TASK_012's "Before the first sealed run" (item 1 as amended above).

---

## Owner-applied workflow changes (`.github/`, applied by the owner's delegate, never by an agent)

### W4: `gate1-tune.yml` (the redo, in parallel)

Changes to the existing workflow:

1. **Get code, git only.** Remove the tarball fallback and its local commit. Fetch `github.sha` with git, retried up to 5 times with a growing delay (for example 10, 20, 40, 80 s), each attempt under the existing timeouts. If every attempt fails, the job fails. After checkout, `test "$(git rev-parse HEAD)" = "${{ github.sha }}"` must pass, or the job fails.
2. **Parallel pieces.** One job, one dispatch. The Tune step starts one `evolvets gate1 tune` process per `(dataset, pool spec)` in `etth1, ecl, synthetic-dev` × `simple, simple+foundation`, six in all, as background processes. Each gets `--dataset <d> --pool-spec <p>`, `--run-id "gate1-tune-<UTC stamp>-<d>-<p with + replaced by ->"` (shared stamp, taken once), the same `--seed`, the same `--max-wall-seconds`, and the existing root arguments. Each writes its log to its own file under `$base/logs/` and prints it when it ends. The step waits for all six and prints one line per piece: run id and exit code. The job fails if any exit code is not 0 or 2.
   - Each process runs with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, so the six do not oversubscribe the node's cores.
   - A new input `pieces` (default `all`) may list a subset as `dataset:pool_spec` pairs separated by spaces. This lets a single failed piece be re-run on its own. The report then names the new run instead of the failed one; the failed run stays in the ledger and is listed.
3. **Concurrency** stays `gate1-tune` with `cancel-in-progress: false`. One runner suffices, because the parallelism is inside the job.
4. Every other rule of the workflow stays: owner-only `workflow_dispatch`, `[self-hosted, gpu]`, `CUDA_VISIBLE_DEVICES: ""`, the ledger on the volume, and `timeout-minutes: 3600`.

Rough cost: the grid grows from 80 to 108 points, so about 1.35 × 16 h ≈ 22 h of CPU in total. The slowest piece bounds the wall-clock (likely an `ecl` piece).

### W5: `gate1-dev-report.yml` (the dev report over several runs)

The `tune_run_id` input becomes `tune_run_ids`: space-separated, and empty means the six newest `gate1-tune-*` runs sharing the newest stamp. The commit-check step checks **every** run's `git_commit` against the repository with `git fetch -q --depth 1 origin <commit>`, and fails if any fetch fails. The report step passes one `--run` per id. Nothing else changes.

### W3 amendment: `gate1-sealed.yml`

- The input `tune_run_id` becomes `tune_run_ids` (space-separated, required).
- Before step 3, each tune run's `git_commit` is read from its `main.jsonl` (as W5 does) and fetched with `git fetch -q --depth 1 origin <commit>`. A failed fetch fails the job, before anything is written. This is the guard against a tarball commit like the first run's.
- Step 3 passes one `--tune-run` per id. Step 4 passes one `--run` per tune id, in the given order, then `--run <sealed_run_id>`.

---

## After the redo

1. W4 has ended with all six pieces `completed` (exit 0). W5 then builds the dev report over the six run ids.
2. **Grid edges:** if the report still shows a grid-edge line for Hedge or Fixed-Share at the **lowest** eta (now `2**-10`), or any edge that the owner considers material, the reviewer or the owner's delegate opens a `needs-human` issue quoting the line. The grid is not extended again without the owner's decision. Extending it would mean another redo.
3. **Timeout review:** TASK_012's "Before the first sealed run," item 2, reads `## Per-series wall-clock` of this report. The six pieces share the node's CPU, so their per-series times may be longer than they would be run one at a time. That makes the review conservative, not lenient.
4. **The first run:** `gate1-tune-20261013T151300Z-b36261f7` stays in the ledger and is never tabled. The `exp/gate1-report` PR (TASK_012 "After the sealed run") names it in its description, with the two reasons it was superseded: a non-repository commit and grid-edge selections. That way every run that started is accounted for (AGENTS.md 2.8).

## Not in this task

- **The Gate 1 decision** and anything in Phase 1: a research gate, escalated after the report is committed (TASK_012).
- **Changing any grid other than `ETA_GRID`**, or extending it further.
- **Splitting a tune run below `(dataset, pool spec)`**, for example per baseline. If one piece is still too slow, the owner decides.
- **A report option to list superseded completed runs.** The first run is accounted for in the results PR description instead.
- **Entry 13 (`feat/live-commit`)** and tagging (#55), as in TASK_012.
