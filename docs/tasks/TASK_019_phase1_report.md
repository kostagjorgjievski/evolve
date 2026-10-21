Status: active

# Task 019: the Phase 1 pilot report (entry 20, `exp/phase1-report`)

Read `AGENTS.md` and `docs/ROADMAP.md` first. This task covers roadmap entry
20: the report of the W12 Phase 1 pilot — best-so-far dev MASE gain per unit
of compute across conditions, self-model calibration, and the
safe-inheritance rate — regenerated from ledger records plus a git commit. It
is a stack of three PRs, opened in order, one open PR at a time. **PRs A and
B are report tooling**: they are built and tested on `tmp_path` fixture
ledgers now, before the pilot's numbers exist, exactly as TASK_014 PR E and
TASK_015 PR C were specified against fixtures and only later filled with
real runs. **PR C commits the real report** as
`docs/results/phase1-pilot.md`; it starts only when the owner has posted the
pilot's run ids and the workflow URL (issue #174 asked for exactly this
sequencing), and the real report run itself is the owner's — the developer
never touches the sp-1 ledgers.

## Why

TASK_018 can run the pilot: nine lineage conditions on `synthetic-shift3`
pool `simple+foundation` seed 0 (`predictive`, `evaluate_all`, `fixed`,
`random`, `bandit_ucb`, `bo_gp`, `shuffled`, `weights_only`, `random_param`),
one replay of `predictive`, and one holdout run per lineage. Every number
the pilot produces is in a ledger; what is missing is the reader that turns
those records into the paper's first Phase 1 table, with the refusal
discipline that makes "the ledger is the paper" checkable:

- PR A: `runner/phase1_report_data.py` — the checks and numbers as
  dataclasses, reading lineage, replay and holdout runs.
- PR B: `runner/phase1_report.py` and `evolvets phase1 report` — the
  rendered Markdown and the CLI (`--out` / `--check`, byte-identical
  regeneration).
- PR C: `docs/results/phase1-pilot.md` — the real report, committed with its
  provenance, plus the task-closing status flips.

The report reads ledgers only. It takes no cache root, no data root and no
foundation root, spawns no process and calls no LLM: a copied ledger
directory regenerates the report byte-identically, and that is a required
test, not a hope.

## Decisions already made (do not reopen)

- **Owner plan #142:** the Phase 1 metrics are best-so-far dev MASE gain per
  unit of compute, self-model calibration, and the safe-inheritance rate
  (the share of accepted edits that hurt on the fresh-seed holdout); the
  baselines run at equal budget.
- **Owner issue #174:** entry 20 is next; its task consumes the run ids and
  the workflow URL the owner posts after W12 is dispatched and the pilot
  completes; entry 21 (`feat/meta-replay`) follows.
- **Owner decision #165 (PR #173):** the recorded client is per-key FIFO, so
  a replay of 3-proposal generations is faithful; `matches_source_run`
  false or `timeout_status_mismatches` above zero is an audit finding the
  report states, never something it repairs or hides.
- **TASK_018:** `budget_exhausted` is the designed end of a lineage under
  `max_evaluations` (exit 0, `validate_run` clean) — the report tables it
  like `completed`; `wall_seconds_total` is a run property, and gains divide
  by the ledger `budget_spent` sums, never by it; the holdout scores the
  starting artifact and each accepted edit, nothing else; `weights_only`'s
  reference is the weights reference, so its starting MASE is not the other
  eight conditions'; the lineage baseline is a `baseline` record charged
  `evaluations = 1`, so `budget_spent.evaluations` exceeds the count of
  `evaluation` records by exactly one.
- **TASK_015 / TASK_014 precedent:** report tooling is built and tested on
  fixture ledgers before the real runs exist; the results note lands only
  after the owner posts the runs' provenance. All report rules — tables,
  refusals, bootstrap, rounding — are therefore fixed by this file before
  the pilot's numbers are read by anyone.

## The report's definitions (fixed by this file, before the runs)

One pilot, one table per metric, one row per condition. The named runs are
the nine lineage runs, the one replay run, and the nine holdout runs; the
order of `--run` flags does not matter (rows order by the `CONDITIONS`
registry's iteration order, restricted to the named conditions).

1. **Gain per unit of compute.** Per condition: `gain = starting_mase −
   best_mase`, both recomputed from records (the baseline record's
   `metrics["mase"]`; the running accepted value — the baseline's, then each
   accepted child's evaluation MASE in ledger order) and cross-checked
   against `run_end.summary`. The denominators are the run's `budget_spent`
   sums: `gain / evaluations`, `gain * 1_000_000.0 / (input tokens +
   output tokens)` (`None` when the token sum is 0 — `random_param` proposes
   without tokens), and `gain * 3600.0 / wall_seconds`, each written exactly
   as spelled here. The nine lineages ran under identical `budget_limits`
   (a refusal checks it); their actual spends differ by design, and the
   report's comparison is the explicit ratio, not an assumption.
2. **Gain uncertainty (series bootstrap).** Per condition, the per-series
   differences `start_mase_s − final_mase_s` from the baseline record's and
   the final artifact's evaluation record's `per_task_blob` result objects
   (`result["series"]` entries carry `series_id` and `mase`), through
   `_gap_rows([(condition, differences)], bootstrap_seed)` — one fresh
   `random.Random(bootstrap_seed)` per condition, 2000 resamples, the gate1
   interval indexes. The pilot is one seed and one run per condition: the
   interval is a series-resampling interval only, and the Notes say so.
   Fewer than two series gives `None` (rendered `n/a`). A condition that
   accepted nothing has all-zero differences: mean 0.0, interval [0.0, 0.0].
3. **Self-model calibration.** Over every proposal of the condition that
   has both a prediction record and an `ok` evaluation: realized gain = the
   parent's measured dev MASE at proposal time (the running value above —
   exactly the quantity the engine's fit history stores) − the child's
   evaluation MASE. Coverage is the share of realized gains inside
   `[q05, q95]` (nominal 0.90) and `[q25, q75]` (nominal 0.50); the table
   also shows the loop-mean of `q50`, of the realized gain, and of `q50 −
   realized`. Proposals evaluated `failed`/`timeout`/`invalid_weights` and
   proposals never evaluated (allocated away) are counted in their own
   columns and excluded from every average: the allocator's choices shape
   what is measurable, and the three counts are shown side by side. All
   conditions log predictions; a condition with no proposals has zero
   everywhere and `None` coverages (rendered `n/a`).
4. **Safe-inheritance rate.** From the holdout run paired by
   `from_run`: accepted edit *i* **hurt** when
   `holdout_mase(accept-i) > holdout_mase(accept-(i−1))` (`start` for
   *i* = 1). The rate is `hurt / judged`, where *judged* counts accepted
   edits whose holdout row and whose predecessor's row are both `ok`;
   accepted edits with a not-`ok` row on either side are *unjudgeable*,
   counted and excluded. The section also reports the holdout MASE of
   `start` and of the final accepted artifact and their difference. The
   rate is a share of accepted edits, never of evaluations. A lineage with
   no accepted edits has `judged = 0` and rate `None` (`n/a`).
5. **Replay audit.** One row, never a condition row: the replay run's
   `matches_source_run`, `timeout_status_mismatches`, its token sums (0 by
   construction), the evaluations it honestly re-charged, and its wall
   sum. A false match or a positive mismatch count is stated in fixed
   sentences with their meaning (temperature noise and timeout mismatches
   are audit findings, per #165 and TASK_018 PR A2's notes).

## Rules for all three PRs

- **No protocol change.** No edit under `arena/`, `ledger/`, `cache/`,
  `updater/`, `improver/`, `selfmodel/`, `allocator/`, `.github/`,
  `docs/agents/`, `AGENTS.md` or `.importlinter`. New code is
  `runner/phase1_report_data.py`, `runner/phase1_report.py`, the
  `phase1 report` subcommand, tests under `tests/runner/`, and PR C's
  `docs/results/phase1-pilot.md`.
- **Ledgers only.** The new modules import nothing from `evolvets.arena`,
  `evolvets.improver`, `evolvets.selfmodel`, `evolvets.allocator` or
  `evolvets.updater`. They read records and blobs through `LedgerReader`
  and `BlobStore`, and import `CONDITIONS` from `runner.phase1` for the row
  order (transitive imports of that module are the engine's, not the
  report's). A test pins that the builder opens no file outside the ledger
  root.
- **Reuse, do not duplicate.** Import the private helpers of
  `gate1_report_data` (`_obj_dict`, `_obj_str`, `_obj_float`,
  `_obj_str_list`, `_obj_float_list`, `_loop_sum`, `_loop_mean`,
  `_gap_rows`, `BOOTSTRAP_RESAMPLES`, the interval indexes);
  `gate1_report_data.py` itself does not change. The bootstrap is
  `_gap_rows`'s, never a re-implementation.
- **Float discipline.** Loop sums and loop means, left to right; no builtin
  `sum()` on floats, no `math.fsum`, no `statistics` (TASK_011 rule). The
  divisions are written exactly as the definitions spell them. Counts and
  token sums are ints. Rendered floats use `f"{x:.4f}"` as gate1 does.
- **Determinism.** The report depends only on the ledger content and
  `bootstrap_seed`: no clock, no environment, no randomness of its own, and
  no dict-iteration order the records do not already pin (sort any key the
  records leave unordered).
- **Standard library only.** New `evolvets.runner.*` modules sit under the
  **default** mypy profile (TASK_018's pin); the strict list in
  `pyproject.toml` is untouched.
- **Tests use `tmp_path` fixtures.** Reuse the fixture builders of
  `tests/runner/test_phase1.py` and `tests/runner/test_phase1_holdout.py`
  (the `ScriptedClient`, the stubbed evaluator, the shift3 lineage with two
  accepted climbs); move shared builders into `tests/runner/conftest.py` if
  that reads better. Refusals are tested on hand-built ledgers
  (`LedgerWriter` + `BlobStore`, copying a real fixture run and editing one
  thing). The developer never runs the report against the sp-1 ledgers.
- **Time box (each PR):** one developer run is 60 minutes; if 40 minutes in
  CI is not green locally, open the PR with what works and list what is
  missing in the description.

---

## PR A: `exp/phase1-report-data`

The checks and numbers of the pilot, as dataclasses. No Markdown, no CLI
change.

### Files and interfaces

`src/evolvets/runner/phase1_report_data.py`:

```python
class Phase1ReportError(Exception):
    """Raised when the named runs cannot form one faithful Phase 1 report."""


@dataclass(frozen=True)
class ConditionRow:
    condition: str
    interface: str
    run_id: str
    status: str  # run_end status: "completed" | "budget_exhausted"
    proposals: int
    predictions: int
    allocations: int
    evaluations: int  # evaluation records (the summary's count)
    accepted: int
    starting_mase: float
    best_mase: float
    gain: float  # starting_mase - best_mase
    input_tokens: int
    output_tokens: int
    wall_seconds: float  # budget_spent wall sum
    evaluations_charged: int  # budget_spent evaluations (baseline included)
    gain_per_evaluation: float
    gain_per_million_tokens: float | None  # None when the token sum is 0
    gain_per_wall_hour: float
    gain_interval: GapRow | None  # series bootstrap; None under two series


@dataclass(frozen=True)
class CalibrationRow:
    condition: str
    predictions: int
    realized: int  # ok evaluations paired with a prediction
    not_ok_evaluated: int
    never_evaluated: int
    coverage_90: float | None  # inside [q05, q95]; None when realized == 0
    coverage_50: float | None  # inside [q25, q75]
    mean_q50: float | None
    mean_gain: float | None
    mean_q50_error: float | None  # loop mean of q50 - realized


@dataclass(frozen=True)
class InheritanceRow:
    condition: str
    lineage_run_id: str
    holdout_run_id: str
    accepted: int
    judged: int
    hurt: int
    safe_rate: float | None  # hurt / judged; None when judged == 0
    unjudgeable: int
    not_ok_rows: int  # holdout rows whose status is not ok
    start_holdout_mase: float | None
    final_holdout_mase: float | None
    holdout_gain: float | None  # start - final; None unless both are ok


@dataclass(frozen=True)
class ReplayAudit:
    run_id: str
    source_run_id: str
    matches_source_run: bool
    timeout_status_mismatches: int
    input_tokens: int
    output_tokens: int
    wall_seconds: float
    evaluations_charged: int


@dataclass(frozen=True)
class Phase1ReportData:
    dataset: str
    pool_spec: str
    foundation_models: tuple[str, ...] | None
    seed: int
    git_commits: tuple[str, ...]  # distinct run_start commits, sorted
    rows: tuple[ConditionRow, ...]  # CONDITIONS registry order
    calibration: tuple[CalibrationRow, ...]  # same conditions, same order
    inheritance: tuple[InheritanceRow, ...]  # conditions with a holdout run
    missing_holdout: tuple[str, ...]  # conditions without one, registry order
    replay: ReplayAudit | None
    distinct_forecast_start_mase: tuple[float, ...]  # across forecast-v2 rows
    untabled: tuple[tuple[str, str], ...]  # (run_id, reason), sorted by run id


def build_phase1_report_data(
    ledger_root: Path, run_ids: Sequence[str], *, bootstrap_seed: int
) -> Phase1ReportData: ...
```

### Exact behaviour: reading and classification

Every refusal is a `Phase1ReportError` raised before any number is built,
naming the run (and record) it refuses.

1. **Read.** Each named run id is read through `LedgerReader` and
   `verify(check_blobs=True)` must pass; a duplicate run id, a missing run
   or a failing verify is refused first. Only the main stream exists in
   these runs; a run with any other stream is refused.
2. **Classify** from `run_start`:
   - `payload.condition` must start `"phase1-"`; anything else is refused
     as not a Phase 1 run.
   - `condition == "phase1-holdout"` is a **holdout** run; its config blob
     holds exactly `{from_run, dataset, pool_spec, foundation_models,
     interface}` and `payload.requires_predictions` must be `False`.
   - otherwise the suffix must be a `CONDITIONS` key (unknown → refusal);
     `payload.requires_predictions` must be `True`.
   - a lineage run whose config `replay_of` is not `None` is the **replay**
     run; at most one may be named (two → refusal). Its `replay_of` must
     name a named lineage run (not a holdout, not itself) → otherwise
     refusal.
   - two named lineage runs of one condition are refused (name one).
   - a holdout's `from_run` must name a named lineage run that is not the
     replay (an orphan holdout is refused); two holdouts for one lineage
     are refused.
3. **Tabled or listed.** A lineage run whose last record is a `run_end`
   with status `completed` or `budget_exhausted` is tabled. `aborted` or a
   missing `run_end` puts the run in `untabled` with the reason (the
   summary's `error` when present, else the status, else `"no run_end"`)
   and nowhere else; the same for an aborted holdout or replay. If no
   lineage run remains tabled, refuse: there is nothing to report.
4. **Cross-run consistency** (over tabled runs; refusals):
   - `protocol_versions` equal across every named run;
   - `dataset`, `pool_spec` equal across tabled lineage runs (one table);
   - `budget_limits` equal across tabled lineage runs — the equal-budget
     rule made checkable (the replay's inflated wall limit is expected and
     exempt: it is the source's spend plus headroom);
   - a holdout's config `pool_spec` and `foundation_models` equal its
     lineage's.

### Exact behaviour: per-run checks (refusals)

For each tabled lineage run:

- exactly one `baseline` record, immediately after `run_start`, with
  `updater_id` equal to the condition's `reference_updater_id`, status
  `ok` and a finite `metrics["mase"]` (a not-`ok` baseline ends a run
  `aborted`, so a tabled run cannot carry one — refuse if it does);
- every `prediction` record's `refers_to` names a `proposal` record of the
  run; every `acceptance`'s `refers_to` names exactly that run's
  `proposal` and `evaluation` records, in that order;
- every `per_task_blob` parses as an object whose read fields are finite
  (the `_obj_*` helpers).

For each holdout run:

- the `updater_id` sequence is exactly `"start"`, `"accept-01"`,
  `"accept-02"`, … contiguous from 1, and the count of `accept-NN` rows
  equals the lineage's accepted count (refusal naming both runs);
- each `accept-NN` row's `per_task_blob` linkage names the lineage run, the
  record id of the *N*-th accepted proposal in ledger order, and that
  proposal's generation; the `start` row names the lineage run, and its
  `proposal_record_id` is null exactly when the lineage proposed nothing
  (refusal on any mismatch).

### Exact behaviour: the summary cross-check

For each tabled lineage run, recompute from the records — `condition`,
`proposals`, `predictions`, `allocations`, `evaluations` (evaluation
records only), `accepted`, `starting_mase`, `best_mase`, `replay_of` and
the four `budget_spent` sums, where `budget_spent.evaluations` equals the
charges of the `baseline` and `evaluation` records (baseline included,
hence one more than the `evaluations` count) — and refuse, naming the run
and the key, on any mismatch with `run_end.summary`. The summary is a
derived view; the report's numbers come from the records, and this check is
what makes that a fact rather than a claim.

### Exact behaviour: the numbers

As the definitions section spells them. `best_mase` is the running accepted
value (the baseline's, then each accepted child's evaluation MASE in ledger
order); `starting_mase` the baseline's. The bootstrap differences pair the
baseline record's and the final artifact's evaluation record's
`result["series"]` by `series_id` (equal ids in equal order, else refusal);
`GapRow` comes from `_gap_rows([(condition, differences)],
bootstrap_seed)`. `distinct_forecast_start_mase` is the tuple of distinct
`starting_mase` values over the tabled `forecast-v2` rows (the
`weights-v1` row excluded), first-seen order in registry order. Calibration
and inheritance follow the definitions exactly; `mean_*` are `_loop_mean`s.

### Tests for PR A (all required; `tests/runner/test_phase1_report_data.py`)

Fixtures: lineages built with the `test_phase1.py` machinery over the
shift3 fixture — `predictive` with two accepted climbs, `fixed` with no
acceptance, `random_param`, `weights_only` — each with its holdout run from
the `test_phase1_holdout.py` builders, one replay of `predictive`, one
aborted lineage; refusals on hand-built copies.

1. **Rows and order:** rows follow the `CONDITIONS` registry order over the
   named conditions; every count equals the record-kind count;
   `evaluations_charged` is one more than `evaluations`; `starting_mase`,
   `best_mase`, the token sums and wall come from the records, and
   `evaluate_all`'s token spend exceeds `fixed`'s on the fixtures.
2. **Summary cross-check fires:** hand-built runs with a wrong `best_mase`,
   a wrong token sum and a wrong `accepted` are each refused, naming run
   and key.
3. **Per-compute columns:** `gain` is `starting − best` as written;
   `gain_per_evaluation` divides by `evaluations_charged`;
   `random_param`'s `gain_per_million_tokens` is `None` and the others'
   match values recomputed in the test with the spelled division;
   `gain_per_wall_hour` likewise.
4. **Series bootstrap:** the interval equals an independent test-side
   recomputation with `random.Random(bootstrap_seed)` over the blob's
   per-series differences; the no-acceptance condition gives mean `0.0`
   and interval `[0.0, 0.0]`; a one-series fixture gives `None`.
5. **Starting values:** the forecast-v2 fixtures share one bit-equal
   starting MASE (so `distinct_forecast_start_mase` has one entry) while
   `weights_only`'s differs.
6. **Calibration:** counts of realized, not-ok and never-evaluated over the
   fixture's proposals; coverages equal hand counts inside the quantile
   pairs; means are loop means; a no-proposal lineage (output budget paid
   for none) gives zeros and `None` coverages.
7. **Parent at proposal:** a generation-2 proposal's realized gain uses the
   accepted generation-1 child's MASE, not the baseline's (the two-climb
   fixture).
8. **Inheritance:** `judged`, `hurt` and `safe_rate` follow the chain rule
   over the holdout rows; a not-ok `accept-NN` row (the weights flat child)
   makes that pair `unjudgeable` and is counted in `not_ok_rows`; a
   no-acceptance lineage gives `judged = 0`, `safe_rate = None`; a missing
   holdout names its condition in `missing_holdout` and adds no row.
9. **Replay audit:** the replay run appears in `replay` and in no `rows`
   entry; tokens are 0; `evaluations_charged` is the re-charged count;
   `matches_source_run` and `timeout_status_mismatches` copy `run_end`.
10. **Untabled:** the aborted lineage and a no-`run_end` hand-built run are
    listed with reasons and appear nowhere else; naming only such runs is a
    refusal.
11. **Equal budgets:** a hand-built pair whose `budget_limits` differ is
    refused; the replay's inflated wall limit is not.
12. **One table:** lineage runs disagreeing on `dataset` or `pool_spec` are
    refused.
13. **Pairing refusals:** a duplicate condition, a second replay, two
    holdouts for one lineage, an orphan holdout, a replay whose
    `replay_of` is not named, a holdout/replay mismatch of `pool_spec` or
    `foundation_models` — each refused.
14. **Shape refusals:** a non-`phase1-` run, an unknown condition suffix, a
    lineage with `requires_predictions` false, a holdout with it true, a
    missing or extra `baseline` record, a `prediction` pointing outside
    the run, a holdout whose `updater_id` sequence is not `start`,
    `accept-01`, … contiguous, a holdout whose linkage `proposal_record_id`
    disagrees with the ledger order, mismatched per-series ids between
    baseline and final evaluation — each refused.
15. **Protocol mix:** a hand-built run under a different `protocol_versions`
    is refused.
16. **Ledgers only:** with `builtins.open` wrapped to record paths, every
    path the builder opens lies under the ledger root; an AST walk of the
    module names no `evolvets.arena`, `evolvets.improver`,
    `evolvets.selfmodel`, `evolvets.allocator` or `evolvets.updater`
    module.
17. **Regenerable:** two builds over the same runs are equal, and a build
    over a copied ledger directory is equal.

### Dependencies

None.

### Size

About 450 lines of implementation. If it grows past ~700 with tests or the
time box hits, split `feat/phase1-report-read` (reading, classification,
refusals, `ConditionRow`, `ReplayAudit`; tests 1, 2, 9, 10–16) from
`feat/phase1-report-numbers` (bootstrap, calibration, inheritance; tests
3–8, 17), in that order.

---

## PR B: `exp/phase1-report`

Rendering and the CLI. `build_phase1_report_data` keeps its signature.

### Files and interfaces

`src/evolvets/runner/phase1_report.py`:

```python
def build_phase1_report(data: Phase1ReportData) -> str: ...
```

`runner/cli.py` gains, under the existing `phase1` subcommand:

`evolvets phase1 report --run ID [--run ID ...] [--bootstrap-seed 0]
[--ledger-root ledgers] (--out PATH | --check PATH)`

— the same flag set and semantics as `gate1 report`: the data is built
before anything is written, so a refusal prints one `FAIL` line, exits 1
and leaves `--out` unwritten; `--check` compares the regenerated report
against its file and reports the first differing line. No cache, data,
foundation or manifest flags: the report reads the ledger root only.

### Exact behaviour: the document

- `# Phase 1 report`, then, in order:
  1. `## What this is` — fixed sentences: one lineage per named condition
     on `<dataset>`, pool `<pool_spec>`, seed `<seed>`, under identical
     budget limits; the gain columns divide by the ledger `budget_spent`
     sums; every number regenerates from these ledger records at git
     commits `<git_commits>`. Then `## Runs`, one line per named run: id,
     kind (condition / replay / holdout of `<condition>`), status — every
     run that started is in the report (AGENTS.md 2.8).
  2. `## Conditions` — one table, rows in registry order: condition,
     interface, status, proposals, predictions, evaluations (charged),
     accepted, start MASE, best MASE, gain, gain/evaluation, gain per 1M
     tokens (`n/a` when `None`), gain per wall hour, input tokens, output
     tokens, wall seconds. Floats `.4f`; counts and tokens as ints.
  3. `## Gain uncertainty (series bootstrap)` — one row per condition:
     mean and `[lower, upper]` as gate1 renders them; the section opens
     with the fixed sentence that these are series-resampling intervals
     over the dev pool, not seed replicates.
  4. `## Self-model calibration` — one row per condition: predictions,
     realized, not-ok evaluated, never evaluated, coverage 90 (`n/a` when
     `None`, else `.4f` as a share of 1), coverage 50, mean q50, mean
     realized gain, mean q50 error; opened by the fixed sentence that
     coverage is only measurable where an evaluation happened.
  5. `## Safe inheritance (holdout)` — one row per condition with a
     holdout: accepted, judged, hurt, safe-inheritance rate (`n/a` when
     `None`), unjudgeable, not-ok rows, start holdout MASE, final holdout
     MASE, holdout gain (`n/a` unless both ok). If `missing_holdout` is
     non-empty, the fixed sentence `No holdout run named for: <conditions>.`
  6. `## Replay audit` — the replay row and the budget it re-spent, with
     the fixed sentences: tokens are zero by construction, evaluations are
     honestly re-charged; `matches_source_run` true means the aligned
     records agree bit-for-bit; false, or a positive
     `timeout_status_mismatches`, is an audit finding (temperature noise
     around one prompt, or a timeout that did not reproduce), stated, not
     repaired. No replay named → the fixed sentence saying so.
  7. `## Runs not tabled` — one line per `untabled` entry, or `None.`
  8. `## Notes` — fixed sentences: one seed, one run per condition, so
     cross-condition differences carry no seed-level uncertainty;
     `budget_exhausted` is the designed end of a lineage under its
     evaluation limit, tabled like `completed`; `weights_only` starts from
     the weights reference, so its starting MASE is not the other
     conditions' and each gain is measured against its own condition's
     start; when `distinct_forecast_start_mase` has more than one entry,
     the sentence `The forecast-v2 conditions' starting MASE values are not
     bit-equal: <values>.` appears, and it never appears otherwise;
     `random_param` proposes without LLM tokens, so its gain per 1M tokens
     is `n/a`; the safe-inheritance rate is a share of accepted edits,
     never of evaluations.

### Tests for PR B (all required; `tests/runner/test_phase1_report.py`)

Over the PR A fixtures:

1. **Regenerable:** two builds are byte-identical; a build from a copied
   ledger directory is byte-identical; `--out` then `--check` exits 0;
   `--check` against a one-character edit exits 1 naming the first
   differing line.
2. **Refusal:** a duplicate condition among `--run` flags prints `FAIL`,
   exits 1 and leaves `--out` unwritten.
3. **Sections and order:** the eight headings in the listed order; the
   `## Runs` block lists every named run.
4. **Conditions table:** registry row order; `.4f` floats, int counts,
   `n/a` in `random_param`'s gain-per-1M-tokens cell; a `budget_exhausted`
   run is tabled with its status shown.
5. **Calibration:** coverages as shares `.4f`, `n/a` for the no-proposal
   condition; the three count columns hold the fixture's hand counts.
6. **Inheritance:** the rate is `hurt / judged` rendered `.4f`, `n/a` when
   `judged == 0`; `unjudgeable` and `not_ok_rows` shown; the
   missing-holdout sentence appears exactly when it should.
7. **Replay:** the audit row and the fixed sentences in both the
   `matches_source_run` true and false cases.
8. **Notes:** each fixed sentence appears exactly when its condition
   holds, including the start-values sentence's absence when the values
   agree (the fixture case) and presence on a hand-built variant.
9. **CLI defaults:** `--ledger-root` defaults to `ledgers`,
   `--bootstrap-seed` to `0`; `--out` and `--check` are mutually
   exclusive and one is required, as `gate1 report`'s are.

### Dependencies

PR A. Nothing else.

### Size

About 250 lines of implementation. No split expected.

---

## PR C: `exp/phase1-pilot-note` (starts only when the owner posts the runs)

**Do not open this PR before the owner has posted, on an issue, the nine
lineage run ids, the replay run id, the nine holdout run ids, the workflow
URL of the W12 dispatch, and the report file's SHA-256.** Until then the
stack stops after PR B; the reviewer's merge of PR B is not a trigger.

The owner generates the report wherever the ledgers are — on sp-1 through
the `cli` wrapper of `phase1-detached.sh`, or on any machine holding a copy
of the ledger root — with:

```
evolvets phase1 report --ledger-root <ledger root> \
  --run <predictive> --run <evaluate_all> --run <fixed> --run <random> \
  --run <bandit_ucb> --run <bo_gp> --run <shuffled> --run <weights_only> \
  --run <random_param> --run <replay run id> \
  --run <the nine holdout run ids> \
  --out phase1-pilot.md
```

This task specifies no workflow: if the owner wants a W-numbered report
job, that is their `fyi-human` call, not the developer's.

The PR:

1. commits the generated file verbatim as `docs/results/phase1-pilot.md`,
   prefixed by a provenance header in the voice of
   `docs/results/headroom-scan.md`: what the pilot is, the run ids, the
   workflow URL, the git commits the runs pinned, and the report's
   SHA-256 (which must equal the owner's posted digest — the PR
   description states the equality);
2. appends a `## Reading` section that uses only comparisons fixed here:
   for each rival condition, the ratios `predictive_gain_per_evaluation /
   rival_gain_per_evaluation` and
   `predictive_gain_per_million_tokens / rival_gain_per_million_tokens`
   (each `n/a` when the rival's denominator is 0), with the bootstrap
   intervals' overlap acknowledged; the calibration coverages against
   their nominal 0.90 / 0.50; the safe-inheritance rates. Descriptive
   sentences only — no verdict, no ranking language beyond the ratios,
   and one fixed closing sentence: the continuation of Phase 1 (entry 21)
   is the owner's decision;
3. changes this file's first line to `Status: done` and sets roadmap entry
   20 (and only entry 20) to `merged (#<A>, #<B>, #<C>; results note in
   docs/results/phase1-pilot.md)`.

No code, no tests, no ledger change. The numbers in the note are the
report's, pasted — never retyped or recomputed.

### Size

One file, the header and the reading section. No split.

---

## After PR C: the Phase 1 continuation gate

Once the note is merged, the planner opens a `needs-human` issue holding
the headline numbers — `predictive`'s gain per evaluation and per 1M tokens
against each rival, the calibration coverages, the safe-inheritance rates —
and asks the owner for the Phase 1 continuation decision: entry 21
(`feat/meta-replay`), a re-spec, or more pilots (more seeds or datasets,
dispatched by the owner with the same CLI). That is a research gate: the
planner does not plan entry 21 before the owner answers, and the developer
opens nothing beyond this task in the meantime.

## Not in this task

- Entry 21 (`feat/meta-replay`) and anything past the continuation gate.
- Any change to the lineage engine, the holdout, the conditions, the
  ledger, the arena or any frozen component; any new dependency.
- A report workflow (`.github/`): the owner's call, as above.
- Multi-seed or multi-dataset reports: the builder refuses a dataset mix
  by design; wider matrices are a later owner decision and use this same
  CLI unchanged.
- Sealed or live evaluation of Phase 1 artifacts: a later gate.

## Closing the task

PR C (or the last PR of a split) sets `Status: done` on this file's first
line and roadmap entry 20 to its merged PR numbers. The real report, the
committed note and the continuation decision follow outside the
developer's stack, as described above.
