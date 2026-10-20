Status: active

# Task 018: the Phase 1 lineage runner (entry 19, item f)

## Why

Entries 16 to 18 built the parts: the improver harness (`proposal` records,
recorded prompts and responses, replay from records), the conformal-ridge
self-model (`prediction` records flushed before evaluation) and the allocator
slot with its seven allocators. Nothing yet drives them as a lineage. This
task adds the propose → predict → allocate → evaluate-under-timeout → accept
loop, the `evolvets phase1` run CLI with the experimental conditions of the
owner's plan and one budget accounting for all of them, the weights-only
object-space condition, and holdout scoring of accepted edits in a run
separate from the lineage. It ends with the pilot the owner dispatches on
sp-1 (shift3, one seed, lineages one at a time, detached) — the ledger that
entry 20's report reads.

## Decisions already made (do not reopen)

- Owner plan #141/#142: the setting is `synthetic-shift3` (+ `synthetic-shift2`
  for the full runs later), pool `simple+foundation` (one evaluation ≈ 163 s
  on shift3, per the headroom scan's results note), option-B forecast-output
  updaters under the `forecast-v2` interface, arena `4`; fresh-seed holdouts
  (`synthetic-shift3-holdout`, `synthetic-shift2-holdout`, scan indexes from
  1000) score accepted edits only and are never used for selection; the
  metrics are best-so-far dev MASE gain per unit compute, self-model
  calibration, safe-inheritance rate.
- Owner answers #143 (carried in TASK_017): LLM `glm-5.3` via
  `https://open.bigmodel.cn/api/anthropic`, key from `EVOLVETS_LLM_API_KEY`
  (the env-var *name* only in any config blob; the value never in a ledger,
  blob or log), budgets are tokens and calls in `run_start.budget_limits`,
  backoff waits recorded but not charged, the pilot comes first (shift3,
  1 seed), lineages run one at a time detached on sp-1, and reproducibility
  means re-running a lineage from its recorded responses with no API call.
- PR #155's review note: the artifact validator is a grammar, not a
  CPU/memory sandbox (`**` is banned but `range(1000000000000)` is
  grammatical), so every evaluation of a child artifact runs in a subprocess
  under a wall deadline *and* an address-space cap.
- The ledger schema (`ledger-schema-v1`) and the arena (`arena-v1`, arena
  version `4`) do not change. `proposal`, `prediction`, `allocation`,
  `evaluation`, `acceptance` and `baseline` records are used exactly as
  TASK_017 and Gate 1 defined them. If any change to a payload or the schema
  seems necessary, stop: that is a `protocol/` PR and a `needs-human` issue.
- `runner/` is instrument code, not agent-editable code; it imports the
  improver, the self-model, the allocator and the arena's dev side, and the
  `.importlinter` contracts stay untouched and green.

## Settling TASK_017's "Notes for the (f) task"

1. **Early history.** The run's first action after `run_start` is one
   evaluation of the *starting artifact* (the pinned reference artifact at
   its placeholder config), recorded as a `baseline` record (`updater_id`
   `"reference"`, or `"weights_reference"` in the weights-only condition),
   charged `evaluations=1` plus wall. The prompt's early history still starts
   thin: `observation_object`'s merged semantics keep generation-0 proposals
   out of the history because the starting artifact is no proposal's child,
   and that is accepted. The runner-side history is different: the self-model
   and the allocator rivals fit *every* evaluated proposal whose status is
   `ok`, generation 0 included, with gain computed against the starting
   artifact's baseline MASE. This asymmetry (prompt history vs fit history)
   is deliberate and is documented in the engine's docstring.
2. **`parent_metrics`.** Hill-climbing only: the lineage always proposes from
   the latest accepted artifact, so the merged `observation_object` rule
   (parent metrics = the most recent accepted proposal's evaluation) is
   exactly right. No branching from non-latest artifacts in this task.
3. **`accepted` pending.** The runner appends the `acceptance` record
   immediately after each `evaluation`, inside the same generation, before
   any later observation can be built — so no observation can ever see a
   pending acceptance. Every evaluated proposal gets exactly one acceptance
   decision, including failures: a non-`ok` evaluation gets an acceptance
   record with `accepted=False`, `rule="hill_climb_v1"`, `new_artifact=None`.
   A crashed lineage is never resumed: the run stays `aborted`, and a fresh
   lineage under the same seed is a new run (append-only ledger, failures
   reported).
4. **Uncharged parse/validate time.** Accepted and bounded: the proposal
   record's wall covers the HTTP calls by the merged proposer's design; the
   parse/validate/blob-write slice is the same code path on same-shaped
   parents in every condition and is millisecond-scale. `run_end.summary`
   carries `wall_seconds_total` (runner-measured, monotonic across the run)
   so the total is auditable; entry 20's report divides by the ledger budget
   sums, not by the run wall.

## Rules for all PRs

- Branches `feat/phase1-lineage`, `feat/phase1-weights-space`,
  `feat/phase1-holdout` from up-to-date `main`, one PR at a time; the
  developer opens, the reviewer squash-merges; stop after opening each PR.
- Commit scopes: `runner` for the engine, evaluator, holdout and CLI;
  `improver` for PR B's artifact and prompt additions. Conventional Commits.
- No edits under `arena/`, `ledger/`, `cache/`, `updater/`, `selfmodel/`,
  `allocator/` (PR B adds to `improver/` only), none under `.github/` (the
  launcher is **W12**, owner-applied, below), none to `.importlinter`.
- Python 3.11 stdlib only in the new modules: `subprocess`, `resource`,
  `os`, `signal` for the evaluator; no new dependency of any weight.
- Tests in `tests/runner/` (new: `test_phase1_eval.py`, `test_phase1.py`,
  `test_phase1_holdout.py`) and `tests/improver/` (additions to
  `test_artifacts.py`), fixtures in `tmp_path`, no real datasets, no LLM
  network calls (a scripted fake client exposes `last_call` like
  TASK_017's tests do). The real runs are W12, owner-dispatched.
- ruff, ruff-format and mypy as everywhere; `improver/` stays in the strict
  override list; the new `runner/` modules follow the existing runner style
  (typed defs, one-line docstrings) under the default profile.
- The ledger is append-only; every run the CLI starts ends with a `run_end`
  (`completed`, `budget_exhausted` or `aborted`) — the `Run` context manager
  guarantees the aborted case on a crash.

## PR A: `feat/phase1-lineage`

The subprocess evaluator and the lineage engine, with the conditions of the
`forecast-v2` object space, plus the CLI. Expect the size rule to split this
into A1 (the evaluator, its tests) and A2 (engine, conditions, CLI, tests);
open A2 only after A1 merges, exactly as TASK_015 split its PR A.

### Files and interfaces

- `src/evolvets/runner/phase1_eval.py` — evaluating one artifact source in a
  fresh subprocess under a wall deadline and an address-space cap.

  ```python
  WORKER_PROTOCOL = "1"           # stamped on every worker reply
  MAX_REPLY_BYTES = 64 << 20      # a reply larger than this is a failure

  @dataclass(frozen=True)
  class EvalOutcome:
      status: str        # "ok" | "failed" | "invalid_weights" | "timeout"
      result_obj: dict[str, object] | None  # result_to_obj(...) when ok
      series_wall_seconds: tuple[float, ...]
      error: str | None
      wall_seconds: float          # parent-measured, whole request

  def evaluate_source(request: Mapping[str, object], *, timeout_seconds: float,
                      memory_gib: float) -> EvalOutcome: ...
  ```

  `request` is the JSON object sent to the child on stdin:
  `{"source": str, "config": {...}, "seed": int, "interface": "forecast-v2" |
  "weights-v1", "dataset": str, "pool_spec": str, "foundation_models": [...],
  "cache_root": str, "data_root": str, "manifest_dir": str,
  "foundation_root": str}`. The child (`_main()` under
  `if __name__ == "__main__":`, run as `python -m evolvets.runner.phase1_eval`)
  reads exactly one request, loads the dev pool with `load_dev_pool`, builds
  the updater with `artifact_updater(source)`, replays it with
  `replay_dataset(factory, config, seed, pool, interface=interface)` through a
  timed factory (per-series wall like gate1's), and writes one JSON reply on
  stdout: `{"worker": WORKER_PROTOCOL, "status": ..., "result": ...,
  "series_wall_seconds": [...], "error": ...}`. The parent spawns the child
  with `start_new_session=True` and
  `preexec_fn` setting `resource.setrlimit(RLIMIT_AS, (cap, cap))` where
  `cap = memory_gib << 30`, then `communicate(request, timeout=...)`:
  - reply ok → `EvalOutcome` from the reply;
  - `TimeoutExpired` → kill the process group with SIGKILL, status
    `"timeout"`, error naming the deadline;
  - nonzero exit, empty/garbage/oversized reply → status `"failed"`, error
    `worker: <exit code or stderr tail, capped at 2000 chars>`.

- `src/evolvets/runner/phase1.py` — the conditions and the loop.

  ```python
  @dataclass(frozen=True)
  class Condition:
      proposer_id: str                 # "llm_edit" | "random_param"
      allocator_id: str                # one of ALLOCATOR_IDS
      interface: str                   # "forecast-v2" | "weights-v1"
      reference_updater_id: str        # "reference" | "weights_reference"
      system_prompt: str               # IMPROVER_SYSTEM_PROMPT in PR A

  CONDITIONS: dict[str, Condition]    # id -> condition, PR A adds the
                                      # forecast-v2 space: predictive (the
                                      # treatment), evaluate_all, fixed,
                                      # random, bandit_ucb, bo_gp, shuffled,
                                      # random_param (the non-LLM track of
                                      # the treatment: random_param proposer,
                                      # predictive allocator). PR B adds
                                      # weights_only.

  @dataclass(frozen=True)
  class LineageSpec:
      condition: str; dataset: str; pool_spec: str
      foundation_models: tuple[str, ...] | None
      seed: int; proposals_per_generation: int; eval_slots: int
      max_evaluations: int; max_generations: int
      budget_limits: Budget
      eval_timeout_seconds: float; eval_memory_gib: float
      client_config: LLMClientConfig | None   # None iff random_param
      replay_of: str | None                    # run id when replaying

  def run_lineage(spec: LineageSpec, *, ledger_root: Path, cache_root: Path,
                  data_root: Path, manifest_dir: Path, foundation_root: Path,
                  run_id: str) -> int: ...
  ```

  `run_lineage` returns the process exit code: 0 for `completed` and
  `budget_exhausted` (exhaustion is the designed end of a lineage), 2
  otherwise.

- `src/evolvets/runner/cli.py` — a `phase1` subcommand with one subcommand
  `lineage`:
  `evolvets phase1 lineage --condition ID --dataset synthetic-shift3
  --pool-spec simple+foundation [--foundation-model NAME ...] --seed 0
  --proposals-per-generation 3 --eval-slots 1 --max-evaluations 16
  --max-generations 64 --llm-max-input-tokens 2000000
  --llm-max-output-tokens 200000 --max-wall-seconds 43200
  --eval-timeout-seconds 1800 --eval-memory-gib 8 --replay-from RUN_ID
  --run-id ID` plus the usual `--ledger-root --cache-root
  --foundation-root --data-root --manifest-dir` with gate1's defaults.
  `--run-id` defaults to `new_run_id(f"phase1-{condition}")`.
- `src/evolvets/runner/phase1_eval.py` and `phase1.py` tests as above.

### Exact behaviour

- **Start.** Load the pool *before* `Run.start` (the gate1 rule: no run
  starts that cannot run). `Run.start(condition=f"phase1-{condition_id}",
  seed, config=<the whole spec plus client-config fields, api_key_env name
  only>, budget_limits, git_commit=current_git_commit(repo),
  requires_predictions=True)`; `protocol_versions()` from `arena.versions`.
  For llm_edit conditions the client is built before `run_start`: a missing
  `EVOLVETS_LLM_API_KEY` fails fast with a clear message and nothing written.
  With `--replay-from RUN_ID`, build `RecordedLLMClient(recorded_responses(
  blobs, records))` from that run instead (same condition, dataset,
  pool_spec and seed required, else refuse), and record
  `"replay_of": RUN_ID` in the config.
- **Generation 0 baseline.** One `evaluate_source` request for the pinned
  reference artifact at its placeholder config; append a `baseline` record
  (`updater_id` from the condition, `hyperparams` = the reference config,
  `pool="dev"`, status/metrics/error from the outcome, `per_task_blob` =
  `{"series_wall_seconds": [...], "result": result_obj}`), charged
  `evaluations=1` and wall. If the baseline itself fails, the run ends
  `failed` (summary says why) — a lineage cannot start without a parent
  score. The runner keeps `current_source`, `current_record_id=None`,
  `current_mase` (from the baseline), `current_gain=0.0`, `records`
  (every record appended so far), `history: list[HistoryPoint]`, `spent`
  (budget accumulator).
- **One generation**, while `spent.evaluations < max_evaluations`,
  `spent.llm_* < limits`, run wall `< max_wall_seconds`, and
  `generation < max_generations`:
  1. *Propose* `proposals_per_generation` times with `propose_and_record(
     run, proposer_id=..., parent_source=current_source,
     parent_record_id=current_record_id, generation, parent_gain=
     current_gain, history_records=records, rng, client, client_config,
     max_repair_rounds=1)` — the merged signature unchanged. Budget is
     re-checked between proposals; a limit binding mid-round ends the round
     with the proposals made so far (a round needs at least one proposal to
     continue, else the run ends `budget_exhausted`).
  2. *Predict* for every proposal: build `ProposalFeatures` (family via
     `artifact_family`, `changed_lines` via `len(diff_lines(...))`, const
     stats via `const_change_stats`, `generation`, `parent_gain` clipped to
     [-1, 1], `is_llm`), refit `ConformalRidgeSelfModel(history)` (defaults),
     `identify()` then `predict(features)`, and
     `append_prediction(run, predictor=..., proposal_record_id=...,
     quantiles=..., wall_seconds=...)` — durable before any evaluation.
  3. *Allocate* once: `make_allocator(allocator_id, k=eval_slots)`; candidates
     carry the predicted quantiles for `predictive`, `shuffled` and (PR B)
     `weights_only`, and `{}` for every prediction-blind rival
     (`evaluate_all`, `fixed`, `random`, `bandit_ucb`, `bo_gp`); history is
     `history`; `evaluations_left = max_evaluations - spent.evaluations`.
     The engine truncates `decision.chosen` to `evaluations_left`
     defensively, then `append_allocation(run, allocator=identify(),
     candidate_ids=[...], chosen_ids=[...], scores=dict(decision.scores),
     wall_seconds=...)`.
  4. *Evaluate* each chosen proposal in order: `evaluate_source` with the
     child source (from its blob), the reference config, the run seed, the
     condition's interface; append an `evaluation` record (`pool="dev"`,
     status/metrics/error from the outcome, `per_task_blob` as the
     baseline's, refers_to the proposal), charged `evaluations=1` and the
     parent-measured wall. A timeout is charged like any evaluation: the
     slot was spent.
  5. *Accept* immediately after each evaluation: `accepted = status == "ok"
     and mase < current_mase` (strict), `rule="hill_climb_v1"`,
     `new_artifact=child_artifact` iff accepted, refers_to
     `[proposal, evaluation]`. On acceptance advance `current_source`,
     `current_record_id`, `current_mase`, and `current_gain = clip(
     parent_mase_at_evaluation - child_mase, -1, 1)`. With `eval_slots=1`
     (the pilot) at most one acceptance per generation; with more, the same
     rule applies per proposal in evaluation order and the lineage advances
     to the last accepted child.
  6. *History*: for each `ok` evaluation append
     `HistoryPoint(features, parent_mase_at_proposal_time - child_mase)` —
     generation 0 included (parent = the baseline MASE); failed, timed-out
     and invalid-weights evaluations join no fit history.
- **End.** `run.end("completed" | "budget_exhausted", summary)` where the
  summary has exactly the keys `condition`, `generations`, `proposals`,
  `predictions`, `allocations`, `evaluations`, `accepted`, `starting_mase`,
  `best_mase`, `budget_spent` (the four budget sums), `wall_seconds_total`,
  `replay_of` (or `null`) and — for a replay — `matches_source_run`
  (true iff the source run's proposal child-artifact digests, evaluation
  statuses and metrics compare equal; the replay re-evaluates for real and
  must be bit-equal). The CLI then runs the same ledger `verify` gate1 runs
  and exits non-zero on violations.
- **The RNG.** One `random.Random(seed)` for the whole run, threaded to the
  proposer and the allocator; no other randomness anywhere (the LLM path
  never draws).
- **Look-ahead, isolation.** The engine touches dev pools only, through
  `load_dev_pool`; `observation_object`'s non-dev guard can therefore never
  fire, and the child worker holds no ledger path at all — the ledger stays
  parent-only.

### Tests for PR A (all required)

1. `evaluate_source` ok path: tiny fixture pool + the reference artifact —
   status `ok`, metrics and per-series detail bit-equal to an in-process
   `replay_dataset` call, `series_wall_seconds` non-empty and finite.
2. `evaluate_source` timeout: a grammatical artifact whose `update` loops
   over `range(1000000000000)` — status `timeout`, bounded wall, the error
   names the deadline, and a later evaluation still works (the kill leaves
   no zombie).
3. `evaluate_source` memory burn: a grammatical artifact allocating a giant
   list — the RLIMIT_AS cap makes it fail inside the deadline (MemoryError
   → worker exit → `failed`), never the parent.
4. `evaluate_source` artifact crash: `raise ValueError` inside `update` —
   status `failed` with the replay's error surfaced.
5. `evaluate_source` worker death: a request with a nonexistent cache path —
   nonzero child exit → `failed` with the `worker:` error.
6. `evaluate_source` garbage reply: a stubbed child printing noise —
   `failed`, no exception escapes.
7. interface pass-through: the reference artifact (a flat-forecast updater)
   under `interface="weights-v1"` — `invalid_weights`.
8. end-to-end tiny lineage, condition `predictive`, scripted fake client,
   2 generations × 2 proposals × 1 slot: ledger order is exactly
   run_start, baseline, then per generation proposal × P, prediction × P,
   allocation, evaluation, acceptance per evaluated proposal; `validate_run`
   with `requires_predictions=True` reports no violations; run ends
   `budget_exhausted` at `max_evaluations`; exit code 0.
9. hill-climb: a scripted better child is accepted (`new_artifact` set,
   next generation's proposals refer to it and edit its source), a worse
   child is recorded `accepted=False` and the current artifact stays.
10. non-ok evaluation: a fake client child that times out (stub
    `evaluate_source`) — evaluation `timeout`, acceptance `False`, the run
    continues, the failed proposal joins no fit history but its prediction
    exists.
11. equal budget: condition `evaluate_all`, `max_evaluations=3`,
    `proposals_per_generation=2` — exactly 3 evaluations charged, the second
    generation evaluates only 1 of its 2 candidates, run ends
    `budget_exhausted`.
12. generation-0 gains: after the first generation, `history` includes the
    evaluated generation-0 proposals with gain against the baseline MASE
    (assert via a spy allocator capturing the history it was handed).
13. prediction-blind rivals: condition `fixed` — the spy allocator's
    candidates carry `{}` quantiles while `prediction` records exist for
    every proposal.
14. budget binding mid-round: a tiny token limit — the round stops between
    proposals, the made proposals still get predictions/allocation/
    evaluation, run ends `budget_exhausted`.
15. crash safety: an exception mid-generation (stubbed evaluator raising) —
    the run ends `aborted`, exit 2, the ledger verifies.
16. replay: re-run a tiny lineage with `--replay-from` — zero llm tokens
    charged anywhere in the replay run, identical child digests,
    `matches_source_run` true.
17. CLI: unknown condition refused; llm_edit without `EVOLVETS_LLM_API_KEY`
   fails before any ledger write; `random_param` needs no key; the config
   blob contains the env-var name but not the value.

### Dependencies

None. Standard library only (`subprocess`, `resource`, `os`, `signal`).

### Size

~750 changed lines expected; split into A1 (evaluator + tests 1–7) and A2
(engine, conditions, CLI + tests 8–17) at the size rule, A2 after A1 merges.

## PR B: `feat/phase1-weights-space`

The weights-only object-space condition: a weights-v1 reference artifact, a
weights system prompt, one keyword on `propose_and_record`, and the
`weights_only` condition in the registry.

### Files and interfaces

- `improver/artifacts.py`:
  `WEIGHTS_REFERENCE_ARTIFACT_CONFIG: Mapping[str, float]` (Fixed-Share's
  placeholder config, `{"eta": 1.0, "alpha": 0.01}` — confirm against
  `updater/baselines.py` `_share_step` while translating) and
  `WEIGHTS_REFERENCE_ARTIFACT_SOURCE: str` — `fixed_share` translated into
  the artifact grammar operation for operation (Fixed-Share only, `update`
  returning weight rows), normalized, pinned by digest in the tests exactly
  as `REFERENCE_ARTIFACT_SOURCE` is.
- `improver/prompt.py`: `WEIGHTS_SYSTEM_PROMPT: str` — the same contract
  with "forecast-v2" → "weights-v1" and one added rule: `update` must return
  weight rows over the experts (finite, non-negative, each horizon summing
  to 1); a flat forecast reply is invalid here.
- `improver/propose.py`: `propose_and_record` gains
  `system_prompt: str = IMPROVER_SYSTEM_PROMPT` (keyword-only), threaded to
  `_llm_proposal`; no other change. Prompt blobs record the system text
  actually sent, so `recorded_responses`/`prompt_key` replay unchanged.
- `runner/phase1.py`: `CONDITIONS["weights_only"] = Condition("llm_edit",
  "predictive", "weights-v1", "weights_reference", WEIGHTS_SYSTEM_PROMPT)`,
  and the engine's generation-0 baseline uses the weights reference source
  and config for it.

### Exact behaviour

- Everything else about the condition is the treatment's: same predictive
  allocator with visible quantiles, same budgets, same loop — the object
  space is the only difference, which is what makes it an ablation.
- The `system_prompt` keyword flows into every prompt object of the
  lineage's proposals; the default leaves every existing caller and test
  byte-identical.
- An edit that makes `update` return a flat forecast evaluates as
  `invalid_weights` under `weights-v1` (the arena's rule, not a new one):
  recorded honestly, accepted=False, joins no fit history. This is the
  mechanism that keeps the object space honest (PR A1's note).

### Tests for PR B (all required)

1. `WEIGHTS_REFERENCE_ARTIFACT_SOURCE` validates, is normalized, and its
   digest matches the pinned constant in the test.
2. Parity: replaying the artifact under `weights-v1` equals replaying the
   class `fixed_share` under `weights-v1` on a fixture pool — bit-equal
   pooled metrics and per-series scores.
3. `WEIGHTS_SYSTEM_PROMPT` states the weights-v1 task and the rows rule;
   the default prompt is unchanged (byte-compare against the merged text).
4. `propose_and_record` default: the prompt blob's system text equals
   `IMPROVER_SYSTEM_PROMPT`; with the weights prompt it equals
   `WEIGHTS_SYSTEM_PROMPT`; both replay through `recorded_responses`.
5. `weights_only` end-to-end on a tiny pool: a rows-output child is
   evaluated `ok` and can be accepted; a flat-forecast child is
   `invalid_weights`, acceptance `False`, no fit-history entry.
6. Registry: `weights_only` uses the predictive allocator with visible
   quantiles (spy allocator sees all five quantile keys).

### Dependencies

None.

### Size

~450 changed lines expected; single PR.

## PR C: `feat/phase1-holdout`

Holdout scoring of accepted edits, in a run separate from the lineage.

### Files and interfaces

- `src/evolvets/runner/phase1_holdout.py`:

  ```python
  def run_holdout(from_run: str, *, dataset: str, pool_spec: str,
                  foundation_models: tuple[str, ...] | None,
                  ledger_root: Path, cache_root: Path, data_root: Path,
                  manifest_dir: Path, foundation_root: Path,
                  run_id: str) -> int: ...
  ```

- `cli.py`: `evolvets phase1 holdout --from-run RUN_ID --dataset
  synthetic-shift3-holdout --pool-spec simple+foundation
  [--foundation-model NAME ...] [--run-id ID]` and the usual path flags;
  `--run-id` defaults to `new_run_id("phase1-holdout")`.

### Exact behaviour

- Read the lineage run (both streams, `LedgerReader`, blobs verified).
  Require exactly one `run_start` whose config names the same dataset family
  the holdout belongs to (shift3 lineage → `synthetic-shift3-holdout`,
  shift2 → `synthetic-shift2-holdout`; anything else refused) and a
  `run_end` of `completed` or `budget_exhausted`; an aborted or missing run
  is refused before anything is written.
- Artifacts to score, in order: the starting artifact (the first proposal's
  `parent_artifact` blob, or the pinned reference source when the lineage
  has no proposals), then every `acceptance` with `accepted=True`, by ledger
  order, each contributing its `new_artifact` blob (asserted equal to that
  proposal's `child_artifact`).
- Load the holdout pool *before* `Run.start`. The run:
  `condition="phase1-holdout"`, seed 0, `requires_predictions=False`,
  config = `{"from_run": ..., "dataset": ..., "pool_spec": ...,
  "foundation_models": [...], "interface": <the lineage's interface>}`.
- Score each artifact with the same `evaluate_source` (the lineage's
  interface, its reference config, seed 0) and append a `baseline` record:
  `updater_id="start"` for the first, then `"accept-01"`, `"accept-02"`, …
  in acceptance order; `hyperparams={}`; `pool="dev"`;
  `per_task_blob={"series_wall_seconds": [...], "result": result_obj,
  "lineage_run_id": ..., "proposal_record_id": ..., "generation": ...}`;
  charged `evaluations=1` and wall. A non-ok holdout score is recorded as
  it is (status, error) — the safe-inheritance rate is entry 20's to compute.
- The holdout run contains only `run_start`, `baseline` records and
  `run_end`: no proposals, predictions, allocations, evaluations or
  acceptances, and it never calls anything in `improver/` — holdout records
  can never feed an observation. End `completed`, exit 0, then the same
  ledger `verify` the lineage CLI runs.

### Tests for PR C (all required)

1. happy path on a tiny finished lineage: one `baseline` per accepted edit
   plus `"start"`, in acceptance order, `pool="dev"`, `evaluations=1`
   charged each; `run_end` `completed`; exit 0; ledger verifies.
2. refuses an aborted lineage (nothing written) and a lineage whose dataset
   family does not match the holdout dataset.
3. a lineage with no proposals: only the `"start"` record, replayed from the
   pinned reference source.
4. a `weights_only` lineage: holdout replays under `weights-v1`; a
   flat-forecast accepted artifact scores `invalid_weights` on the holdout.
5. linkage keys present and correct in every `per_task_blob`
   (`lineage_run_id`, `proposal_record_id`, `generation`).
6. the holdout run has no records of any improver-driven kind and imports
   nothing from `evolvets.improver` (assert the module's imports).
7. CLI: defaults, path flags, missing foundation cache surfaces as a clear
   `ArenaError` before any run starts.

### Dependencies

None.

### Size

~400 changed lines expected; single PR.

## Not in this task

- Entry 20 (`exp/phase1-report`): the report's design waits for real runs.
- Entry 21 (`feat/meta-replay`).
- Any sealed or live evaluation of Phase 1 artifacts (a later gate), any
  protocol change, any new dependency, GPU jobs (the lineage replays cached
  forecasts on CPU only), resuming crashed lineages, and multi-dataset or
  multi-seed runs beyond the pilot (the owner dispatches those with this
  same CLI — nothing here is shift3-specific).

## W12: the detached sp-1 launcher (owner-applied)

A human (or their delegate) adds `.github/scripts/phase1-detached.sh`
following `foundation-detached.sh` line for line — `docker exec -d` survival,
`${EVOLVETS_DATA:-/data}/evolve` base, work dir `phase1-<tag>` with
`status.txt` + `logs/`, clone-then-tarball fetch of `main`,
`PYTHONPATH="$work/src/src"`, `cli()` wrapper, no GPU scripts — and the owner
dispatches the pilot on sp-1 with `EVOLVETS_LLM_API_KEY` already in the
container environment (the script checks it exists and never logs it):

```
for c in predictive evaluate_all fixed random bandit_ucb bo_gp shuffled \
         weights_only random_param; do
  cli phase1 lineage --condition "$c" --dataset synthetic-shift3 \
    --pool-spec simple+foundation --seed 0 \
    --proposals-per-generation 3 --eval-slots 1 --max-evaluations 16 \
    --eval-timeout-seconds 1800 --eval-memory-gib 8 \
    --ledger-root "$base/ledgers" --cache-root "$base/cache_store" \
    --foundation-root "$base/foundation_cache" --data-root "$base/data" \
    --manifest-dir data/manifests   # one at a time, status.txt per step
done
cli phase1 lineage --condition predictive ... --replay-from <predictive run id>
for r in <all nine run ids>; do
  cli phase1 holdout --from-run "$r" --dataset synthetic-shift3-holdout \
    --pool-spec simple+foundation <the same path flags>
done
```

The budget guards (`--llm-max-input-tokens 2000000`,
`--llm-max-output-tokens 200000`, `--max-wall-seconds 43200`) are the CLI
defaults and identical for every condition: equal limits, honest charges,
the evaluation limit designed to bind. The owner posts the run ids and the
workflow URL on the task-closing issue; entry 20's task consumes them.

## Closing the task

1. The developer lands PRs A (possibly A1/A2), B and C in order, one open PR
   at a time.
2. The last merged PR sets `Status: done` on this file's first line and
   sets roadmap entries 19 (and only 19) to its merged PR numbers.
3. The owner's delegate applies **W12**; the owner dispatches the pilot and
   posts the run ids. W12 and the runs are outside the developer's stack;
   the results themselves belong to the entry 20 task.
