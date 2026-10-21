# EvolveTS Roadmap

Owner of this file: the planner agent (`docs/agents/planner.md`). The developer agent implements one PR at a time from the active task file in `docs/tasks/`, and the reviewer agent reviews and squash-merges. Do not start a PR that is not yet specified in a task file. The status column records which task covers each entry, or its merged PR numbers.

## The research questions, in dependency order

1. **Headroom:** can learned adaptation beat the best hand-designed online updaters (Fixed-Share, AdaHedge)? Measured against the true-best forecast on synthetic worlds and the best switching oracle on real data.
2. **Self-prediction:** can the system predict the gain of a change to itself before testing it, and does that accuracy hold as it changes?
3. **Prediction-guided allocation:** does using those predictions improve capability per unit compute, versus fixed improvers, Bayesian optimization, bandits and shuffled predictions, without the system playing it safe?
4. **Self-play on planted mismatches:** does training an improvement model on generated world/system mismatches produce skill that holds up on real systems?
5. **Recursion:** does the improver become better at improving itself (fair races, knockouts, many lineages)?
6. **Transfer:** does improving skill transfer to time-series tasks it never trained on?

## Phase 0: the instrument (done)

Goal: answer question 1 and have every piece of measurement infrastructure frozen and trustworthy before any agent edits anything.

| # | Branch | Content | Status |
|---|---|---|---|
| 1 | `chore/bootstrap` | Repo, tooling, CI, rules | merged |
| 2 | `chore/lockfile-and-probe` | Commit `uv.lock`, CI `uv sync --locked`, tighten violation probe | merged (#2) |
| 3 | `feat/ledger-core` + `feat/ledger-payloads` | Schema v1, canonical JSON, hash-chained append-only writer/reader; all payload kinds | merged (#3, #4) |
| 4 | `feat/ledger-blobs` | Content-addressed blob store, ordering validator, `evolvets ledger verify` | merged (#5) |
| 4b | `feat/requires-predictions` | Explicit `requires_predictions` in `run_start`; PR #5 review nits; then human tags `ledger-schema-v1` | merged (#13) |
| 4c | `feat/ledger-baseline-status` | `Baseline` gains `status`, `error`, `per_task_blob` (issue #38, option A); then human tags `ledger-schema-v1` | merged (#41) |
| 5 | `feat/generator-core` + `feat/generator-grammar` | Synthetic world/system mismatch grammar with true-best forecast per world | merged (#18, #19) |
| 6 | `feat/real-data` | ETTh1 + ECL subset download, manifests, chronological splits | merged (#22, #23, #24) |
| 7 | `feat/experts-cache` | Five simple experts, look-ahead-safe; prediction cache | merged (#28, #29, #30) |
| 8 | `feat/arena-replay` | Replay loop, horizon-aware label release, metrics | TASK_007 |
| 9 | `feat/arena-sealed` | Sealed/live pools in a separate process; scores written straight to the sealed ledger stream | merged (#43, #46, #51, #54) |
| 10 | `feat/updater-baselines` | Equal, FTL, Hedge, Fixed-Share, AdaHedge updaters; oracles (interface: entry 8) | merged (#79, #81, #82) |
| 11 | `exp/gate1` | Tuning on dev, sealed evaluation, Gate 1 report regenerable from the ledger | dev half done: TASK_011 (#85, #86, #87, #91, #92); first results in `docs/results/gate1-dev-first-run.md`; sealed half (TASK_012) superseded by the owner on 2026-10-14 (faithful and reproducible results only) |
| 11b | `exp/learned-probe` | Learned-updater probe on dev (research question 1, owner request #110): two learned updaters (`ridge_stack`, `loss_gate`) replayed in recorded runs against the best tuned baseline of the first dev tune run, with per-pool bootstrap verdicts in the dev report and a results note | merged (#112, #113, #114, #115, #117, #119; results note in #128, `docs/results/gate1-learned-probe.md`) |
| 11c | `exp/headroom-scan` | Headroom scan before Phase 1 (owner request #125): seeded generator presets of increasing severity plus a wide replicate as new dev datasets, tuned with `gate1 tune`, and a ranked headroom table (best tuned baseline vs switching oracles, best expert per series, true-best) with paired bootstrap intervals and a recommendation by a pre-fixed rule. Hard real domains (#125 candidate 2) wait for owner decisions | merged (#130, #132, #133, #134; results note in #140, `docs/results/headroom-scan.md`) |

**Gate 1 decision:** if headroom over the best tuned baseline is small, harden the generator (PR 5) before Phase 1. *Decided by the owner on 2026-10-19 (#141, #142): Phase 1 runs on `synthetic-shift3` and `synthetic-shift2`, pool `simple+foundation`, with forecast-output updaters (option B); `synthetic-shift1` and `synthetic-wide` are secondary checks.*

Parallel track (PR 8 has landed; entry 12 approved by the owner on 2026-10-12):

| # | Branch | Content | Status |
|---|---|---|---|
| 12 | `feat/foundation-experts` | Chronos-2, TiRex, TimesFM 2.5 as cached experts, built on the lab GPU (approved; TASK_009). Toto dropped: needs torch 2.7 or Python 3.12 (issue #64). Moirai 2.0 dropped: uni2ts needs torch<2.5 and the license is non-commercial (issue #66). Revisit if the project's torch/Python pins change | merged (#56, #57, #58, #61, #63, #70, #72, #76) |
| 13 | `feat/live-commit` | Daily public, timestamped forecast commitments from generation 0 onward | spec pending |

## Phase 1: the loop with an API LLM (questions 2 and 3) (current)

Improver (proposal program, editable), self-model (per-proposal gain quantiles), allocator slot with rival allocators (BO, bandit, random, shuffled), runner with experimental conditions and equal budget accounting, Goodhart diagnostics.

Owner plan (#142): the object is the updater, under the option-B interface. Level L2 is an LLM improver proposing edits, a self-model predicting each edit's gain before evaluation (logged and flushed first, AGENTS.md 2.3), and an allocator choosing what to evaluate under a fixed evaluation budget. Level L5 revises the allocator and the self-model by replaying the ledger's improvement history against the frozen arena. The baselines run at equal budget: evaluate-everything with replay selection, a fixed allocator, random, a bandit, Bayesian optimisation, and a weights-only object space. The metrics are best-so-far dev MASE gain per unit of compute, self-model calibration, and the safe-inheritance rate (the share of accepted edits that hurt on the fresh-seed holdout). The entries are in #142's order.

| # | Branch | Content | Status |
|---|---|---|---|
| 14 | `protocol/forecast-updaters` + `feat/forecast-updater-reference` | (a) Updater interface `forecast-v2` beside `weights-v1`, `ARENA_VERSION = "4"` with weights-only scores unchanged; reference forecast-output updater `bias_share` | merged (#145, #146, #147) |
| 15 | `feat/holdout-replicates` | (b) Fresh-seed holdout replicates `synthetic-shift2-holdout` and `synthetic-shift3-holdout` (scan indexes from 1000), never tuned on; owner-dispatched cache builds (W11) | merged (#152, #153) |
| 16 | `feat/improver-harness` | (c) LLM improver: recorded prompts and responses as blobs, token accounting, `proposal` records, replay from recorded responses | merged (#155, #156) |
| 17 | `feat/selfmodel` | (d) Self-model: per-proposal gain quantiles, `prediction` records flushed before evaluation | merged (#157) |
| 18 | `feat/allocators` | (e) Allocator slot, the self-model allocator and the rival allocators (fixed, random, bandit, BO, evaluate-everything with replay selection) | merged (#160) |
| 19 | `feat/phase1-runner` | (f) Run CLI with conditions and equal budget accounting; holdout scoring of accepted edits only; detached sp-1 launcher (`.github/scripts/`, owner-applied) | merged (#163, #164, #166, #172, #173) |
| 20 | `exp/phase1-report` | (g) Report: gain per unit of compute, calibration, safe-inheritance rate, regenerable from the ledger | TASK_019 (#174) |
| 21 | `feat/meta-replay` | (h) Meta level: revise the allocator and the self-model by replaying the ledger history against the frozen arena | spec pending: after TASK_019's Phase 1 continuation gate (owner decision on the pilot report) |
| 22 | `feat/practice-worlds` | (L3) Allocator also chooses series, presets and seeds per candidate; the system generates new practice worlds with the seeded grammar, aimed at where the current updater fails. Generated worlds use scan indexes below 1000, so they may coincide with dev scan worlds (they are never part of the holdout); any report comparing dev-scan scores with practice-world scores states this overlap | later task, after (e) |
| 23 | `feat/live-loop` | (L4) Live loop: forecast live public series (ETT/electricity-style feeds), commit forecasts before targets arrive (builds on entry 13), revise from matured outcomes. Same architecture, later stage | later task, after (f) |
| 24 | `feat/meta-improver-strategy` | (L5) Meta level also revises the improver's strategy (prompt, edit types, use of history) by replaying the ledger. The verifier is not freely self-edited: a verifier change must agree with fixed ground-truth anchors, the frozen arena metric on fixed fresh-seed sets the system never generates or selects on | later task, after (h) |

Owner direction (#142, 2026-10-19): the goal is the full self-improving system (L2 to L5), not a paper. TASK_016 does not preclude entries 22 to 24: `forecast-v2` is the object interface for all levels, and the holdout replicates use a reserved seed range (from 1000) that practice-world generation and live-loop revision must not use or select on.

## Phase 2: training our own improvement model (questions 4 and 5)

Time-series-aware LLM (code backbone + ChatTS-style encoder, gain head), supervised warm-up on planted problems and ledgers, RL on fixing, self-play generator, recursion. Specs written after Phase 1 results.

## Phase 3: validation (question 6 and SOTA)

GIFT-Eval and fev-bench, year-long live record, improvement benchmark versus frontier LLMs and humans, transfer to anomaly detection and imputation.

## Standing review criteria

Every PR is checked for: CI green including the violation probe; no look-ahead path; predictions logged before evaluation; seeds from config; budgets accounted; no data or secrets committed; tests that would catch the failure the PR is meant to prevent; size under ~400 lines or an explicit stack.
