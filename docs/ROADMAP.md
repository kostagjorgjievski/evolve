# EvolveTS Roadmap

Owner of this file: the planner agent (`docs/agents/planner.md`). The developer agent implements one PR at a time from the active task file in `docs/tasks/`, and the reviewer agent reviews and squash-merges. Do not start a PR that is not yet specified in a task file. The status column records which task covers each entry, or its merged PR numbers.

## The research questions, in dependency order

1. **Headroom:** can learned adaptation beat the best hand-designed online updaters (Fixed-Share, AdaHedge)? Measured against the true-best forecast on synthetic worlds and the best switching oracle on real data.
2. **Self-prediction:** can the system predict the gain of a change to itself before testing it, and does that accuracy hold as it changes?
3. **Prediction-guided allocation:** does using those predictions improve capability per unit compute, versus fixed improvers, Bayesian optimization, bandits and shuffled predictions, without the system playing it safe?
4. **Self-play on planted mismatches:** does training an improvement model on generated world/system mismatches produce skill that holds up on real systems?
5. **Recursion:** does the improver become better at improving itself (fair races, knockouts, many lineages)?
6. **Transfer:** does improving skill transfer to time-series tasks it never trained on?

## Phase 0: the instrument (current)

Goal: answer question 1 and have every piece of measurement infrastructure frozen and trustworthy before any agent edits anything.

| # | Branch | Content | Status |
|---|---|---|---|
| 1 | `chore/bootstrap` | Repo, tooling, CI, rules | merged |
| 2 | `chore/lockfile-and-probe` | Commit `uv.lock`, CI `uv sync --locked`, tighten violation probe | merged (#2) |
| 3 | `feat/ledger-core` + `feat/ledger-payloads` | Schema v1, canonical JSON, hash-chained append-only writer/reader; all payload kinds | merged (#3, #4) |
| 4 | `feat/ledger-blobs` | Content-addressed blob store, ordering validator, `evolvets ledger verify` | TASK_002 |
| 5 | `feat/generator` | Synthetic world/system mismatch grammar with true-best forecast per world | spec pending |
| 6 | `feat/real-data` | ETTh1 + ECL subset download, manifests, chronological splits | spec pending |
| 7 | `feat/experts-cache` | Five simple experts, look-ahead-safe; prediction cache | spec pending |
| 8 | `feat/arena-replay` | Replay loop, horizon-aware label release, metrics | spec pending |
| 9 | `feat/arena-sealed` | Sealed/live pools in a separate process; scores written straight to the sealed ledger stream | spec pending |
| 10 | `feat/updater-baselines` | Updater interface; equal, FTL, Hedge, Fixed-Share, AdaHedge; oracles | spec pending |
| 11 | `exp/gate1` | Tuning on dev, sealed evaluation, Gate 1 report regenerable from the ledger | spec pending |

**Gate 1 decision:** if headroom over the best tuned baseline is small, harden the generator (PR 5) before Phase 1.

Parallel track, started as soon as PR 8 lands:

| # | Branch | Content |
|---|---|---|
| 12 | `feat/foundation-experts` | Chronos-2, TiRex, Toto, Moirai, TimesFM in the expert pool (needs approval for heavy dependencies and GPU) |
| 13 | `feat/live-commit` | Daily public, timestamped forecast commitments from generation 0 onward |

## Phase 1: the loop with an API LLM (questions 2 and 3)

Improver (proposal program, editable), self-model (per-proposal gain quantiles), allocator slot with rival allocators (BO, bandit, random, shuffled), runner with experimental conditions and equal budget accounting, Goodhart diagnostics. Specs written after Gate 1.

## Phase 2: training our own improvement model (questions 4 and 5)

Time-series-aware LLM (code backbone + ChatTS-style encoder, gain head), supervised warm-up on planted problems and ledgers, RL on fixing, self-play generator, recursion. Specs written after Phase 1 results.

## Phase 3: validation (question 6 and SOTA)

GIFT-Eval and fev-bench, year-long live record, improvement benchmark versus frontier LLMs and humans, transfer to anomaly detection and imputation.

## Standing review criteria

Every PR is checked for: CI green including the violation probe; no look-ahead path; predictions logged before evaluation; seeds from config; budgets accounted; no data or secrets committed; tests that would catch the failure the PR is meant to prevent; size under ~400 lines or an explicit stack.
