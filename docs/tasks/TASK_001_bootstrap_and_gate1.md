# Task 001 — Bootstrap EvolveTS and reach Gate 1

Read `AGENTS.md` first. It contains the git, PR and research-integrity rules for this task.

## Goal

Set up the EvolveTS repository and build the minimum needed to answer **Gate 1**:

> Is there headroom? On cached expert predictions, how far are hand-designed online updaters (Hedge, Fixed-Share, AdaHedge) from the best achievable switching combination, on dev and sealed tasks?

If hand-designed updaters already sit near the ceiling, the project needs a harder setup before anything else is built. This task produces the evidence for that decision.

Out of scope for this task: LLM calls, the improver, the self-model, the allocator, live data. Do not stub them beyond empty packages.

## Step 0 — Inspect before building

1. Check whether an EvolveTS (or TSCode) repository or directory already exists in the workspace.
   - If it does: read its structure, README and any existing evolution loop, archive or evaluation code. Write a short note (in the first PR description) mapping what exists onto the layout below, and **ask before** moving or rewriting existing code.
   - If it does not: create a new repository `evolvets`.
2. Confirm `git`, `uv` and the `gh` CLI are available and authenticated. If `gh` is not authenticated, stop and ask.

## Target layout

```
evolvets/
├── src/evolvets/
│   ├── arena/        # replay, label release, dev/sealed/live pools, metrics
│   ├── cache/        # expert prediction store
│   ├── experts/      # expert forecasters used to fill the cache
│   ├── updater/      # updater interface + hand-designed baselines
│   ├── improver/     # empty package (future)
│   ├── selfmodel/    # empty package (future)
│   ├── allocator/    # empty package (future)
│   ├── ledger/       # append-only record store
│   └── runner/       # CLI entry points
├── tests/
├── data/manifests/
├── configs/
├── reports/
├── .github/
│   ├── workflows/ci.yml
│   ├── pull_request_template.md
│   └── CODEOWNERS
├── AGENTS.md
├── pyproject.toml
├── .pre-commit-config.yaml
├── .importlinter
├── .gitignore
└── .env.example
```

## PR stack

Open these as separate PRs, in order. After each PR, stop and report its link with a short summary, per `AGENTS.md`. Continue to the next only when told.

### PR 1 — `chore/bootstrap`: repository, tooling, rules

- Initialise the repo with the layout above (empty packages with `__init__.py`).
- `pyproject.toml` with `uv`, Python 3.11+, dev dependencies: `ruff`, `mypy`, `pytest`, `pre-commit`, `import-linter`.
- `.pre-commit-config.yaml`: ruff, ruff-format, mypy, trailing whitespace, end-of-file, check-added-large-files (limit 500 KB).
- `.gitignore` covering `data/` (except `data/manifests/`), `cache_store/`, `ledgers/`, `runs/`, `.env`, notebooks' checkpoints.
- `.importlinter` contract: `evolvets.updater`, `evolvets.improver`, `evolvets.selfmodel`, `evolvets.allocator` may not import `evolvets.arena.sealed` or `evolvets.arena.live`.
- GitHub Actions `ci.yml`: on every PR and push to `main`, run ruff, ruff-format check, mypy, import-linter and pytest.
- `.github/pull_request_template.md` with exactly these sections:

  ```
  ## What and why
  ## How it was tested
  ## Part of a stack?
  (Which PR number in the stack, what comes before and after.)
  ## Integrity checklist
  - [ ] No agent-editable package can reach sealed or live data (import-linter passes)
  - [ ] No look-ahead: targets only reach code through arena label release
  - [ ] No frozen component changed (or this is a `protocol/` PR with version bump)
  - [ ] All randomness is seeded from config
  - [ ] No data, cache, ledger, secret or large file committed
  - [ ] No test weakened, skipped or deleted to make CI pass
  ## Notes for the reviewer
  ```

- `CODEOWNERS`: assign `src/evolvets/arena/`, `src/evolvets/ledger/` and `.importlinter` to the human maintainers (ask for their GitHub handles; leave a TODO if not provided).
- Commit `AGENTS.md` as provided.
- Push and create the GitHub repository (private). Then **ask the human** to enable branch protection on `main`: require PRs, require CI to pass, require one approving review, require linear history, disallow force pushes. If `gh` has admin rights, propose the exact `gh api` command and wait for approval before running it.
- One placeholder test so CI runs green.

### PR 2 — `feat/ledger`: append-only ledger, schema v1

- `ledger/schema.py`: a versioned dataclass for one record. Fields:
  - `record_id` (UUID), `schema_version`, `created_at` (UTC)
  - `run_id`, `condition`, `seed`, `git_commit`
  - `kind` (`proposal`, `evaluation`, `prediction`, `baseline`, `correction`)
  - `parent_id`, `updater_hash`, `improver_version` (nullable for now)
  - `prediction` (quantiles, nullable), `dev_score`, `sealed_score` (nullable)
  - `budget` (tokens, seconds, evaluations)
  - `payload` (free-form dict for kind-specific data)
  - `refers_to` (for corrections)
- `ledger/store.py`: append-only JSONL writer with fsync on each write, and a reader. No update or delete API.
- Tests: records round-trip; the store cannot overwrite or delete; a record written before an exception is still on disk.

### PR 3 — `feat/data-and-cache`: datasets, splits, expert cache

- Datasets:
  - **Synthetic regime-switching generator** (`arena/synthetic.py`): piecewise processes with sudden, gradual, recurring and false (noise-only) drift; seeded; dev and sealed tasks come from **disjoint seed ranges**.
  - **ETTh1** and an **Electricity (ECL)** subset (e.g. 20 series). Download scripts write a manifest with source URL, date and SHA-256 per file.
- Chronological splits per real series: earliest segment for expert warm-up, then **dev**, then **sealed**, with a gap of at least one forecast horizon between segments. Record boundaries in the manifest. Leave the most recent segment unused and reserved for future live-style checks.
- Experts (`experts/`), first set of five: Naive, SeasonalNaive, AutoETS (via `statsforecast`), Ridge on lag features, LightGBM on lag features. Each is a class with `fit` / `predict` that only uses data available at the forecast origin.
- `cache/`: run each expert once over every origin of every task and store predictions (Parquet or `.npz`) in `cache_store/`, keyed by `(dataset, series, expert, horizon)`, with a manifest hash. Default horizon H = 24; origins every step.
- Tests: an expert's prediction at origin t is unchanged if data after t is altered (look-ahead test); cache reads match direct predictions on a small fixture.

### PR 4 — `feat/arena`: replay, label release, metrics

- `arena/replay.py`: steps through origins; at each origin gives the updater the experts' cached forecasts and the **feedback that has just matured** (errors for origins whose full horizon has now passed). Targets are never passed directly.
- Pool access: `arena/dev.py` (importable by anyone) and `arena/sealed.py` (served only via a separate process: a small local server or subprocess that accepts an updater reference and returns scores straight to the ledger). Keep the process boundary even though it is simple now.
- Metrics: cumulative prequential MAE normalised per series (MASE-style, scaled by in-sample seasonal-naive error on the warm-up segment), plus recovery time after known change points for synthetic tasks.
- Tests: label release never exposes a target early (including the last H origins); sealed scores are not returned to the calling process.

### PR 5 — `feat/updater-baselines`: interface, baselines, Gate 1 report

- `updater/interfaces.py`: the updater contract, a single pure function plus an init:

  ```python
  def init(n_experts: int, config: dict) -> State: ...
  def update(state: State, feedback: Feedback, context: Context) -> tuple[np.ndarray, State]:
      """Return simplex weights over experts for the current origin, and the new state."""
  ```

  `Feedback` holds per-expert errors for newly matured origins; `Context` holds the current origin index and simple features (recent volatility, seasonality strength). Weights must be non-negative and sum to 1; the arena validates this and scores a violation as a failed evaluation.
- Baselines (each a module implementing the contract):
  1. Equal weights
  2. Follow-the-leader (best recent expert)
  3. Hedge / exponential weights
  4. Fixed-Share
  5. AdaHedge
- Reference ceilings, computed offline and clearly labelled as **oracles, not baselines**:
  - best single expert in hindsight per series;
  - best switching sequence with a small switching penalty (dynamic programming).
- Tune each baseline's hyperparameters on **dev only** with a small grid; record every grid point in the ledger as `kind="baseline"`.
- Run tuned baselines on **sealed** through the sealed process; scores go to the ledger.
- `runner/gate1.py` produces `reports/gate1.md` with:
  - a table per dataset: each baseline's dev and sealed normalised loss, and the two oracle ceilings;
  - **headroom** = gap between the best tuned baseline and the switching oracle, on sealed, with block-bootstrap confidence intervals over series;
  - recovery-time results on synthetic drift types;
  - a plain statement of whether the headroom looks large enough to continue, and what you would change if it doesn't (harder drift, more diverse experts, longer horizons).

## Definition of done

- Five PRs open in order, each with CI green and the template fully filled in.
- `reports/gate1.md` exists on the PR 5 branch and every number in it can be regenerated from the ledger with one command (`uv run evolvets gate1 --from-ledger <path>`).
- A final message to the human: the five PR links, the Gate 1 headline (headroom per dataset with intervals), and anything you were unsure about.
