# EvolveTS

EvolveTS is the self-evolution component of TimeSeries Code (TSCode). It studies whether a time-series system can learn to improve itself: an improver proposes edits to the system around frozen forecasters, a self-model predicts each edit's gain before it is tested, an allocator spends the evaluation budget, and a sealed arena scores the result. Every attempt is recorded in an append-only ledger.

Read [`AGENTS.md`](AGENTS.md) before contributing. It holds the git workflow, PR rules and the research-integrity rules.

## Layout

| Package | Role |
|---|---|
| `arena/` | Replay, horizon-aware label release, dev/sealed/live pools, metrics |
| `cache/` | Store of expert predictions, computed once |
| `experts/` | Forecasters that fill the cache |
| `updater/` | The editable rule that combines experts over time, plus hand-designed baselines |
| `improver/` | Proposes edits (future) |
| `selfmodel/` | Predicts an edit's gain before evaluation (future) |
| `allocator/` | Chooses which edits get evaluated (future) |
| `ledger/` | Append-only record of every proposal, prediction and result |
| `runner/` | CLI entry points |

## Setup

```bash
uv sync --all-groups
uv run pre-commit install
```

## Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run lint-imports
uv run pytest
```

## Current task

[`docs/tasks/TASK_001_bootstrap_and_gate1.md`](docs/tasks/TASK_001_bootstrap_and_gate1.md): bootstrap the repository and answer Gate 1 (is there headroom over hand-designed online updaters?).
