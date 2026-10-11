Status: done

# Task 003: Explicit `requires_predictions` flag, then tag `ledger-schema-v1`

Read `AGENTS.md` and `docs/ROADMAP.md` first. One PR. This is the last change to the ledger schema before it is tagged `ledger-schema-v1` (issue #10).

## Why

`validate_run` currently guesses whether the prediction-before-evaluation rule applies: it enforces the rule only if the run contains at least one prediction record. So a run that was supposed to log predictions but logged none validates as clean, which is exactly the failure the rule exists to catch (raised in the PR #5 review). The run must declare it up front.

The flag is named `requires_predictions`, not `uses_predictions`, because what matters is whether predictions must be logged, not whether the allocator acts on them. The "self-model predicts but doesn't act" condition logs predictions (`requires_predictions = true`) even though its allocator ignores them.

## PR: `feat/requires-predictions`

### Schema (`ledger/payloads.py`)

1. Add a required field `requires_predictions: bool` to `RunStart`, validated with `_require_bool`. No default; omitting it is a construction error like any other missing field.
2. `schema_version` stays `"1.0"`. Nothing has been tagged and no real runs exist yet, so this is still part of v1. Update test fixtures and the `run_start` builders accordingly.

### Validator (`ledger/validate.py`)

3. Remove the `uses_predictions` parameter of `validate_run` and the auto-detection. The rule is read from the run's `run_start.requires_predictions`.
4. If the run has no valid `run_start`, the existing `run_start` violation is reported and the prediction rule is skipped (no crash).
5. When `requires_predictions` is `true`: every `evaluation` must have a `prediction` for the same proposal logged before it (same stream: smaller `seq`; other stream: strictly earlier `created_at`). This is the current rule, unchanged.
6. When `requires_predictions` is `false`: any `prediction` record in the run is a violation (`prediction at seq N in a run that declares requires_predictions=false`). A run cannot quietly mix conditions.
7. `evolvets ledger verify` needs no new arguments; it gets the rule from the ledger.

### Small fixes from the PR #5 review (same PR)

8. `BlobStore.put`: if the target file already exists, re-hash it. If the hash doesn't match the ref, atomically replace it with the correct content (temp file, fsync, rename). If it matches, it stays a no-op.
9. `LedgerWriter.break_stale_lock` docstring: mention that empty locks older than `STALE_EMPTY_LOCK_SECONDS` can also be broken.
10. `LedgerReader.verify`: a blob failure reports the running record total in `records`, like other failures, not 0.

### Tests (all required; the reviewer will check each)

1. `RunStart` without `requires_predictions` raises; with a non-bool value (`1`, `"true"`, `None`) raises; `True` and `False` round-trip.
2. `requires_predictions=true`, an evaluation with no prediction anywhere → violation. This is the case auto-detection used to miss.
3. `requires_predictions=true`, prediction before evaluation in the same stream → clean.
4. `requires_predictions=true`, prediction in `main` with a strictly earlier `created_at` than an evaluation in `sealed` → clean; equal timestamps → violation.
5. `requires_predictions=false`, proposals and evaluations with no predictions → clean.
6. `requires_predictions=false`, one prediction record present → violation naming its seq.
7. Run with no `run_start`: `validate_run` returns the `run_start` violation and does not raise.
8. `evolvets ledger verify` exits non-zero on the fixture from test 2 and zero on the fixture from test 3.
9. `BlobStore.put` repairs a corrupted existing blob: corrupt the file, `put` the correct bytes, then `get` succeeds and `verify(check_blobs=True)` passes.
10. Blob failure in `verify` reports a non-zero `records` count when earlier records were read.

### Constraints

- Standard library only. No new dependencies.
- In the same PR, change this file's first line to `Status: done`.
- Size: well under 400 lines of implementation.

## After merge

The reviewer comments on issue #10 that the schema is final and ready for the human to tag `ledger-schema-v1` on the merge commit. From then on, changes to `ledger/` need a `protocol/` PR.
