Status: active

# Task 002: Lockfile, then the ledger (schema v1)

Read `AGENTS.md` and `docs/ROADMAP.md` first. This task replaces the PR 2 section of TASK_001. It is a stack of three PRs; open them in order and stop after each one for review.

## Why the ledger matters

The ledger is the only source of truth for every result, and it is also the **training data** for the improvement model we will train later. So it must record not just outcomes, but everything a model needs to learn from each attempt: what the improver saw, what it concluded, what it changed, what it predicted, and what happened. Fields added after the schema is frozen would split the dataset, so they all go in now.

Use the standard library only (`dataclasses`, `json`, `hashlib`, `uuid`, `os`, `datetime`). No new dependencies.

---

## PR A: `chore/lockfile-and-probe`

1. Commit `uv.lock`. Change the CI install step to `uv sync --locked --all-groups`.
2. Tighten the violation probe: the step must pass only if `lint-imports` exits non-zero **and** its output contains the contract name `Agent-editable packages never reach sealed or live data` marked as broken. Any other failure (crash, missing config) must fail the step.

---

## PR B: `feat/ledger-core`

### Layout on disk

```
ledgers/<run_id>/
├── main.jsonl      # written by the runner and agent-side processes
├── sealed.jsonl    # written only by the sealed arena process (PR 9)
└── blobs/          # content-addressed store (PR C)
```

Each stream is its own append-only file with its own hash chain, so the two processes never write the same file and agent-side code never needs to open `sealed.jsonl`.

### Envelope: `ledger/schema.py`

Every line is one `Record`, a frozen dataclass:

| Field | Type | Notes |
|---|---|---|
| `schema_version` | str | `"1.0"` |
| `record_id` | str | UUID4 |
| `stream` | str | `"main"` or `"sealed"` |
| `seq` | int | 0-based position in the stream |
| `prev_hash` | str | `record_hash` of the previous record in the stream; 64 zeros for `seq == 0` |
| `record_hash` | str | SHA-256 of the canonical JSON of every other field |
| `created_at` | str | UTC ISO-8601 with `Z`, microseconds |
| `run_id` | str | |
| `kind` | str | one of the kinds below |
| `refers_to` | list[str] | record_ids this record depends on (may be empty) |
| `budget` | `Budget` | cost charged by this record |
| `payload` | kind-specific dataclass | see below |

`Budget`: `llm_input_tokens`, `llm_output_tokens`, `wall_seconds`, `gpu_seconds`, `evaluations` (all non-negative numbers, default 0).

A blob reference is a string `"sha256:<64 hex>"`. Large or binary content (code, arrays, traces, prompts) is always stored as a blob and referenced; payloads hold only small fields. Until PR C lands, blob refs are validated for format only.

### Kinds and payloads

Each kind has its own frozen payload dataclass in `ledger/payloads.py`. Unknown kinds and unknown or missing fields are rejected.

- **`run_start`**: `condition`, `seed`, `config_blob`, `git_commit`, `protocol_versions` (dict: `ledger`, `arena`, `generator`, `cache_manifest`), `budget_limits` (a `Budget`).
- **`proposal`**: one candidate change.
  - `target` (`"updater"` or `"improver"`)
  - `parent_artifact` and `child_artifact` (blob refs to the full code before and after)
  - `diff_blob` (unified diff)
  - `observation_blob`: everything the proposer was shown (behaviour traces, error series, prior attempts, as one serialized bundle)
  - `prompt_blob` and `response_blob` (exact model input and raw output; null for non-LLM proposers)
  - `diagnosis` (short text: what the proposer thinks is wrong)
  - `proposer` (dict: `id`, `version`, `model`, `temperature`)
  - `improver_version`
  - `planted_defect`: nullable dict for synthetic tasks (`defect_id`, `family`, `params`); null for real tasks
- **`prediction`**: refers to one proposal. `predictor` (`id`, `version`), `metric`, `horizon_generations` (1 for per-proposal), `quantiles` (dict with keys `q05`, `q25`, `q50`, `q75`, `q95`; monotone non-decreasing).
- **`allocation`**: `allocator` (`id`, `version`), `candidates` (list of proposal record_ids), `chosen` (subset), `scores` (dict id → float).
- **`evaluation`**: refers to one proposal. `pool` (`"dev"`, `"sealed"`, `"live"`), `status` (`"ok"`, `"failed"`, `"timeout"`, `"invalid_weights"`), `metrics` (dict str → float), `per_task_blob` (nullable), `error` (nullable text). Records with `pool` other than `"dev"` are valid only in the `sealed` stream.
- **`acceptance`**: refers to a proposal and its evaluation. `accepted` (bool), `rule` (str), `new_artifact` (nullable blob ref).
- **`baseline`**: `updater_id`, `hyperparams` (dict), `pool`, `metrics`. Same stream rule as `evaluation`.
- **`commitment`**: `subject` (str), `content_hash` (blob ref), `external_timestamp` (nullable str; for later public timestamping).
- **`correction`**: refers to exactly one record. `reason`, `replacement` (nullable payload dict). The original stays untouched.
- **`run_end`**: `status` (`"completed"`, `"aborted"`, `"budget_exhausted"`), `summary` (dict).

All floats must be finite. NaN and infinities are rejected at construction, not at write time.

### Canonical JSON: `ledger/canonical.py`

`json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)`, UTF-8. `record_hash` is SHA-256 over the canonical JSON of the record with `record_hash` removed. The same record must always produce the same bytes.

### Writer and reader: `ledger/store.py`

- `LedgerWriter(root, run_id, stream)`: opens `<root>/<run_id>/<stream>.jsonl` in append mode. On open it reads the existing file, verifies the chain and resumes `seq` and `prev_hash` from the last record. `append(kind, payload, refers_to=(), budget=Budget())` builds the record, writes one line, flushes and `os.fsync`s before returning the record. No update, delete or truncate API exists.
- A per-stream lock file (`O_CREAT | O_EXCL`) prevents two writers on one stream; a second writer raises.
- `LedgerReader(root, run_id)`: iterates records of one or both streams, `verify()` checks every `record_hash`, every `prev_hash` link and `seq` continuity, and reports the first broken position.

### Tests for PR B (`tests/ledger/`)

1. Every kind round-trips: construct, write, read back, equal.
2. Constructing a payload with NaN, an unknown field, a missing field, an unknown kind, non-monotone quantiles, or a sealed-pool evaluation in the `main` stream raises.
3. Canonical serialization is byte-identical across two constructions of the same record.
4. Tamper detection: `verify()` fails after editing one byte of a line, deleting a line, swapping two lines, or appending a hand-written line with a wrong `prev_hash`.
5. Reopening a ledger resumes `seq` and the chain correctly.
6. A record appended in a subprocess that then raises is present and verifies afterwards.
7. A second concurrent writer on the same stream raises.
8. The store module exposes no function or method whose name contains `delete`, `update`, `remove`, `truncate` or `rewrite`.

If PR B exceeds ~400 lines excluding tests, split payloads into a follow-up and say so in the description.

---

## PR C: `feat/ledger-blobs`

1. **Blob store** (`ledger/blobs.py`): `put(data: bytes) -> "sha256:..."` writes `blobs/<first 2 hex>/<hash>` atomically (temp file, fsync, rename); writing existing content is a no-op. `get(ref) -> bytes` re-hashes and raises on mismatch. No delete API.
2. **Reference check**: `LedgerReader.verify(check_blobs=True)` also confirms every blob ref in every payload exists and hashes correctly.
3. **Ordering validator** (`ledger/validate.py`), returning a list of violations:
   - every `evaluation` of a proposal in a run whose condition uses predictions has a `prediction` for that proposal with a smaller `seq` in the same stream, or an earlier `created_at` if in another stream;
   - every `refers_to` id exists and points backwards in time;
   - every `acceptance` refers to an existing proposal and evaluation;
   - `run_start` is the first record of `main` and `run_end`, if present, is the last;
   - summed budgets do not exceed `budget_limits` from `run_start`.
4. **CLI**: `uv run evolvets ledger verify <root> <run_id>` runs chain, blob and ordering checks, prints a summary, and exits non-zero on any violation.
5. **Tests**: blob round-trip and dedup; a corrupted blob is detected; each validator rule has one passing and one failing fixture; the CLI exits 0 on a clean ledger and non-zero on each failure type.

After PR C merges, a human tags `ledger-schema-v1`. From then on schema changes need a `protocol/` PR.
