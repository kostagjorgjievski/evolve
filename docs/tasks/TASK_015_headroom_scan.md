Status: active

# Task 015: headroom scan before Phase 1

Read `AGENTS.md` and `docs/ROADMAP.md` first. The owner asked for this task in issue #125. It is a stack of three small PRs. Open them in order and stop after each one for review. Three `.github/` workflow changes are specified at the end. **The owner's delegate applies them** (AGENTS.md section 7), and the developer never writes them.

This task starts after TASK_014 is done. The developer always works on the lowest-numbered active task, and TASK_014 PR E (the results note) is still open. #125 says that PR E does not block *planning* this task.

## Why

The TASK_014 learned-updater probe (report run https://github.com/kostagjorgjievski/evolve/actions/runs/38704951541) found almost no room above the tuned baselines:

- `ridge_stack` beats `fixed_share` only on `synthetic-dev`/`simple`: +0.0036 MASE, 95 % interval [0.0008, 0.0063], with no multiple-comparison correction.
- `ridge_stack` is inside the interval or worse on the other five pools, and `loss_gate` is worse on all six.
- On real data the best tuned updater is already within 0.011 to 0.030 MASE of the switching oracle with at most 8 switches (first run, `docs/results/gate1-dev-first-run.md`).

Weighting a fixed pool is close to saturation, so a self-model would have no meaningful gains to predict there. Phase 1 needs a setting with real headroom, and with evaluations expensive enough that choosing what to evaluate matters.

This task builds a cheap, recorded and reproducible scan that measures headroom for each candidate setting. It reuses the arena, the oracles, the caches, `gate1 tune` and the ledger. For each pool:

> **headroom** = (best tuned baseline MASE − target MASE) / best tuned baseline MASE

There are three targets: the switching oracles (8 and 32 switches), the best expert per series and, on synthetic pools, the true-best forecast. Each comes with a paired bootstrap interval over series, computed as in the Gate 1 report.

- **PR A** `feat/synthetic-scan-presets`: named, seeded generator presets of increasing severity, plus a wide replicate of the default grammar. They become cache datasets that every existing tool (`cache build`, `foundation build`, `arena`, `gate1 tune`) accepts.
- **PR B** `feat/gate1-headroom-report`: `evolvets gate1 headroom`, a ranked headroom table computed from completed tune runs.
- **PR C** `docs/headroom-results`: `docs/results/headroom-scan.md`, written after the owner's run, with the ranked table and a recommendation made by the rule fixed in this file.
- **W8, W9 and W10** (owner-applied): foundation caches for the new datasets, the tune runs and the report.

The output is a ranked table plus a recommendation. **The owner chooses the Phase 1 setting.** This task does not plan it.

## Candidate settings and what this task covers

#125 lists the candidates in priority order. None of them needs an arena or protocol change.

1. **Harder generator settings: covered (PR A).** These are planted changes, regime shifts and non-stationarity of increasing severity, written with the existing grammar. No new `GrammarConfig` field is added: the presets only set existing fields.
2. **Hard real domains for foundation models: not covered. They need owner decisions first** (see "Open questions for the owner"). The arena hard-codes `period = 24` and the cache hard-codes horizon 24 for real data. Short-history and intermittent series do not fit the warmup/dev split. GIFT-Eval is distributed as Arrow files, so reading it may need a new dependency. Each of these needs an owner decision (AGENTS.md section 7). That work becomes a later task.
3. **Larger and more varied pools across many series: partly covered (PR A).** `synthetic-wide` is an independent replicate of the default grammar, with 3x the series of `synthetic-dev`. It tests whether the `synthetic-dev` headroom holds over more series. A larger or more varied *expert* pool changes the simple cache's fixed expert list, which means a `CACHE_VERSION` bump. That is a protocol change and an owner decision, so it is listed as a follow-up.

## Decisions already made (do not reopen)

- **Owner, #109, #110 and #125:**
  - Results must be faithful and reproducible.
  - There is no sealed ceremony and no redo for process reasons.
  - The first dev tune run `gate1-tune-20261013T151300Z-b36261f7` is the reference for ETTh1, ECL and `synthetic-dev`. It is not re-run.
- **TASK_007/010/011:**
  - `replay_dataset` is the only way to score an updater.
  - `select_configs` is the selection rule.
  - The oracles bound single-expert selection only.
  - Reports never mix protocol versions and table only `completed` runs.
  - Floats are accumulated in explicit left-to-right loops.
  - The bootstrap uses 2000 resamples, one `random.Random(seed)` per table, and interval indexes 50 and 1949 (`gate1_report_data._gap_rows`).
- **TASK_004:**
  - `FAMILIES`, `GrammarConfig()` defaults, `dev_tasks`, `world_seed` and the draw order are unchanged.
  - `GENERATOR_VERSION` stays `"1"`, because no existing (spec, seed) changes its output. The golden-fingerprint tests must pass unchanged.
- **Out of scope** (#125), listed in PR C as follow-up decisions for the owner:
  - updaters that output forecasts rather than weights;
  - probabilistic metrics (CRPS, calibration).

  Both need an `updater/interfaces.py` or arena protocol change.

## Rules for all three PRs

- **No protocol change.**
  - No change to `ledger/`, `PROTOCOL_KEYS`, `SCHEMA_VERSION`, `ARENA_VERSION`, `GENERATOR_VERSION`, `CACHE_VERSION`, any payload or `updater/`.
  - No change to replay, label release, metrics or oracles.
  - The arena changes in PR A are additive: `arena/dev.py` learns to load the new datasets, and every existing pool loads exactly as before.
- **`all` keeps its meaning.** Every `all` on the command line still expands to exactly `CACHE_DATASETS == ("etth1", "ecl", "synthetic-dev")`. This keeps `gate1-tune.yml` and the cache commands unchanged.
- **Standard library only.** No new dependency.
- **Seeds:**
  - Scan worlds use seeds derived from the preset name, family and index (PR A).
  - Tune runs record their seed as before.
  - The only other randomness is the report's bootstrap, seeded from its CLI.
- **Equal accounting.** Scan tune runs are ordinary `gate1 tune` runs, with the same grids, budgets and records as the reference run.
- **Agents never run the scan on lab data.** Tests use fixtures in `tmp_path`. The real runs are W8 to W10, dispatched by the owner.
- **Strict mypy.** `evolvets.arena.synthetic.presets` is already covered by `evolvets.arena.*`. Add `evolvets.runner.gate1_headroom` to the strict override list in `pyproject.toml`. Do not weaken any override.
- **Import contracts.**
  - `arena/synthetic/presets.py` imports only the standard library and `evolvets.arena.synthetic.*`.
  - The import-linter contracts stay unchanged and must pass.
- Do not edit `.importlinter`, `.github/`, `AGENTS.md` or `docs/agents/`.

---

## PR A: `feat/synthetic-scan-presets`

### Files and interfaces

`src/evolvets/arena/synthetic/presets.py` (new):

```python
@dataclass(frozen=True)
class ScanPreset:
    name: str  # the cache dataset name
    families: tuple[str, ...]  # non-empty, a subsequence of FAMILIES in FAMILIES order
    config: GrammarConfig


SCAN_DATASETS: tuple[str, ...] = (
    "synthetic-shift1",
    "synthetic-shift2",
    "synthetic-shift3",
    "synthetic-wide",
)
SCAN_PRESETS: Mapping[str, ScanPreset]  # MappingProxyType, SCAN_DATASETS order


def scan_world_seed(preset: str, family: str, index: int) -> int: ...
def scan_tasks(preset: str, family: str, n: int) -> list[SyntheticTask]: ...
def preset_to_dict(preset: ScanPreset) -> dict[str, Any]: ...
def preset_sha256(preset: ScanPreset) -> str: ...
```

`src/evolvets/cache/datasets.py`:

```python
CACHE_DATASETS = ("etth1", "ecl", "synthetic-dev")  # unchanged; what "all" expands to
DEV_DATASETS: tuple[str, ...] = (*CACHE_DATASETS, *SCAN_DATASETS)  # every loadable dev dataset
```

### Exact behaviour: the presets

`synthetic-shift1` to `synthetic-shift3` form a severity ladder. They use the families `("sudden", "gradual", "recurring", "false_drift", "mixed")`, which leaves out `stationary`, the family with nothing planted. Every field not listed below is the `GrammarConfig()` default. In particular `length = 2000`, `period = 24` and `warmup = 300` stay, so `checked_synthetic_series` and the foundation context logic work unchanged.

| field | shift1 | shift2 | shift3 |
| --- | --- | --- | --- |
| `max_changes` | 6 | 12 | 24 |
| `min_segment` | 120 | 70 | 48 |
| `shift_range` | (3.0, 6.0) | (4.0, 8.0) | (5.0, 10.0) |
| `new_slope_range` | (-0.02, 0.02) | (-0.03, 0.03) | (-0.05, 0.05) |
| `ramp_range` | (10, 100) | (5, 50) | (4, 30) |
| `max_bursts` | 4 | 6 | 8 |
| `burst_length_range` | (10, 60) | (5, 50) | (5, 30) |
| `burst_mult_range` | (2.0, 5.0) | (3.0, 6.0) | (3.0, 8.0) |

`synthetic-wide` uses all six `FAMILIES` with `config = GrammarConfig()`. It differs from `synthetic-dev` only through its seeds, and through `per_family` at build time (W8 builds it with 60).

**Seeds.** `scan_world_seed(preset, family, index)`:

- `preset` must be in `SCAN_PRESETS`, `family` in that preset's `families`, and `index` an int `>= 0` (bools rejected). Anything else raises `ValueError`.
- The seed is the first 8 bytes, big-endian, of `sha256(b"\x00".join((b"evolvets/generator/v1", b"scan", preset, family, str(index))))`, masked with `2**63 - 1`. Every part is UTF-8.
- `world_seed` accepts only the pools `dev` and `sealed`, so the `scan` field keeps these seeds domain-separated from dev and sealed seeds.

**Tasks.** `scan_tasks(preset, family, n)` returns `[make_task(family, scan_world_seed(preset, family, i), SCAN_PRESETS[preset].config) for i in range(n)]`. `n` is an int `>= 0`.

**Hashing.** `preset_to_dict` returns `{"name": ..., "families": [...], "config": config_to_dict(config)}`. `preset_sha256` is `sha256_hex(canonical_bytes(preset_to_dict(preset)))`. This module may import `evolvets.ledger.canonical`, as `arena/dev.py` already does. No import contract forbids it.

### Exact behaviour: datasets, pools and the CLI

**`public_inputs(dataset, ..., per_family)`** for a dataset in `SCAN_DATASETS`:

- `per_family` is checked as for `synthetic-dev`.
- The series are `SeriesInput(f"{family}/{i}", task.world.y, task.world.spec.period)`, for each family in the preset's `families` order, then each `i`.
- The source is exactly `{"kind": "synthetic", "pool": "scan", "preset": name, "preset_sha256": ..., "generator_version": GENERATOR_VERSION, "per_family": str(count)}`.
- Output for `etth1`, `ecl` and `synthetic-dev` is unchanged.

**`load_dev_pool`** accepts every dataset in `DEV_DATASETS`. A scan dataset loads its series through a new `_scan_series(manifest)`, which raises `ArenaError` unless all of these hold:

- `source["pool"] == "scan"`;
- `source["preset"] == manifest.dataset`;
- `source["preset_sha256"]` equals the current `preset_sha256`. A preset edited after the cache was built is refused, not reinterpreted;
- `generator_version` is current;
- `per_family` is an int `>= 1`, and the series count matches.

It then builds the series with `checked_synthetic_series(manifest, expected)`, so the oracle, change points and boundaries come from the existing code.

**Everywhere else**, "a known dataset" now means `DEV_DATASETS`, while `all` expands to `CACHE_DATASETS` only:

- the `ValueError` checks in `gate1_tune.TuneSpec` validation and in `gate1_probe`;
- the CLI `choices` of `cache build|verify`, `foundation build|verify|compare`, `arena replay|oracle`, `gate1 tune` and `gate1 probe`;
- the help text of `gate1 tune --dataset`, which becomes "`all` expands to etth1, ecl and synthetic-dev";
- the help text of `--per-family`, which becomes "required for synthetic datasets".

The sealed side (`arena/sealed/`, `gate1_sealed`) is not touched, and scan datasets have no sealed counterpart. `pool_leakage_risk` needs no change: no model's `pretraining_overlap` names a scan dataset.

### Tests for PR A (all required)

In `tests/arena/test_synthetic_presets.py`:

1. **Registry:**
   - `tuple(SCAN_PRESETS) == SCAN_DATASETS`;
   - each `preset.name` equals its key, and passes the cache's dataset-name check;
   - `SCAN_DATASETS` is disjoint from `CACHE_DATASETS` and from `arena.sealed.names.SEALED_DATASETS` (the test may import it);
   - the mapping is read-only.
2. **Preset rules:**
   - Each preset's families are non-empty, have no duplicates, and form a subsequence of `FAMILIES` in order.
   - Each config has `length`, `period` and `warmup` equal to the `GrammarConfig()` defaults.
   - `synthetic-wide` has all `FAMILIES` and `config == GrammarConfig()`.
   - The three shift presets match the table above field by field.
3. **Ladder:** from shift1 to shift3, `max_changes` and `max_bursts` strictly increase, `min_segment` strictly decreases, and `shift_range[0]` strictly increases. Over `scan_tasks(p, "sudden", 20)`, the mean number of change points strictly increases from shift1 to shift2 to shift3.
4. **Seeds:**
   - The result is deterministic and in `[0, 2**63)`.
   - A golden value for `scan_world_seed("synthetic-shift1", "sudden", 0)` is pinned as a literal.
   - The values are pairwise distinct over every preset, every one of its families and indexes `0..49`.
   - They are disjoint from `world_seed("dev", f, i)` for every family and `i < 50`.
   - An unknown preset, a family not in the preset (for example `stationary` for shift1), and an index of `-1` or `True` each raise `ValueError`.
5. **Tasks:**
   - `scan_tasks(p, f, 3)[i]` equals `make_task(f, scan_world_seed(p, f, i), config)`, compared by `world.y` and `change_kinds`.
   - `scan_tasks(p, f, 0) == []`.
   - `synthetic-wide` index 0 differs from `dev_tasks(f, 1)[0]` for every family.
6. **Generator unchanged:**
   - `GENERATOR_VERSION == "1"`.
   - Every existing test in `tests/arena/test_synthetic_*.py` passes without modification, golden fingerprints included.
7. **Hash:** `preset_sha256` changes when any one config field or the family list changes (`dataclasses.replace`), and is stable across two calls.

In `tests/cache/test_datasets.py` and `tests/arena/test_dev.py`:

8. **`public_inputs`:**
   - For each scan dataset with `per_family = 2`, the ids, order, values, period and source keys are exactly as specified.
   - `per_family` of `None`, `0` or `True` raises `CacheError`.
   - The `synthetic-dev` output (series and source) is unchanged.
9. **Pool load:** a scan cache built in `tmp_path` with `per_family = 1` gives one series per preset family. Each series has `score_start == scale_end == 300`, change points equal to the world's, and an oracle equal to `true_best_forecast` at every scored origin. `replay_dataset(BASELINES["fixed_share"], DEFAULT_CONFIGS["fixed_share"], 0, pool)` is `ok`, with an `oracle_mase` per series.
10. **Refusals** (`ArenaError` for each):
    - a cache built while `SCAN_PRESETS[name]` is monkeypatched to a different config, then loaded with the real preset (`preset_sha256` mismatch);
    - a manifest whose `source.preset` names another scan dataset;
    - a `generator_version` mismatch;
    - a `per_family` that does not match the series count.

    Build hand-made manifests the way the existing synthetic-dev refusal tests in `tests/arena/test_dev.py` do.
11. **Foundation pool:** `build_foundation_cache` with the fake adapter on `synthetic-shift2` (`per_family = 1`), then `load_dev_pool(..., pool_spec="simple+foundation", foundation_models=["fake"])`, loads with `pool_leakage_risk == False`.

In `tests/runner/` and `tests/cache/test_cli.py`:

12. **CLI and `all`:**
    - `cache build synthetic-shift1 --per-family 1` prints its `OK` line.
    - `_gate1_axes` expands `--dataset all` to exactly `CACHE_DATASETS`, and keeps `--dataset synthetic-wide` as given.
    - `arena oracle synthetic-shift3` (with a `per_family = 1` cache) succeeds.
    - An unknown dataset is still rejected by argparse.
13. **Tune over a scan dataset:** `tune` with small grids over `synthetic-shift1` (`per_family = 1`) and `simple` ends `completed`. Its config blob's `simple_cache_ids` holds the scan cache id, and its records carry `oracle_mase`. Every existing test in `tests/runner/` passes unchanged.

### Dependencies

None.

### Size

About 200 lines of implementation. If it exceeds ~400 lines with tests, split it:

- `feat/synthetic-scan-presets`: `presets.py` with tests 1 to 7;
- `feat/scan-datasets`: datasets, pool loading and the CLI, with tests 8 to 13.

### After PR A merges

The reviewer opens an `fyi-human` issue that asks the owner's delegate to apply **W8**. It also asks the owner to dispatch W8 for each of the 4 scan datasets x 3 models (Chronos-2, TiRex, TimesFM 2.5), with `per_family` 20 for the shift presets and 60 for `synthetic-wide`, and never two builds of one dataset at once. PR B does not wait for this.

---

## PR B: `feat/gate1-headroom-report`

### Files and interfaces

`src/evolvets/runner/gate1_headroom.py` (new). It may import the private helpers of `gate1_report_data`.

```python
HEADROOM_TARGETS: tuple[str, ...] = (
    "switch_mase_m8",
    "switch_mase_m32",
    "best expert per series",
    "true-best forecast",
)
RANK_TARGET = "switch_mase_m8"


@dataclass(frozen=True)
class HeadroomRow:
    target: str  # a HEADROOM_TARGETS label, as the Gaps table names it
    target_mase: float  # loop-mean of t_s
    gap: float  # loop-mean of b_s - t_s, equal to the Gate 1 report's GapRow.mean
    headroom: float  # (B - T) / B, where B and T are the loop-sums of b_s and t_s
    lower: float | None
    upper: float | None  # None with a single series


@dataclass(frozen=True)
class HeadroomSection:
    run_id: str
    dataset: str
    pool_spec: str
    leakage_risk: bool
    series: int
    best_baseline: str | None
    best_baseline_mase: float | None
    replay_wall_seconds: float | None  # budget.wall_seconds of the best baseline's selected record
    tune_wall_seconds: float  # loop-sum of budget.wall_seconds over every baseline record of the pool
    rows: tuple[HeadroomRow, ...]  # HEADROOM_TARGETS order; () without a best baseline


@dataclass(frozen=True)
class HeadroomData:
    run_rows: tuple[RunRow, ...]
    protocol_versions: dict[str, str]
    ranked: tuple[HeadroomSection, ...]  # leakage_risk False, with rows, ranked
    leakage_ranked: tuple[HeadroomSection, ...]  # leakage_risk True, with rows, ranked
    unranked: tuple[HeadroomSection, ...]  # no best baseline, in config order


def build_headroom_data(
    ledger_root: Path, run_ids: Sequence[str], *, bootstrap_seed: int
) -> HeadroomData: ...
def render_headroom(data: HeadroomData, run_ids: Sequence[str], *, bootstrap_seed: int) -> str: ...
```

`runner/cli.py` gains `evolvets gate1 headroom --run ID [--run ID ...] [--bootstrap-seed 0] [--ledger-root ledgers] (--out PATH | --check PATH)`. It behaves exactly as `gate1 report` does for these flags.

The following refactors of `gate1_report_data.py` are allowed, provided every existing test in `tests/runner/` passes unchanged and `build_report` output is unchanged:

- Split `_labeled_differences` into `_labeled_targets(...)`, which returns the per-series target values `t_s` under the same labels in the same order, and a difference step `d_s = b_s - t_s`. This is the same arithmetic as now, so the numbers are bit-identical.
- Extract the lookup of the best baseline's selected record from `_build_section` into a helper both modules use.
- Extract the per-run loop of `build_report_data` (read, verify, status, pools, claims) into a helper both modules use.
- **Family gaps on scan pools.** Replace `dataset == _SYNTHETIC_DATASET` in `_build_section` with a check that the dataset is `synthetic-dev` or in `SCAN_DATASETS`, so that scan pools get per-family Gaps tables too. Output for existing datasets is unchanged.

### Exact behaviour

**Refusals.** `build_headroom_data` first calls `build_report_data(ledger_root, run_ids, bootstrap_seed=bootstrap_seed)`. Every refusal of the Gate 1 report therefore applies unchanged, with the same `ReportError` message. This includes a probe run passed as `--run`, a protocol-version mix, and two complete runs over one pool. `run_rows` and `protocol_versions` are copied from that result. Incomplete runs are listed and never ranked.

**For each pool of each completed run**, in the same order as the report:

- The best tuned baseline is `_best_baseline`. `b_s` is its selected record's per-series MASE, and `t_s` is each target's per-series value from `_labeled_targets`.
- The `true-best forecast` row exists only when every series has `oracle_mase` (synthetic pools). The other three always exist. Rows keep the `HEADROOM_TARGETS` order.
- `headroom = (B - T) / B`, where `B` and `T` are the left-to-right loop-sums over series. If `B == 0.0`, raise `ReportError` ("headroom undefined: zero baseline MASE").
- **The bootstrap** uses one `random.Random(bootstrap_seed)` per section, drawn in row order. Each row takes 2000 resamples. Each resample draws `n` indexes with `rng.randrange(n)` and accumulates `sb` and `st` left to right over the same indexes. The resample's value is `(sb - st) / sb`, and a zero `sb` raises the same `ReportError`. The sorted values give `lower = values[50]` and `upper = values[1949]`. With `n == 1`, both are `None`.
- `replay_wall_seconds` is the `budget.wall_seconds` of the selected record. `tune_wall_seconds` is the loop-sum over every baseline record of the pool, the oracle record excluded.

**Ranking** applies separately to the non-leakage and the leakage-risk sections that have rows. Let `r` be the section's `RANK_TARGET` row. The sort key, descending, is:

1. `r.lower`, or `r.headroom` when `r.lower is None`;
2. then `r.headroom`;
3. ties keep the report's order (a stable sort).

Sections without a best baseline go to `unranked`.

**Rendering:**

```
# Headroom scan (dev)

generated by `evolvets gate1 headroom --run <id> [--run <id> ...] --bootstrap-seed N`

## Runs
| run id | condition | status | git commit | records | validate_run |

## Protocol versions
| protocol | version |

## Ranked headroom
| rank | dataset | pool | series | best tuned baseline | baseline MASE | switch-8 headroom | 95% interval | switch-32 headroom | best-per-series headroom | true-best headroom | replay seconds | tune seconds |

## Ranked headroom, leakage-risk pools
(same table; the heading and table only when there is such a section)

## Pools without a tuned baseline
(a bullet list "<dataset>, pool <pool_spec> (run <id>)"; the heading and list only when non-empty)

## Per-pool detail

### <dataset>, pool <pool_spec>[ (leakage risk)]
Best tuned baseline: <id> (dev MASE <x>), run <run id>
| target | target MASE | gap | headroom | 95% interval |

## Notes
```

- Ranked sections come first, then leakage-risk sections, both in rank order. A missing `true-best` row renders as `n/a`, and a missing interval as `n/a`. Every float is `f"{x:.4f}"`, and intervals are `[lower, upper]`.
- The Notes section has exactly these lines:
  - Headroom is (best tuned baseline MASE − target MASE) / best tuned baseline MASE over the pool's series; positive means the target is better. Intervals are paired bootstrap intervals over series (2000 resamples) of the same ratio.
  - The baselines were tuned, and the targets computed, on these same dev series.
  - The switching oracles and the best expert per series select single experts in hindsight: no online updater can reach them, and they do not bound weighted combinations.
  - Replay seconds is the wall-clock of one evaluation (one replay of the selected configuration over every series of the pool); tune seconds is the whole grid.
  - Pools are ranked by the lower end of the switch-8 headroom interval; there is no multiple-comparison correction.
- Two renders of the same ledger and seed are byte-identical.

### Tests for PR B (all required; `tests/runner/test_gate1_headroom.py`, plus tests in `tests/runner/test_gate1_report_data.py`)

The fixtures are tune runs in `tmp_path` (PR A's `synthetic-shift1`, `synthetic-dev` and the `etth1_env` fixture of `test_gate1_report_data.py`), plus hand-built ledgers for edge cases.

1. **Gate 1 report unchanged:** every existing test in `test_gate1_report.py`, `test_gate1_report_data.py` and `test_gate1_probe_data.py` passes without modification.
2. **Numbers:**
   - `headroom`, `target_mase` and `gap` equal hand computations from the selected blob and the oracle blob.
   - `gap` equals the matching `GapRow.mean` of `build_report_data` exactly.
   - `best_baseline` equals the report's best baseline.
3. **Bootstrap:**
   - The bounds equal an independent re-implementation in the test of the stated draw order and seed.
   - The same seed gives identical data, and a different seed gives equal `headroom` values.
   - A single-series pool gives `None` bounds, rendered as `n/a`.
4. **Rows:** a synthetic pool has 4 rows in `HEADROOM_TARGETS` order, and an ETTh1 pool has 3, with no `true-best forecast` row.
5. **Zero baseline:** the ratio helper raises `ReportError` when `B == 0.0`, and when a resample's `sb == 0.0`.
6. **Ranking:** hand-built sections give a descending order by `lower`. A tie on `lower` is broken by `headroom`, and a full tie keeps input order. A single-series section ranks by `headroom`. Leakage-risk sections are ranked separately, and a pool where every baseline failed is in `unranked` and rendered in the bullet list.
7. **Refusals pass through:**
   - a duplicate run id;
   - a protocol-version mix;
   - a probe run passed as `--run`;
   - two complete runs over one pool.

   Each raises the same `ReportError` message as `build_report_data`.
8. **Incomplete run:** an `aborted` tune run is listed in `run_rows` and appears in no section.
9. **Costs:** `replay_wall_seconds` equals the selected record's `budget.wall_seconds`, and `tune_wall_seconds` equals the loop-sum over the pool's baseline records.
10. **Rendering:**
    - the section order, headings and table headers are exactly as specified;
    - the leakage heading appears only with leakage sections;
    - the Notes lines are present;
    - the generating line lists every `--run`;
    - two renders are byte-identical.
11. **CLI:**
    - `--run` is repeatable and `--out` writes the file.
    - `--check` passes on the written file and fails on a one-character change.
    - A refusal prints `FAIL` and exits 1, with `--out` not written.
12. **Family gaps on scan pools:** the report data of a `synthetic-shift1` pool has one `FamilyGaps` per preset family present, in `FAMILIES` order. `synthetic-dev` sections are unchanged.

### Dependencies

None.

### Size

About 250 lines of implementation. If it exceeds ~400 lines with tests, split it:

- `feat/gate1-headroom-data`: the refactors, `build_headroom_data`, and tests 1 to 9 and 12;
- `feat/gate1-headroom-report`: the rendering, the CLI, and tests 10 and 11.

### After PR B merges

The reviewer opens an `fyi-human` issue with these requests:

1. The owner's delegate applies **W9** and **W10**.
2. Once every W8 build has finished, the owner dispatches W9.
3. Once W9 has finished, the owner dispatches W10 with `gate1-tune-20261013T151300Z-b36261f7` and every completed W9 tune run id.
4. The owner posts the W9 and W10 workflow run URLs and the report artifact's SHA-256 on that issue.

---

## PR C: `docs/headroom-results`

PR C starts only after the owner has posted the W9 and W10 run URLs, as above. Until then, the developer has nothing to do on this task.

### Files

`docs/results/headroom-scan.md`, structured like `docs/results/gate1-dev-first-run.md`:

1. **What this is:** two or three sentences on which pools were scanned and against which targets.
2. **Ranked headroom:** the report's `## Ranked headroom` table (and the leakage-risk table, if present), copied verbatim.
3. **Recommendation**, made by this rule and no other:
   - Take the highest-ranked **non-leakage `simple+foundation`** pool whose switch-8 headroom lower bound is **at least 0.02** (2 % of the best tuned baseline's MASE). Write exactly one sentence: "By the rule fixed in TASK_015, the recommended Phase 1 setting is <dataset>, pool simple+foundation (switch-8 headroom <x>, 95 % interval [<l>, <u>]; one evaluation takes <s> seconds)."
   - If no pool qualifies, write: "By the rule fixed in TASK_015, no scanned setting has meaningful headroom; the next candidate is hard real domains (#125, candidate 2)."
   - Below it, add one sentence for each of the other qualifying pools, in rank order, in the same format, starting "Also qualifying: ...".
   - The rule uses `simple+foundation` because a Phase 1 setting without the foundation models would not reflect the system. The rule is fixed here, before the run. The owner may override it, but the note does not change it after seeing the numbers.
   - The final choice is the owner's. The note says so in one sentence and adds no other interpretation.
4. **Caveats**, all of these:
   - dev numbers, with the baselines tuned on the same series;
   - the oracles are single-expert hindsight bounds, not reachable by an online updater and not bounds on weighted combinations;
   - ETTh1 and ECL foundation results are leakage-risk;
   - several comparisons, uncorrected;
   - the baselines' grid-edge note from the first run;
   - the scan presets were designed in TASK_015 before any run, and none was changed after.
5. **Follow-up decisions for the owner** (one line each, no recommendation):
   - hard real domains (candidate 2, with the open questions below);
   - larger or more varied expert pools (a `CACHE_VERSION` bump);
   - updaters that output forecasts;
   - probabilistic metrics (CRPS, calibration).
6. **Provenance:**
   - every W8 build (dataset, model, per_family, manifest artifact);
   - every W9 tune run id with its status, including any aborted piece;
   - each run's recorded git commit, and its mapping to the real `main` SHA if the tarball fallback was used;
   - the commands and seeds;
   - the W9 and W10 run URLs;
   - the artifact's SHA-256.
7. **The full report**, verbatim and byte-identical to the W10 artifact. Download it with `gh run download` and check `sha256sum` before pasting.

The same PR changes this file's first line to `Status: done` and sets roadmap entry 11c's status to the merged PR numbers.

### Checks

These are documentation-only checks, stated in the PR description:

- the artifact's SHA-256 matches the posted one;
- the embedded report equals the artifact;
- every tune run id from W9's log is listed.

### After PR C merges

The reviewer opens a `needs-human` issue titled `Phase 1 setting after the headroom scan`. It holds:

- the ranked table;
- the recommendation sentence;
- a link to the results note.

No agent plans past it. Choosing the Phase 1 setting and starting Phase 1 are owner decisions.

---

## Owner-applied workflows (`.github/`, applied by the owner's delegate, never by an agent)

W9 and W10 follow the rules of `gate1-tune.yml` and `gate1-probe.yml`:

- `workflow_dispatch` only, with `if: github.actor == github.repository_owner`;
- `runs-on: [self-hosted, gpu]` with `CUDA_VISIBLE_DEVICES: ""` (CPU work, which never touches a GPU);
- the same paths and `gpu-env.sh dev "$VENV"`;
- `concurrency` with `cancel-in-progress: false`;
- the same git-then-tarball checkout, printing `::warning` with `github.sha` on the tarball fallback;
- dispatch inputs reach `run:` scripts through `env:`, never through `${{ }}`.

Their file names start with `gate1-`, so under #123 the reviewer may review and merge them.

**W8: amendment to `gpu-foundation-cache.yml`.** This is not a `gate1-*` file, so a human applies it.

- The `dataset` choices gain the four `SCAN_DATASETS`.
- Every dataset whose name starts with `synthetic-` takes the synthetic path. That covers `cache build "$DATASET" --per-family "$PER_FAMILY"` and `--per-family` on `foundation build`. Today the script tests `= synthetic-dev`, so the new datasets would fall into the real-data branch.
- Everything else is unchanged, including `gpu-pick.sh` and `gpu-release.sh`.

**W9 `gate1-headroom-tune.yml`: the scan tune runs.**

- **Inputs:**
  - `max_wall_seconds` (default `172800`, per piece);
  - `seed` (default `0`);
  - `datasets` (default `synthetic-shift1 synthetic-shift2 synthetic-shift3 synthetic-wide`).
- **Settings:** `timeout-minutes: 3600`, concurrency group `gate1-headroom-tune`.

Steps:

1. Checkout, paths and environment, as `gate1-tune.yml`. `git status --porcelain` must be empty.
2. For each dataset, start `evolvets gate1 tune --dataset <d> --pool-spec simple --pool-spec simple+foundation --seed <seed> --max-wall-seconds <n> --ledger-root "$LEDGER_ROOT" --cache-root "$CACHE_ROOT" --foundation-root "$FM_ROOT" --data-root "$DATA_ROOT" --manifest-dir data/manifests` in the background, with its output in its own log file. Then `wait` for each and collect its exit code.
3. Print each piece's `OK` or `FAIL` line, and one line `TUNE_RUN_IDS=<space-separated ids>` covering every run that started.
4. The job succeeds only if every exit code is 0 or 2.

A piece that ended `aborted` may be re-dispatched alone through `datasets`. A piece that completed is never re-run. The report refuses two complete runs over one pool.

**W10 `gate1-headroom-report.yml`: the report.**

- **Inputs:** `run_ids` (space-separated, required, each passed as `--run`) and `bootstrap_seed` (default `0`).
- It runs `evolvets gate1 headroom ... --out headroom-scan.md`.
- It prints each run's recorded commit and its GitHub lookup status, as `gate1-dev-report.yml` does.
- It uploads the file as an artifact and prints its `sha256sum`.

**Cost estimate**, from the first run's wall seconds:

- `synthetic-dev` (120 series) took about 8,300 s on `simple` and 13,000 s on `simple+foundation`.
- That gives about 5 h per shift preset (100 series each) and about 18 h for `synthetic-wide` (360 series), with the four pieces running in parallel.
- The W8 GPU builds scale the same way, relative to the `synthetic-dev` builds.

---

## Open questions for the owner (candidate 2, not planned here)

1. **Which real domains?** Hourly series whose behaviour is dominated by exogenous effects (for example solar or traffic subsets of GIFT-Eval), or intermittent or short-history series. The second kind needs a different period, horizon and split rule, which means an arena protocol change.
2. **Data format.** GIFT-Eval ships as Arrow files. Either a reader dependency is added (`pyarrow`, a large native build, so section 7 applies), or the original CSV or `.tsf` sources are used where they exist (stdlib only).
3. **Licences and manifests.** For each source: the licence, the pinned URL and the content hash, as `data/manifests/` already records.

## Not in this task

- Choosing the Phase 1 setting, and anything in Phase 1.
- Hard real domains (candidate 2), until the questions above are answered.
- New `GrammarConfig` fields or families, a `GENERATOR_VERSION` bump, a larger expert pool, forecast-output updaters, probabilistic metrics.
- Re-running the reference tune run, a wider baseline grid, sealed evaluation.
- Learned-updater probes on the scan pools. Once a setting is chosen, `gate1 probe` already works on any tuned pool.

## Closing the task

PR C sets `Status: done` and the roadmap status, as above. W8 to W10 and the runs themselves are outside the developer's stack.
