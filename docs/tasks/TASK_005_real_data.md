Status: active

# Task 005: Real data (ETTh1 + ECL subset): fetch, manifests, chronological splits

Read `AGENTS.md` and `docs/ROADMAP.md` first. This task covers roadmap entry 6 (`feat/real-data`). It is a stack of two PRs; open them in order and stop after each one for review.

## Why

Research question 1 is measured on synthetic worlds against the true-best forecast (TASK_004) and on real data against the best switching oracle. This task brings in the real data: two public hourly datasets, each pinned by a content hash, cut into chronological segments so that tuning (dev) and the final Gate 1 evaluation (sealed) never share a target.

## Rules for both PRs

- Code lives in `src/evolvets/arena/realdata/` (strict mypy, as for all of `arena/`). Tests live in `tests/arena/`. The CLI additions live in `runner/cli.py`.
- **Standard library only** (`urllib.request`, `zipfile`, `io`, `csv` if wanted, `hashlib`, `json`, `os`, `math`, `array`, `datetime`, `dataclasses`). No numpy, no pandas, no PyYAML, no `requests`.
- **Manifests are JSON written to `.yaml` files.** JSON is a subset of YAML 1.2, so `data/manifests/*.yaml` (AGENTS.md section 6) stays valid YAML with no new dependency. Write them with `json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"`, read them with `json.loads`. Every write of a manifest or a data file goes to a temporary file in the same directory, is flushed and `fsync`ed, then moved into place with `os.replace`. No reader ever sees a partial file.
- `REALDATA_VERSION = "1"` in `arena/realdata/__init__.py`. Any change that alters a processed series, the subset or a split boundary must bump it; the golden tests enforce this. **No ledger change in this task:** nothing in PR A or PR B writes ledger records, and `run_start.protocol_versions` has no `realdata` key — `PROTOCOL_KEYS` in `ledger/payloads.py` is exactly `ledger`, `arena`, `generator`, `cache_manifest`. Runs pin the real data through the split manifest's `realdata_version` field plus `run_start.git_commit`. If runs are ever to record a real-data protocol version directly, the key must be added to `PROTOCOL_KEYS` in a `protocol/` PR before `ledger-schema-v1` is tagged (issue #10); that decision belongs to the PR 8/11 specs, not to this task.
- **Tests never touch the network.** Downloads in tests use `file://` URLs pointing at fixtures built in `tmp_path`, or a monkeypatched `urllib.request.urlopen`.
- Never commit anything under `data/` except `data/manifests/*.yaml` (already git-ignored; do not change `.gitignore`).
- Errors: `FetchError(Exception)` and `ManifestError(ValueError)` (PR A); `ParseError(ValueError)`, `SplitError(ValueError)` and `PrepareError(Exception)` (PR B). Bools are rejected wherever an int is expected.

### The two sources

| name | url | file | license |
|---|---|---|---|
| `etth1` | `https://raw.githubusercontent.com/zhouhaoyi/ETDataset/1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv` | `ETTh1.csv` | `CC-BY-ND-4.0` |
| `ecl` | `https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip` | `electricityloaddiagrams20112014.zip` | `CC-BY-4.0` |

The ETTh1 URL is pinned to a commit, not a branch. The planner downloaded both on 2026-10-11 and got:

- `ETTh1.csv`: 2,589,657 bytes, SHA-256 `f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066`; 17,420 data rows, hourly from `2016-07-01 00:00:00` to `2018-06-26 19:00:00`, columns `date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`.
- the ECL zip: 261,335,609 bytes, SHA-256 `f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1`; contains `LD2011_2014.txt` (plus `__MACOSX/` entries to ignore): a header `"";"MT_001";...;"MT_370"`, then 140,256 rows at 15-minute steps from `"2011-01-01 00:15:00"` to `"2015-01-01 00:00:00"`, `;`-separated, decimal commas, values in kW.

If your download gives a different hash or size, **do not commit a manifest**: open a `needs-human` issue with both hashes and stop. Redistribution is not needed and not allowed: the data stays in git-ignored `data/`.

---

## PR A: `feat/real-data-fetch`

Download and pin the raw files. No parsing.

### Files and interfaces

`arena/realdata/sources.py`

```python
@dataclass(frozen=True)
class Source:
    name: str  # lowercase [a-z0-9_]+
    url: str  # http(s) or file
    filename: str  # bare file name, no path separators
    license: str


SOURCES: Mapping[str, Source]  # exactly "etth1" and "ecl", as in the table above
```

`arena/realdata/manifest.py`

```python
MANIFEST_VERSION = "1"


@dataclass(frozen=True)
class RawManifest:
    dataset: str
    url: str
    filename: str
    license: str
    sha256: str  # 64 lowercase hex chars
    size_bytes: int  # > 0
    downloaded_on: str  # UTC date, "YYYY-MM-DD"
    manifest_version: str = MANIFEST_VERSION


def raw_manifest_path(
    manifest_dir: Path, dataset: str
) -> Path: ...  # <manifest_dir>/<dataset>.yaml
def write_raw_manifest(path: Path, manifest: RawManifest) -> None: ...
def read_raw_manifest(path: Path, source: Source) -> RawManifest: ...
```

`arena/realdata/fetch.py`

```python
def raw_path(data_root: Path, source: Source) -> Path: ...  # <data_root>/raw/<name>/<filename>
def fetch(
    source: Source,
    data_root: Path,
    manifest_dir: Path,
    *,
    today: date | None = None,
    timeout: float = 60.0,
) -> RawManifest: ...
def verify_raw(source: Source, data_root: Path, manifest_dir: Path) -> RawManifest: ...
def file_sha256(path: Path) -> tuple[str, int]: ...  # (hex digest, size), read in 1 MiB chunks
```

CLI: `evolvets data fetch DATASET` and `evolvets data verify DATASET`, `DATASET` in `{etth1, ecl, all}`, options `--data-root` (default `data`) and `--manifest-dir` (default `data/manifests`). Print `OK <dataset>: <sha256> <size> bytes` or `FAIL <dataset>: <reason>` per dataset, in the order `etth1, ecl`; exit 1 if any failed, else 0. Errors are caught per dataset and reported, not raised as tracebacks.

### Exact behaviour

- `RawManifest` validates in `__post_init__` (`ManifestError`): all strings non-empty; `sha256` matches `^[0-9a-f]{64}$`; `size_bytes` is an `int`, not a `bool`, and `> 0`; `downloaded_on` parses with `date.fromisoformat` and is exactly 10 characters; `manifest_version == "1"`.
- `write_raw_manifest` writes the JSON form described in the rules (keys = the dataclass fields), atomically.
- `read_raw_manifest(path, source)`: the file must be a JSON object with exactly the dataclass keys, else `ManifestError`. `dataset`, `url`, `filename` and `license` must equal the source's, else `ManifestError` ("manifest is for a different source"). This catches a registry edit that was not accompanied by a new manifest.
- `fetch`:
  1. If the manifest exists, read it (errors propagate). If the target file exists: hash it; if it matches the manifest, return the manifest without any network access. If it does not match, raise `FetchError` naming both hashes. Never delete or overwrite an existing target file.
  2. If the manifest does not exist and the target file exists, raise `FetchError` (an unpinned file of unknown origin; the user removes it by hand).
  3. Otherwise download: `urllib.request.urlopen(source.url, timeout=timeout)`, read in 1 MiB chunks into `<target>.part` (created or truncated; parent dirs created), hashing and counting as it goes; flush and `fsync`. On any exception during the download, delete `<target>.part` and raise `FetchError` from it. A zero-byte download raises `FetchError` and leaves nothing behind.
  4. If a manifest exists and the hash or size differs: delete `.part`, raise `FetchError` ("source content changed"), leave the manifest byte-for-byte unchanged and the target absent.
  5. Otherwise `os.replace(.part, target)`. If no manifest existed, write one with `downloaded_on = (today or datetime.now(UTC).date()).isoformat()`. The target is moved into place **before** the manifest is written, so a crash between the two leaves case 2, which refuses to proceed rather than silently pinning a file.
  - A stale `.part` left by a hard kill is simply truncated by the next download.
- `verify_raw`: reads the manifest and hashes the target; raises `FetchError` if the file is missing or differs; returns the manifest otherwise. No network.

### Committed manifests

Run `uv run evolvets data fetch all` and commit `data/manifests/etth1.yaml` and `data/manifests/ecl.yaml`. Their `sha256` and `size_bytes` must equal the values in "The two sources". Paste the CLI output in the PR description.

### Tests for PR A (all required; `tests/arena/test_realdata_fetch.py`, `test_realdata_manifest.py`)

1. Fresh fetch from a `file://` fixture with an injected `today`: target exists with the fixture bytes, manifest written with the right `sha256`, `size_bytes`, `url`, `dataset`, `filename`, `license`, `downloaded_on`, `manifest_version`; no `.part` file left.
2. Second fetch with the file and manifest in place does no network access: the source fixture is deleted first (or `urlopen` monkeypatched to raise) and fetch still returns the manifest.
3. Manifest present, target missing, source has the same bytes: the target is restored and the manifest file is byte-identical to before (including `downloaded_on`, even with a different `today`).
4. Manifest present, target missing, source content changed: `FetchError`; target absent; no `.part`; manifest byte-identical.
5. Target present with content that does not match the manifest: `FetchError`; the target file is unchanged.
6. Target present with no manifest: `FetchError`; no manifest written; the target is unchanged.
7. Download failure mid-stream (monkeypatched `urlopen` whose `read` returns one chunk and then raises `OSError`): `FetchError`; no target, no `.part`, no manifest. A stale `.part` with junk bytes, created by hand, is overwritten by the next successful fetch.
8. Empty source: `FetchError`; no target, no `.part`, no manifest.
9. Manifest round trip: write then read gives an equal `RawManifest`; the file text equals `json.dumps(..., sort_keys=True, indent=2, ...) + "\n"`. Each of these raises `ManifestError`: missing key, extra key, non-object JSON, invalid JSON, uppercase or 63-char `sha256`, `size_bytes` of `0`, `-1` and `True`, `downloaded_on` of `"2026-1-1"` and `"2026-02-30"`, `manifest_version` `"2"`, and `dataset`, `url`, `filename` or `license` differing from the source (one case each).
10. `verify_raw`: passes on a good file; raises on a missing file and on a file with one byte changed.
11. CLI on fixtures (monkeypatch `SOURCES` to `file://` sources): `data fetch all` then `data verify all` exit 0 with one `OK` line per dataset in order; after tampering one file, `data verify all` exits 1 with a `FAIL` line for that dataset and an `OK` line for the other. An unknown dataset name is rejected by argparse (exit code 2).
12. Registry: `SOURCES` has exactly the keys `etth1` and `ecl` with the URLs, file names and licenses in the table; the ETTh1 URL contains the commit `1d16c8f4f943005d613b5bc962e9eeb06058cf07`.
13. Committed manifests: `data/manifests/etth1.yaml` and `ecl.yaml` read without error against `SOURCES`, and their `sha256` and `size_bytes` equal the values pinned in the test file (the values in "The two sources").

### Size

About 200 lines of implementation. If it grows past ~400, split the CLI and its tests into a follow-up PR.

---

## PR B: `feat/real-data-splits`

Parse the raw files into hourly series, pick the ECL subset, cut chronological splits, and write processed files plus a split manifest per dataset.

### Files and interfaces

`arena/realdata/parse.py`

```python
@dataclass(frozen=True)
class HourlySeries:
    series_id: str  # "etth1/OT", "ecl/MT_001"
    start: str  # "YYYY-MM-DD HH:MM:SS", timestamp of values[0]; later values are consecutive hours
    values: tuple[float, ...]  # all finite


def parse_etth1(path: Path) -> tuple[HourlySeries, ...]: ...
def parse_ecl(
    path: Path, window_start: str = "2012-01-01 00:00:00", window_hours: int = 26304
) -> tuple[HourlySeries, ...]: ...
def select_ecl_subset(series: Sequence[HourlySeries], n: int = 20) -> tuple[HourlySeries, ...]: ...
```

`arena/realdata/splits.py`

```python
MAX_HORIZON = 168
SPLIT_PERCENT = (
    20,
    35,
    30,
)  # warmup, dev, sealed, as integer percent of usable length; reserve gets the rest


@dataclass(frozen=True)
class Segment:
    name: Literal["warmup", "dev", "sealed", "reserve"]
    start: int
    end: int  # exclusive


@dataclass(frozen=True)
class Split:
    length: int
    gap: int
    segments: tuple[Segment, Segment, Segment, Segment]  # warmup, dev, sealed, reserve


def chronological_split(length: int, gap: int = MAX_HORIZON) -> Split: ...
```

`arena/realdata/prepare.py`

```python
@dataclass(frozen=True)
class PublicSeries:
    series_id: str
    start: str
    values: tuple[float, ...]  # values[0 : split.sealed.start] only
    split: Split


def splits_manifest_path(
    manifest_dir: Path, dataset: str
) -> Path: ...  # <manifest_dir>/<dataset>.splits.yaml
def prepare(dataset: str, data_root: Path, manifest_dir: Path) -> None: ...
def load_public(dataset: str, data_root: Path, manifest_dir: Path) -> tuple[PublicSeries, ...]: ...
```

CLI: `evolvets data prepare DATASET` with the same options and output conventions as PR A (`OK <dataset>: <n> series, processed <sha256 of public file>`). `evolvets data verify` additionally checks, when a split manifest exists, that both processed files exist and match their hashes.

### Exact behaviour: parsing

- `parse_etth1`: UTF-8 text. The header must be exactly `date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT`. At least one data row. Every row has 8 fields; the first is `YYYY-MM-DD HH:MM:SS` and each is exactly one hour after the previous (no gaps, no duplicates); every other field parses with `float()` to a finite value. Any violation raises `ParseError` with the 1-based line number. Returns 7 series, in header order, with ids `etth1/<column>` and `start` = the first row's timestamp.
- `parse_ecl`: opens the zip and reads `LD2011_2014.txt` (other members ignored; missing member raises `ParseError`). The header's first field is `""` and the rest are quoted client ids; every row has the same number of fields. The first field is a quoted `YYYY-MM-DD HH:MM:SS`; consecutive rows are exactly 15 minutes apart. Values: replace `,` by `.` and parse with `float()`; must be finite and `>= 0`.
  - **Hourly aggregation.** A reading stamped `T` covers `(T - 15 min, T]`. The hourly value for hour `H` (covering `[H, H + 1 h)`) is `(v1 + v2 + v3 + v4) / 4` for the readings stamped `H+15m, H+30m, H+45m, H+60m`, summed left to right: the mean load in kW over the hour, which equals kWh. Hours run from `window_start` for `window_hours` hours (defaults: all of 2012–2014, 26,304 hours). Readings outside the window are read, validated and ignored. If the file does not cover the whole window, raise `ParseError`.
  - Keep peak memory well under 1 GB: accumulate in `array.array("d")` per client and convert to tuples only at the end. Expect about a minute on the full file.
  - Known artefacts in the source, **kept as is** (they hit every method equally): DST transitions produce irregular values around 01:00–02:00 local time. The aggregation formula applies unchanged; do not write tests asserting specific values for those hours.
  - Returns one series per client, in header order, ids `ecl/<client>`, `start = window_start`.
- `select_ecl_subset`: a series is **eligible** iff it has no run of 24 or more consecutive values exactly equal to `0.0` (this drops clients that start late or stop early). Rank eligible series by `sha256(b"evolvets/ecl-subset/v1/" + client_id.encode("utf-8")).hexdigest()` (client id without the `ecl/` prefix), take the first `n`, and return them sorted by `series_id`. Fewer than `n` eligible raises `ParseError`. The rule never looks at forecast performance.
  - On the real file the planner's computation gave 289 eligible clients and this subset: `MT_001, MT_010, MT_011, MT_018, MT_052, MT_067, MT_090, MT_119, MT_138, MT_157, MT_172, MT_193, MT_194, MT_208, MT_227, MT_238, MT_251, MT_284, MT_297, MT_306`. If your implementation gives anything else, stop and open a `needs-human` issue rather than adjust either side.

### Exact behaviour: splits

- `chronological_split(length, gap)`: `length` and `gap` are ints (not bools), `gap >= 1`. `usable = length - 3 * gap`; `w = usable * 20 // 100`, `d = usable * 35 // 100`, `s = usable * 30 // 100`, `r = usable - w - d - s`. Segments: warmup `[0, w)`, dev `[w + gap, w + gap + d)`, sealed `[dev.end + gap, dev.end + gap + s)`, reserve `[sealed.end + gap, length)`. If any segment would be empty, raise `SplitError` (so `length = 3 * gap + 5` is the smallest valid length).
- Meaning: segments are ranges of **forecast origins**. With `gap = MAX_HORIZON`, for every origin `t` in a segment and every `h <= MAX_HORIZON`, the target `t + h` is before the next segment's start. Dev targets therefore never fall in the sealed segment. Experts may still read all history before their origin; that is not leakage.
- Reserve is never used in Phase 0. It is kept for later live-style checks.
- Expected boundaries: ETTh1 (`length = 17420`): warmup `[0, 3383)`, dev `[3551, 9471)`, sealed `[9639, 14713)`, reserve `[14881, 17420)`. ECL (`length = 26304`): warmup `[0, 5160)`, dev `[5328, 14358)`, sealed `[14526, 22266)`, reserve `[22434, 26304)`.

### Exact behaviour: prepare and load

- `prepare.py` has module constants `ECL_WINDOW_START = "2012-01-01 00:00:00"`, `ECL_WINDOW_HOURS = 26304` and `ECL_SUBSET_SIZE = 20`, which `prepare` passes to `parse_ecl` and `select_ecl_subset`. Tests monkeypatch them to use small fixtures; nothing else changes them.
- `prepare(dataset, ...)`: unknown dataset raises `PrepareError`. First `verify_raw` (PR A); on failure nothing is written. Parse (for `ecl`, then select the subset). Compute each series' split with `chronological_split(len(values))`.
- It writes two processed files, atomically, under `<data_root>/processed/`:
  - `<dataset>.public.json`: `canonical_bytes` (from `ledger/canonical.py`) of `{"dataset", "realdata_version", "series": [{"series_id", "start", "values": values[:sealed.start]}, ...]}`;
  - `<dataset>.sealed.json`: the same with full `values`. Only PR 9's sealed process will read it; nothing in this task does except `prepare`'s own hashing and `data verify`.
  - Floats go through `json` as `repr`, so the bytes are exact and platform-independent (no `libm` involved).
- Then the split manifest `<manifest_dir>/<dataset>.splits.yaml` (JSON-as-YAML, as above) with exactly these keys: `manifest_version` (`"1"`), `realdata_version`, `dataset`, `raw_sha256` (from the raw manifest), `public_sha256`, `sealed_sha256` (SHA-256 of the two processed files' bytes), `max_horizon`, `split_percent` (list), `series`: a list, in output order, of `{"series_id", "start", "length", "segments": {name: {"start", "end", "first_time", "last_time"}}}` where `first_time` and `last_time` are the timestamps of indices `start` and `end - 1`.
- The manifest contains no dates, so `prepare` is fully deterministic. If the split manifest already exists, the newly computed manifest text must be byte-identical; if not, raise `PrepareError`, leave the existing manifest unchanged, and write no processed files (compute both files and the manifest in memory first, compare, then write). Processed files are written before the manifest.
- `load_public(dataset, ...)`: reads the split manifest and **only** `<dataset>.public.json`. It checks the file's SHA-256 against `public_sha256` (mismatch or missing file raises `PrepareError`), and returns `PublicSeries` with the split rebuilt from the manifest; each series' `len(values)` must equal its sealed start. It never opens the sealed file or the raw file.

### Committed manifests

Run `uv run evolvets data prepare all` and commit `data/manifests/etth1.splits.yaml` and `data/manifests/ecl.splits.yaml`. Paste the CLI output and the `data verify all` output in the PR description.

### Tests for PR B (all required; `tests/arena/test_realdata_parse.py`, `test_realdata_splits.py`, `test_realdata_prepare.py`)

1. `chronological_split`: the ETTh1 and ECL boundaries above exactly; for every `length` from `3*gap + 5` to `3*gap + 400` with `gap = 7`, segments are in order, non-empty, separated by exactly `gap`, warmup starts at 0, reserve ends at `length`, and every origin `t` in a segment has `t + gap <` the next segment's start; `length = 3*gap + 4`, `gap = 0` and `bool` arguments raise `SplitError`.
2. `parse_etth1` on a small fixture: exact values and `start`; each of these raises `ParseError`: wrong header, header only, a skipped hour, a duplicated hour, a row with 7 fields, an empty field, `nan`, `inf`.
3. `parse_ecl` on a small zip fixture (3 clients, a window of a few hours): hourly means exact, including that a reading stamped `H:00` belongs to hour `H - 1`; decimal commas and bare integers parse; `__MACOSX` members ignored; each of these raises `ParseError`: missing member, a skipped 15-minute step, a row with the wrong field count, a negative value, a window that starts before or ends after the data.
4. `select_ecl_subset`: a zero run of exactly 23 is eligible and of exactly 24 is not; ranking follows the SHA-256 rule (check with a hand-computed order on 5 fixture ids); output is sorted by id; fewer than `n` eligible raises.
5. `prepare` on fixture data (fixture sources monkeypatched into `SOURCES`, short enough series with `MAX_HORIZON` kept at 168, so fixtures need more than `3*168 + 5` hours): writes both processed files and the manifest; running it again leaves all three byte-identical; the manifest has exactly the listed keys.
6. `prepare` with an existing split manifest that differs (one boundary edited by hand): `PrepareError`; manifest and both processed files unchanged.
7. `prepare` with a raw file that fails `verify_raw`: error, and no processed file or split manifest is created.
8. Crash safety: with `os.replace` monkeypatched to raise when moving the split manifest, no partial manifest exists afterwards, and a rerun succeeds.
9. No look-ahead into sealed: two fixture sources that differ only at indices `>= sealed.start` of a series produce byte-identical public files and different sealed files, and `load_public` returns equal results for both.
10. `load_public`: returns exactly `values[:sealed.start]` per series and the manifest's splits; still works with the sealed file and the raw file deleted; records every file open (monkeypatch `builtins.open`, `io.open`, `os.open` and `pathlib.Path.open`) and none is the sealed or raw file; a public file with one byte changed raises `PrepareError`.
11. Golden: the SHA-256 of the public and sealed files for the fixture in test 5 are pinned in the test file; the docstring says changing them requires bumping `REALDATA_VERSION`.
12. CLI on fixtures: `data prepare all` then `data verify all` exit 0; deleting a processed file makes `data verify all` exit 1 with a `FAIL` line for that dataset.
13. Committed split manifests: both read; ETTh1 lists the 7 series `etth1/HUFL ... etth1/OT` in header order, each of length 17,420 starting `2016-07-01 00:00:00` with the boundaries above; ECL lists exactly the 20 pinned clients, each of length 26,304 starting `2012-01-01 00:00:00` with the boundaries above; each `raw_sha256` equals the committed raw manifest's `sha256`; `max_horizon == 168` and `realdata_version == "1"`.

### Size

About 300 lines of implementation. If it grows past ~400, split `prepare.py`, the CLI and their tests into a follow-up PR (`feat/real-data-prepare`).

---

## Not in this task

- **Serving sealed real data.** Reading `<dataset>.sealed.json` from a separate process, and writing sealed scores to the sealed ledger stream, is PR 9 (`feat/arena-sealed`). Note that the public/sealed file split here is a convenience, not isolation: any process can read any file in `data/`. Isolation comes from PR 9's process boundary.
- **Import contracts.** Still pending from TASK_004: agent-editable packages must not import `evolvets.arena.synthetic` (oracle) once updaters exist; the same question applies to `evolvets.arena.realdata.prepare`'s file paths. A new `.importlinter` contract needs human review, so the planner raises it when specifying PR 8 or PR 10.
- Experts and the prediction cache (PR 7), replay and label release (PR 8), the switching oracle (PR 10).

## Closing the task

PR B (`feat/real-data-splits`, or its follow-up if split) changes this file's first line to `Status: done`.
