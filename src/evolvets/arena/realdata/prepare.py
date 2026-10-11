"""Processed real-data files and the split manifests that pin them.

``prepare`` turns a verified raw file into hourly series, cuts each series
into chronological segments and writes, under ``<data_root>/processed/``:

- ``<dataset>.public.json`` — values up to the sealed segment start only;
- ``<dataset>.sealed.json`` — the full values.

plus a split manifest ``<manifest_dir>/<dataset>.splits.yaml`` (JSON in a
``.yaml`` file, as for the raw manifests). The manifest contains no dates,
so ``prepare`` is fully deterministic: re-running it on the same raw file
reproduces byte-identical outputs, and an existing manifest that differs
from the recomputed one is an error, never something edited in place.

``load_public`` reads the manifest and the public file only — it never
opens the sealed or the raw file, so nothing outside the sealed process
(added in a later task) reaches sealed targets through this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from evolvets.arena.realdata import REALDATA_VERSION, sources
from evolvets.arena.realdata.fetch import file_sha256, raw_path, verify_raw
from evolvets.arena.realdata.manifest import manifest_text, write_atomic
from evolvets.arena.realdata.parse import (
    HourlySeries,
    ParseError,
    format_timestamp,
    parse_ecl,
    parse_etth1,
    parse_timestamp,
    select_ecl_subset,
)
from evolvets.arena.realdata.splits import (
    MAX_HORIZON,
    SPLIT_PERCENT,
    Segment,
    SegmentName,
    Split,
    SplitError,
    chronological_split,
)
from evolvets.ledger.canonical import canonical_bytes, sha256_hex

ECL_WINDOW_START = "2012-01-01 00:00:00"
ECL_WINDOW_HOURS = 26304
ECL_SUBSET_SIZE = 20

SPLIT_MANIFEST_VERSION = "1"

_MANIFEST_KEYS = (
    "manifest_version",
    "realdata_version",
    "dataset",
    "raw_sha256",
    "public_sha256",
    "sealed_sha256",
    "max_horizon",
    "split_percent",
    "series",
)
_SERIES_KEYS = ("series_id", "start", "length", "segments")
_SEGMENT_KEYS = ("start", "end", "first_time", "last_time")
_SEGMENT_NAMES: tuple[SegmentName, ...] = ("warmup", "dev", "sealed", "reserve")
_HEX = set("0123456789abcdef")


class PrepareError(Exception):
    """Processed files or a split manifest are missing, tampered or stale."""


@dataclass(frozen=True)
class PublicSeries:
    """One series as seen outside the sealed process: public values + split."""

    series_id: str
    start: str
    values: tuple[float, ...]  # values[0 : split.sealed.start] only
    split: Split


@dataclass(frozen=True)
class SplitManifest:
    """The parsed contents of a ``<dataset>.splits.yaml`` file."""

    dataset: str
    realdata_version: str
    raw_sha256: str
    public_sha256: str
    sealed_sha256: str
    series: tuple[tuple[str, str, Split], ...]  # (series_id, start, split) in file order


def splits_manifest_path(manifest_dir: Path, dataset: str) -> Path:
    """Return ``<manifest_dir>/<dataset>.splits.yaml``."""
    return manifest_dir / f"{dataset}.splits.yaml"


def prepare(dataset: str, data_root: Path, manifest_dir: Path) -> None:
    """Build the public/sealed files and the split manifest for ``dataset``.

    The raw file is verified first; on failure nothing is written. All
    outputs are computed in memory before anything is written, and the
    processed files are written before the manifest.
    """
    source = sources.SOURCES.get(dataset)
    if source is None:
        raise PrepareError(
            f"unknown dataset {dataset!r}; expected one of {sorted(sources.SOURCES)}"
        )
    raw_manifest = verify_raw(source, data_root, manifest_dir)
    if dataset == "etth1":
        series = parse_etth1(raw_path(data_root, source))
    else:
        parsed = parse_ecl(raw_path(data_root, source), ECL_WINDOW_START, ECL_WINDOW_HOURS)
        series = select_ecl_subset(parsed, ECL_SUBSET_SIZE)
    splits = [chronological_split(len(entry.values)) for entry in series]
    public_bytes = _processed_bytes(dataset, series, splits, sealed=False)
    sealed_bytes = _processed_bytes(dataset, series, splits, sealed=True)
    manifest_bytes = manifest_text(
        {
            "manifest_version": SPLIT_MANIFEST_VERSION,
            "realdata_version": REALDATA_VERSION,
            "dataset": dataset,
            "raw_sha256": raw_manifest.sha256,
            "public_sha256": sha256_hex(public_bytes),
            "sealed_sha256": sha256_hex(sealed_bytes),
            "max_horizon": MAX_HORIZON,
            "split_percent": list(SPLIT_PERCENT),
            "series": [
                {
                    "series_id": entry.series_id,
                    "start": entry.start,
                    "length": len(entry.values),
                    "segments": _segments_obj(entry, split),
                }
                for entry, split in zip(series, splits, strict=True)
            ],
        }
    ).encode("utf-8")
    manifest_file = splits_manifest_path(manifest_dir, dataset)
    if manifest_file.exists() and manifest_file.read_bytes() != manifest_bytes:
        raise PrepareError(
            f"{manifest_file} exists with different content: split manifests are never "
            "edited in place; if the data changed deliberately, bump REALDATA_VERSION "
            "and remove the old artefacts"
        )
    processed_dir = data_root / "processed"
    write_atomic(processed_dir / f"{dataset}.public.json", public_bytes)
    write_atomic(processed_dir / f"{dataset}.sealed.json", sealed_bytes)
    write_atomic(manifest_file, manifest_bytes)


def sealed_start(split: Split) -> int:
    """Return the start index of the sealed segment of ``split``."""
    return split.segments[2].start


def _segments_obj(entry: HourlySeries, split: Split) -> dict[str, dict[str, object]]:
    """Return the per-segment manifest entry for one series."""
    start = parse_timestamp(entry.start)
    return {
        segment.name: {
            "start": segment.start,
            "end": segment.end,
            "first_time": format_timestamp(start + timedelta(hours=segment.start)),
            "last_time": format_timestamp(start + timedelta(hours=segment.end - 1)),
        }
        for segment in split.segments
    }


def _processed_bytes(
    dataset: str, series: tuple[HourlySeries, ...], splits: list[Split], *, sealed: bool
) -> bytes:
    """Return the canonical bytes of a processed (public or sealed) file."""
    return canonical_bytes(
        {
            "dataset": dataset,
            "realdata_version": REALDATA_VERSION,
            "series": [
                {
                    "series_id": entry.series_id,
                    "start": entry.start,
                    "values": list(entry.values)
                    if sealed
                    else list(entry.values[: sealed_start(split)]),
                }
                for entry, split in zip(series, splits, strict=True)
            ],
        }
    )


def read_split_manifest(manifest_dir: Path, dataset: str) -> SplitManifest:
    """Read and validate ``<dataset>.splits.yaml``; raise PrepareError on any fault.

    The stored segments must match :func:`chronological_split` recomputed
    from the stored length and ``max_horizon``, so a hand-edited boundary
    cannot pass unnoticed.
    """
    path = splits_manifest_path(manifest_dir, dataset)
    if not path.exists():
        raise PrepareError(f"{path} is missing; run 'evolvets data prepare {dataset}'")
    try:
        obj: object = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise PrepareError(f"{path}: cannot be read: {exc}") from exc
    if not isinstance(obj, dict):
        raise PrepareError(f"{path}: manifest must be a JSON object")
    where = f"{path}: "
    obj = {str(key): value for key, value in obj.items()}
    if set(obj) != set(_MANIFEST_KEYS):
        raise PrepareError(f"{where}keys must be exactly {sorted(_MANIFEST_KEYS)}")
    if obj["manifest_version"] != SPLIT_MANIFEST_VERSION:
        raise PrepareError(f"{where}manifest_version must be {SPLIT_MANIFEST_VERSION!r}")
    if obj["dataset"] != dataset:
        raise PrepareError(f"{where}manifest is for dataset {obj['dataset']!r}, not {dataset!r}")
    if obj["realdata_version"] != REALDATA_VERSION:
        raise PrepareError(
            f"{where}realdata_version {obj['realdata_version']!r} != current {REALDATA_VERSION!r}"
        )
    if obj["max_horizon"] != MAX_HORIZON:
        raise PrepareError(f"{where}max_horizon must be {MAX_HORIZON}, got {obj['max_horizon']!r}")
    percent = obj["split_percent"]
    if not isinstance(percent, list) or tuple(percent) != SPLIT_PERCENT:
        raise PrepareError(f"{where}split_percent must be {list(SPLIT_PERCENT)}, got {percent!r}")
    for key in ("raw_sha256", "public_sha256", "sealed_sha256"):
        value = obj[key]
        if not isinstance(value, str) or len(value) != 64 or not set(value) <= _HEX:
            raise PrepareError(f"{where}{key} must be 64 lowercase hex chars, got {value!r}")
    series_obj = obj["series"]
    if not isinstance(series_obj, list) or not series_obj:
        raise PrepareError(f"{where}series must be a non-empty list")
    return SplitManifest(
        dataset=dataset,
        realdata_version=REALDATA_VERSION,
        raw_sha256=obj["raw_sha256"],
        public_sha256=obj["public_sha256"],
        sealed_sha256=obj["sealed_sha256"],
        series=tuple(
            _parse_series_entry(path, entry, index) for index, entry in enumerate(series_obj)
        ),
    )


def _parse_series_entry(path: Path, entry: object, index: int) -> tuple[str, str, Split]:
    """Validate one ``series`` entry of a split manifest and return its split."""
    where = f"{path}: series[{index}]: "
    if not isinstance(entry, dict):
        raise PrepareError(f"{where}must be a JSON object")
    entry = {str(key): value for key, value in entry.items()}
    if set(entry) != set(_SERIES_KEYS):
        raise PrepareError(f"{where}keys must be exactly {sorted(_SERIES_KEYS)}")
    series_id = entry["series_id"]
    if not isinstance(series_id, str) or not series_id:
        raise PrepareError(f"{where}series_id must be a non-empty string")
    start = entry["start"]
    if not isinstance(start, str):
        raise PrepareError(f"{where}start must be a string")
    try:
        start_time = parse_timestamp(start)
    except ParseError as exc:
        raise PrepareError(f"{where}{exc}") from exc
    length = entry["length"]
    if isinstance(length, bool) or not isinstance(length, int):
        raise PrepareError(f"{where}length must be an int, got {length!r}")
    segments_obj = entry["segments"]
    if not isinstance(segments_obj, dict) or set(map(str, segments_obj)) != set(_SEGMENT_NAMES):
        raise PrepareError(f"{where}segments must have exactly {list(_SEGMENT_NAMES)}")
    segments: list[Segment] = []
    for name in _SEGMENT_NAMES:
        segment_obj = segments_obj[name]
        if not isinstance(segment_obj, dict) or set(map(str, segment_obj)) != set(_SEGMENT_KEYS):
            raise PrepareError(
                f"{where}segment {name}: keys must be exactly {sorted(_SEGMENT_KEYS)}"
            )
        seg_start = segment_obj["start"]
        seg_end = segment_obj["end"]
        for key, value in (("start", seg_start), ("end", seg_end)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise PrepareError(f"{where}segment {name}: {key} must be an int, got {value!r}")
        if not 0 <= seg_start < seg_end <= length:
            raise PrepareError(f"{where}segment {name}: need 0 <= start < end <= {length}")
        expected_times = {
            "first_time": format_timestamp(start_time + timedelta(hours=seg_start)),
            "last_time": format_timestamp(start_time + timedelta(hours=seg_end - 1)),
        }
        for key, want in expected_times.items():
            if segment_obj[key] != want:
                raise PrepareError(
                    f"{where}segment {name}: {key} must be {want!r}, got {segment_obj[key]!r}"
                )
        segments.append(Segment(name, seg_start, seg_end))
    split = Split(
        length=length,
        gap=MAX_HORIZON,
        segments=(segments[0], segments[1], segments[2], segments[3]),
    )
    try:
        recomputed = chronological_split(length, MAX_HORIZON)
    except SplitError as exc:
        raise PrepareError(f"{where}{exc}") from exc
    if split != recomputed:
        raise PrepareError(
            f"{where}segments do not match the recomputed chronological split {recomputed}"
        )
    return series_id, start, split


def load_public(dataset: str, data_root: Path, manifest_dir: Path) -> tuple[PublicSeries, ...]:
    """Load the public series of ``dataset``; never opens sealed or raw files.

    The public file's SHA-256 must match the split manifest, and each
    series' length must equal its sealed segment start.
    """
    manifest = read_split_manifest(manifest_dir, dataset)
    public_file = data_root / "processed" / f"{dataset}.public.json"
    if not public_file.exists():
        raise PrepareError(f"{public_file} is missing; run 'evolvets data prepare {dataset}'")
    digest, _ = file_sha256(public_file)
    if digest != manifest.public_sha256:
        raise PrepareError(
            f"{public_file} does not match the split manifest: file sha256 {digest}, "
            f"manifest sha256 {manifest.public_sha256}"
        )
    try:
        obj: object = json.loads(public_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise PrepareError(f"{public_file}: cannot be read: {exc}") from exc
    if not isinstance(obj, dict):
        raise PrepareError(f"{public_file}: must be a JSON object")
    obj = {str(key): value for key, value in obj.items()}
    if set(obj) != {"dataset", "realdata_version", "series"}:
        raise PrepareError(f"{public_file}: keys must be exactly dataset, realdata_version, series")
    if obj["dataset"] != dataset:
        raise PrepareError(f"{public_file}: dataset must be {dataset!r}, got {obj['dataset']!r}")
    if obj["realdata_version"] != REALDATA_VERSION:
        raise PrepareError(
            f"{public_file}: realdata_version must be {REALDATA_VERSION!r}, "
            f"got {obj['realdata_version']!r}"
        )
    entries = obj["series"]
    if not isinstance(entries, list) or len(entries) != len(manifest.series):
        raise PrepareError(
            f"{public_file}: series count does not match the manifest's {len(manifest.series)}"
        )
    out: list[PublicSeries] = []
    for index, (entry, (series_id, start, split)) in enumerate(
        zip(entries, manifest.series, strict=True)
    ):
        where = f"{public_file}: series[{index}]: "
        if not isinstance(entry, dict):
            raise PrepareError(f"{where}must be a JSON object")
        entry = {str(key): value for key, value in entry.items()}
        if set(entry) != {"series_id", "start", "values"}:
            raise PrepareError(f"{where}keys must be exactly series_id, start, values")
        if entry["series_id"] != series_id or entry["start"] != start:
            raise PrepareError(
                f"{where}must be {series_id!r} starting {start!r}, got "
                f"{entry['series_id']!r} starting {entry['start']!r}"
            )
        raw_values = entry["values"]
        if not isinstance(raw_values, list):
            raise PrepareError(f"{where}values must be a list")
        checked: list[float] = []
        for value in raw_values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PrepareError(f"{where}values must be numbers, got {value!r}")
            checked.append(float(value))
        public_len = sealed_start(split)
        if len(checked) != public_len:
            raise PrepareError(f"{where}length {len(checked)} != sealed segment start {public_len}")
        out.append(PublicSeries(series_id, start, tuple(checked), split))
    return tuple(out)
