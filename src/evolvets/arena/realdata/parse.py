"""Parse the pinned raw files into hourly series.

ETTh1 is already hourly. The ECL zip holds 15-minute readings stamped with
the end of the interval they cover, so hour ``H`` (covering
``[H, H + 1h)``) aggregates the readings stamped ``H+15m .. H+60m``: their
mean is the load in kW over the hour, which equals the energy in kWh.
Accumulation uses one ``array.array("d")`` per client over the window only,
so peak memory stays well under 1 GB on the full file.
"""

from __future__ import annotations

import hashlib
import io
import math
import zipfile
from array import array
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"
_HOUR = timedelta(hours=1)
_QUARTER = timedelta(minutes=15)
_ETTH1_HEADER = "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT"
_ECL_MEMBER = "LD2011_2014.txt"
_SUBSET_SALT = b"evolvets/ecl-subset/v1/"
_ZERO_RUN = 24


class ParseError(ValueError):
    """A raw data file violates the pinned format."""


@dataclass(frozen=True)
class HourlySeries:
    """One hourly series: consecutive hours starting at ``start``."""

    series_id: str  # "etth1/OT", "ecl/MT_001"
    start: str  # "YYYY-MM-DD HH:MM:SS", timestamp of values[0]
    values: tuple[float, ...]  # all finite


def parse_timestamp(text: str) -> datetime:
    """Parse a strict ``YYYY-MM-DD HH:MM:SS`` timestamp or raise ParseError."""
    if len(text) != 19:
        raise ParseError(f"timestamp must be YYYY-MM-DD HH:MM:SS, got {text!r}")
    try:
        return datetime.strptime(text, _TIMESTAMP_FORMAT)
    except ValueError as exc:
        raise ParseError(f"timestamp must be YYYY-MM-DD HH:MM:SS, got {text!r}") from exc


def format_timestamp(moment: datetime) -> str:
    """Format ``moment`` as a strict ``YYYY-MM-DD HH:MM:SS`` timestamp."""
    return moment.strftime(_TIMESTAMP_FORMAT)


def _decode(path: Path, data: bytes) -> str:
    """Decode raw bytes as UTF-8 text, raising ParseError on failure."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ParseError(f"{path}: not valid UTF-8: {exc}") from exc


def parse_etth1(path: Path) -> tuple[HourlySeries, ...]:
    """Parse ``ETTh1.csv`` into its 7 hourly series, in header order.

    Every row must have 8 fields, hourly consecutive timestamps and finite
    values; any violation raises :class:`ParseError` with the 1-based line
    number.
    """
    try:
        text = _decode(path, path.read_bytes())
    except OSError as exc:
        raise ParseError(f"cannot read {path}: {exc}") from exc
    lines = text.splitlines()
    if not lines or lines[0] != _ETTH1_HEADER:
        raise ParseError(f"{path}: line 1: header must be {_ETTH1_HEADER!r}")
    columns = _ETTH1_HEADER.split(",")[1:]
    values: list[list[float]] = [[] for _ in columns]
    start: str | None = None
    prev: datetime | None = None
    for lineno, line in enumerate(lines[1:], start=2):
        fields = line.split(",")
        where = f"{path}: line {lineno}"
        if len(fields) != len(columns) + 1:
            raise ParseError(f"{where}: expected {len(columns) + 1} fields, got {len(fields)}")
        try:
            stamp = parse_timestamp(fields[0])
        except ParseError as exc:
            raise ParseError(f"{where}: {exc}") from exc
        if prev is not None and stamp - prev != _HOUR:
            raise ParseError(
                f"{where}: {fields[0]!r} is not one hour after the previous row"
                f" ({format_timestamp(prev)!r})"
            )
        for index, raw in enumerate(fields[1:]):
            value = _finite_float(raw, f"{where}: column {columns[index]}")
            values[index].append(value)
        if start is None:
            start = fields[0]
        prev = stamp
    if start is None:
        raise ParseError(f"{path}: no data rows after the header")
    return tuple(
        HourlySeries(f"etth1/{column}", start, tuple(column_values))
        for column, column_values in zip(columns, values, strict=True)
    )


def _finite_float(raw: str, where: str) -> float:
    """Parse ``raw`` with ``float()`` and require a finite value."""
    try:
        value = float(raw)
    except ValueError as exc:
        raise ParseError(f"{where}: not a number: {raw!r}") from exc
    if not math.isfinite(value):
        raise ParseError(f"{where}: not a finite value: {raw!r}")
    return value


def parse_ecl(
    path: Path, window_start: str = "2012-01-01 00:00:00", window_hours: int = 26304
) -> tuple[HourlySeries, ...]:
    """Parse the ECL zip into one hourly series per client, in header order.

    Readings outside ``[window_start, window_start + window_hours)`` are
    read and validated but ignored; the file must cover the whole window.
    Hour ``H`` is the left-to-right mean of the readings stamped
    ``H+15m .. H+60m``. Values use decimal commas and must be finite and
    ``>= 0``.
    """
    start = parse_timestamp(window_start)
    if isinstance(window_hours, bool) or not isinstance(window_hours, int):
        raise ParseError(f"window_hours must be an int, got {window_hours!r}")
    if window_hours < 1:
        raise ParseError(f"window_hours must be >= 1, got {window_hours}")
    window_quarters = window_hours * 4
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise ParseError(f"{path}: not a zip file: {exc}") from exc
    with archive:
        try:
            lines = io.TextIOWrapper(archive.open(_ECL_MEMBER), encoding="utf-8", newline="")
        except KeyError as exc:
            raise ParseError(f"{path}: no {_ECL_MEMBER!r} member in the zip") from exc
        header = lines.readline()
        if not header:
            raise ParseError(f"{path}: file is empty")
        header_fields = header.rstrip("\r\n").split(";")
        client_ids = _parse_ecl_header(path, header_fields)
        sums = [array("d", [0.0]) * window_hours for _ in client_ids]
        counts = array("i", [0]) * window_hours
        field_count = len(header_fields)
        prev: datetime | None = None
        for lineno, line in enumerate(lines, start=2):
            where = f"{path}: line {lineno}"
            fields = line.rstrip("\r\n").split(";")
            if len(fields) != field_count:
                raise ParseError(f"{where}: expected {field_count} fields, got {len(fields)}")
            stamp_field = fields[0]
            if len(stamp_field) < 2 or stamp_field[0] != '"' or stamp_field[-1] != '"':
                raise ParseError(f"{where}: timestamp must be quoted, got {stamp_field!r}")
            try:
                stamp = parse_timestamp(stamp_field[1:-1])
            except ParseError as exc:
                raise ParseError(f"{where}: {exc}") from exc
            if prev is None:
                if (stamp - start) % _QUARTER != timedelta(0):
                    raise ParseError(
                        f"{where}: {window_start!r} is not on the file's 15-minute grid"
                    )
            elif stamp - prev != _QUARTER:
                raise ParseError(
                    f"{where}: {stamp_field[1:-1]!r} is not 15 minutes after the previous row"
                    f" ({format_timestamp(prev)!r})"
                )
            prev = stamp
            quarter = (stamp - start) // _QUARTER
            in_window = 1 <= quarter <= window_quarters
            hour = (quarter - 1) // 4 if in_window else -1
            for index, raw in enumerate(fields[1:]):
                value = _ecl_value(raw, f"{where}: client {client_ids[index]}")
                if in_window:
                    sums[index][hour] += value
            if in_window:
                counts[hour] += 1
        if prev is None:
            raise ParseError(f"{path}: no data rows after the header")
    _check_window_coverage(path, counts, window_start, window_hours)
    return tuple(
        HourlySeries(f"ecl/{client_id}", window_start, tuple(value / 4.0 for value in client_sums))
        for client_id, client_sums in zip(client_ids, sums, strict=True)
    )


def _parse_ecl_header(path: Path, header_fields: list[str]) -> list[str]:
    """Return the client ids from a split ECL header line."""
    if len(header_fields) < 2 or header_fields[0] != '""':
        raise ParseError(
            f'{path}: line 1: header must be an empty "" field followed by quoted client ids'
        )
    client_ids: list[str] = []
    for field in header_fields[1:]:
        if len(field) < 3 or field[0] != '"' or field[-1] != '"' or '"' in field[1:-1]:
            raise ParseError(
                f"{path}: line 1: client id must be non-empty and quoted, got {field!r}"
            )
        client_ids.append(field[1:-1])
    return client_ids


def _ecl_value(raw: str, where: str) -> float:
    """Parse an ECL reading (decimal comma) and require finite and ``>= 0``."""
    try:
        value = float(raw.replace(",", "."))
    except ValueError as exc:
        raise ParseError(f"{where}: not a number: {raw!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ParseError(f"{where}: value must be finite and >= 0, got {raw!r}")
    return value


def _check_window_coverage(
    path: Path, counts: array[int], window_start: str, window_hours: int
) -> None:
    """Raise ParseError unless every window hour has exactly 4 readings."""
    for hour, count in enumerate(counts):
        if count != 4:
            raise ParseError(
                f"{path}: the file does not cover the whole window "
                f"{window_start!r} + {window_hours}h: hour {hour} has {count} of 4 readings"
            )


def select_ecl_subset(series: Sequence[HourlySeries], n: int = 20) -> tuple[HourlySeries, ...]:
    """Pick the pinned ECL subset without looking at forecast performance.

    A series is eligible iff it has no run of 24 or more consecutive values
    exactly equal to ``0.0`` (this drops clients that start late or stop
    early). Eligible series are ranked by
    ``sha256("evolvets/ecl-subset/v1/" + client_id)`` and the first ``n``
    are returned sorted by ``series_id``.
    """
    if isinstance(n, bool) or not isinstance(n, int):
        raise ParseError(f"n must be an int, got {n!r}")
    if n < 1:
        raise ParseError(f"n must be >= 1, got {n}")
    eligible = [entry for entry in series if _has_no_long_zero_run(entry.values)]
    if len(eligible) < n:
        raise ParseError(f"only {len(eligible)} of {len(series)} series are eligible; need {n}")
    ranked = sorted(
        eligible,
        key=lambda entry: hashlib.sha256(
            _SUBSET_SALT + entry.series_id.removeprefix("ecl/").encode("utf-8")
        ).hexdigest(),
    )
    return tuple(sorted(ranked[:n], key=lambda entry: entry.series_id))


def _has_no_long_zero_run(values: Sequence[float]) -> bool:
    """Return True iff ``values`` has no run of 24+ consecutive zeros."""
    run = 0
    for value in values:
        if value == 0.0:
            run += 1
            if run >= _ZERO_RUN:
                return False
        else:
            run = 0
    return True
