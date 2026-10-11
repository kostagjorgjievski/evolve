"""Parser tests: ETTh1 rows, ECL quarter-to-hour aggregation, subset rule."""

from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from evolvets.arena.realdata.parse import (
    HourlySeries,
    ParseError,
    parse_ecl,
    parse_etth1,
    parse_timestamp,
    select_ecl_subset,
)

_ETTH1_HEADER = "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT"
_TS_FORMAT = "%Y-%m-%d %H:%M:%S"


def _etth1_text(rows: int, *, start: str = "2016-07-01 00:00:00", variant: int = 0) -> str:
    """Build deterministic ETTh1 text; ``variant`` shifts values at index >= 344 only."""
    moment = datetime.strptime(start, _TS_FORMAT)
    lines = [_ETTH1_HEADER]
    for index in range(rows):
        values = ",".join(
            str(column + index / 100 + (variant if index >= 344 else 0)) for column in range(1, 8)
        )
        lines.append(moment.strftime(_TS_FORMAT) + "," + values)
        moment += timedelta(hours=1)
    return "\n".join(lines) + "\n"


def _write_etth1(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_parse_etth1_values_and_start(tmp_path: Path) -> None:
    path = _write_etth1(tmp_path / "ETTh1.csv", _etth1_text(3))
    series = parse_etth1(path)
    assert [entry.series_id for entry in series] == [
        "etth1/HUFL",
        "etth1/HULL",
        "etth1/MUFL",
        "etth1/MULL",
        "etth1/LUFL",
        "etth1/LULL",
        "etth1/OT",
    ]
    assert all(entry.start == "2016-07-01 00:00:00" for entry in series)
    assert all(len(entry.values) == 3 for entry in series)
    assert series[0].values == (1.0, 1.01, 1.02)
    assert series[6].values == (7.0, 7.01, 7.02)


@pytest.mark.parametrize(
    ("bad", "pattern"),
    [
        (
            "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,WRONG\n2016-07-01 00:00:00,1,2,3,4,5,6,7\n",
            "header",
        ),
        ("date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n", "no data rows"),
        (
            "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n"
            "2016-07-01 00:00:00,1,2,3,4,5,6,7\n"
            "2016-07-01 02:00:00,1,2,3,4,5,6,7\n",  # skipped hour
            r"line 3: .*not one hour after",
        ),
        (
            "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n"
            "2016-07-01 00:00:00,1,2,3,4,5,6,7\n"
            "2016-07-01 00:00:00,1,2,3,4,5,6,7\n",  # duplicated hour
            r"line 3: .*not one hour after",
        ),
        (
            "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n2016-07-01 00:00:00,1,2,3,4,5,6\n",
            r"line 2: expected 8 fields",
        ),
        (
            "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n2016-07-01 00:00:00,1,2,3,4,5,,7\n",
            r"line 2: .*not a number",
        ),
        (
            "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n2016-07-01 00:00:00,nan,2,3,4,5,6,7\n",
            r"line 2: .*not a finite",
        ),
        (
            "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n2016-07-01 00:00:00,inf,2,3,4,5,6,7\n",
            r"line 2: .*not a finite",
        ),
        (
            "date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT\n"
            "2016-07-01 00:00:00,1,2,3,4,5,6,7\n"
            "2016-07-01 00:00:60,1,2,3,4,5,6,7\n",  # malformed timestamp
            r"line 3: .*timestamp",
        ),
    ],
)
def test_parse_etth1_rejects_bad_rows(tmp_path: Path, bad: str, pattern: str) -> None:
    path = _write_etth1(tmp_path / "ETTh1.csv", bad)
    with pytest.raises(ParseError, match=pattern):
        parse_etth1(path)


def test_parse_timestamp_is_strict() -> None:
    assert parse_timestamp("2016-07-01 00:00:00") == datetime(2016, 7, 1)
    for bad in ("2016-7-1 00:00:00", "2016-07-01 00:00", "2016-07-01T00:00:00", ""):
        with pytest.raises(ParseError):
            parse_timestamp(bad)


def _ecl_zip(path: Path, text: str, *, include_member: bool = True) -> Path:
    """Write an ECL-style zip; ``__MACOSX`` junk members are always present."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("__MACOSX/LD2011_2014.txt", "junk to be ignored")
        archive.writestr("__MACOSX/._whatever", "more junk")
        if include_member:
            archive.writestr("LD2011_2014.txt", text)
    path.write_bytes(buffer.getvalue())
    return path


def _ecl_text(start: str, quarters: int, values: list[tuple[str, ...]]) -> str:
    """Build ECL text: ``quarters`` rows of 15 minutes from ``start``."""
    moment = datetime.strptime(start, _TS_FORMAT)
    lines = ['"";"MT_A";"MT_B"']
    for index in range(quarters):
        stamp = moment.strftime(_TS_FORMAT)
        lines.append(f'"{stamp}";{values[index][0]};{values[index][1]}')
        moment += timedelta(minutes=15)
    return "\n".join(lines) + "\n"


def test_parse_ecl_hourly_means_and_grid(tmp_path: Path) -> None:
    # 12 readings = 3 hours; the reading stamped H:00 belongs to hour H - 1.
    values = [
        ("0,25", "1"),  # hour 0: MT_A mean (0.25+1+2+3)/4, MT_B (1+2+3+4)/4
        ("1", "2"),
        ("2", "3"),
        ("3", "4"),
        ("4", "5"),  # hour 1
        ("5", "6"),
        ("6", "7"),
        ("7", "8"),
        ("8", "9"),  # hour 2
        ("9", "10"),
        ("10", "11"),
        ("11", "12"),
    ]
    path = _ecl_zip(tmp_path / "ecl.zip", _ecl_text("2012-01-01 00:15:00", 12, values))
    series = parse_ecl(path, "2012-01-01 00:00:00", 3)
    assert [entry.series_id for entry in series] == ["ecl/MT_A", "ecl/MT_B"]
    assert all(entry.start == "2012-01-01 00:00:00" for entry in series)
    assert series[0].values == (1.5625, 5.5, 9.5)  # decimal commas and bare integers parse
    assert series[1].values == (2.5, 6.5, 10.5)


def test_parse_ecl_ignores_readings_outside_the_window(tmp_path: Path) -> None:
    values = [(str(q), str(q)) for q in range(20)]
    # data 2012-01-01 00:15 .. 05:00; window 01:00 .. 04:00 (3 hours)
    path = _ecl_zip(tmp_path / "ecl.zip", _ecl_text("2012-01-01 00:15:00", 20, values))
    series = parse_ecl(path, "2012-01-01 01:00:00", 3)
    # hour 0 = readings 01:15..02:00 = values 4..7; hour 1 = 8..11; hour 2 = 12..15
    assert series[0].values == ((4 + 5 + 6 + 7) / 4, (8 + 9 + 10 + 11) / 4, (12 + 13 + 14 + 15) / 4)


@pytest.mark.parametrize(
    "window_start,window_hours",
    [
        ("2011-12-31 23:00:00", 3),  # window starts before the data
        ("2012-01-01 03:00:00", 3),  # window ends after the data
    ],
)
def test_parse_ecl_rejects_uncovered_windows(
    tmp_path: Path, window_start: str, window_hours: int
) -> None:
    values = [(str(q), str(q)) for q in range(12)]
    path = _ecl_zip(tmp_path / "ecl.zip", _ecl_text("2012-01-01 00:15:00", 12, values))
    with pytest.raises(ParseError, match="does not cover the whole window"):
        parse_ecl(path, window_start, window_hours)


def test_parse_ecl_rejects_bad_files(tmp_path: Path) -> None:
    values = [(str(q), str(q)) for q in range(12)]
    # missing member
    path = _ecl_zip(tmp_path / "empty.zip", "", include_member=False)
    with pytest.raises(ParseError, match="member"):
        parse_ecl(path, "2012-01-01 00:00:00", 1)
    # skipped 15-minute step
    text = _ecl_text("2012-01-01 00:15:00", 4, values[:4])
    skip = text.replace('"2012-01-01 01:00:00"', '"2012-01-01 01:15:00"')
    path = _ecl_zip(tmp_path / "skip.zip", skip)
    with pytest.raises(ParseError, match="not 15 minutes after"):
        parse_ecl(path, "2012-01-01 00:00:00", 1)
    # wrong field count
    bad_fields = '"";"MT_A";"MT_B"\n"2012-01-01 00:15:00";1\n'
    path = _ecl_zip(tmp_path / "fields.zip", bad_fields)
    with pytest.raises(ParseError, match="fields"):
        parse_ecl(path, "2012-01-01 00:00:00", 1)
    # negative value
    moment = datetime(2012, 1, 1, 0, 15)
    rows = []
    for quarter in range(4):
        rows.append(f'"{moment.strftime(_TS_FORMAT)}";-{quarter + 1};0')
        moment += timedelta(minutes=15)
    negative = '"";"MT_A";"MT_B"\n' + "\n".join(rows) + "\n"
    path = _ecl_zip(tmp_path / "negative.zip", negative)
    with pytest.raises(ParseError, match=">= 0"):
        parse_ecl(path, "2012-01-01 00:00:00", 1)


def _series(series_id: str, values: tuple[float, ...]) -> HourlySeries:
    return HourlySeries(series_id, "2012-01-01 00:00:00", values)


def test_select_ecl_subset_zero_run_boundary() -> None:
    zeros_23 = _series("ecl/MT_A", (1.0,) + (0.0,) * 23 + (2.0,))
    zeros_24 = _series("ecl/MT_B", (1.0,) + (0.0,) * 24 + (2.0,))
    assert select_ecl_subset((zeros_23,), 1) == (zeros_23,)
    with pytest.raises(ParseError, match="eligible"):
        select_ecl_subset((zeros_24,), 1)


def test_select_ecl_subset_ranking_follows_the_hash_rule() -> None:
    ids = ["ecl/MT_A", "ecl/MT_B", "ecl/MT_C", "ecl/MT_D", "ecl/MT_E"]
    entries = tuple(_series(series_id, (1.0, 2.0)) for series_id in ids)
    ranked = sorted(
        ids,
        key=lambda series_id: hashlib.sha256(
            b"evolvets/ecl-subset/v1/" + series_id.removeprefix("ecl/").encode("utf-8")
        ).hexdigest(),
    )
    subset = select_ecl_subset(entries, 3)
    assert [entry.series_id for entry in subset] == sorted(ranked[:3])


def test_select_ecl_subset_output_is_sorted_and_validated() -> None:
    entries = tuple(_series(f"ecl/MT_{index:03d}", (float(index),)) for index in range(5, 0, -1))
    subset = select_ecl_subset(entries, 5)
    assert [entry.series_id for entry in subset] == sorted(entry.series_id for entry in subset)
    with pytest.raises(ParseError, match="n must be"):
        select_ecl_subset(entries, True)  # type: ignore[arg-type]
    with pytest.raises(ParseError, match="eligible"):
        select_ecl_subset(entries, 6)
