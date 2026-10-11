"""The two pinned real-data sources.

The ETTh1 URL is pinned to a commit, not a branch, so the content hash in
the committed manifest is stable. The raw files are never redistributed:
they stay in git-ignored ``data/``.
"""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    """A pinned public dataset file."""

    name: str  # lowercase [a-z0-9_]+
    url: str  # http(s) or file
    filename: str  # bare file name, no path separators
    license: str


SOURCES: Mapping[str, Source] = {
    "etth1": Source(
        name="etth1",
        url=(
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
            "1d16c8f4f943005d613b5bc962e9eeb06058cf07/ETT-small/ETTh1.csv"
        ),
        filename="ETTh1.csv",
        license="CC-BY-ND-4.0",
    ),
    "ecl": Source(
        name="ecl",
        url="https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
        filename="electricityloaddiagrams20112014.zip",
        license="CC-BY-4.0",
    ),
}
