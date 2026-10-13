"""Gate 1 report: rendering, leakage split, notes and the CLI (TASK_011 PR C2).

The rendering half of the PR C test list: regenerable from the ledger alone
(1), synthetic-only rows (4), the leakage-risk split (5), the per-series
wall-clock section (10), the Notes sentences (11), the ``gate1 report`` CLI
(12) and the subprocess import check of (13). Numbers tracing to records,
bootstrap intervals, refusals and the AST import guard are PR C1's, in
``test_gate1_report_data.py``; the fixture environments are the same ones
that module and ``test_gate1_tune.py`` build (each test module builds its
own, as elsewhere in the suite).
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from evolvets.arena.sealed.client import SERIES_TIMEOUT_SECONDS
from evolvets.arena.synthetic.grammar import FAMILIES
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import build_cache
from evolvets.foundation.fake import FAKE_SPEC, FakeAdapter
from evolvets.foundation.store import build_foundation_cache
from evolvets.runner import cli
from evolvets.runner.gate1_report import build_report
from evolvets.runner.gate1_report_data import build_report_data
from evolvets.runner.gate1_tune import DevPaths, TuneSpec, tune
from tests.arena.test_dev import build_real_env

COMMIT = "d" * 40
SMALL_GRIDS: dict[str, list[dict[str, float]]] = {
    "equal": [{}],
    "hedge": [{"eta": 0.5}, {"eta": 1.0}],
}


@pytest.fixture(scope="module")
def synthetic_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """A synthetic-dev cache with one series per family, and the paths to load it."""
    base = tmp_path_factory.mktemp("render-synthetic")
    cache_root = base / "cache_store"
    source, series = public_inputs("synthetic-dev", base, base, per_family=1)
    build_cache("synthetic-dev", source, series, cache_root)
    return {
        "base": base,
        "cache_root": cache_root,
        "foundation_root": base / "foundation_cache",
        "data_root": base,
        "manifest_dir": base,
    }


@pytest.fixture(scope="module")
def etth1_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The fixture ETTh1 prepared and cached, plus a leaking pooled model's cache.

    The ``leaky`` model records ETTh1 overlap ``"yes"``, so results replayed
    over ``simple+foundation`` carry ``leakage_risk`` true while the simple
    ones stay false (as ``real_env`` in tests/arena/test_pool.py).
    """
    base = tmp_path_factory.mktemp("render-etth1")
    monkeypatch = pytest.MonkeyPatch()
    try:
        data_root, manifest_dir, cache_root = build_real_env(base, monkeypatch)
    finally:
        monkeypatch.undo()
    foundation_root = base / "foundation_cache"
    leaky = replace(FAKE_SPEC, name="leaky", pretraining_overlap={"etth1": "yes", "ecl": "no"})
    build_foundation_cache(
        "etth1",
        FakeAdapter(leaky.name),
        leaky,
        foundation_root,
        cache_root,
        data_root,
        manifest_dir,
        64,
    )
    return {
        "base": base,
        "cache_root": cache_root,
        "foundation_root": foundation_root,
        "data_root": data_root,
        "manifest_dir": manifest_dir,
    }


def _dev_paths(env: dict[str, Path]) -> DevPaths:
    """The DevPaths of one fixture environment."""
    return DevPaths(
        cache_root=env["cache_root"],
        foundation_root=env["foundation_root"],
        data_root=env["data_root"],
        manifest_dir=env["manifest_dir"],
    )


def _tune_run(env: dict[str, Path], ledger_root: Path, run_id: str, spec: TuneSpec) -> str:
    """Run one tuning sweep against a fixture environment; returns the run_end status."""
    return tune(
        spec,
        _dev_paths(env),
        ledger_root=ledger_root,
        run_id=run_id,
        git_commit=COMMIT,
        grids=SMALL_GRIDS,
    )


_SYNTHETIC_SPEC = TuneSpec(
    datasets=("synthetic-dev",),
    pool_specs=("simple",),
    foundation_models=None,
    seed=7,
    max_wall_seconds=10000.0,
)
_ETTH1_SPEC = TuneSpec(
    datasets=("etth1",),
    pool_specs=("simple", "simple+foundation"),
    foundation_models=("leaky",),
    seed=7,
    max_wall_seconds=10000.0,
)


@pytest.fixture(scope="module")
def synthetic_ledger(
    synthetic_env: dict[str, Path], tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """A completed tune run on the synthetic fixture, small grid."""
    ledger = tmp_path_factory.mktemp("render-synthetic-ledger") / "ledgers"
    assert _tune_run(synthetic_env, ledger, "run-1", _SYNTHETIC_SPEC) == "completed"
    return ledger


@pytest.fixture(scope="module")
def etth1_ledger(etth1_env: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A completed tune run on ETTh1, simple and pooled with the leaking model."""
    ledger = tmp_path_factory.mktemp("render-etth1-ledger") / "ledgers"
    assert _tune_run(etth1_env, ledger, "run-leak", _ETTH1_SPEC) == "completed"
    return ledger


# --- Test 1: regenerable -------------------------------------------------------------------------


def test_the_report_is_regenerable_from_the_ledger_alone(
    synthetic_ledger: Path, tmp_path: Path
) -> None:
    """Two builds give identical strings, from this directory or a copy of the ledger."""
    first = build_report(synthetic_ledger, ["run-1"], bootstrap_seed=5)
    assert build_report(synthetic_ledger, ["run-1"], bootstrap_seed=5) == first
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(synthetic_ledger, elsewhere)
    assert build_report(elsewhere, ["run-1"], bootstrap_seed=5) == first
    assert first.endswith("\n")
    # the generating line names the command's run ids and seed, and nothing else
    assert "generated by `evolvets gate1 report --run run-1 --bootstrap-seed 5`" in first


# --- Test 4: synthetic-only rows -----------------------------------------------------------------


def test_true_best_and_family_rows_appear_only_on_synthetic(
    synthetic_ledger: Path, etth1_ledger: Path
) -> None:
    """The true-best forecast and per-family gaps are synthetic-only."""
    synthetic = build_report(synthetic_ledger, ["run-1"], bootstrap_seed=0)
    real = build_report(etth1_ledger, ["run-leak"], bootstrap_seed=0)
    assert "true-best forecast" in synthetic
    headings = [f"### Family gaps: {family}" for family in FAMILIES]
    assert all(heading in synthetic for heading in headings)
    positions = [synthetic.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "true-best forecast" not in real
    assert "Family gaps:" not in real


# --- Test 5: the leakage split -------------------------------------------------------------------


def test_leakage_risk_results_sit_in_their_own_section(
    synthetic_ledger: Path, etth1_ledger: Path
) -> None:
    """The pooled subsection follows the leakage heading; simple ones stay out of it."""
    lines = build_report(etth1_ledger, ["run-leak"], bootstrap_seed=0).splitlines()
    split = lines.index("## Leakage-risk results")
    assert lines.index("## etth1, pool simple") < split
    assert lines.index("## etth1, pool simple+foundation") > split
    sentence = "A pooled foundation model may have seen this dataset in pretraining."
    assert lines[split + 2] == sentence
    clean = build_report(synthetic_ledger, ["run-1"], bootstrap_seed=0).splitlines()
    split = clean.index("## Leakage-risk results")
    assert clean[split + 2] == "No leakage-risk results."


# --- Test 10: per-series wall-clock --------------------------------------------------------------


def _four(value: float) -> str:
    """Format as the report does, for asserting against rendered cells."""
    return f"{value:.4f}"


def test_the_wall_clock_section_shows_the_max_and_the_timeout(synthetic_ledger: Path) -> None:
    """Each pool's largest series_wall_seconds, next to SERIES_TIMEOUT_SECONDS as .4f."""
    data = build_report_data(synthetic_ledger, ["run-1"], bootstrap_seed=0)
    report = build_report(synthetic_ledger, ["run-1"], bootstrap_seed=0)
    section = report[report.index("## Per-series wall-clock") :]
    pool = data.sections[0]
    assert pool.max_series_wall_seconds is not None
    row = f"| {pool.dataset} | {pool.pool_spec} | {_four(pool.max_series_wall_seconds)} |"
    assert row in section
    assert data.series_timeout_seconds == SERIES_TIMEOUT_SECONDS
    assert f"| {_four(SERIES_TIMEOUT_SECONDS)} |" in section


# --- Test 11: the Notes --------------------------------------------------------------------------


def test_the_notes_carry_the_two_fixed_sentences(synthetic_ledger: Path) -> None:
    """The oracle bound and the dev-selection caveats are both stated."""
    report = build_report(synthetic_ledger, ["run-1"], bootstrap_seed=0)
    assert "not a floor for weighted combinations" in report
    assert "chosen on these same series" in report
    assert "The Gate 1 decision is the owner's." in report


# --- Test 12: the CLI ----------------------------------------------------------------------------


def _report_args(ledger: Path, *extra: str) -> list[str]:
    """A gate1 report invocation over run-1, plus the output arguments."""
    return ["gate1", "report", "--ledger-root", str(ledger), "--run", "run-1", *extra]


def test_cli_report_out_writes_and_check_accepts(
    synthetic_ledger: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """--out writes the built report; --check against the same file exits 0."""
    out = tmp_path / "report.md"
    assert cli.main(_report_args(synthetic_ledger, "--out", str(out), "--bootstrap-seed", "3")) == 0
    expected = build_report(synthetic_ledger, ["run-1"], bootstrap_seed=3)
    assert out.read_text(encoding="utf-8") == expected
    checking = _report_args(synthetic_ledger, "--check", str(out), "--bootstrap-seed", "3")
    assert cli.main(checking) == 0
    assert capsys.readouterr().out.startswith("OK ")


def test_cli_report_check_names_the_first_differing_line(
    synthetic_ledger: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A one-character change fails the check at that line's number."""
    lines = build_report(synthetic_ledger, ["run-1"], bootstrap_seed=0).splitlines()
    target = next(i for i, line in enumerate(lines) if line.startswith("| equal |"))
    lines[target] = lines[target][:-1] + ("0" if lines[target].endswith("1") else "1")
    tampered = tmp_path / "tampered.md"
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert cli.main(_report_args(synthetic_ledger, "--check", str(tampered))) == 1
    assert f"differs at line {target + 1}" in capsys.readouterr().out


def test_cli_report_requires_a_run(synthetic_ledger: Path, tmp_path: Path) -> None:
    """Without --run argparse rejects the call."""
    with pytest.raises(SystemExit):
        cli.main(
            [
                "gate1",
                "report",
                "--ledger-root",
                str(synthetic_ledger),
                "--out",
                str(tmp_path / "r.md"),
            ]
        )


def test_cli_report_refusal_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ReportError is one FAIL line, exit 1, and --out stays unwritten."""
    out = tmp_path / "report.md"
    args = [
        "gate1",
        "report",
        "--ledger-root",
        str(tmp_path / "ledgers"),
        "--run",
        "nope",
        "--out",
        str(out),
    ]
    assert cli.main(args) == 1
    assert capsys.readouterr().out.startswith("FAIL ")
    assert not out.exists()


# --- Test 13 (subprocess part): sealed imports ---------------------------------------------------


def test_importing_the_report_module_loads_no_sealed_server_or_pools() -> None:
    """A fresh interpreter importing gate1_report loads neither the server nor the pools."""
    probe = (
        "import sys, evolvets.runner.gate1_report; "
        "print(sorted(name for name in sys.modules if name.startswith('evolvets.arena.sealed')))"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    loaded = ast.literal_eval(done.stdout.strip())
    assert "evolvets.arena.sealed.server" not in loaded
    assert "evolvets.arena.sealed.pools" not in loaded
