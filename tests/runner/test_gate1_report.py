"""Gate 1 report: rendering, leakage split, notes and the CLI (TASK_011 PR C2).

The rendering half of the PR C test list: regenerable from the ledger alone
(1), synthetic-only rows (4), the leakage-risk split (5), the per-series
wall-clock section (10), the Notes sentences (11), the ``gate1 report`` CLI
(12) and the subprocess import check of (13). Numbers tracing to records,
bootstrap intervals, refusals and the AST import guard are PR C1's, in
``test_gate1_report_data.py``; the fixture environments are the same ones
that module and ``test_gate1_tune.py`` build (each test module builds its
own, as elsewhere in the suite).

TASK_014 PR D adds this module's probe-rendering tests: the report without
probe runs is unchanged (test 1), the leakage order of the probe sections
(test 8's rendering part), the section layout, Notes and generating line
(test 9) and the ``--probe-run`` CLI surface (test 10). The probe's numbers,
verdicts and refusals are ``test_gate1_probe_data.py``'s, as are the
rendering halves of tests 6 and 7, asserted over those tests' own ledgers.
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
from evolvets.ledger.canonical import canonical_json
from evolvets.runner import cli
from evolvets.runner.gate1_probe import ProbeSpec, probe
from evolvets.runner.gate1_probe_data import build_probe_data
from evolvets.runner.gate1_report import build_report
from evolvets.runner.gate1_report_data import build_report_data
from evolvets.runner.gate1_tune import DevPaths, TuneSpec, tune
from evolvets.updater.interfaces import ConfigValue
from evolvets.updater.learned import LEARNED_IDS
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


# --- TASK_014 PR D: probe fixtures ----------------------------------------------------------------


#: The small learned grids the rendering fixtures probe with (two ridge_stack
#: points, one loss_gate point), as ``SMALL_GRIDS`` is for the baselines.
_PROBE_GRIDS: dict[str, list[dict[str, ConfigValue]]] = {
    "ridge_stack": [
        {"half_life": 48.0, "shrink": 1.0, "refit_every": 24},
        {"half_life": 336.0, "shrink": 0.1, "refit_every": 24},
    ],
    "loss_gate": [{"beta": 4.0, "half_life": 336.0, "ridge": 0.01, "refit_every": 24}],
}


def _probe_run(env: dict[str, Path], ledger_root: Path, run_id: str, spec: ProbeSpec) -> str:
    """Run one probe against a fixture environment; returns the run_end status."""
    return probe(
        spec,
        _dev_paths(env),
        ledger_root=ledger_root,
        run_id=run_id,
        git_commit=COMMIT,
        grids=_PROBE_GRIDS,
    )


def _probe_spec(
    tune_run_id: str, dataset: str, pool_specs: tuple[str, ...], models: tuple[str, ...] | None
) -> ProbeSpec:
    """A probe spec pinned to one fixture tune run, over some of its pool specs."""
    return ProbeSpec(
        tune_run_id=tune_run_id,
        datasets=(dataset,),
        pool_specs=pool_specs,
        foundation_models=models,
        seed=3,
        max_wall_seconds=10000.0,
    )


@pytest.fixture(scope="module")
def synthetic_probe(synthetic_env: dict[str, Path], synthetic_ledger: Path) -> Path:
    """The synthetic tune ledger plus one completed probe run pinned to run-1."""
    spec = _probe_spec("run-1", "synthetic-dev", ("simple",), None)
    assert _probe_run(synthetic_env, synthetic_ledger, "probe-1", spec) == "completed"
    return synthetic_ledger


@pytest.fixture(scope="module")
def etth1_probe(etth1_env: dict[str, Path], etth1_ledger: Path) -> Path:
    """The ETTh1 tune ledger plus disjoint probe runs over simple and the pooled spec."""
    simple = _probe_spec("run-leak", "etth1", ("simple",), ("leaky",))
    assert _probe_run(etth1_env, etth1_ledger, "probe-leak", simple) == "completed"
    pooled = _probe_spec("run-leak", "etth1", ("simple+foundation",), ("leaky",))
    assert _probe_run(etth1_env, etth1_ledger, "probe-leak-fm", pooled) == "completed"
    return etth1_ledger


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


# --- TASK_014 PR D: the probe rendering (tests 1, 8, 9 and 10) -----------------------------------


#: The verdict words a cell may hold, plus "none" for a missing selection.
_VERDICTS = {"beats", "worse", "inside", "n/a", "none"}


def test_a_report_without_probe_runs_is_unchanged(synthetic_ledger: Path) -> None:
    """Naming no probe run gives main's report byte for byte, and no probe anywhere."""
    default = build_report(synthetic_ledger, ["run-1"], bootstrap_seed=5)
    explicit = build_report(synthetic_ledger, ["run-1"], bootstrap_seed=5, probe_run_ids=())
    assert default == explicit
    assert "probe" not in default


def test_the_probe_section_lays_out_as_specified(synthetic_probe: Path) -> None:
    """Probe runs, verdicts, then one subsection per pool, between wall-clock and Notes."""
    report = build_report(synthetic_probe, ["run-1"], bootstrap_seed=5, probe_run_ids=["probe-1"])
    lines = report.splitlines()
    probe_at = lines.index("## Learned-updater probe")
    assert lines.index("## Per-series wall-clock") < probe_at < lines.index("## Notes")
    assert lines[probe_at + 2] == "### Probe runs"
    assert (
        lines[probe_at + 4]
        == "| run id | condition | status | git commit | tune run | records | validate_run |"
    )
    assert f"| probe-1 | gate1-dev-probe | completed | {COMMIT} | run-1 | 5 | none |" in lines
    verdict_at = lines.index("### Probe verdicts")
    assert lines[verdict_at + 2] == (
        "| dataset | pool | leakage risk | " + " | ".join(LEARNED_IDS) + " |"
    )
    verdict_rows = [line for line in lines[verdict_at:] if line.startswith("| synthetic-dev |")]
    assert len(verdict_rows) == 1
    cells = [cell.strip() for cell in verdict_rows[0].strip("|").split("|")]
    assert cells[:3] == ["synthetic-dev", "simple", "no"]
    assert len(cells) == 3 + len(LEARNED_IDS)
    assert all(cell in _VERDICTS for cell in cells[3:])
    # the subsection: the best baseline sentence, the learned rows, the gains
    data = build_probe_data(synthetic_probe, ["run-1"], ["probe-1"], bootstrap_seed=5)
    (section,) = (*data.sections, *data.leakage_sections)
    assert section.best_baseline is not None and section.best_baseline_mase is not None
    pool_at = lines.index("### synthetic-dev, pool simple")
    assert lines[pool_at + 2] == (
        f"Best tuned baseline: {section.best_baseline} (dev MASE {section.best_baseline_mase:.4f})"
    )
    assert lines[pool_at + 4] == (
        "| updater | selected config | dev MASE | configurations | "
        "failed | evaluations | wall seconds |"
    )
    rows = lines[pool_at + 6 : pool_at + 6 + len(section.learned)]  # +5 is the table rule
    for line, tuned in zip(rows, section.learned, strict=True):
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert cells[0] == tuned.baseline
        assert cells[1] == ("none" if tuned.config is None else canonical_json(tuned.config))
        assert cells[2] == ("none" if tuned.mase is None else f"{tuned.mase:.4f}")
        assert cells[3:] == [
            str(tuned.tried),
            str(tuned.failed),
            f"{tuned.evaluations:.4f}",
            f"{tuned.wall_seconds:.4f}",
        ]
    assert f"| updater | mean gain over {section.best_baseline} | 95% interval | verdict |" in lines
    for gain in section.gains:
        interval = "n/a" if gain.lower is None else f"[{gain.lower:.4f}, {gain.upper:.4f}]"
        assert f"| {gain.updater} | {gain.mean:.4f} | {interval} | {gain.verdict} |" in lines


def test_the_probe_notes_and_generating_line_appear_only_with_probe_runs(
    synthetic_probe: Path,
) -> None:
    """The four extra Notes and the --probe-run ids in the command, only with probe runs."""
    with_probe = build_report(
        synthetic_probe, ["run-1"], bootstrap_seed=5, probe_run_ids=["probe-1"]
    )
    without = build_report(synthetic_probe, ["run-1"], bootstrap_seed=5)
    assert (
        "generated by `evolvets gate1 report --run run-1 --probe-run probe-1 "
        "--bootstrap-seed 5`" in with_probe
    )
    assert "generated by `evolvets gate1 report --run run-1 --bootstrap-seed 5`" in without
    sentences = (
        "positive favours the learned updater",
        "the configurations column shows how many grid points each had",
        "no multiple-comparison correction is applied",
        "learners that output forecasts directly are not tested here",
    )
    for sentence in sentences:
        assert sentence in with_probe
        assert sentence not in without
    notes = with_probe[with_probe.index("## Notes") :]
    assert notes.index("The Gate 1 decision is the owner's.") < notes.index(
        "positive favours the learned updater"
    )


def test_two_renders_of_the_same_probe_ledger_are_byte_identical(
    synthetic_probe: Path, tmp_path: Path
) -> None:
    """The probe report is regenerable: same ledger and seed, same bytes, even elsewhere."""
    first = build_report(synthetic_probe, ["run-1"], bootstrap_seed=5, probe_run_ids=["probe-1"])
    assert (
        build_report(synthetic_probe, ["run-1"], bootstrap_seed=5, probe_run_ids=["probe-1"])
        == first
    )
    elsewhere = tmp_path / "elsewhere"
    shutil.copytree(synthetic_probe, elsewhere)
    assert build_report(elsewhere, ["run-1"], bootstrap_seed=5, probe_run_ids=["probe-1"]) == first


def test_probe_leakage_sections_follow_the_clean_ones(etth1_probe: Path) -> None:
    """The pooled subsection carries (leakage risk) and comes last, in the verdict table too."""
    report = build_report(
        etth1_probe, ["run-leak"], bootstrap_seed=0, probe_run_ids=["probe-leak", "probe-leak-fm"]
    )
    lines = report.splitlines()
    clean = lines.index("### etth1, pool simple")
    pooled = lines.index("### etth1, pool simple+foundation (leakage risk)")
    assert lines.index("### Probe runs") < clean < pooled
    assert any(line.startswith("| probe-leak |") for line in lines)
    assert any(line.startswith("| probe-leak-fm |") for line in lines)
    verdict_at = lines.index("### Probe verdicts")
    verdict_rows = [line for line in lines[verdict_at:clean] if line.startswith("| etth1 |")]
    assert len(verdict_rows) == 2
    assert verdict_rows[0].startswith("| etth1 | simple | no |")
    assert verdict_rows[1].startswith("| etth1 | simple+foundation | yes |")
    for row in verdict_rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 3 + len(LEARNED_IDS)
        assert all(cell in _VERDICTS for cell in cells[3:])


def _probe_report_args(ledger: Path, *extra: str) -> list[str]:
    """A gate1 report invocation over run-leak and both probe runs, plus the output arguments."""
    return [
        "gate1",
        "report",
        "--ledger-root",
        str(ledger),
        "--run",
        "run-leak",
        "--probe-run",
        "probe-leak",
        "--probe-run",
        "probe-leak-fm",
        *extra,
    ]


def test_cli_probe_run_is_repeatable(
    etth1_probe: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two --probe-run flags name both probe runs, and --out carries their section."""
    out = tmp_path / "report.md"
    args = _probe_report_args(etth1_probe, "--bootstrap-seed", "3", "--out", str(out))
    assert cli.main(args) == 0
    expected = build_report(
        etth1_probe, ["run-leak"], bootstrap_seed=3, probe_run_ids=["probe-leak", "probe-leak-fm"]
    )
    assert out.read_text(encoding="utf-8") == expected
    assert capsys.readouterr().out.startswith("OK ")


def test_cli_probe_refusal_writes_nothing(
    synthetic_probe: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A probe refusal is one FAIL line, exit 1, and --out stays unwritten."""
    out = tmp_path / "report.md"
    args = [
        "gate1",
        "report",
        "--ledger-root",
        str(synthetic_probe),
        "--run",
        "run-1",
        "--probe-run",
        "nope",
        "--out",
        str(out),
    ]
    assert cli.main(args) == 1
    assert capsys.readouterr().out.startswith("FAIL ")
    assert not out.exists()


def test_cli_check_on_the_written_probe_report(
    etth1_probe: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--check accepts the freshly written probe report and rejects a one-character change."""
    out = tmp_path / "report.md"
    assert cli.main(_probe_report_args(etth1_probe, "--out", str(out))) == 0
    assert cli.main(_probe_report_args(etth1_probe, "--check", str(out))) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    target = next(i for i, line in enumerate(lines) if line.startswith("| probe-leak |"))
    lines[target] = lines[target].replace("completed", "completee", 1)
    tampered = tmp_path / "tampered.md"
    tampered.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert cli.main(_probe_report_args(etth1_probe, "--check", str(tampered))) == 1
    assert f"differs at line {target + 1}" in capsys.readouterr().out
