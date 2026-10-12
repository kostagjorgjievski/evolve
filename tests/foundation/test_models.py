"""Model-adapter tests for TASK_009 PR C1/C2: load_adapter, the model pins, smoke.

The real-model tests need the library and model weights: they carry the
``foundation`` marker, skip without the library (``importorskip``) and run in
the GPU workflow, not in plain CI. The missing-library tests assert a
behaviour that only exists without the library, so they skip when it is
installed. Everything else runs torch-free in plain CI.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from evolvets.foundation.errors import FoundationError
from evolvets.foundation.fake import FAKE_SPEC
from evolvets.foundation.interfaces import QUANTILES
from evolvets.foundation.models import load_adapter
from evolvets.foundation.registry import get_spec
from evolvets.foundation.smoke import run_smoke
from evolvets.runner import cli

#: Tests of the no-library refusal path; the fm-chronos group installs the library.
_needs_library_absent = pytest.mark.skipif(
    importlib.util.find_spec("chronos") is not None,
    reason="behaviour under test needs chronos absent (fm-chronos installs it)",
)
#: The same for tirex; the fm-tirex group installs the library.
_needs_tirex_absent = pytest.mark.skipif(
    importlib.util.find_spec("tirex") is not None,
    reason="behaviour under test needs tirex absent (fm-tirex installs it)",
)


# --- C1 test 1: the chronos2 registry entry is verified and pinned -----------------------


def test_chronos2_entry_is_pinned() -> None:
    """hf_revision is a 40-hex commit hash; max_context, license and overlap_source are set."""
    spec = get_spec("chronos2")
    assert spec.hf_model_id == "amazon/chronos-2"
    assert re.fullmatch(r"[0-9a-f]{40}", spec.hf_revision)
    assert spec.hf_revision != "unpinned"
    assert spec.max_context == 4096  # the model card's 8192, capped by TASK_009
    assert spec.quantile_method == "native"
    assert spec.license  # "apache-2.0", read from the model card
    assert spec.overlap_source.startswith(("http://", "https://"))


def test_other_models_remain_unpinned_placeholders() -> None:
    """C1 and C2 fill in chronos2 and tirex; the rest arrive with their own PRs (C3 to C5)."""
    for name in ("toto", "moirai2", "timesfm25"):
        spec = get_spec(name)
        assert spec.hf_revision == "unpinned"
        assert spec.quantile_method == "unknown"


# --- C2 test 1: the tirex registry entry is verified and pinned ---------------------------


def test_tirex_entry_is_pinned() -> None:
    """hf_revision is a 40-hex commit hash; max_context, license and overlap_source are set."""
    spec = get_spec("tirex")
    assert spec.hf_model_id == "NX-AI/TiRex"
    assert re.fullmatch(r"[0-9a-f]{40}", spec.hf_revision)
    assert spec.hf_revision != "unpinned"
    assert spec.max_context == 2048  # the library FAQ's maximum, under the task's 4096 cap
    assert spec.quantile_method == "native"
    assert spec.license  # "nx-ai-community-license", read from the model card
    assert spec.overlap_source.startswith(("http://", "https://"))


# --- C2 test 2: the real smoke check (library and weights; GPU workflow) -----------------


@pytest.mark.foundation
def test_tirex_smoke_passes_on_cpu() -> None:
    """The smoke check passes with the real adapter on CPU, a context of 128.

    C1's chronos2 test uses a context of 64; TiRex's released model (patch
    size 32) cannot carry the smoke series' 0.15/step trend from only two
    patches of in-context signal — it tracks the sine's phase but damps the
    trend (measured 2026-10-12: MAE 1.87 vs naive 1.26 at 64; 0.23 at 128 and
    0.24 at the registry context). 128 is four patches, still a short context,
    and every run_smoke check (shapes, finiteness, batch invariance, MAE beats
    naive) runs unchanged.
    """
    pytest.importorskip("tirex")
    adapter = load_adapter("tirex", "cpu")
    spec = dataclasses.replace(get_spec("tirex"), max_context=128)
    report = run_smoke(spec, adapter)
    assert report.ok, report.problems
    assert report.device == "cpu"
    assert report.n_origins == 20
    assert report.max_rel_batch_diff <= 1e-3


# --- C2 test 3: build names the missing dependency group (plain CI) ----------------------


@_needs_tirex_absent
def test_build_without_tirex_library_names_the_group(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """build --model tirex with the library absent exits nonzero naming fm-tirex."""
    argv = [
        "foundation",
        "build",
        "--dataset",
        "etth1",
        "--model",
        "tirex",
        "--root",
        str(tmp_path / "foundation_cache"),
        "--cache-root",
        str(tmp_path / "cache_store"),
        "--data-root",
        str(tmp_path / "data"),
        "--manifest-dir",
        str(tmp_path / "manifests"),
    ]
    assert cli.main(argv) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].startswith("FAIL etth1/tirex: ")
    assert "fm-tirex" in out[0]
    assert not (tmp_path / "foundation_cache").exists()


# --- C1 test 2: the real smoke check (library and weights; GPU workflow) -----------------


@pytest.mark.foundation
def test_chronos2_smoke_passes_on_cpu() -> None:
    """The smoke check passes with the real adapter on CPU, a context of 64."""
    pytest.importorskip("chronos")
    adapter = load_adapter("chronos2", "cpu")
    spec = dataclasses.replace(get_spec("chronos2"), max_context=64)
    report = run_smoke(spec, adapter)
    assert report.ok, report.problems
    assert report.device == "cpu"
    assert report.n_origins == 20
    assert report.max_rel_batch_diff <= 1e-3


# --- C1 test 3: build names the missing dependency group (plain CI) ----------------------


@_needs_library_absent
def test_build_without_library_names_the_group(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """build --model chronos2 with the library absent exits nonzero naming fm-chronos."""
    argv = [
        "foundation",
        "build",
        "--dataset",
        "etth1",
        "--model",
        "chronos2",
        "--root",
        str(tmp_path / "foundation_cache"),
        "--cache-root",
        str(tmp_path / "cache_store"),
        "--data-root",
        str(tmp_path / "data"),
        "--manifest-dir",
        str(tmp_path / "manifests"),
    ]
    assert cli.main(argv) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].startswith("FAIL etth1/chronos2: ")
    assert "fm-chronos" in out[0]
    assert not (tmp_path / "foundation_cache").exists()


# --- C1 test 4: importing the adapter module needs no torch ------------------------------


def test_import_models_without_torch() -> None:
    """import evolvets.foundation.models and its adapter modules succeed with torch absent."""
    code = (
        "import sys; "
        "import evolvets.foundation.models, evolvets.foundation.models.chronos2, "
        "evolvets.foundation.models.tirex; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


# --- C1 test 5: load_adapter refusals ----------------------------------------------------


def test_load_adapter_unknown_name() -> None:
    """An unregistered name raises FoundationError."""
    with pytest.raises(FoundationError, match="unknown foundation model"):
        load_adapter("nope", "cpu")


def test_load_adapter_module_not_yet_added() -> None:
    """A registered model whose adapter module is absent says which PR brings it."""
    with pytest.raises(FoundationError, match="arrives with TASK_009 PR C"):
        load_adapter("toto", "cpu")


@_needs_library_absent
def test_load_adapter_library_missing() -> None:
    """A known model whose library is not installed names its dependency group."""
    with pytest.raises(FoundationError, match="fm-chronos"):
        load_adapter("chronos2", "cpu")


# --- the smoke check's own behaviour (torch-free, fake and naive adapters) ---------------


class _NaiveAdapter:
    """The worst honest forecaster: every quantile is the context's last value."""

    def __init__(self) -> None:
        self.name = "fake"

    def forecast(
        self, contexts: Sequence[Sequence[float]], horizon: int
    ) -> list[list[list[float]]]:
        return [[[float(context[-1])] * horizon for _ in QUANTILES] for context in contexts]


def test_smoke_cli_with_fake(capsys: pytest.CaptureFixture[str]) -> None:
    """The smoke command prints one JSON line and exits 0 for the fake adapter."""
    assert cli.main(["foundation", "smoke", "--model", "fake", "--device", "cpu"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert report["model"] == "fake"
    assert report["ok"] is True
    assert report["problems"] == []
    assert report["n_origins"] == 20


def test_smoke_flags_a_forecast_no_better_than_naive() -> None:
    """An adapter that repeats the last value fails the MAE check, not the run."""
    report = run_smoke(FAKE_SPEC, _NaiveAdapter())
    assert not report.ok
    assert any("does not beat the naive MAE" in problem for problem in report.problems)
