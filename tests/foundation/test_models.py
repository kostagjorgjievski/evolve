"""Model-adapter tests for TASK_009 PR C1: load_adapter, the chronos2 pin, smoke.

Test 2 needs the real library and model weights: it carries the ``foundation``
marker, skips without the library (``importorskip``) and runs in the GPU
workflow, not in plain CI. The missing-library tests assert a behaviour that
only exists without the library, so they skip when it is installed. Everything
else runs torch-free in plain CI.
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
    """C1 fills in chronos2 only; the other four arrive with their own PRs (C2 to C5)."""
    for name in ("tirex", "toto", "moirai2", "timesfm25"):
        spec = get_spec(name)
        assert spec.hf_revision == "unpinned"
        assert spec.quantile_method == "unknown"


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
    """import evolvets.foundation.models and .chronos2 succeed with torch absent."""
    code = (
        "import sys; "
        "import evolvets.foundation.models, evolvets.foundation.models.chronos2; "
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
        load_adapter("tirex", "cpu")


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
