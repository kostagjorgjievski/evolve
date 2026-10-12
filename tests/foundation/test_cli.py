"""CLI tests for ``evolvets foundation``: build, verify, compare with the fake adapter.

TASK_009 PR B test 10, updated by PR C: real models resolve through their
adapters (see test_models.py), while a dropped model whose adapter module was
never added still fails, naming the drop as permanent. Every command runs
end to end with ``--model fake`` on the deterministic ETTh1 fixture (no
network).
"""

from __future__ import annotations

import re
import struct
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolvets.arena.realdata.fetch import fetch
from evolvets.arena.realdata.prepare import prepare
from evolvets.cache.datasets import public_inputs
from evolvets.cache.store import DEFAULT_HORIZON, build_cache
from evolvets.foundation.interfaces import QUANTILES
from evolvets.foundation.store import foundation_cache_id, read_foundation_manifest
from evolvets.runner import cli
from tests.arena.test_realdata_prepare import _build_etth1_fixture, _install

_PUBLIC_ROWS = 344
_START = 287  # first_origin(24)
_POINT = QUANTILES.index(0.5)  # 4
_N_SERIES = 7
_ORIGINS = _N_SERIES * (_PUBLIC_ROWS - _START)  # 399
_SYNTHETIC_ORIGINS = 6 * (2000 - _START)  # 6 families, 2000 rows each
_OK = re.compile(rf"^OK etth1/fake: {_N_SERIES} series, {_ORIGINS} origins, cache [0-9a-f]{{64}}$")


@pytest.fixture()
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Prepared fixture ETTh1, its simple cache, and an empty foundation cache root."""
    etth1 = _build_etth1_fixture(tmp_path / "fixtures")
    _install(monkeypatch, etth1=etth1)
    data_root, manifest_dir = tmp_path / "data", tmp_path / "manifests"
    fetch(etth1, data_root, manifest_dir, today=date(2026, 10, 11))
    prepare("etth1", data_root, manifest_dir)
    cache_root = tmp_path / "cache_store"
    source, series = public_inputs("etth1", data_root, manifest_dir)
    build_cache("etth1", source, series, cache_root)
    return SimpleNamespace(
        data_root=data_root,
        manifest_dir=manifest_dir,
        cache_root=cache_root,
        root=tmp_path / "foundation_cache",
        tmp=tmp_path,
    )


def _data_args(env: SimpleNamespace) -> list[str]:
    return ["--data-root", str(env.data_root), "--manifest-dir", str(env.manifest_dir)]


def _foundation_args(command: str, env: SimpleNamespace, model: str = "fake") -> list[str]:
    """An ``etth1`` argv for ``foundation <command>``; tests append their own extra flags."""
    args = ["foundation", command, "--dataset", "etth1", "--model", model, "--root", str(env.root)]
    if command == "build":
        args += ["--cache-root", str(env.cache_root), *_data_args(env)]
    elif command == "compare":
        args += _data_args(env)
    return args


def test_build_verify_compare_with_fake(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """Build, verify and compare run end to end with --model fake."""
    assert cli.main([*_foundation_args("build", env), "--batch-size", "16"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and _OK.match(out[0]), out
    manifest = read_foundation_manifest(env.root, "etth1", "fake")
    assert (manifest.batch_size, manifest.device, manifest.dtype) == (16, "cpu", "float64")

    assert cli.main(_foundation_args("verify", env)) == 0
    assert capsys.readouterr().out.splitlines() == out  # the same line again

    assert cli.main([*_foundation_args("compare", env), "--sample", "20", "--seed", "3"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"OK etth1/fake: compared 20/{_ORIGINS} origins, max_abs 0.000e+00, "
        "max_rel 0.000e+00, 0 non-finite values"
    ]


def test_build_with_device_cuda_is_refused_for_the_fake(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """--device cuda means a real model's device; the fake runs on cpu."""
    assert cli.main([*_foundation_args("build", env), "--device", "cuda"]) == 1
    assert capsys.readouterr().out.splitlines() == [
        "FAIL etth1/fake: --device cuda needs a real model; the fake adapter runs on cpu"
    ]
    assert not (env.root / "etth1").exists()


def test_dropped_model_names_refuse_commands(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """Commands refuse a dropped model's absent adapter module, naming the drop as permanent."""
    for model, argv in (
        ("toto", [*_foundation_args("compare", env, model="toto"), "--sample", "5", "--seed", "0"]),
        ("moirai2", _foundation_args("build", env, model="moirai2")),
    ):
        assert cli.main(argv) == 1
        out = capsys.readouterr().out.splitlines()
        assert len(out) == 1, out
        assert out[0].startswith(f"FAIL etth1/{model}: ")
        assert "TASK_009 PR C" in out[0] and "none is planned" in out[0]
    assert not env.root.exists()


def test_verify_a_corrupt_cache_fails(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify names the fault and exits nonzero; a restored cache verifies again."""
    assert cli.main(_foundation_args("build", env)) == 0
    capsys.readouterr()
    series_file = env.root / "etth1" / "fake" / "series" / "0000.f64"
    raw = series_file.read_bytes()
    series_file.write_bytes(raw[:-1] + bytes([raw[-1] ^ 0xFF]))
    assert cli.main(_foundation_args("verify", env)) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].startswith("FAIL etth1/fake: ")

    series_file.write_bytes(raw)  # restored
    assert cli.main(_foundation_args("verify", env)) == 0


def test_compare_rejects_a_nonpositive_sample(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """A bad --sample is a FAIL line, not a traceback."""
    assert cli.main(_foundation_args("build", env)) == 0
    capsys.readouterr()
    assert cli.main([*_foundation_args("compare", env), "--sample", "0", "--seed", "1"]) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].startswith("FAIL etth1/fake: sample must be")


def test_compare_prints_nonfinite_drift_as_inf_and_a_count(
    env: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """A NaN stored 0.5 value reaches the compare line as inf maxima and a count."""
    assert cli.main(_foundation_args("build", env)) == 0
    capsys.readouterr()
    series_file = env.root / "etth1" / "fake" / "series" / "0000.f64"
    raw = bytearray(series_file.read_bytes())
    struct.pack_into("<d", raw, _POINT * DEFAULT_HORIZON * 8, float("nan"))
    series_file.write_bytes(bytes(raw))
    assert (
        cli.main([*_foundation_args("compare", env), "--sample", str(_ORIGINS), "--seed", "0"]) == 0
    )
    assert capsys.readouterr().out.splitlines() == [
        f"OK etth1/fake: compared {_ORIGINS}/{_ORIGINS} origins, "
        "max_abs inf, max_rel inf, 1 non-finite values"
    ]


def test_build_synthetic_dev_needs_per_family(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The synthetic-dev foundation build requires --per-family, like the simple cache."""
    root, cache_root = tmp_path / "foundation_cache", tmp_path / "cache_store"
    source, series = public_inputs("synthetic-dev", tmp_path / "data", tmp_path / "manifests", 1)
    build_cache("synthetic-dev", source, series, cache_root)
    missing = [
        "foundation",
        "build",
        "--dataset",
        "synthetic-dev",
        "--model",
        "fake",
        "--root",
        str(root),
        "--cache-root",
        str(cache_root),
        *_data_args(
            SimpleNamespace(data_root=tmp_path / "data", manifest_dir=tmp_path / "manifests")
        ),
    ]
    assert cli.main(missing) == 1
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 and out[0].startswith(
        "FAIL synthetic-dev/fake: dataset 'synthetic-dev': per_family"
    )
    assert not (root / "synthetic-dev").exists()

    assert cli.main([*missing, "--per-family", "1"]) == 0
    out = capsys.readouterr().out.splitlines()
    ident = foundation_cache_id(root, "synthetic-dev", "fake")
    assert out == [f"OK synthetic-dev/fake: 6 series, {_SYNTHETIC_ORIGINS} origins, cache {ident}"]


def test_unknown_model_is_rejected_by_argparse() -> None:
    """A model outside the registry and 'fake' never reaches the command."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["foundation", "build", "--dataset", "etth1", "--model", "nope"])
    assert excinfo.value.code == 2


def test_model_is_required_by_argparse() -> None:
    """Leaving out --model is argparse's exit 2 on every subcommand, like --dataset."""
    for argv in (
        ["foundation", "build", "--dataset", "etth1"],
        ["foundation", "verify", "--dataset", "etth1"],
        ["foundation", "compare", "--dataset", "etth1", "--sample", "5", "--seed", "0"],
    ):
        with pytest.raises(SystemExit) as excinfo:
            cli.main(argv)
        assert excinfo.value.code == 2
