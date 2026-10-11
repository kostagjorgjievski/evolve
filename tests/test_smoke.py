"""Placeholder tests so CI has something to run during bootstrap."""

import importlib

import pytest

import evolvets
from evolvets.runner.cli import main

PACKAGES = [
    "arena",
    "cache",
    "experts",
    "updater",
    "improver",
    "selfmodel",
    "allocator",
    "ledger",
    "runner",
]


def test_version_is_set() -> None:
    assert evolvets.__version__


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    importlib.import_module(f"evolvets.{name}")


def test_cli_runs_without_command() -> None:
    assert main([]) == 0
