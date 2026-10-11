"""Shared fixtures for ledger tests."""

import hashlib

import pytest

from evolvets.ledger.payloads import RunEnd, RunStart
from evolvets.ledger.schema import Budget

PROTOCOL_VERSIONS = {
    "ledger": "1.0",
    "arena": "0",
    "generator": "0",
    "cache_manifest": "0",
}


@pytest.fixture()
def blob_ref():
    """Return a helper producing valid ``sha256:...`` blob references."""

    def make(seed: str) -> str:
        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        return f"sha256:{digest}"

    return make


@pytest.fixture()
def root(tmp_path):
    return tmp_path / "ledgers"


@pytest.fixture()
def budget_limits():
    return Budget(llm_input_tokens=1000, wall_seconds=1.5, evaluations=2)


@pytest.fixture()
def run_start(budget_limits, blob_ref):
    return RunStart(
        condition="baseline-hedge",
        seed=7,
        config_blob=blob_ref("config"),
        git_commit="abc123",
        protocol_versions=dict(PROTOCOL_VERSIONS),
        budget_limits=budget_limits,
    )


@pytest.fixture()
def run_end():
    return RunEnd(status="completed", summary={"records": 2, "note": "done"})
