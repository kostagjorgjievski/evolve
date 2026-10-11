"""Shared fixtures for ledger tests."""

import hashlib
import uuid

import pytest

from evolvets.ledger.payloads import (
    Acceptance,
    Allocation,
    Baseline,
    Commitment,
    Correction,
    Evaluation,
    Prediction,
    Proposal,
    RunEnd,
    RunStart,
)
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
        requires_predictions=True,
        config_blob=blob_ref("config"),
        git_commit="abc123",
        protocol_versions=dict(PROTOCOL_VERSIONS),
        budget_limits=budget_limits,
    )


@pytest.fixture()
def run_end():
    return RunEnd(status="completed", summary={"records": 2, "note": "done"})


@pytest.fixture()
def proposal(blob_ref):
    return Proposal(
        target="updater",
        parent_artifact=blob_ref("parent"),
        child_artifact=blob_ref("child"),
        diff_blob=blob_ref("diff"),
        observation_blob=blob_ref("observation"),
        prompt_blob=None,
        response_blob=None,
        diagnosis="weights adapt too slowly after regime shifts",
        proposer={"id": "improver", "version": "3", "model": None, "temperature": None},
        improver_version="3",
        planted_defect=None,
    )


@pytest.fixture()
def prediction():
    return Prediction(
        predictor={"id": "selfmodel", "version": "1"},
        metric="mase",
        horizon_generations=1,
        quantiles={"q05": 0.1, "q25": 0.2, "q50": 0.3, "q75": 0.4, "q95": 0.6},
    )


@pytest.fixture()
def allocation():
    first, second = str(uuid.uuid4()), str(uuid.uuid4())
    return Allocation(
        allocator={"id": "thompson", "version": "1"},
        candidates=[first, second],
        chosen=[first],
        scores={first: 0.8, second: 0.3},
    )


@pytest.fixture()
def evaluation(blob_ref):
    return Evaluation(
        pool="dev",
        status="ok",
        metrics={"mase": 0.72, "recovery_steps": 4.0},
        per_task_blob=None,
        error=None,
    )


@pytest.fixture()
def acceptance(blob_ref):
    return Acceptance(accepted=True, rule="beats-parent-on-dev", new_artifact=blob_ref("new"))


@pytest.fixture()
def baseline():
    return Baseline(
        updater_id="hedge",
        hyperparams={"learning_rate": 0.05, "share": 0.1},
        pool="dev",
        metrics={"mase": 0.81},
    )


@pytest.fixture()
def commitment(blob_ref):
    return Commitment(
        subject="daily-forecast", content_hash=blob_ref("commitment"), external_timestamp=None
    )


@pytest.fixture()
def correction():
    return Correction(reason="wrong pool recorded", replacement={"pool": "dev"})
