"""The registry of foundation models a forecast cache can be built from.

Every value here was read from the model's official Hugging Face card, its
training-corpus listing on the Hub, or PyPI. Where a fact could not be
verified it is the literal string ``"unknown"`` (TASK_009: never guess a
value); PR C re-verifies each entry and pins ``hf_revision`` to a commit
hash. ``max_context`` is ``0`` (unset) until PR C reads it from the model
card, capped at 4096.
"""

from dataclasses import dataclass

from evolvets.foundation.errors import FoundationError


@dataclass(frozen=True)
class ModelSpec:
    """One foundation model: where it lives, how it is installed, what it saw in pretraining.

    ``pretraining_overlap`` records, per real dataset of this repo, whether the
    model's published pretraining corpus includes it (``"yes"``), provably
    excludes it (``"no"``), or cannot be decided from the published sources
    (``"unknown"``); ``overlap_source`` is the page the entries were read from.
    """

    name: str  # "chronos2", "tirex", "toto", "moirai2", "timesfm25"
    hf_model_id: str  # verified from the model card, or "unknown" until PR C
    hf_revision: str  # commit hash pinned in PR C; "unpinned" until then
    pip_package: str
    dependency_group: str  # "fm-chronos", ...
    max_context: int  # L; 0 (unset) until PR C verifies the model card
    license: str  # SPDX id or the license's name, as on the model card
    pretraining_overlap: dict[str, str]  # dataset -> "yes" | "no" | "unknown"
    overlap_source: str  # URL of the page the overlap entries were read from


MODELS: dict[str, ModelSpec] = {
    "chronos2": ModelSpec(
        name="chronos2",
        hf_model_id="amazon/chronos-2",
        hf_revision="unpinned",
        pip_package="chronos-forecasting",
        dependency_group="fm-chronos",
        max_context=0,
        license="apache-2.0",
        pretraining_overlap={"etth1": "unknown", "ecl": "yes"},
        overlap_source="https://huggingface.co/datasets/autogluon/chronos_datasets",
    ),
    "tirex": ModelSpec(
        name="tirex",
        hf_model_id="NX-AI/TiRex",
        hf_revision="unpinned",
        pip_package="tirex-ts",
        dependency_group="fm-tirex",
        max_context=0,
        license="nx-ai-community-license",
        pretraining_overlap={"etth1": "unknown", "ecl": "yes"},
        overlap_source="https://huggingface.co/datasets/autogluon/chronos_datasets",
    ),
    "toto": ModelSpec(
        name="toto",
        # two generations exist (1.0 "toto-ts", 2.0 "toto-models"); TASK_009 names no version
        hf_model_id="unknown",
        hf_revision="unpinned",
        pip_package="unknown",  # depends on the generation chosen for hf_model_id
        dependency_group="fm-toto",
        max_context=0,
        license="apache-2.0",  # both generations' cards say apache-2.0
        pretraining_overlap={"etth1": "unknown", "ecl": "yes"},
        overlap_source="https://huggingface.co/datasets/autogluon/chronos_datasets",
    ),
    "moirai2": ModelSpec(
        name="moirai2",
        hf_model_id="Salesforce/moirai-2.0-R-small",
        hf_revision="unpinned",
        pip_package="uni2ts",
        dependency_group="fm-moirai",
        max_context=0,
        license="cc-by-nc-4.0",
        pretraining_overlap={"etth1": "yes", "ecl": "yes"},
        overlap_source="https://huggingface.co/Salesforce/moirai-2.0-R-small",
    ),
    "timesfm25": ModelSpec(
        name="timesfm25",
        hf_model_id="google/timesfm-2.5-200m-pytorch",
        hf_revision="unpinned",
        pip_package="timesfm",
        dependency_group="fm-timesfm",
        max_context=0,
        license="apache-2.0",
        pretraining_overlap={"etth1": "no", "ecl": "no"},
        overlap_source="https://huggingface.co/google/timesfm-2.5-200m-pytorch",
    ),
}


def get_spec(name: str) -> ModelSpec:
    """Return the spec registered under ``name``, raising FoundationError if there is none."""
    try:
        return MODELS[name]
    except KeyError:
        known = ", ".join(MODELS)
        raise FoundationError(f"unknown foundation model {name!r}; known models: {known}") from None
