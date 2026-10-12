"""The registry of foundation models a forecast cache can be built from.

Every value here was read from the model's official Hugging Face card, its
training-corpus listing on the Hub, PyPI, or the model library's own
documentation. Where a fact could not be verified it is the literal string
``"unknown"`` (TASK_009: never guess a value). PR C pins each model's
``hf_revision`` to a commit hash, verifies ``max_context`` from the model
card (capped at 4096) and fills ``quantile_method`` from what the library
actually returns; a model still waiting for its PR keeps its PR A
placeholder entry, with ``hf_revision = "unpinned"``.
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
    quantile_method: str  # how the adapter obtains QUANTILES; "unknown" until PR C
    license: str  # SPDX id or the license's name, as on the model card
    pretraining_overlap: dict[str, str]  # dataset -> "yes" | "no" | "unknown"
    overlap_source: str  # URL of the page the overlap entries were read from


MODELS: dict[str, ModelSpec] = {
    # Facts verified 2026-10-12 for PR C1 (feat/foundation-models-chronos2):
    # model card https://huggingface.co/amazon/chronos-2 (max context 8192,
    # capped at 4096 by TASK_009; Apache-2.0; F32 weights; pip package
    # "chronos-forecasting>=2.0", whose Chronos2Pipeline.predict_quantiles
    # natively returns the training levels 0.1 .. 0.9 = QUANTILES), commit
    # hash from https://huggingface.co/api/models/amazon/chronos-2 field
    # "sha". Overlap read from the card's corpus listing: "electricity_15min"
    # (UCI ElectricityLoadDiagrams2011_2014, the ECL source) is included, so
    # ecl is "yes"; etth1 is not in that listing, but the card names a second
    # corpus (Salesforce/GiftEvalPretrain) whose contents were not verified
    # within the lookup time box, so etth1 stays "unknown".
    "chronos2": ModelSpec(
        name="chronos2",
        hf_model_id="amazon/chronos-2",
        hf_revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
        pip_package="chronos-forecasting",
        dependency_group="fm-chronos",
        max_context=4096,
        quantile_method="native",
        license="apache-2.0",
        pretraining_overlap={"etth1": "unknown", "ecl": "yes"},
        overlap_source="https://huggingface.co/datasets/autogluon/chronos_datasets",
    ),
    # Facts verified 2026-10-12 for PR C2 (feat/foundation-models-tirex):
    # model card https://huggingface.co/NX-AI/TiRex (pip "tirex-ts"; weights
    # under license_name "nx-ai-community-license"; 35M-parameter xLSTM),
    # commit hash from https://huggingface.co/api/models/NX-AI/TiRex field
    # "sha". Max context 2048 from the library FAQ ("TiRex supports a maximum
    # context length of up to 2,048"; the paper trains with a context window
    # of 2048); the adapter validates max_context against the checkpoint's own
    # train_ctx_len. Quantile method native: the paper trains on the levels
    # Q = {0.1, 0.2, ..., 0.9} = QUANTILES, and the forecast API returns
    # exactly the checkpoint's config.quantiles (asserted at load time).
    # Overlap: the card names the same two corpora as chronos-2 —
    # autogluon/chronos_datasets (includes "electricity_15min", the UCI
    # ElectricityLoadDiagrams2011_2014 source, so ecl is "yes") and
    # Salesforce/GiftEvalPretrain. ETTh is not in the chronos_datasets repo
    # (it lives in the companion autogluon/chronos_datasets_extra, which the
    # card does not name), and GiftEvalPretrain's card does not enumerate its
    # datasets, so etth1 stays "unknown".
    "tirex": ModelSpec(
        name="tirex",
        hf_model_id="NX-AI/TiRex",
        hf_revision="63c740922493f5fbe60b277609ec62babfba2762",
        pip_package="tirex-ts",
        dependency_group="fm-tirex",
        max_context=2048,
        quantile_method="native",
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
        quantile_method="unknown",
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
        quantile_method="unknown",
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
        quantile_method="unknown",
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
