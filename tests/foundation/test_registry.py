"""Registry tests: order, dependency groups, overlap shape, unknown names (TASK_009 PR A 8)."""

import pytest

from evolvets.foundation.errors import FoundationError
from evolvets.foundation.registry import MODELS, get_spec

GROUPS = ("fm-chronos", "fm-tirex", "fm-toto", "fm-moirai", "fm-timesfm")


def test_registry_order_and_fields():
    """The five models appear in task order, each with a valid group and overlap map."""
    assert tuple(MODELS) == ("chronos2", "tirex", "toto", "moirai2", "timesfm25")
    for spec in MODELS.values():
        assert spec.dependency_group in GROUPS
        assert set(spec.pretraining_overlap) == {"etth1", "ecl"}
        assert all(value in {"yes", "no", "unknown"} for value in spec.pretraining_overlap.values())
        assert spec.overlap_source.startswith(("http://", "https://"))
        assert spec.quantile_method  # "native", "closed-form", ...; "unknown" until PR C
        assert MODELS[spec.name] is spec


def test_get_spec():
    """get_spec returns the registered spec and raises FoundationError for anything else."""
    assert get_spec("chronos2").pip_package == "chronos-forecasting"
    with pytest.raises(FoundationError, match="unknown foundation model"):
        get_spec("gpt-5-forecaster")
