"""A deterministic torch-free adapter for tests, CI and smoke runs."""

from __future__ import annotations

from collections.abc import Sequence

from evolvets.foundation.interfaces import QUANTILES
from evolvets.foundation.registry import ModelSpec


class FakeAdapter:
    """Quantile forecasts from a closed form: ``last + (q - 0.5) * spread + h * slope``.

    ``last`` is the context's last value, ``spread`` its max minus min, and
    ``slope`` is ``(last - first) / len(context)`` for the context's first
    value; ``h`` counts 1 .. horizon and ``q`` runs over :data:`QUANTILES`.
    The result is deterministic in the context alone, so equal contexts give
    equal forecasts whatever the batching. Every context received is recorded,
    in order, in ``seen``.

    It also carries the run metadata the cache builder records in the
    manifest — ``device = "cpu"``, ``dtype = "float64"`` and
    ``library_versions = {}`` — because it uses no libraries and no GPU.
    """

    def __init__(self, name: str = "fake") -> None:
        self.name = name
        self.seen: list[tuple[float, ...]] = []
        self.device = "cpu"
        self.dtype = "float64"
        self.library_versions: dict[str, str] = {}

    def forecast(
        self, contexts: Sequence[Sequence[float]], horizon: int
    ) -> list[list[list[float]]]:
        """Return one ``[quantile][h]`` row per context, recording each context in ``seen``."""
        out: list[list[list[float]]] = []
        for context in contexts:
            values = tuple(float(v) for v in context)
            self.seen.append(values)
            last = values[-1]
            spread = max(values) - min(values)
            slope = (last - values[0]) / len(values)
            out.append(
                [
                    [last + (q - 0.5) * spread + h * slope for h in range(1, horizon + 1)]
                    for q in QUANTILES
                ]
            )
        return out


#: The spec ``--model fake`` builds with (``evolvets foundation build``, for
#: tests and CI only). No Hugging Face id, package or license applies — the
#: fake is generated code, not a pretrained model, so it saw nothing in
#: pretraining — and its context window is 512 values. Real models arrive
#: with TASK_009 PR C, which replaces the registry's placeholder entries.
FAKE_SPEC = ModelSpec(
    name="fake",
    hf_model_id="none",
    hf_revision="none",
    pip_package="none",
    dependency_group="none",
    max_context=512,
    license="none",
    pretraining_overlap={"etth1": "no", "ecl": "no"},
    overlap_source="none",
)
