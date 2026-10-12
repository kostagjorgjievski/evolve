"""Chronos-2 adapter (TASK_009 PR C1): Amazon's 120M quantile forecaster.

Registry facts, verified 2026-10-12 from the sources named in the spec: the
model is ``amazon/chronos-2`` (Apache-2.0, F32 safetensors weights, loaded
uncast so the manifest records dtype ``"float32"``), run through the
``chronos-forecasting`` package (>= 2.0) in the ``fm-chronos`` dependency
group. The model card puts the maximum context length at 8192; TASK_009
caps ``max_context`` at 4096.

Quantile method: **native** (``spec.quantile_method == "native"``).
``Chronos2Pipeline.predict_quantiles`` is asked for exactly
:data:`evolvets.foundation.interfaces.QUANTILES`, which are Chronos-2's
training levels (the library's default ``quantile_levels`` is the same
0.1 .. 0.9), so the model returns the nine quantiles directly — no
interpolation between levels and no empirical quantiles of samples.

Batching: every context is passed as its own univariate series and the
library left-pads the variable lengths itself, as its documentation
specifies; ``cross_learning=False`` keeps the contexts in one batch
independent of each other (no cross-series attention or context). The
model weights come from whatever ``HF_ENDPOINT`` points at — never
hard-coded here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evolvets.foundation.errors import FoundationError
from evolvets.foundation.interfaces import QUANTILES, Adapter
from evolvets.foundation.models._device import library_versions, resolve_device
from evolvets.foundation.registry import ModelSpec


def make_adapter(spec: ModelSpec, device: str) -> Adapter:
    """Load the pinned Chronos-2 checkpoint on ``device`` and wrap it as an :class:`Adapter`."""
    try:
        from chronos import Chronos2Pipeline
    except ModuleNotFoundError as exc:
        raise FoundationError(
            f"model {spec.name!r} needs the {spec.dependency_group!r} dependency group: "
            "install it and run through 'uv run --group fm-chronos ...'"
        ) from exc

    resolved = resolve_device(device)
    pipeline = Chronos2Pipeline.from_pretrained(
        spec.hf_model_id,
        revision=spec.hf_revision,
        device_map=resolved.device_map,
    )
    context_limit = int(pipeline.model_context_length)
    if spec.max_context > context_limit:
        raise FoundationError(
            f"model {spec.name!r}: max_context {spec.max_context} exceeds the checkpoint's "
            f"context length {context_limit}; verify the registry entry against the model card"
        )
    return Chronos2Adapter(spec=spec, pipeline=pipeline, device=resolved.label)


class Chronos2Adapter:
    """Adapter over ``Chronos2Pipeline`` exposing the run metadata the manifest records."""

    def __init__(self, spec: ModelSpec, pipeline: Any, device: str) -> None:
        self.name = spec.name
        self._pipeline = pipeline
        self.device = device
        self.dtype = "float32"  # amazon/chronos-2 ships F32 weights, loaded uncast
        self.library_versions = library_versions((spec.pip_package, "torch", "numpy"))

    def forecast(
        self, contexts: Sequence[Sequence[float]], horizon: int
    ) -> list[list[list[float]]]:
        """Return one ``[quantile][h]`` row per context, in input order."""
        import torch

        tensors = []
        for context in contexts:
            values = [float(value) for value in context]
            if not values:
                raise FoundationError(f"{self.name}: refused an empty context")
            tensors.append(torch.tensor(values, dtype=torch.float32))
        quantiles, _mean = self._pipeline.predict_quantiles(
            tensors,
            prediction_length=horizon,
            quantile_levels=list(QUANTILES),
            cross_learning=False,
        )
        rows: list[list[list[float]]] = []
        for i, forecast in enumerate(quantiles):
            shape = tuple(forecast.shape)
            expected = (1, horizon, len(QUANTILES))
            if shape != expected:
                raise FoundationError(
                    f"{self.name}: context {i}: forecast shape {shape}, expected {expected}"
                )
            rows.append(
                [[float(forecast[0, h, q]) for h in range(horizon)] for q in range(len(QUANTILES))]
            )
        return rows
