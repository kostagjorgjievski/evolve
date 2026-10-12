"""TiRex adapter (TASK_009 PR C2): NX-AI's 35M xLSTM quantile forecaster.

Registry facts, verified 2026-10-12 from the sources named in the spec: the
model is ``NX-AI/TiRex`` (``nx-ai-community-license`` weights, a 35M-parameter
xLSTM), run through the ``tirex-ts`` package (>= 1.4) in the ``fm-tirex``
dependency group. The library's FAQ puts the maximum context length at 2048
(the paper trains with a context window of 2048); the adapter validates the
registry's ``max_context`` against the checkpoint's own ``train_ctx_len`` so
the model never silently truncates a context.

Quantile method: **native** (``spec.quantile_method == "native"``). TiRex is
trained on the levels ``Q = {0.1, 0.2, ..., 0.9}`` (paper, section 3) —
exactly :data:`evolvets.foundation.interfaces.QUANTILES` — and its forecast
API takes no quantile argument: the checkpoint's ``config.quantiles`` is what
it returns. The adapter asserts at load time that the pinned checkpoint's
levels are our nine; a checkpoint that differs is refused rather than
silently re-mapped.

Batching: ``forecast`` receives each context as its own univariate 1D tensor
and the library left-pads the variable lengths with NaN itself (the model
masks the padding), as its documentation specifies; the batch dimension only
batches series — each row is forecast from its own values alone, with no
cross-series attention or context. The model weights come from whatever
``HF_ENDPOINT`` points at — never hard-coded here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evolvets.foundation.errors import FoundationError
from evolvets.foundation.interfaces import QUANTILES, Adapter
from evolvets.foundation.models._device import library_versions, resolve_device
from evolvets.foundation.registry import ModelSpec


def make_adapter(spec: ModelSpec, device: str) -> Adapter:
    """Load the pinned TiRex checkpoint on ``device`` and wrap it as an :class:`Adapter`."""
    try:
        from tirex import load_model
    except ModuleNotFoundError as exc:
        raise FoundationError(
            f"model {spec.name!r} needs the {spec.dependency_group!r} dependency group: "
            "install it and run through 'uv run --group fm-tirex ...'"
        ) from exc

    resolved = resolve_device(device)
    model = load_model(
        spec.hf_model_id,
        device=resolved.device_map,
        hf_kwargs={"revision": spec.hf_revision},
    )
    model.eval()
    trained_quantiles = [float(level) for level in model.config.quantiles]
    if trained_quantiles != list(QUANTILES):
        raise FoundationError(
            f"model {spec.name!r}: checkpoint quantile levels {trained_quantiles} are not "
            f"the registry's {list(QUANTILES)}; the registry entry (quantile_method "
            f"{spec.quantile_method!r}) no longer matches the pinned revision"
        )
    context_limit = int(model.config.train_ctx_len)
    if spec.max_context > context_limit:
        raise FoundationError(
            f"model {spec.name!r}: max_context {spec.max_context} exceeds the checkpoint's "
            f"context length {context_limit}; verify the registry entry against the model card"
        )
    return TiRexAdapter(spec=spec, model=model, device=resolved.label)


class TiRexAdapter:
    """Adapter over ``ForecastModel.forecast`` exposing the run metadata the manifest records."""

    def __init__(self, spec: ModelSpec, model: Any, device: str) -> None:
        self.name = spec.name
        self._model = model
        self.device = device
        self.dtype = "float32"  # TiRex casts contexts to float32 and forecasts float32
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
        quantiles, _mean = self._model.forecast(
            context=tensors,
            prediction_length=horizon,
            output_type="torch",
            batch_size=len(tensors),
        )
        rows: list[list[list[float]]] = []
        for i, forecast in enumerate(quantiles):
            shape = tuple(forecast.shape)
            expected = (horizon, len(QUANTILES))
            if shape != expected:
                raise FoundationError(
                    f"{self.name}: context {i}: forecast shape {shape}, expected {expected}"
                )
            rows.append(
                [[float(forecast[h, q]) for h in range(horizon)] for q in range(len(QUANTILES))]
            )
        return rows
