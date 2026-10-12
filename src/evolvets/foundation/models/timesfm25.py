"""TimesFM 2.5 adapter (TASK_009 PR C5): Google's 200M decoder-only quantile forecaster.

Registry facts, verified 2026-10-12 from the sources named in the spec: the
model is ``google/timesfm-2.5-200m-pytorch`` (Apache-2.0, F32 safetensors
weights), run through the ``timesfm`` package (>= 3.0, the first PyPI line
with the TimesFM 2.5 classes; the card's "pip install from PyPI coming soon"
note predates it) in the ``fm-timesfm`` dependency group. The checkpoint's own
``config.json`` puts ``context_length`` at 16384; TASK_009 caps
``max_context`` at 4096. The torch class picks its device itself (``cuda:0``
when CUDA is available, else CPU) and takes no device argument, so the
adapter moves the module to the requested device after load — the manifest
must record the device the forecasts actually used.

Quantile method: **native** (``spec.quantile_method == "native"``). The
checkpoint — its ``config.json`` and the library's model definition both —
trains the levels ``[0.1, 0.2, ..., 0.9]`` = exactly
:data:`evolvets.foundation.interfaces.QUANTILES`, and ``forecast`` returns
them as columns 1..9 of its quantile output (column 0 is the mean). The
adapter asserts the levels at load time; a checkpoint that differs is
refused rather than silently re-mapped. The card's ``ForecastConfig`` is
used as documented: ``use_continuous_quantile_head=True`` (the non-median
levels come from the continuous spread head — still the same nine levels,
not an interpolation between foreign levels and not empirical quantiles of
samples), ``fix_quantile_crossing=True`` (the library's own monotone
rearrangement; the builder's crossing counter remains the check on top),
``normalize_inputs=True``, ``force_flip_invariance=True`` and
``infer_is_positive=True`` (which clamps only rows whose own context is
nonnegative).

Batching: ``forecast`` left-pads each context to the compiled ``max_context``
with zeros behind a mask the model excludes from its attention, statistics
and normalization, and truncates nothing the builder sends (contexts arrive
already sliced to ``max_context``). The batch dimension only batches series
— every operation inside the decoder acts on one row's patches, with no
cross-series attention or context; the base class pads a call up to
``per_core_batch_size`` with dummy series whose rows are dropped from the
output. The adapter compiles once with ``max_context`` from the spec and
grows ``per_core_batch_size`` to the largest batch it is handed, so a build
decodes its builder batches whole; horizons compile up to the model's
output-patch quantum of 128, and a longer horizon is a
:class:`~evolvets.foundation.errors.FoundationError`.

The checkpoint is loaded **eager** (``torch_compile=False``, a speed knob
only — the checkpoint's own config.json records the same). The card's
example passes ``torch_compile=True``, but inductor's C++ codegen imports
``setuptools`` at compile time; it is not in the ``fm-timesfm`` group and
TASK_009 names no such dependency to add, so the compiled path is not
available in these environments and eager keeps CPU and GPU runs alike. The
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

#: Horizons the compiled decode covers: the model's output-patch length, the
#: quantum the library rounds any ``max_horizon`` up to (DEFAULT_HORIZON is 24).
HORIZON_LIMIT = 128


def make_adapter(spec: ModelSpec, device: str) -> Adapter:
    """Load the pinned TimesFM 2.5 checkpoint on ``device`` and wrap it as an Adapter."""
    try:
        import timesfm
    except ModuleNotFoundError as exc:
        raise FoundationError(
            f"model {spec.name!r} needs the {spec.dependency_group!r} dependency group: "
            "install it and run through 'uv run --group fm-timesfm ...'"
        ) from exc

    resolved = resolve_device(device)
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
        spec.hf_model_id,
        revision=spec.hf_revision,
        torch_compile=False,
    )
    _move_to_requested_device(model, resolved.device_map)
    definition = model.model.config
    trained_quantiles = [float(level) for level in definition.quantiles]
    if trained_quantiles != list(QUANTILES):
        raise FoundationError(
            f"model {spec.name!r}: checkpoint quantile levels {trained_quantiles} are not "
            f"the registry's {list(QUANTILES)}; the registry entry (quantile_method "
            f"{spec.quantile_method!r}) no longer matches the pinned revision"
        )
    context_limit = int(definition.context_limit)
    if spec.max_context > context_limit:
        raise FoundationError(
            f"model {spec.name!r}: max_context {spec.max_context} exceeds the checkpoint's "
            f"context length {context_limit}; verify the registry entry against the model card"
        )
    patch_len = int(definition.input_patch_len)
    if spec.max_context % patch_len != 0:
        raise FoundationError(
            f"model {spec.name!r}: max_context {spec.max_context} is not a multiple of the "
            f"checkpoint's patch size {patch_len}; the compiled padding would change it"
        )
    return TimesFM25Adapter(spec=spec, model=model, device=resolved.label)


def _move_to_requested_device(model: Any, device_map: str) -> None:
    """Move the self-placed torch module onto the device the caller asked for.

    ``TimesFM_2p5_200M_torch`` has no device argument: its module takes
    ``cuda:0`` whenever CUDA is available and CPU otherwise, and the plain
    ``device``/``device_count`` attributes the decoder reads are not updated
    by :meth:`nn.Module.to`, so a CPU request on a CUDA machine needs all
    three set by hand.
    """
    import torch

    module = model.model
    if device_map == "cpu":
        module.to("cpu")
        module.device = torch.device("cpu")
        module.device_count = 1
    elif module.device.type != "cuda":
        raise FoundationError("--device cuda but the loaded module sits on cpu")


class TimesFM25Adapter:
    """Adapter over ``TimesFM_2p5_200M_torch`` exposing the run metadata the manifest records."""

    def __init__(self, spec: ModelSpec, model: Any, device: str) -> None:
        self.name = spec.name
        self._model = model
        self._spec = spec
        self.device = device
        parameters = iter(model.model.parameters())
        self.dtype = str(next(parameters).dtype).removeprefix("torch.")
        self.library_versions = library_versions((spec.pip_package, "torch", "numpy"))
        self._compiled_batch = 0
        self._compile(batch_size=1)

    def _compile(self, batch_size: int) -> None:
        """Compile the library's forecast config for ``batch_size`` contexts per decode."""
        import timesfm

        self._model.compile(
            timesfm.ForecastConfig(
                max_context=self._spec.max_context,
                max_horizon=HORIZON_LIMIT,
                normalize_inputs=True,
                per_core_batch_size=batch_size,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        self._compiled_batch = batch_size

    def forecast(
        self, contexts: Sequence[Sequence[float]], horizon: int
    ) -> list[list[list[float]]]:
        """Return one ``[quantile][h]`` row per context, in input order."""
        import numpy as np

        if horizon > HORIZON_LIMIT:
            raise FoundationError(
                f"{self.name}: horizon {horizon} exceeds the compiled maximum "
                f"{HORIZON_LIMIT} (the checkpoint's output-patch quantum)"
            )
        if len(contexts) > self._compiled_batch:
            self._compile(batch_size=len(contexts))
        inputs = []
        for i, context in enumerate(contexts):
            if not len(context):
                raise FoundationError(f"{self.name}: context {i}: refused an empty context")
            inputs.append(np.asarray([float(value) for value in context], dtype=np.float32))
        _point, quantiles = self._model.forecast(horizon=horizon, inputs=inputs)
        expected = (len(contexts), horizon, len(QUANTILES) + 1)  # column 0 is the mean
        if tuple(quantiles.shape) != expected:
            raise FoundationError(
                f"{self.name}: forecast shape {tuple(quantiles.shape)}, expected {expected}"
            )
        return [
            [[float(quantiles[i, h, 1 + q]) for h in range(horizon)] for q in range(len(QUANTILES))]
            for i in range(len(contexts))
        ]
