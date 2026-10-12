"""Device resolution shared by the model adapters (torch is imported lazily)."""

from __future__ import annotations

from dataclasses import dataclass

from evolvets.foundation.errors import FoundationError

_ALLOWED = ("auto", "cpu", "cuda")


@dataclass(frozen=True)
class ResolvedDevice:
    """Where an adapter runs: the ``device_map`` to load with, and the manifest label.

    ``label`` is what the cache manifest records in its ``device`` field, for
    example ``"cpu"`` or ``"cuda:NVIDIA GeForce RTX 3080 Ti"``; it names the
    hardware the forecasts were computed on.
    """

    device_map: str
    label: str


def resolve_device(device: str) -> ResolvedDevice:
    """Map ``auto``/``cpu``/``cuda`` to a concrete device, needing torch only for GPUs.

    ``auto`` uses CUDA when available and falls back to CPU; ``cuda`` without
    an available CUDA device is a :class:`FoundationError`, not a silent CPU
    run — the manifest must never record a device the forecasts did not use.
    """
    if device not in _ALLOWED:
        raise FoundationError(f"device must be one of {_ALLOWED}, got {device!r}")
    if device == "cpu":
        return ResolvedDevice(device_map="cpu", label="cpu")
    import torch

    if not torch.cuda.is_available():
        if device == "cuda":
            raise FoundationError("--device cuda but torch.cuda.is_available() is False")
        return ResolvedDevice(device_map="cpu", label="cpu")
    return ResolvedDevice(device_map="cuda", label=f"cuda:{torch.cuda.get_device_name()}")


def library_versions(packages: tuple[str, ...]) -> dict[str, str]:
    """Return ``{package: version}`` for ``packages`` read with ``importlib.metadata``.

    Every package must be importable metadata-wise in the running group
    environment; a missing one is an environment fault, so its
    :class:`importlib.metadata.PackageNotFoundError` propagates unchanged
    rather than being recorded as a made-up version.
    """
    from importlib.metadata import version

    return {package: version(package) for package in packages}
