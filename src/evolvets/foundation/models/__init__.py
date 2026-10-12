"""Adapters around the pretrained model libraries, one module per model.

``foundation/models/<name>.py`` defines ``make_adapter(spec, device)`` and
does every heavy import inside that function, so importing this package (or
any adapter module) never needs torch or a model library. The libraries are
installed per model, in the ``fm-*`` dependency groups (TASK_009), never in
``dev``; tests that exercise a real adapter carry the ``foundation`` marker
and run in the GPU workflow.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Protocol, cast

from evolvets.foundation.errors import FoundationError
from evolvets.foundation.interfaces import Adapter
from evolvets.foundation.registry import ModelSpec, get_spec

__all__ = ["load_adapter", "require_adapter_module"]


class _AdapterModule(Protocol):
    """The interface every ``foundation/models/<name>.py`` module exposes."""

    def make_adapter(self, spec: ModelSpec, device: str) -> Adapter: ...


def require_adapter_module(name: str) -> str:
    """Return the adapter module name for registered ``name``, raising if there is none.

    A registered model whose adapter module is not in this checkout yet is
    TASK_009 PR C work that has not landed: the five real adapters arrive one
    PR at a time (``feat/foundation-models-<name>``). This check needs no
    heavy import, so commands that must refuse such a model without loading
    anything (``verify``) use it directly.
    """
    spec = get_spec(name)
    module_name = f"{__name__}.{spec.name}"
    if importlib.util.find_spec(module_name) is None:
        raise FoundationError(
            f"model {spec.name!r} arrives with TASK_009 PR C "
            f"(feat/foundation-models-{spec.name}); no adapter module exists in this checkout yet"
        )
    return module_name


def load_adapter(name: str, device: str) -> Adapter:
    """Return the adapter for the registered model ``name``, running on ``device``.

    The adapter module is imported lazily and built through its
    ``make_adapter``; a missing model library raises :class:`FoundationError`
    naming the dependency group that installs it.
    """
    spec = get_spec(name)
    module_name = require_adapter_module(name)
    module = cast(_AdapterModule, importlib.import_module(module_name))
    return module.make_adapter(spec, device)
