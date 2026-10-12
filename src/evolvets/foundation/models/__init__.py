"""Adapters around the pretrained model libraries, one module per model.

``foundation/models/<name>.py`` defines ``make_adapter(spec, device)`` and
does every heavy import inside that function, so importing this package (or
any adapter module) never needs torch or a model library. The libraries are
installed per model, in the ``fm-*`` dependency groups (TASK_009), never in
``dev``; tests that exercise a real adapter carry the ``foundation`` marker
and run in the GPU workflow. TASK_009 PR C ended with three adapters
(``chronos2``, ``tirex``, ``timesfm25``); ``toto`` and ``moirai2`` were
dropped and keep placeholder registry entries only.
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

    After TASK_009 PR C, the only registered models without an adapter module
    are ``toto`` and ``moirai2``, which the task dropped over the torch and
    Python pins (issues #64 and #66); their placeholder registry entries stay
    for the record and no adapter will arrive. This check needs no heavy
    import, so commands that must refuse such a model without loading
    anything (``verify``) use it directly.
    """
    spec = get_spec(name)
    module_name = f"{__name__}.{spec.name}"
    if importlib.util.find_spec(module_name) is None:
        raise FoundationError(
            f"model {spec.name!r} has no adapter module in this checkout and none is "
            "planned: TASK_009 PR C built chronos2, tirex and timesfm25, and dropped "
            "toto (issue #64) and moirai2 (issue #66) over the torch and Python pins"
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
