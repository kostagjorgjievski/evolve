"""Foundation-model experts, run offline on a GPU into content-hashed caches.

Instrument code (not agent-editable). Heavy libraries (torch, model packages)
are imported lazily inside ``foundation/models/`` only, so importing this
package never needs them. See ``docs/tasks/TASK_009_foundation_experts.md``.

``FOUNDATION_CACHE_VERSION`` covers the cache file layout, manifest fields
and quantile levels; bump it on any change to them. Version 2 (TASK_009 PR
C1) adds ``model.quantile_method`` to the manifest's model object; no cache
built under version 1 existed outside tests.
"""

FOUNDATION_CACHE_VERSION = "2"
