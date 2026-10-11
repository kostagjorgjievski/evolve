"""The arena: the only package that may load sealed or live pools.

``ARENA_VERSION`` is the protocol version runs put in
``run_start.protocol_versions["arena"]``. The arena protocol covers
``updater/interfaces.py``, ``arena/labels.py``, ``arena/replay.py``,
``arena/dev.py``, ``arena/metrics.py`` and ``arena/evaluate.py``: any change
that alters what an updater receives, what it may return, which origins are
scored or any metric value must bump it. This file stays import-free.
"""

ARENA_VERSION = "1"
