"""The arena: the only package that may load sealed or live pools.

``ARENA_VERSION`` is the protocol version runs put in
``run_start.protocol_versions["arena"]``. The arena protocol covers
``updater/interfaces.py``, ``arena/labels.py``, ``arena/replay.py``,
``arena/dev.py``, ``arena/pool.py``, ``arena/metrics.py``,
``arena/evaluate.py`` and ``arena/sealed/``: any change that alters what an
updater receives, what it may return, which origins are scored, any metric
value, or what the sealed process layout promises a caller must bump it.
Version 2 added the pool spec: a replay may now drive the simple five plus
cached foundation experts (``simple+foundation``); a ``simple`` replay is
bit-identical to version 1. This file stays import-free.
"""

ARENA_VERSION = "2"
