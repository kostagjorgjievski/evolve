"""The arena: the only package that may load sealed or live pools.

``ARENA_VERSION`` is the protocol version runs put in
``run_start.protocol_versions["arena"]``. The arena protocol covers
``updater/interfaces.py``, ``arena/labels.py``, ``arena/replay.py``,
``arena/dev.py``, ``arena/pool.py``, ``arena/metrics.py``,
``arena/evaluate.py``, ``arena/oracles.py`` and ``arena/sealed/``: any change
that alters what an updater receives, what it may return, which origins are
scored, any metric value, or what the sealed process layout promises a caller
must bump it. Version 2 added the pool spec: a replay may now drive the
simple five plus cached foundation experts (``simple+foundation``); a
``simple`` replay is bit-identical to version 1. The sealed pools serve the
same spec — their foundation caches are built and joined under the sealed
cache root, and the sealed server's request carries the spec and the model
names. Version 3 added the hindsight oracles over the expert pool
(``arena/oracles.py``); they change no replayed score, only what the arena
can report about a pool. Version 4 added the forecast-v2 updater interface
(``updater/interfaces.py``): a forecast-v2 updater may return its own point
forecast instead of weights; weights-v1 replays are bit-identical to version 3
apart from the version string and the recorded interface. This file stays
import-free.
"""

ARENA_VERSION = "4"
