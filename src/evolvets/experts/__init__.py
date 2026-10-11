"""Streaming, deterministic experts over a single series.

``EXPERTS_VERSION`` is the protocol version runs record in
``run_start.protocol_versions["experts"]``. Any change that alters any
forecast value must bump it; the golden-hash tests enforce this.

Experts see values only: nothing in this package imports from another
``evolvets`` package, and an expert never receives a world, a mean path,
a split or a timestamp — just floats, one at a time.
"""

EXPERTS_VERSION = "1"
