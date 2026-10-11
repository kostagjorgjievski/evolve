"""Synthetic worlds with a known true-best forecast.

``GENERATOR_VERSION`` is the protocol version runs record in
``run_start.protocol_versions["generator"]``. Any change that alters the
output for an existing (spec, seed) must bump it; the golden-fingerprint
tests enforce this.
"""

GENERATOR_VERSION = "1"
