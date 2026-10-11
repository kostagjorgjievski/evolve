"""Content-addressed cache of expert forecasts over public series.

``CACHE_VERSION`` is the protocol version runs record in
``run_start.protocol_versions["cache_manifest"]``; the specific cache a run
used is identified by the cache id, the SHA-256 of the dataset's
``manifest.json`` bytes. Any change to the cache file layout or manifest must
bump ``CACHE_VERSION``; a change that alters a forecast value bumps the
experts' ``EXPERTS_VERSION`` instead.

``cache_store/`` is readable by any process, which is why only public data
may be cached in it; sealed caches are served by the arena's sealed process.
"""

CACHE_VERSION = "1"
