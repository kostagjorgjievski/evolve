"""The protocol versions a run records, in one place."""

from evolvets.arena import ARENA_VERSION
from evolvets.arena.synthetic import GENERATOR_VERSION
from evolvets.cache import CACHE_VERSION
from evolvets.ledger.schema import SCHEMA_VERSION


def protocol_versions() -> dict[str, str]:
    """Return the current version of every protocol a run pins.

    The keys are exactly the ledger's ``PROTOCOL_KEYS``: ``ledger``,
    ``arena``, ``generator`` and ``cache_manifest``.
    """
    return {
        "ledger": SCHEMA_VERSION,
        "arena": ARENA_VERSION,
        "generator": GENERATOR_VERSION,
        "cache_manifest": CACHE_VERSION,
    }
