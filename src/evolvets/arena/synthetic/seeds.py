"""Per-pool seed derivation for synthetic tasks.

Dev and sealed tasks come from disjoint, domain-separated seed spaces: the
pool, family, index and salt are joined with NUL bytes before hashing, so
no combination leaks into another. Sealed seeds need a salt of at least 32
bytes; without the salt, sealed tasks cannot be regenerated from the
repository even though the grammar is public. This module never creates,
stores or reads a salt — where the sealed salt lives is decided by the
``feat/arena-sealed`` spec.
"""

import hashlib
from typing import Literal

from evolvets.arena.synthetic.grammar import FAMILIES, GrammarConfig, SyntheticTask, make_task

_POOLS = ("dev", "sealed")
_PREFIX = b"evolvets/generator/v1"
_SEALED_SALT_BYTES = 32
_SEED_MASK = 2**63 - 1
_DEFAULT_CONFIG = GrammarConfig()


def world_seed(pool: Literal["dev", "sealed"], family: str, index: int, salt: bytes = b"") -> int:
    """Return the world seed for ``pool``/``family``/``index``, an int in ``[0, 2**63)``.

    The ``dev`` pool takes no salt (its tasks are public and reproducible
    from the repository); the ``sealed`` pool requires a salt of at least
    32 bytes.
    """
    if pool not in _POOLS:
        raise ValueError(f"pool must be one of {_POOLS}, got {pool!r}")
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError(f"index must be an int, got {index!r}")
    if index < 0:
        raise ValueError(f"index must be >= 0, got {index}")
    if pool == "dev" and salt:
        raise ValueError("the dev pool takes no salt")
    if pool == "sealed" and len(salt) < _SEALED_SALT_BYTES:
        raise ValueError(f"the sealed pool needs a salt of at least {_SEALED_SALT_BYTES} bytes")
    payload = b"\x00".join(
        (
            _PREFIX,
            pool.encode("utf-8"),
            family.encode("utf-8"),
            str(index).encode("utf-8"),
            salt,
        )
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _SEED_MASK


def dev_tasks(family: str, n: int, config: GrammarConfig = _DEFAULT_CONFIG) -> list[SyntheticTask]:
    """Return the first ``n`` public dev tasks for ``family``, in index order.

    There is deliberately no sealed counterpart in this task: sealed tasks
    are created only by the arena's sealed pool, from the sealed salt.
    """
    return [make_task(family, world_seed("dev", family, i), config) for i in range(n)]
