"""Sealed evaluation pools.

Only this package may open sealed data — the sealed segment of the real
datasets and the salted synthetic worlds — and only trusted processes may
import :mod:`evolvets.arena.sealed.pools`: the human-started ``evolvets
sealed`` CLI and the sealed server it spawns. This ``__init__`` deliberately
imports nothing, so importing the package loads no sealed file and stays
cheap for the caller-side code that reads
:mod:`evolvets.arena.sealed.names` and
:mod:`evolvets.arena.sealed.client`.
"""
