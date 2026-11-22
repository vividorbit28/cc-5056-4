"""LRU cache for assembled sparse systems and their factorizations.

The cache stores, per structural fingerprint, the assembled matrix ``A``, the
normal-equation matrix ``A^T A`` and its sparse LU factorization.  A
fingerprint captures everything that changes ``A`` (unknown pixels, edge
topology, row weights, anchor positions/weights, gauge choice, boundary
mode).  Right-hand-side-only edits (anchor colours, target/source pixel
values) keep the fingerprint stable, so they reuse the factorization and pay
only for RHS assembly and triangular solves.

The cache is deliberately conservative: if any structural input changes the
fingerprint changes and the old entry is never used.  That is what prevents a
resized region from silently reusing a matrix that misses the new pixels.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


def hash_array(hasher: "hashlib._Hash", arr: np.ndarray, tag: str) -> None:
    a = np.ascontiguousarray(arr)
    hasher.update(tag.encode("ascii"))
    hasher.update(str(a.dtype).encode("ascii"))
    hasher.update(repr(a.shape).encode("ascii"))
    hasher.update(a.tobytes())


def structural_fingerprint(**parts) -> str:
    """Stable hash of every input that determines the assembled matrix.

    Arrays are hashed by content; scalars by repr.  Callers must pass all and
    only structural quantities (never anchor colours or target pixels).
    """
    h = hashlib.sha256()
    for key in sorted(parts):
        value = parts[key]
        if isinstance(value, np.ndarray):
            hash_array(h, value, key)
        elif isinstance(value, (list, tuple)) and value and isinstance(
                value[0], (list, tuple)):
            arr = np.asarray(value)
            hash_array(h, arr, key)
        else:
            h.update(key.encode("ascii"))
            h.update(repr(value).encode("utf-8"))
    return h.hexdigest()


@dataclass
class CachedSystem:
    fingerprint: str
    A: object            # csr_matrix, rows already include gauge/anchor rows
    AtA: object          # csc_matrix normal equations
    lu: object           # scipy SuperLU factorization of AtA
    meta: dict


class FactorizationCache:
    """Thread-safe bounded LRU of factorizations."""

    def __init__(self, max_entries: int = 8):
        self.max_entries = int(max_entries)
        self._data: "OrderedDict[str, CachedSystem]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.poisoned = 0

    def get(self, fingerprint: str) -> CachedSystem | None:
        with self._lock:
            entry = self._data.get(fingerprint)
            if entry is None:
                self.misses += 1
                return None
            self._data.move_to_end(fingerprint)
            self.hits += 1
            return entry

    def put(self, entry: CachedSystem) -> None:
        with self._lock:
            if entry.fingerprint in self._data:
                self._data.move_to_end(entry.fingerprint)
                self._data[entry.fingerprint] = entry
                return
            self._data[entry.fingerprint] = entry
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)
                self.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def mark_poisoned(self) -> None:
        """Record a fingerprint collision or structural mismatch.

        A poisoned cache is cleared so no later solve can publish values from
        a matrix that was assembled for a different region.
        """
        with self._lock:
            self.poisoned += 1
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            return {
                "entries": len(self._data),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "poisoned": self.poisoned,
                "hit_rate": (self.hits / (self.hits + self.misses)
                             if (self.hits + self.misses) else 0.0),
            }


DEFAULT_CACHE = FactorizationCache()
