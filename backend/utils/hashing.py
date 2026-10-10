from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def seeded_unit_floats(key: str, n: int) -> list[float]:
    """Deterministic floats in [0,1) derived from a string key (used by the demo simulator)."""
    out: list[float] = []
    counter = 0
    while len(out) < n:
        digest = hashlib.sha256(f"{key}:{counter}".encode()).digest()
        for i in range(0, len(digest) - 3, 4):
            out.append(int.from_bytes(digest[i : i + 4], "big") / 2**32)
            if len(out) >= n:
                break
        counter += 1
    return out
