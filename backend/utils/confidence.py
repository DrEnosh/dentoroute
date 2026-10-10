"""Confidence formulas, taken from the official TypeSafe 'Confidence' documentation.

Jev returns `confidence` itself for Choice and Score answers. These helpers exist so that
(a) the demo simulator emits internally consistent numbers and (b) tests can sanity-check
the formulas. They are NOT used to override anything Jev returns.
"""
from __future__ import annotations

from typing import Sequence


def choice_confidence(probabilities: Sequence[float]) -> float:
    """(p_max - 1/n) / (1 - 1/n)"""
    n = len(probabilities)
    if n < 2:
        return 1.0
    return max(0.0, min(1.0, (max(probabilities) - 1 / n) / (1 - 1 / n)))


def score_confidence(probabilities: Sequence[float]) -> float:
    """1 - E|i - mode| / MAD_uniform, floored at 0 (ordered levels)."""
    n = len(probabilities)
    if n < 2:
        return 1.0
    m = max(range(n), key=lambda i: probabilities[i])
    spread = sum(p * abs(i - m) for i, p in enumerate(probabilities))
    even_spread = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, min(1.0, 1 - spread / even_spread))


def noul_certainty(p: float) -> float:
    """|2p - 1| - the docs' optional confidence-style number for a Noul."""
    return abs(2 * p - 1)


def confidence_band(confidence: float, high: float = 0.8, medium: float = 0.5) -> str:
    if confidence >= high:
        return "high"
    if confidence >= medium:
        return "medium"
    return "low"
