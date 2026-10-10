"""Calibration analysis: do the probabilities mean what they say, and does confidence track accuracy?

Pure functions over benchmark rows. All "confidence" values are the provider's own numbers:
 - Jev: probability distributions, the documented `confidence` (Score/Choice), and Noul probabilities.
 - Claude: *verbalized* confidences only (no distributions), so far fewer samples exist.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from backend.services.metrics_service import CATEGORICAL, QUESTIONS, is_correct, valid_rows
from backend.schemas.labels import BOOLEAN_QUESTIONS

BAND_EDGES = {"low": (0.0, 0.5), "medium": (0.5, 0.8), "high": (0.8, 1.0001)}
MIN_N_FOR_CLAIM = 30


# ------------------------------------------------------------------ samples -----
def top_class_samples(rows: list[dict[str, Any]], q: str) -> list[tuple[float, int]]:
    """(probability of the predicted answer, correct?) per case. Needs the provider's probabilities."""
    out: list[tuple[float, int]] = []
    for r in valid_rows(rows):
        probs = r.get("probabilities", {}).get(q)
        if probs is None and q not in BOOLEAN_QUESTIONS and r.get("confidence_kind") == "verbalized":
            c = r.get("confidences", {}).get(q)  # Claude: only a self-reported number, no distribution
            if c is not None:
                out.append((float(c), int(is_correct(r, q))))
            continue
        if probs is None:
            continue
        if q in BOOLEAN_QUESTIONS:
            p_yes = float(probs)
            p_top = max(p_yes, 1 - p_yes)
        else:
            p_top = float(probs[r["predictions"][q]])
        out.append((p_top, int(is_correct(r, q))))
    return out


def confidence_samples(rows: list[dict[str, Any]], q: str) -> list[tuple[float, int]]:
    """(provider's confidence number, correct?). Booleans use certainty |2p-1| (docs' Noul convention)."""
    out: list[tuple[float, int]] = []
    for r in valid_rows(rows):
        if q in BOOLEAN_QUESTIONS:
            p = r.get("probabilities", {}).get(q)
            c = abs(2 * p - 1) if p is not None else None
        else:
            c = r.get("confidences", {}).get(q)
        if c is not None:
            out.append((float(c), int(is_correct(r, q))))
    return out


# --------------------------------------------------------------- reliability ----
def reliability_bins(samples: list[tuple[float, int]], n_bins: int = 10, lo: float = 0.0) -> list[dict[str, Any]]:
    """Equal-width bins on [lo, 1]. Each bin: n, mean confidence, observed accuracy."""
    edges = np.linspace(lo, 1.0, n_bins + 1)
    bins = []
    for i in range(n_bins):
        left, right = edges[i], edges[i + 1]
        members = [(c, y) for c, y in samples if (left <= c < right) or (i == n_bins - 1 and c == right)]
        bins.append({
            "lo": float(left), "hi": float(right), "n": len(members),
            "mean_confidence": float(np.mean([c for c, _ in members])) if members else None,
            "accuracy": float(np.mean([y for _, y in members])) if members else None,
        })
    return bins


def expected_calibration_error(bins: list[dict[str, Any]]) -> float | None:
    n = sum(b["n"] for b in bins)
    if not n:
        return None
    return float(sum(b["n"] / n * abs(b["accuracy"] - b["mean_confidence"]) for b in bins if b["n"]))


def brier_score(rows: list[dict[str, Any]], q: str) -> float | None:
    """Multiclass Brier (sum over classes) for Score/Choice questions; binary Brier for Noul questions."""
    ok = [r for r in valid_rows(rows) if r.get("probabilities", {}).get(q) is not None]
    if not ok:
        return None
    vals = []
    for r in ok:
        p = r["probabilities"][q]
        if q in BOOLEAN_QUESTIONS:
            vals.append((float(p) - float(r["reference"][q])) ** 2)
        else:
            vals.append(sum((float(pk) - (1.0 if k == r["reference"][q] else 0.0)) ** 2 for k, pk in p.items()))
    return float(np.mean(vals))


# ----------------------------------------------------- confidence vs accuracy ---
def band_of(c: float, high: float = 0.8, medium: float = 0.5) -> str:
    return "high" if c >= high else "medium" if c >= medium else "low"


def accuracy_by_band(samples: list[tuple[float, int]], high: float = 0.8, medium: float = 0.5) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for band in ("low", "medium", "high"):
        members = [y for c, y in samples if band_of(c, high, medium) == band]
        out[band] = {"n": len(members), "accuracy": float(np.mean(members)) if members else None}
    return out


def confidence_accuracy_relationship(samples: list[tuple[float, int]], high: float = 0.8, medium: float = 0.5) -> dict[str, Any]:
    """Answer 'does higher confidence mean higher accuracy?' with numbers and an honest caveat."""
    n = len(samples)
    bands = accuracy_by_band(samples, high, medium)
    occupied = [bands[b]["accuracy"] for b in ("low", "medium", "high") if bands[b]["n"] > 0]
    monotonic = all(x <= y + 1e-9 for x, y in zip(occupied, occupied[1:])) if len(occupied) >= 2 else None
    rho = p_value = None
    conf = np.array([c for c, _ in samples], dtype=float)
    corr = np.array([y for _, y in samples], dtype=float)
    if n >= 3 and np.ptp(conf) > 1e-9 and np.ptp(corr) > 0:
        res = stats.spearmanr(conf, corr)
        rho, p_value = float(res.statistic), float(res.pvalue)
    if n < MIN_N_FOR_CLAIM:
        answer = f"Not enough data to say (n={n}, need at least {MIN_N_FOR_CLAIM})."
    elif rho is None:
        answer = "Cannot tell: confidence or correctness is constant in this sample."
    elif rho > 0.1 and p_value is not None and p_value < 0.05 and monotonic is not False:
        answer = f"Yes in this sample: accuracy rises with confidence (Spearman rho={rho:.2f}, p={p_value:.3g})."
    elif rho > 0.1 and p_value is not None and p_value < 0.05:
        answer = f"Partly: positive correlation (rho={rho:.2f}) but accuracy is not monotonic across bands."
    else:
        answer = f"No clear relationship in this sample (rho={rho:.2f}, p={p_value:.3g})."
    return {"n": n, "bands": bands, "monotonic": monotonic, "spearman_rho": rho, "p_value": p_value, "answer": answer}


# ---------------------------------------------------------------- histograms ----
def histogram(values: list[float], n_bins: int = 10) -> dict[str, Any]:
    counts, edges = np.histogram(values, bins=n_bins, range=(0.0, 1.0)) if values else (np.zeros(n_bins, int), np.linspace(0, 1, n_bins + 1))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def confidence_histograms(rows: list[dict[str, Any]], q: str) -> dict[str, Any]:
    samples = confidence_samples(rows, q)
    return {
        "all": histogram([c for c, _ in samples]),
        "correct": histogram([c for c, y in samples if y]),
        "incorrect": histogram([c for c, y in samples if not y]),
    }


def probability_histogram(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Every probability the provider emitted (all classes, all questions), to show how peaked its outputs are."""
    vals: list[float] = []
    for r in valid_rows(rows):
        for q, p in r.get("probabilities", {}).items():
            if isinstance(p, dict):
                vals.extend(float(v) for v in p.values())
            elif p is not None:
                vals.extend([float(p), 1 - float(p)])
    return histogram(vals)


# ------------------------------------------------------------ risk / coverage ---
def risk_coverage(rows: list[dict[str, Any]], thresholds: list[float] | None = None) -> list[dict[str, Any]]:
    """Confidence-gated routing curve. Gate = min(urgency confidence, specialty confidence) >= t.

    coverage = share of cases that would be automated; accuracy = share of those where BOTH urgency and
    specialty match the reference. Shows what you buy (accuracy) and pay (more human review) per threshold.
    """
    thresholds = thresholds if thresholds is not None else [round(t, 2) for t in np.linspace(0, 0.95, 20)]
    ok = [r for r in valid_rows(rows) if r.get("confidences", {}).get("urgency") is not None and r.get("confidences", {}).get("specialty") is not None]
    out = []
    for t in thresholds:
        sel = [r for r in ok if min(r["confidences"]["urgency"], r["confidences"]["specialty"]) >= t]
        acc = float(np.mean([is_correct(r, "urgency") and is_correct(r, "specialty") for r in sel])) if sel else None
        out.append({
            "threshold": float(t), "n_automated": len(sel), "coverage": len(sel) / len(ok) if ok else None,
            "joint_accuracy": acc, "escalated": len(ok) - len(sel),
        })
    return out


# ----------------------------------------------------------------- aggregate ----
def calibration_report(rows: list[dict[str, Any]], n_bins: int = 10, high: float = 0.8, medium: float = 0.5) -> dict[str, Any]:
    per_q: dict[str, Any] = {}
    for q in QUESTIONS:
        top = top_class_samples(rows, q)
        bins = reliability_bins(top, n_bins, lo=0.0)
        conf = confidence_samples(rows, q)
        per_q[q] = {
            "n_probability_samples": len(top), "reliability_bins": bins, "ece": expected_calibration_error(bins),
            "brier": brier_score(rows, q),
            "confidence": confidence_accuracy_relationship(conf, high, medium),
            "confidence_histograms": confidence_histograms(rows, q),
        }
    pooled = [s for q in CATEGORICAL for s in confidence_samples(rows, q)]
    pooled_top = [s for q in QUESTIONS for s in top_class_samples(rows, q)]
    pooled_bins = reliability_bins(pooled_top, n_bins)
    return {
        "per_question": per_q,
        "pooled": {
            "n": len(pooled_top), "reliability_bins": pooled_bins, "ece": expected_calibration_error(pooled_bins),
            "confidence": confidence_accuracy_relationship(pooled, high, medium),
        },
        "probability_histogram": probability_histogram(rows),
        "risk_coverage": risk_coverage(rows),
        "note": (
            "ECE uses the probability of the predicted answer. For Score/Choice the 'confidence' is the "
            "provider-documented measure; for Noul it is |2p-1|. Results describe THIS run on THIS synthetic "
            "rubric only."
        ),
    }
