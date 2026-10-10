"""Policy features and the small learned 'reference-risk' score.

The safety gate needs one function of Jev's output that works both on a live `JevDecision` and on stored
benchmark rows, so both are turned into the same flat feature dict here.

The risk score is a plain logistic regression over those features, fit on the TRAIN split only to predict
"the rule-based reference says a human is needed" (urgent/emergency, infection concern, same-day review or
human review). At run time it is just a dot product with stored coefficients (no scikit-learn needed).
Research-demo model: synthetic data, not clinically validated.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from backend.schemas.decision import JevDecision

NUMERIC_FEATURES = [
    "urgent_tail",          # P(URGENT) + P(EMERGENCY)
    "urgency_score",        # Jev Score, 0..3
    "urgency_confidence",
    "specialty_confidence",
    "specialty_top_probability",
    "p_infection",
    "p_same_day",
    "p_human_review",
    "complexity_score",
    "referral_score",
]


def features_from_decision(d: JevDecision) -> dict[str, Any]:
    up = d.urgency.probabilities
    return {
        "urgent_tail": up.get("URGENT", 0.0) + up.get("EMERGENCY", 0.0),
        "urgency_score": d.urgency.score,
        "urgency_confidence": d.urgency.confidence,
        "specialty_confidence": d.specialty.confidence,
        "specialty_top_probability": d.specialty.top_probability,
        "p_infection": d.infection_concern.probability,
        "p_same_day": d.same_day_review.probability,
        "p_human_review": d.human_review.probability,
        "complexity_score": d.complexity.score,
        "referral_score": d.referral_priority.score,
        "specialty": d.specialty.label,
    }


def features_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Same features from a stored Jev benchmark row (see benchmark_service.jev_row)."""
    p, c, s = row["probabilities"], row["confidences"], row["scores"]
    up = p["urgency"]
    return {
        "urgent_tail": up.get("URGENT", 0.0) + up.get("EMERGENCY", 0.0),
        "urgency_score": s["urgency"],
        "urgency_confidence": c["urgency"],
        "specialty_confidence": c["specialty"],
        "specialty_top_probability": max(p["specialty"].values()),
        "p_infection": p["infection_concern"],
        "p_same_day": p["same_day_review"],
        "p_human_review": p["human_review"],
        "complexity_score": s["complexity"],
        "referral_score": s["referral_priority"],
        "specialty": row["predictions"]["specialty"],
    }


def vectorize(feats: dict[str, Any], names: list[str] | None = None) -> np.ndarray:
    return np.array([float(feats[n]) for n in (names or NUMERIC_FEATURES)])


def fit(rows_feats: list[dict[str, Any]], y: list[bool], c: float = 0.5) -> dict[str, Any]:
    """Fit on TRAIN rows only. Balanced class weights (missing a risky case costs more than a false alarm)."""
    from sklearn.linear_model import LogisticRegression

    X = np.array([vectorize(f) for f in rows_feats])
    y_arr = np.array(y, dtype=int)
    if len(set(y_arr.tolist())) < 2:
        raise ValueError("Risk model needs both risky and non-risky training cases.")
    mean, scale = X.mean(axis=0), X.std(axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=1000).fit((X - mean) / scale, y_arr)
    return {
        "features": list(NUMERIC_FEATURES), "mean": [float(x) for x in mean], "scale": [float(x) for x in scale],
        "coef": [float(x) for x in clf.coef_[0]], "intercept": float(clf.intercept_[0]), "regularization_c": c,
        "n_train": int(len(y_arr)), "n_positive": int(y_arr.sum()),
    }


def score(model: dict[str, Any] | None, feats: dict[str, Any]) -> float | None:
    """P(reference says a human is needed), or None when no model is configured."""
    if not model:
        return None
    z = model["intercept"]
    for name, mu, sd, w in zip(model["features"], model["mean"], model["scale"], model["coef"]):
        z += w * (float(feats[name]) - mu) / sd
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))
