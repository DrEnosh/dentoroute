"""SIMULATED Jev responses for Demo Mode (no API keys needed).

*** Nothing produced here is real Jev output. ***
It is a small deterministic heuristic model with seeded noise, emitting the same raw JSON shape as the
documented TypeSafe System One API so the whole app (parsing, routing, UI) can be exercised offline.
Results derived from it must never be presented as Jev performance, and every result carries mode="demo".

It deliberately does not look at reference labels or at the rubric file.
"""
from __future__ import annotations

import math
from typing import Any

from backend.schemas.case import CaseState
from backend.schemas.labels import SPECIALTIES
from backend.utils.confidence import choice_confidence, score_confidence
from backend.utils.hashing import seeded_unit_floats, stable_hash

SIM_MODEL = "demo-simulator (NOT Jev)"


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _round_probs(values: list[float], digits: int) -> list[float]:
    total = sum(values)
    vals = [round(v / total, digits) for v in values]
    drift = round(1.0 - sum(vals), digits)
    vals[max(range(len(vals)), key=lambda i: vals[i])] = round(vals[max(range(len(vals)), key=lambda i: vals[i])] + drift, digits)
    return vals


def _score_answer(mu: float, sigma: float, levels: int, legend: list[str]) -> dict[str, Any]:
    mu = max(0.0, min(levels - 1.0, mu))
    raw = [math.exp(-((i - mu) ** 2) / (2 * sigma**2)) for i in range(levels)]
    probs = _round_probs(raw, 2)
    score = round(sum(i * p for i, p in enumerate(probs)), 2)
    return {
        "type": "score", "score": score, "confidence": round(score_confidence(probs), 2),
        "legend": {str(i): legend[i] for i in range(levels)},
        "probabilities": {str(i): p for i, p in enumerate(probs)},
    }


def _choice_answer(scores: dict[str, float], temperature: float) -> dict[str, Any]:
    exps = {k: math.exp(v / temperature) for k, v in scores.items()}
    names = list(exps)
    probs = _round_probs([exps[k] for k in names], 3)
    best = names[max(range(len(names)), key=lambda i: probs[i])]
    return {
        "type": "choice", "choice": best, "confidence": round(choice_confidence(probs), 2),
        "probabilities": dict(zip(names, probs)),
    }


def simulate(case: CaseState) -> dict[str, Any]:
    """Return a raw System-One-shaped response for `case` (deterministic per case)."""
    from backend.services.jev_service import QUESTION_SPECS  # lazy: avoids a circular import

    s = case
    h = stable_hash(s.to_jev_state())
    u = seeded_unit_floats(h, 12)
    sparse = s.info_completeness == "sparse"
    partial = s.info_completeness == "partial"
    vague = 2.4 if sparse else 1.5 if partial else 1.0  # uncertainty multiplier

    pain = s.pain_severity
    biting = "biting" in s.pain_trigger
    infect_signal = s.swelling and (s.fever or s.progressive_symptoms or s.lymphadenopathy or s.trismus)
    airway = s.difficulty_breathing or (s.difficulty_swallowing and s.swelling)

    # ---- urgency (0..3) --------------------------------------------------
    mu = (
        0.11 * pain + 0.8 * s.swelling + 0.7 * (s.fever and pain >= 2) + 0.55 * s.progressive_symptoms
        + 0.6 * s.trismus + 1.6 * s.tooth_avulsion + 0.5 * s.trauma + 0.45 * s.tooth_fracture
        + 0.4 * s.pus_or_drainage + 1.5 * s.difficulty_swallowing + 2.2 * s.difficulty_breathing
        + 0.45 * (s.lesion_present and (s.lesion_duration_weeks or 0) >= 3)
        + 0.5 * (s.recent_extraction_or_procedure and pain >= 4) + 0.3 * s.jaw_locking
    )
    mu = min(3.0, mu * 0.78) + (u[0] - 0.5) * 0.9 * (1 if u[1] < 0.35 else 0.25)
    sigma = 0.28 * vague + (0.25 if 1.2 < mu % 1 < 1.8 else 0)
    urgency = _score_answer(mu, sigma, 4, QUESTION_SPECS["urgency"]["criteria"])

    # ---- specialty ---------------------------------------------------------
    sp = {k: 0.0 for k in SPECIALTIES}
    sp["general_dentistry"] = 1.2 + (1.5 if (s.administrative_request or pain == 0) else 0)
    sp["emergency_dental_service"] = 4.2 * airway + 3.6 * (s.swelling and s.fever and s.trismus) + 3.8 * s.tooth_avulsion
    sp["pediatric_dentistry"] = 4.2 * (s.age < 13)
    sp["oral_pathology"] = 3.6 * (s.lesion_present and (s.lesion_duration_weeks or 0) >= 6)
    sp["oral_medicine"] = 3.1 * s.lesion_present + 2.8 * (s.jaw_clicking or s.jaw_locking)
    sp["oral_surgery"] = 3.2 * s.third_molar_symptoms + 2.3 * (s.swelling and not s.gingival_involvement and (s.fever or s.progressive_symptoms)) + 1.8 * (s.recent_extraction_or_procedure and pain >= 5)
    sp["periodontics"] = 3.2 * s.gingival_involvement
    sp["orthodontics"] = 3.8 * s.orthodontic_appliance_issue
    sp["prosthodontics"] = 3.8 * s.restoration_or_prosthesis_issue
    sp["endodontics"] = (
        2.7 * (pain >= 5 and s.pain_character in ("throbbing", "spontaneous", "lingering"))
        + 1.6 * (biting and pain >= 4) + 1.2 * (s.previous_root_canal and pain >= 3)
        + 1.3 * (s.tooth_fracture and pain >= 5)
    )
    for i, k in enumerate(SPECIALTIES):  # seeded per-label jitter
        sp[k] += (u[2 + (i % 8)] - 0.5) * 0.9
    specialty = _choice_answer(sp, 0.55 * vague)

    # ---- booleans (probability of YES) ------------------------------------
    def noul(logit: float, jitter: float, pull: float = 0.0) -> float:
        p = _sigmoid(logit + jitter) * (1 - pull) + 0.5 * pull
        return round(min(0.99, max(0.01, p)), 2)

    infection = noul(
        -3.2 + 3.6 * infect_signal + 2.6 * s.pus_or_drainage + 2.6 * s.difficulty_swallowing
        + 1.0 * (s.swelling and pain >= 6 and (s.previous_root_canal or biting)) + 1.0 * (s.fever and pain >= 4 and bool(s.tooth_region)),
        (u[10] - 0.5) * 1.6, pull=0.4 if sparse else 0.15 if partial else 0.0,
    )
    same_day = noul(
        -2.6 + 1.5 * (mu >= 1.6) + 1.4 * (mu >= 2.2) + 2.6 * s.tooth_avulsion + 1.8 * (infection >= 0.5)
        + 1.2 * (s.bleeding and s.recent_extraction_or_procedure and pain >= 5),
        (u[11] - 0.5) * 1.6, pull=0.4 if sparse else 0.15 if partial else 0.0,
    )
    human = noul(
        -2.3 + 3.2 * (mu >= 2.6) + 2.4 * sparse + 1.3 * (partial and mu >= 0.8)
        + 2.2 * (s.lesion_present and (s.lesion_duration_weeks or 0) >= 3) + 1.8 * (infection >= 0.5 and s.fever)
        + 1.2 * (s.age < 13 and s.trauma) + 1.0 * bool(
            any(k in " ".join(s.medical_history + s.medications).lower() for k in ("anticoag", "warfarin", "immunocomp", "uncontrolled"))
            and (infection >= 0.5 or s.bleeding)
        ),
        (u[0] - 0.5) * 1.4,
        pull=0.0,
    )
    if sparse:  # cap: ambiguous cases should read as 'unsure', not as a confident yes
        human = round(min(human, 0.66), 2)

    # ---- complexity / referral priority ------------------------------------
    n_risk = len(s.medical_history) + (len(s.medications) >= 3) + s.swelling + s.trismus + s.trauma + s.lesion_present + (s.age < 13) + (s.age >= 75) + (infection >= 0.5)
    complexity = _score_answer(0.45 * n_risk + 0.9 * sparse + 1.5 * s.difficulty_breathing + (u[3] - 0.5) * 0.4, 0.45 * vague, 4, QUESTION_SPECS["complexity"]["criteria"])
    referral = _score_answer(0.85 * mu + 0.3 * s.lesion_present + 0.25 * (n_risk >= 3) + (u[4] - 0.5) * 0.3, 0.4 * vague, 4, QUESTION_SPECS["referral_priority"]["criteria"])

    return {
        "model": SIM_MODEL,
        "simulated": True,
        "answers": {
            "urgency": urgency,
            "specialty": specialty,
            "infection_concern": {"type": "noul", "noul": infection},
            "same_day_review": {"type": "noul", "noul": same_day},
            "human_review": {"type": "noul", "noul": human},
            "complexity": complexity,
            "referral_priority": referral,
        },
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }
