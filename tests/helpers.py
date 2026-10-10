"""Test helpers: build JevDecision objects with exact probabilities (no Jev needed)."""
from __future__ import annotations

from backend.schemas.decision import ChoiceDecision, JevDecision, NoulDecision, ScoreDecision
from backend.schemas.labels import COMPLEXITY_LEVELS, REFERRAL_LEVELS, SPECIALTIES, URGENCY_LEVELS
from backend.utils.confidence import choice_confidence, score_confidence


def score(levels: list[str], probs: list[float]) -> ScoreDecision:
    sc = sum(i * p for i, p in enumerate(probs))
    top = max(range(len(probs)), key=lambda i: probs[i])
    return ScoreDecision(
        label=levels[top], score=sc, rounded_label=levels[min(len(levels) - 1, int(round(sc)))],
        confidence=score_confidence(probs), probabilities=dict(zip(levels, probs)),
    )


def choice(label: str, top_p: float) -> ChoiceDecision:
    rest = (1 - top_p) / (len(SPECIALTIES) - 1)
    probs = {s: (top_p if s == label else rest) for s in SPECIALTIES}
    return ChoiceDecision(label=label, confidence=choice_confidence(list(probs.values())), probabilities=probs)


def make_decision(
    *, urgency=(0.05, 0.85, 0.08, 0.02), specialty=("endodontics", 0.9), infection=0.1, same_day=0.1,
    human=0.1, complexity=(0.1, 0.8, 0.1, 0.0), referral=(0.1, 0.8, 0.1, 0.0), mode="demo",
) -> JevDecision:
    return JevDecision(
        mode=mode, model="test-model", urgency=score(URGENCY_LEVELS, list(urgency)),
        specialty=choice(*specialty), infection_concern=NoulDecision(probability=infection),
        same_day_review=NoulDecision(probability=same_day), human_review=NoulDecision(probability=human),
        complexity=score(COMPLEXITY_LEVELS, list(complexity)), referral_priority=score(REFERRAL_LEVELS, list(referral)),
    )


# ---------------------------------------------------------------- policy tests --
def benign(**over):
    """A decision that passes every gate of the conservative policy (override one field to trip one rule)."""
    base = dict(
        urgency=(0.97, 0.03, 0.0, 0.0), specialty=("general_dentistry", 0.97), infection=0.02, same_day=0.02, human=0.02,
        complexity=(0.9, 0.1, 0.0, 0.0), referral=(0.9, 0.1, 0.0, 0.0),
    )
    base.update(over)
    return make_decision(**base)


def row_from_decision(d, ref: dict, case_id: str, split: str) -> dict:
    """A benchmark row (benchmark_service.jev_row shape) built from a decision, for policy-study tests."""
    return {
        "case_id": case_id, "split": split, "category": "x", "provider": "jev", "mode": "demo", "reference": ref,
        "predictions": {"urgency": d.urgency.label, "specialty": d.specialty.label,
                        "infection_concern": d.infection_concern.prediction, "same_day_review": d.same_day_review.prediction,
                        "human_review": d.human_review.prediction, "complexity": d.complexity.label,
                        "referral_priority": d.referral_priority.label},
        "probabilities": {"urgency": d.urgency.probabilities, "specialty": d.specialty.probabilities,
                          "complexity": d.complexity.probabilities, "referral_priority": d.referral_priority.probabilities,
                          "infection_concern": d.infection_concern.probability, "same_day_review": d.same_day_review.probability,
                          "human_review": d.human_review.probability},
        "confidences": {"urgency": d.urgency.confidence, "specialty": d.specialty.confidence,
                        "complexity": d.complexity.confidence, "referral_priority": d.referral_priority.confidence},
        "scores": {"urgency": d.urgency.score, "complexity": d.complexity.score, "referral_priority": d.referral_priority.score},
    }


def reference(urgency="ROUTINE", specialty="general_dentistry", infection=False, same_day=False, human=False,
              complexity="SIMPLE", referral="LOW") -> dict:
    return dict(urgency=urgency, specialty=specialty, infection_concern=infection, same_day_review=same_day,
                human_review=human, complexity=complexity, referral_priority=referral)
