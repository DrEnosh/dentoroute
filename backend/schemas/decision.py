"""Normalized decision objects (what the rest of the app consumes from Jev / demo / Claude)."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ScoreDecision(BaseModel):
    label: str = Field(description="Most likely level (argmax of probabilities)")
    score: float = Field(description="Probability-weighted position, 0..n-1 (Jev 'score')")
    rounded_label: str = Field(description="Level nearest to `score` (docs' suggested rounding)")
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]

    @property
    def top_probability(self) -> float:
        return max(self.probabilities.values())


class ChoiceDecision(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]

    @property
    def top_probability(self) -> float:
        return max(self.probabilities.values())


class NoulDecision(BaseModel):
    probability: float = Field(ge=0, le=1, description="Probability the answer is YES")

    @property
    def prediction(self) -> bool:
        return self.probability >= 0.5

    @property
    def certainty(self) -> float:
        """|2p-1| — the docs' optional confidence-style number for a Noul."""
        return abs(2 * self.probability - 1)


class JevDecision(BaseModel):
    source: Literal["jev", "claude"] = "jev"
    mode: Literal["live", "demo"] = "live"
    model: str
    urgency: ScoreDecision
    specialty: ChoiceDecision
    infection_concern: NoulDecision
    same_day_review: NoulDecision
    human_review: NoulDecision
    complexity: ScoreDecision
    referral_priority: ScoreDecision

    latency_ms: float | None = None
    usage: dict[str, int] | None = None
    request_id: str | None = None
    cached: bool = False
    raw_response: dict[str, Any] | None = None

    def summary_row(self) -> dict[str, Any]:
        return {
            "urgency": self.urgency.label,
            "specialty": self.specialty.label,
            "infection_probability": self.infection_concern.probability,
            "same_day_probability": self.same_day_review.probability,
            "human_review_probability": self.human_review.probability,
            "complexity": self.complexity.label,
            "referral_priority": self.referral_priority.label,
        }


class ClaudeDecision(BaseModel):
    """Claude's structured answers to the same questions (benchmark baseline).

    Claude does not return calibrated distributions, so the numbers below are
    *verbalized* confidences (what the model says), clearly different in nature
    from Jev's probabilities.
    """

    model: str
    urgency: str
    urgency_confidence: float | None = Field(default=None, ge=0, le=1)
    specialty: str
    specialty_confidence: float | None = Field(default=None, ge=0, le=1)
    infection_concern: bool
    infection_probability: float | None = Field(default=None, ge=0, le=1)
    same_day_review: bool
    same_day_probability: float | None = Field(default=None, ge=0, le=1)
    human_review: bool
    human_review_probability: float | None = Field(default=None, ge=0, le=1)
    complexity: str
    referral_priority: str
    latency_ms: float | None = None
    usage: dict[str, int] | None = None
    raw_response: dict[str, Any] | None = None
