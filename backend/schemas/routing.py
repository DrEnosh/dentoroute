from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RouteType = Literal[
    "escalate_to_human", "urgent_review", "automated_specialty_route", "human_review"
]
ConfidenceBand = Literal["high", "medium", "low"]


class RoutingResult(BaseModel):
    route: RouteType
    workflow_label: str = Field(description="e.g. 'ENDODONTIC REVIEW'")
    destination_specialty: str | None = None
    triggered_rule: str
    reasons: list[str]
    confidence_band: ConfidenceBand
    requires_human: bool
    policy_name: str = "conservative"
    thresholds: dict = Field(default_factory=dict, description="Snapshot of config/routing_policy.yaml")
    disclaimer: str = "Research-demo thresholds, not clinical thresholds."


class ExplanationResult(BaseModel):
    source: Literal["claude", "template"]
    model: str | None = None
    summary: str
    engine_findings: str
    why_route: str
    urgency_signals: list[str] = Field(default_factory=list)
    human_review_note: str
    uncertainty: str
    disclaimer: str
    grounding_ok: bool = True
    grounding_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    latency_ms: float | None = None
