"""Request / response models for the FastAPI layer."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.schemas.case import CaseState
from backend.schemas.decision import JevDecision
from backend.schemas.routing import ExplanationResult, RoutingResult


class GenerateCaseRequest(BaseModel):
    category: str | None = None
    seed: int | None = None
    count: int = Field(default=1, ge=1, le=50)
    from_dataset: bool = Field(
        default=True, description="Pick from the saved dataset (True) or generate a fresh case (False)"
    )


class EvaluateCaseRequest(BaseModel):
    case: CaseState | None = None
    case_id: str | None = None
    demo: bool | None = Field(default=None, description="Override demo mode for this request")
    explain: bool = True
    persist: bool = True


class JevEvaluateRequest(BaseModel):
    case: CaseState
    demo: bool | None = None


class ClaudeExplainRequest(BaseModel):
    case: CaseState
    decision: JevDecision
    routing: RoutingResult


class BenchmarkRunRequest(BaseModel):
    provider: Literal["jev", "claude", "both"] = "jev"
    limit: int | None = Field(default=None, ge=1)
    demo: bool | None = None
    workers: int | None = Field(default=None, ge=1, le=32)
    seed: int = 0
    include_rows: bool = Field(default=False, description="Include per-case rows in the response (large)")


class EvaluateCaseResponse(BaseModel):
    patient_id: str
    mode: Literal["live", "demo"]
    decision: JevDecision | None = None
    routing: RoutingResult | None = None
    explanation: ExplanationResult | None = None
    reference: dict[str, Any] | None = None
    error: str | None = None
    error_code: str | None = None
    run_id: int | None = None
    safety_notice: str = "Research prototype. Not for clinical diagnosis or treatment."
