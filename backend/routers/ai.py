from __future__ import annotations

from fastapi import APIRouter

from backend.schemas.api import ClaudeExplainRequest, JevEvaluateRequest
from backend.schemas.routing import ExplanationResult
from backend.services import jev_service, pipeline_service, routing_service
from backend.utils.config import get_settings

router = APIRouter(prefix="/api", tags=["ai"])


@router.post("/jev/evaluate")
def jev_evaluate(req: JevEvaluateRequest) -> dict:
    """Raw decision layer: the 7 Jev answers plus the deterministic routing result. Provider failures -> 502/503."""
    decision = jev_service.get_jev_service().evaluate(req.case, demo=req.demo)
    routing = routing_service.route_decision(decision)
    return {"decision": decision.model_dump(), "routing": routing.model_dump(), "mode": decision.mode}


@router.post("/claude/explain", response_model=ExplanationResult)
def claude_explain(req: ClaudeExplainRequest) -> ExplanationResult:
    """Explanation layer. Degrades to a deterministic template (with a warning) if Claude is unavailable."""
    demo = get_settings().resolve_demo_mode() and req.decision.mode == "demo"
    return pipeline_service._explain(req.case, req.decision, req.routing, demo=demo)
