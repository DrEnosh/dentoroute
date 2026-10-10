from __future__ import annotations

from fastapi import APIRouter

from backend.services import policy_eval_service
from backend.services.policy_service import MAX_LIMITS, MIN_LIMITS, validate_policy
from backend.utils.config import load_routing_policy

router = APIRouter(prefix="/api/policy", tags=["policy"])


@router.get("")
def active_policy() -> dict:
    """The routing policy in force, its hard safety limits and where its thresholds came from."""
    p = validate_policy(load_routing_policy())
    return {
        "name": p.get("name"), "version": p.get("version"), "rules": p["rules"], "thresholds": p["thresholds"],
        "human_only_specialties": p.get("human_only_specialties", []), "risk_model_configured": bool(p.get("risk_model")),
        "provenance": p.get("provenance"), "limits": {"max": MAX_LIMITS, "min": MIN_LIMITS},
        "disclaimer": p.get("disclaimer"),
    }


@router.post("/study")
def run_study() -> dict:
    """Train/calibration/holdout study on the latest stored Jev benchmark. Evaluates a candidate; changes nothing."""
    return policy_eval_service.public_study(policy_eval_service.run_and_store_study())


@router.get("/study")
def latest_study() -> dict:
    return {"study": policy_eval_service.latest_study()}
