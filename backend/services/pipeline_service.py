"""End-to-end pipeline: case -> Jev decisions -> routing policy -> explanation -> persistence.

Never raises for AI/provider failures: they come back as `EvaluateCaseResponse.error` with a readable
message, and the failed run is still logged so Case History shows what happened.
"""
from __future__ import annotations

from typing import Any

from backend.repositories import case_repo
from backend.schemas.api import EvaluateCaseResponse
from backend.schemas.case import CaseState, SyntheticCase
from backend.services import claude_service, ground_truth_service, jev_service, routing_service
from backend.services.synthetic_case_service import get_case
from backend.utils.config import get_settings
from backend.utils.errors import DentoRouteError, RepositoryError
from backend.utils.logging_utils import get_logger, log_event

logger = get_logger("pipeline")


def _explain(case: CaseState, decision, routing, demo: bool):
    """Claude explanation when live and a key exists; otherwise (or on any failure) the template."""
    svc = claude_service.get_claude_service()
    if demo or not svc.available:
        why = "Demo Mode" if demo else "no ANTHROPIC_API_KEY configured"
        return claude_service.template_explanation(
            case, decision, routing, warnings=[f"Template explanation used ({why}); Claude was not called."]
        )
    try:
        return svc.explain(case, decision, routing)
    except DentoRouteError as exc:
        log_event(logger, "explain_fallback", case_id=case.patient_id, success=False, validation=exc.__class__.__name__)
        return claude_service.template_explanation(
            case, decision, routing, warnings=[f"Claude explanation unavailable ({exc.user_message}); template shown instead."]
        )


def evaluate_case(
    case: CaseState, *, demo: bool | None = None, explain: bool = True, persist: bool = True,
    reference: dict[str, Any] | None = None, use_cache: bool = True,
) -> EvaluateCaseResponse:
    settings = get_settings()
    is_demo = settings.resolve_demo_mode(demo)
    mode = "demo" if is_demo else "live"
    decision = routing = explanation = None
    error: str | None = None
    error_code: str | None = None
    try:
        decision = jev_service.get_jev_service().evaluate(case, demo=is_demo, use_cache=use_cache)
        routing = routing_service.route_decision(decision)
        if explain:
            explanation = _explain(case, decision, routing, is_demo)
    except DentoRouteError as exc:
        error, error_code = exc.user_message, exc.code
        log_event(logger, "pipeline_error", case_id=case.patient_id, success=False, validation=exc.__class__.__name__)
    except Exception as exc:  # last-resort guard: the UI must never crash on an AI/provider failure
        error, error_code = "Something unexpected went wrong while evaluating this case. Check the backend logs.", "unexpected_error"
        logger.exception("unexpected pipeline failure: %s", exc.__class__.__name__)

    if routing is None:  # R0 fail-safe: an invalid or missing model output is never routed automatically
        try:
            routing = routing_service.fail_safe_routing(error or "No decision was produced")
        except DentoRouteError:  # a broken policy file must not hide the original error
            routing = None

    run_id = None
    if persist:
        try:
            run_id = case_repo.save_run(
                patient_id=case.patient_id, mode=mode, model=decision.model if decision else settings.jev_model,
                state=case.model_dump(),
                decision=decision.model_dump() if decision else None,
                routing=routing.model_dump() if routing else None,
                explanation=explanation.model_dump() if explanation else None, error=error,
            )
        except RepositoryError as exc:
            logger.warning("could not persist run: %s", exc.user_message)
    return EvaluateCaseResponse(
        patient_id=case.patient_id, mode=mode, decision=decision, routing=routing, explanation=explanation,
        reference=reference, error=error, error_code=error_code, run_id=run_id,
    )


def evaluate_synthetic(synthetic: SyntheticCase, **kw: Any) -> EvaluateCaseResponse:
    ref = synthetic.reference.model_dump() if synthetic.reference else None
    return evaluate_case(synthetic.state, reference=ref, **kw)


def evaluate_by_id(case_id: str, **kw: Any) -> EvaluateCaseResponse:
    return evaluate_synthetic(get_case(case_id), **kw)


def case_detail(case_id: str) -> dict[str, Any]:
    """A case, its rule-based reference labels and its latest stored run (if any)."""
    c = get_case(case_id)
    ref = c.reference or ground_truth_service.label_case(c.state)
    return {
        "patient_id": c.patient_id, "category": c.category, "state": c.state.model_dump(),
        "reference": ref.model_dump(), "latest_run": case_repo.latest_run_for_case(case_id),
    }
