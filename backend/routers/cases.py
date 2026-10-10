from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from backend.repositories import case_repo
from backend.schemas.api import EvaluateCaseRequest, EvaluateCaseResponse, GenerateCaseRequest
from backend.services import export_service, pipeline_service
from backend.services import synthetic_case_service as scs
from backend.utils.config import SAFETY_BANNER
from backend.schemas.labels import CATEGORIES

router = APIRouter(prefix="/api/cases", tags=["cases"])


def error_status(code: str | None) -> int:
    return 503 if code in ("missing_api_key", "database_error") else 502


@router.post("/generate")
def generate(req: GenerateCaseRequest) -> dict:
    if req.category and req.category not in CATEGORIES:
        raise HTTPException(422, f"Unknown category '{req.category}'. Valid: {', '.join(CATEGORIES)}")
    cases = [scs.random_case(req.category, None if req.seed is None else req.seed + i, req.from_dataset) for i in range(req.count)]
    return {"count": len(cases), "cases": [c.model_dump() for c in cases], "safety_notice": SAFETY_BANNER,
            "note": "Synthetic cases only. Reference labels come from an educational rule set, not clinicians."}


@router.get("/demo")
def demo_cases() -> dict:
    return {"cases": [c.model_dump() for c in scs.load_demo_cases()]}


@router.get("/categories")
def categories() -> dict:
    return {"categories": scs.category_counts(scs.load_dataset())}


@router.post("/evaluate", response_model=EvaluateCaseResponse)
def evaluate(req: EvaluateCaseRequest):
    if req.case is None and not req.case_id:
        raise HTTPException(422, "Provide either 'case' (structured state) or 'case_id'.")
    kw = dict(demo=req.demo, explain=req.explain, persist=req.persist)
    res = pipeline_service.evaluate_by_id(req.case_id, **kw) if req.case_id else pipeline_service.evaluate_case(req.case, **kw)
    if res.error:  # same structured body, honest status code
        return JSONResponse(status_code=error_status(res.error_code), content=res.model_dump(mode="json"))
    return res


@router.get("/{case_id}")
def get_case(case_id: str) -> dict:
    return pipeline_service.case_detail(case_id)


@router.get("/{case_id}/report")
def case_report(case_id: str, format: str = Query("md", pattern="^(md|json)$")):
    run = case_repo.latest_run_for_case(case_id)
    if run is None:
        raise HTTPException(404, f"No stored evaluation for '{case_id}'. Evaluate the case first.")
    detail = pipeline_service.case_detail(case_id)
    payload = {"patient_id": case_id, "mode": run["mode"], "decision": run["decision"], "routing": run["routing"],
               "explanation": run["explanation"], "reference": detail["reference"], "error": run["error"]}
    if format == "json":
        return PlainTextResponse(export_service.case_report_json(payload), media_type="application/json")
    return PlainTextResponse(export_service.case_report_markdown(payload), media_type="text/markdown")
