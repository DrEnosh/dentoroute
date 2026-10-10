from __future__ import annotations

from fastapi import APIRouter, Query

from backend.repositories import case_repo

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history")
def history(
    urgency: str | None = None, specialty: str | None = None, min_confidence: float | None = Query(None, ge=0, le=1),
    max_confidence: float | None = Query(None, ge=0, le=1), search: str | None = None, limit: int = Query(200, ge=1, le=1000),
) -> dict:
    runs = case_repo.list_runs(urgency=urgency, specialty=specialty, min_confidence=min_confidence,
                               max_confidence=max_confidence, search=search, limit=limit)
    return {"count": len(runs), "runs": runs}
