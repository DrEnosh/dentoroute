from __future__ import annotations

from fastapi import APIRouter

from backend.repositories import db
from backend.services.synthetic_case_service import load_dataset
from backend.utils.config import DEMO_BANNER, SAFETY_BANNER, TAGLINE, get_settings

router = APIRouter(tags=["system"])


@router.get("/")
def root() -> dict:
    s = get_settings()
    demo = s.resolve_demo_mode()
    return {
        "name": "DentoRoute", "tagline": TAGLINE, "powered_by": "TypeSafe Jev", "version": "1.0.0",
        "safety_notice": SAFETY_BANNER, "mode": "demo" if demo else "live",
        "banner": DEMO_BANNER if demo else None, "docs": "/docs",
        "endpoints": [
            "GET /health", "POST /api/cases/generate", "POST /api/cases/evaluate", "POST /api/jev/evaluate",
            "POST /api/claude/explain", "POST /api/benchmark/run", "GET /api/benchmark/results",
            "GET /api/cases/{case_id}", "GET /api/history",
        ],
    }


@router.get("/health")
def health() -> dict:
    s = get_settings()
    try:
        db.get_engine()
        db_ok = True
    except Exception:  # noqa: BLE001
        db_ok = False
    try:
        n_cases = len(load_dataset())
    except Exception:  # noqa: BLE001
        n_cases = 0
    demo = s.resolve_demo_mode()
    return {
        "status": "ok" if db_ok else "degraded", "mode": "demo" if demo else "live",
        "jev_key_configured": s.has_jev_key, "claude_key_configured": s.has_claude_key,
        "jev_model": s.jev_model, "claude_model": s.claude_model, "dataset_cases": n_cases, "database": "ok" if db_ok else "unavailable",
        "safety_notice": SAFETY_BANNER,
    }
