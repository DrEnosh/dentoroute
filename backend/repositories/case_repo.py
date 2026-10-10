from __future__ import annotations

import json
from typing import Any

from backend.models.orm import CaseRun
from backend.repositories.db import session_scope


def save_run(
    *, patient_id: str, mode: str, model: str, state: dict, decision: dict | None,
    routing: dict | None, explanation: dict | None, error: str | None,
) -> int:
    min_conf = None
    urgency = specialty = None
    latency = None
    if decision:
        urgency = decision["urgency"]["label"]
        specialty = decision["specialty"]["label"]
        min_conf = min(decision["urgency"]["confidence"], decision["specialty"]["confidence"])
        latency = decision.get("latency_ms")
    with session_scope() as s:
        row = CaseRun(
            patient_id=patient_id, mode=mode, model=model, urgency=urgency, specialty=specialty,
            route=routing["route"] if routing else None, min_confidence=min_conf, latency_ms=latency,
            state_json=json.dumps(state),
            decision_json=json.dumps(_strip_raw(decision)) if decision else None,
            routing_json=json.dumps(routing) if routing else None,
            explanation_json=json.dumps(explanation) if explanation else None,
            error=error,
        )
        s.add(row)
        s.flush()
        return int(row.id)


def _strip_raw(decision: dict | None) -> dict | None:
    return decision  # raw responses are kept: Research Mode needs them


def _row_to_dict(row: CaseRun) -> dict[str, Any]:
    return {
        "run_id": row.id, "patient_id": row.patient_id, "created_at": row.created_at.isoformat(),
        "mode": row.mode, "model": row.model, "urgency": row.urgency, "specialty": row.specialty,
        "route": row.route, "min_confidence": row.min_confidence, "latency_ms": row.latency_ms,
        "state": json.loads(row.state_json),
        "decision": json.loads(row.decision_json) if row.decision_json else None,
        "routing": json.loads(row.routing_json) if row.routing_json else None,
        "explanation": json.loads(row.explanation_json) if row.explanation_json else None,
        "error": row.error,
    }


def list_runs(
    *, urgency: str | None = None, specialty: str | None = None, min_confidence: float | None = None,
    max_confidence: float | None = None, search: str | None = None, limit: int = 200,
) -> list[dict[str, Any]]:
    with session_scope() as s:
        q = s.query(CaseRun)
        if urgency:
            q = q.filter(CaseRun.urgency == urgency)
        if specialty:
            q = q.filter(CaseRun.specialty == specialty)
        if min_confidence is not None:
            q = q.filter(CaseRun.min_confidence >= min_confidence)
        if max_confidence is not None:
            q = q.filter(CaseRun.min_confidence <= max_confidence)
        if search:
            q = q.filter(CaseRun.patient_id.ilike(f"%{search}%"))
        rows = q.order_by(CaseRun.created_at.desc(), CaseRun.id.desc()).limit(limit).all()
        return [_row_to_dict(r) for r in rows]


def latest_run_for_case(patient_id: str) -> dict[str, Any] | None:
    with session_scope() as s:
        row = (
            s.query(CaseRun).filter(CaseRun.patient_id == patient_id)
            .order_by(CaseRun.created_at.desc(), CaseRun.id.desc()).first()
        )
        return _row_to_dict(row) if row else None
