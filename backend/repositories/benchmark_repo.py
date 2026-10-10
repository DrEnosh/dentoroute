from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from backend.models.orm import BenchmarkResult, BenchmarkRun
from backend.repositories.db import session_scope


def create_run(provider: str, mode: str, model: str) -> int:
    with session_scope() as s:
        row = BenchmarkRun(provider=provider, mode=mode, model=model)
        s.add(row)
        s.flush()
        return int(row.id)


def save_results(run_id: int, rows: list[dict[str, Any]]) -> None:
    with session_scope() as s:
        s.add_all(
            BenchmarkResult(
                run_id=run_id, patient_id=r["case_id"], payload_json=json.dumps(r), error=r.get("error")
            )
            for r in rows
        )


def finish_run(run_id: int, n_cases: int, n_errors: int, summary: dict[str, Any]) -> None:
    with session_scope() as s:
        run = s.get(BenchmarkRun, run_id)
        if run is not None:
            run.finished_at = datetime.now(timezone.utc)
            run.n_cases, run.n_errors = n_cases, n_errors
            run.summary_json = json.dumps(summary)


def latest_run(provider: str, mode: str | None = None) -> dict[str, Any] | None:
    with session_scope() as s:
        q = s.query(BenchmarkRun).filter(BenchmarkRun.provider == provider, BenchmarkRun.finished_at.is_not(None))
        if mode:
            q = q.filter(BenchmarkRun.mode == mode)
        run = q.order_by(BenchmarkRun.id.desc()).first()
        if run is None:
            return None
        results = s.query(BenchmarkResult).filter(BenchmarkResult.run_id == run.id).all()
        return {
            "run_id": run.id, "provider": run.provider, "mode": run.mode, "model": run.model,
            "started_at": run.started_at.isoformat(),
            "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            "n_cases": run.n_cases, "n_errors": run.n_errors,
            "summary": json.loads(run.summary_json) if run.summary_json else None,
            "rows": [json.loads(r.payload_json) for r in results],
        }
