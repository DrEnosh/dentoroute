from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from backend.schemas.api import BenchmarkRunRequest
from backend.services import benchmark_service, export_service

router = APIRouter(prefix="/api/benchmark", tags=["benchmark"])


def _trim(run: dict | None, include_rows: bool) -> dict | None:
    if run is None or include_rows:
        return run
    return {k: v for k, v in run.items() if k != "rows"}


@router.post("/run")
def run(req: BenchmarkRunRequest) -> dict:
    """Synchronous. Demo mode finishes in about a second; a live 500+ case run can take minutes."""
    out = benchmark_service.run_benchmark(req.provider, req.limit, req.demo, req.workers, req.seed)
    return {"runs": {k: _trim(v, req.include_rows) for k, v in out["runs"].items()},
            "skipped": out["skipped"], "comparison": out["comparison"]}


@router.get("/results")
def results(include_rows: bool = False) -> dict:
    res = benchmark_service.get_results()
    return {"jev": _trim(res["jev"], include_rows), "claude": _trim(res["claude"], include_rows), "comparison": res["comparison"]}


@router.get("/export")
def export(provider: str = Query("jev", pattern="^(jev|claude)$"), format: str = Query("csv", pattern="^(csv|json)$")):
    run = benchmark_service.get_results()[provider]
    if run is None:
        raise HTTPException(404, f"No stored {provider} benchmark run. Run a benchmark first.")
    if format == "csv":
        return PlainTextResponse(export_service.rows_to_csv(run["rows"]), media_type="text/csv")
    return PlainTextResponse(export_service.benchmark_json(run), media_type="application/json")
