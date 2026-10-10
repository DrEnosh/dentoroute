"""DentoRoute API. Research prototype - synthetic data only. Not for clinical diagnosis or treatment."""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routers import ai, benchmark, cases, history, policy, system
from backend.utils.config import SAFETY_BANNER, TAGLINE
from backend.utils.errors import (
    BenchmarkError, CaseNotFoundError, DentoRouteError, MissingApiKeyError, PolicyError, RepositoryError,
)
from backend.utils.logging_utils import get_logger

logger = get_logger("api")

_STATUS = {MissingApiKeyError: 503, RepositoryError: 503, CaseNotFoundError: 404, BenchmarkError: 400, PolicyError: 500}


def create_app() -> FastAPI:
    app = FastAPI(title="DentoRoute", description=f"{TAGLINE} {SAFETY_BANNER}", version="1.0.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(DentoRouteError)
    async def handle_known(_: Request, exc: DentoRouteError) -> JSONResponse:
        status = next((s for t, s in _STATUS.items() if isinstance(exc, t)), 502)
        logger.warning("request failed: %s (%s)", exc.code, exc.user_message)
        return JSONResponse(status_code=status, content={"error": exc.user_message, "code": exc.code, "safety_notice": SAFETY_BANNER})

    @app.exception_handler(Exception)
    async def handle_unknown(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error: %s", exc.__class__.__name__)
        return JSONResponse(status_code=500, content={"error": "Unexpected server error. See the backend logs.", "code": "internal_error"})

    for r in (system.router, cases.router, ai.router, benchmark.router, history.router, policy.router):
        app.include_router(r)
    return app


app = create_app()
