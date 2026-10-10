"""Frontend -> backend client.

Talks to the FastAPI backend over HTTP when it is reachable (DENTOROUTE_API_URL). If it is not, the very same
FastAPI app is run in-process through Starlette's TestClient, so `streamlit run frontend/app.py` works on its own
and every response (status codes, error bodies) is identical either way.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from backend.utils.config import get_settings  # noqa: E402


class ApiError(Exception):
    """A readable error for the UI (never a traceback)."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.message, self.status = message, status


_cache: dict[str, Any] = {"transport": None, "kind": None, "checked": 0.0}


def _http_reachable(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=1.5).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def _transport() -> tuple[Any, str]:
    """('api' over HTTP) or ('in-process'), re-checked every 30 seconds."""
    now = time.time()
    if _cache["transport"] is not None and now - _cache["checked"] < 30:
        return _cache["transport"], _cache["kind"]
    url = get_settings().dentoroute_api_url.rstrip("/")
    if _http_reachable(url):
        t, kind = httpx.Client(base_url=url, timeout=600), "api"
    else:
        from fastapi.testclient import TestClient

        from backend.main import create_app

        t, kind = TestClient(create_app(), raise_server_exceptions=False), "in-process"
    _cache.update(transport=t, kind=kind, checked=now)
    return t, kind


def backend_kind() -> str:
    return _transport()[1]


def request(method: str, path: str, *, json: Any = None, params: dict | None = None, allow_error_body: bool = False) -> Any:
    t, _ = _transport()
    try:
        r = t.request(method, path, json=json, params={k: v for k, v in (params or {}).items() if v not in (None, "")})
    except httpx.HTTPError as exc:
        _cache["checked"] = 0.0  # force a re-check next time
        raise ApiError("Could not reach the DentoRoute backend. Is it running?") from exc
    ctype = r.headers.get("content-type", "")
    if r.status_code >= 400:
        body = r.json() if ctype.startswith("application/json") else {}
        if allow_error_body and isinstance(body, dict) and body.get("error"):
            return body
        msg = body.get("error") or body.get("detail") if isinstance(body, dict) else None
        if isinstance(msg, list):  # FastAPI validation errors
            msg = "; ".join(str(m.get("msg", m)) for m in msg)
        raise ApiError(str(msg or f"Backend error (HTTP {r.status_code})."), r.status_code)
    return r.json() if ctype.startswith("application/json") else r.text


# ------------------------------------------------------------------ endpoints ----
def health() -> dict:
    return request("GET", "/health")


def demo_cases() -> list[dict]:
    return request("GET", "/api/cases/demo")["cases"]


def categories() -> dict[str, int]:
    return request("GET", "/api/cases/categories")["categories"]


def generate_case(category: str | None, seed: int | None, from_dataset: bool = True) -> dict:
    return request("POST", "/api/cases/generate", json={"category": category, "seed": seed, "count": 1, "from_dataset": from_dataset})["cases"][0]


def evaluate(*, case: dict | None = None, case_id: str | None = None, demo: bool | None = None, explain: bool = True) -> dict:
    """Returns the EvaluateCaseResponse body. Provider failures come back with `error` set (not raised)."""
    body = {"case": case, "case_id": case_id, "demo": demo, "explain": explain}
    return request("POST", "/api/cases/evaluate", json=body, allow_error_body=True)


def case_detail(case_id: str) -> dict:
    return request("GET", f"/api/cases/{case_id}")


def case_report(case_id: str, fmt: str = "md") -> str:
    return request("GET", f"/api/cases/{case_id}/report", params={"format": fmt})


def history(**filters: Any) -> list[dict]:
    return request("GET", "/api/history", params=filters)["runs"]


def run_benchmark(provider: str, limit: int | None, demo: bool | None, workers: int | None = None, include_rows: bool = True) -> dict:
    body = {"provider": provider, "limit": limit, "demo": demo, "workers": workers, "include_rows": include_rows}
    return request("POST", "/api/benchmark/run", json=body)


def benchmark_results(include_rows: bool = True) -> dict:
    return request("GET", "/api/benchmark/results", params={"include_rows": str(include_rows).lower()})


def export_benchmark(provider: str, fmt: str) -> str:
    return request("GET", "/api/benchmark/export", params={"provider": provider, "format": fmt})


def active_policy() -> dict:
    return request("GET", "/api/policy")


def policy_study() -> dict | None:
    return request("GET", "/api/policy/study")["study"]


def run_policy_study() -> dict:
    return request("POST", "/api/policy/study")
