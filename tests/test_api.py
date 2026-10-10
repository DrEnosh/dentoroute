"""FastAPI endpoints via TestClient (demo mode; Jev/Claude mocked where live paths matter)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.utils import config


@pytest.fixture
def client():
    return TestClient(create_app(), raise_server_exceptions=False)


def test_root_and_health(client):
    root = client.get("/").json()
    assert root["name"] == "DentoRoute" and root["mode"] == "demo" and "Not for clinical" in root["safety_notice"]
    assert root["banner"].startswith("Demo Mode")
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["mode"] == "demo" and h["dataset_cases"] >= 500
    assert h["jev_key_configured"] is False and "key" not in str(h).lower().replace("_key_configured", "")


def test_health_never_exposes_secrets(client, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-secret-abc")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    config.reset_settings_cache()
    body = client.get("/health").text + client.get("/").text
    assert "sk-secret-abc" not in body and "sk-ant-secret" not in body


def test_generate_cases(client):
    r = client.post("/api/cases/generate", json={"category": "facial_swelling", "count": 3, "seed": 5}).json()
    assert r["count"] == 3 and all(c["category"] == "facial_swelling" for c in r["cases"])
    fresh = client.post("/api/cases/generate", json={"from_dataset": False, "seed": 1}).json()
    assert fresh["cases"][0]["patient_id"].startswith("SYN-GEN")


def test_generate_unknown_category_is_422(client):
    assert client.post("/api/cases/generate", json={"category": "teeth_whitening"}).status_code == 422


def test_evaluate_by_id_returns_full_pipeline(client):
    r = client.post("/api/cases/evaluate", json={"case_id": "DEMO-04"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "demo" and body["routing"]["route"] in ("urgent_review", "escalate_to_human") and body["routing"]["requires_human"]
    assert body["decision"]["urgency"]["label"] and body["explanation"]["source"] == "template"
    assert body["reference"]["urgency"] == "URGENT" and "Not for clinical" in body["safety_notice"]


def test_evaluate_inline_case(client, severe_case):
    r = client.post("/api/cases/evaluate", json={"case": severe_case.model_dump(), "explain": False})
    assert r.status_code == 200 and r.json()["explanation"] is None and r.json()["reference"] is None


def test_evaluate_needs_case_or_id(client):
    assert client.post("/api/cases/evaluate", json={}).status_code == 422


def test_evaluate_rejects_invalid_state(client):
    assert client.post("/api/cases/evaluate", json={"case": {"age": 500}}).status_code == 422
    assert client.post("/api/cases/evaluate", json={"case": {"not_a_field": 1}}).status_code == 422


def test_unknown_case_is_404_with_readable_message(client):
    r = client.post("/api/cases/evaluate", json={"case_id": "NOPE-1"})
    assert r.status_code == 404 and "NOPE-1" in r.json()["error"]
    assert client.get("/api/cases/NOPE-1").status_code == 404


def test_live_mode_without_key_is_503_not_a_crash(client):
    r = client.post("/api/cases/evaluate", json={"case_id": "DEMO-01", "demo": False})
    assert r.status_code == 503
    body = r.json()
    assert "TYPESAFE_API_KEY" in body["error"] and body["decision"] is None
    r2 = client.post("/api/jev/evaluate", json={"case": {"age": 30}, "demo": False})
    assert r2.status_code == 503 and r2.json()["code"] == "missing_api_key"


def test_jev_evaluate_endpoint(client, severe_case):
    r = client.post("/api/jev/evaluate", json={"case": severe_case.model_dump()}).json()
    assert r["mode"] == "demo" and set(r["decision"]) >= {"urgency", "specialty", "infection_concern", "human_review"}
    assert r["routing"]["route"] in {"escalate_to_human", "urgent_review", "automated_specialty_route", "human_review"}


def test_claude_explain_degrades_to_template_without_key(client, severe_case):
    ev = client.post("/api/jev/evaluate", json={"case": severe_case.model_dump()}).json()
    r = client.post("/api/claude/explain", json={"case": severe_case.model_dump(), "decision": ev["decision"], "routing": ev["routing"]})
    assert r.status_code == 200 and r.json()["source"] == "template" and r.json()["warnings"]


def test_case_detail_and_report(client):
    client.post("/api/cases/evaluate", json={"case_id": "DEMO-07"})
    d = client.get("/api/cases/DEMO-07").json()
    assert d["reference"]["urgency"] == "URGENT" and d["latest_run"]["route"]
    md = client.get("/api/cases/DEMO-07/report?format=md")
    assert md.status_code == 200 and "DentoRoute case report" in md.text and "Demo Mode" in md.text
    js = client.get("/api/cases/DEMO-07/report?format=json").json()
    assert js["safety_notice"] and js["patient_id"] == "DEMO-07"
    assert client.get("/api/cases/DEMO-09/report").status_code == 404  # never evaluated


def test_history_filters(client):
    for cid in ("DEMO-01", "DEMO-04", "DEMO-07"):
        client.post("/api/cases/evaluate", json={"case_id": cid})
    allr = client.get("/api/history").json()
    assert allr["count"] == 3
    urgent = client.get("/api/history", params={"urgency": "URGENT"}).json()["runs"]
    assert urgent and all(r["urgency"] == "URGENT" for r in urgent)
    assert client.get("/api/history", params={"search": "DEMO-04"}).json()["count"] == 1
    assert client.get("/api/history", params={"min_confidence": 2}).status_code == 422


def test_benchmark_run_results_and_export(client):
    assert client.get("/api/benchmark/results").json() == {"jev": None, "claude": None, "comparison": None}
    assert client.get("/api/benchmark/export").status_code == 404
    run = client.post("/api/benchmark/run", json={"provider": "jev", "limit": 40, "demo": True}).json()
    assert run["runs"]["jev"]["summary"]["simulated"] is True and "rows" not in run["runs"]["jev"]
    res = client.get("/api/benchmark/results").json()
    assert res["jev"]["n_cases"] == 40 and "rows" not in res["jev"]
    assert len(client.get("/api/benchmark/results?include_rows=true").json()["jev"]["rows"]) == 40
    csv = client.get("/api/benchmark/export?format=csv")
    assert csv.headers["content-type"].startswith("text/csv") and len(csv.text.strip().splitlines()) == 41
    assert client.get("/api/benchmark/export?format=json").json()["n_cases"] == 40


def test_benchmark_claude_without_key_is_503(client):
    r = client.post("/api/benchmark/run", json={"provider": "claude", "limit": 5})
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["error"]


def test_benchmark_validation(client):
    assert client.post("/api/benchmark/run", json={"provider": "gpt"}).status_code == 422
    assert client.post("/api/benchmark/run", json={"limit": 0}).status_code == 422
