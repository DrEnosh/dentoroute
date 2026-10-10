"""Benchmark orchestration + end-to-end pipeline (Jev/Claude mocked, no keys, no network)."""
from __future__ import annotations

import copy
import json

import httpx2
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from backend.repositories import benchmark_repo, case_repo
from backend.services import benchmark_service as bs
from backend.services import claude_service as cs
from backend.services import demo_simulator, export_service, jev_service as js, pipeline_service
from backend.services.synthetic_case_service import load_dataset
from backend.utils import config
from backend.utils.errors import BenchmarkError, MissingApiKeyError
from tests.test_claude_service import FakeClient, good_decision, good_explanation


# --------------------------------------------------------------- sampling ------
def test_select_cases_is_stratified_seeded_and_deterministic():
    a = bs.select_cases(limit=42, seed=1)
    b = bs.select_cases(limit=42, seed=1)
    c = bs.select_cases(limit=42, seed=2)
    assert [x.patient_id for x in a] == [x.patient_id for x in b]
    assert [x.patient_id for x in a] != [x.patient_id for x in c]
    assert len({x.category for x in a}) == 21  # every category represented
    assert len(bs.select_cases()) == len(load_dataset()) >= 500


# ----------------------------------------------------------- demo benchmark ----
def test_demo_benchmark_runs_all_cases_and_is_labelled_simulated():
    out = bs.run_benchmark("jev", demo=True, workers=4)
    run = out["runs"]["jev"]
    assert run["mode"] == "demo" and run["summary"]["simulated"] is True
    assert run["n_cases"] >= 500 and run["n_errors"] == 0
    assert "NOT Jev" in run["model"]
    assert run["summary"]["metrics"]["latency"]["status"] == "not measured"  # no fabricated latency
    assert run["summary"]["metrics"]["cost"]["status"] == "not measured"
    row = run["rows"][0]
    for key in ("case_id", "timestamp", "model", "latency_ms", "predictions", "probabilities", "confidences", "error"):
        assert key in row
    json.dumps(run["summary"], allow_nan=False)  # strictly JSON-safe


def test_demo_benchmark_is_persisted_and_retrievable():
    bs.run_benchmark("jev", limit=30, demo=True)
    stored = benchmark_repo.latest_run("jev")
    assert stored and stored["n_cases"] == 30 and len(stored["rows"]) == 30
    assert bs.get_results()["claude"] is None and bs.get_results()["comparison"] is None


def test_both_in_demo_mode_skips_claude_with_reason():
    out = bs.run_benchmark("both", limit=10, demo=True)
    assert "claude" not in out["runs"] and "Demo Mode" in out["skipped"]["claude"]
    assert out["comparison"] is None  # never compare a simulator with Claude


def test_claude_only_without_key_raises_clear_error():
    with pytest.raises(MissingApiKeyError):
        bs.run_benchmark("claude", limit=5, demo=False)


def test_invalid_provider_and_empty_selection():
    with pytest.raises(BenchmarkError):
        bs.run_benchmark("gpt")
    with pytest.raises(BenchmarkError):
        bs.run_benchmark("jev", demo=True, cases=[])


# ------------------------------------------------------- live (mocked) both ----
def _live_jev(monkeypatch, fail_ids=()):
    monkeypatch.setenv("DENTOROUTE_DEMO_MODE", "false")
    config.reset_settings_cache()

    def handler(request):
        state = json.loads(request.content)["state"]
        from backend.schemas.case import CaseState

        raw = copy.deepcopy(demo_simulator.simulate(CaseState(**{"patient_id": "x", **state})))
        raw.pop("simulated", None)
        raw["model"] = "jev-1.13.0"
        return httpx2.Response(200, json=raw)

    client = TypeSafeClient(api_key="k", transport=httpx2.MockTransport(handler), retry=RetryPolicy(max_retries=0))
    monkeypatch.setattr(js, "_service", js.JevService(client=client))


class OracleClaude(FakeClient):
    """Fake Claude that answers from the rubric, with one malformed answer in the batch."""

    def __init__(self):
        super().__init__()
        self.n = 0

    def _create(self, **kwargs):
        from types import SimpleNamespace

        self.n += 1
        data = good_decision() if self.n != 3 else {**good_decision(), "urgency": "SUPER_URGENT"}
        return SimpleNamespace(content=[SimpleNamespace(type="tool_use", input=data)],
                               usage=SimpleNamespace(input_tokens=100, output_tokens=50))


def test_live_both_produces_comparison_with_honest_failure_accounting(monkeypatch):
    _live_jev(monkeypatch)
    monkeypatch.setattr(cs, "_service", cs.ClaudeService(client=OracleClaude()))
    out = bs.run_benchmark("both", limit=20, demo=False, workers=1)
    jev, claude, cmp_ = out["runs"]["jev"], out["runs"]["claude"], out["comparison"]
    assert jev["mode"] == "live" and jev["summary"]["simulated"] is False
    assert jev["summary"]["metrics"]["latency"]["status"] == "measured"
    assert claude["summary"]["metrics"]["n_invalid_outputs"] == 1
    assert claude["summary"]["metrics"]["schema_compliance"] == pytest.approx(19 / 20)
    assert cmp_["paired"]["n_paired"] == 19
    assert cmp_["table"]["cost"]["claude"]["status"] == "not measured"  # no prices configured -> no invented cost
    assert "same author" in cmp_["caveat"]
    assert bs.get_results()["comparison"] is not None


def test_api_failures_do_not_abort_the_run(monkeypatch):
    monkeypatch.setenv("DENTOROUTE_DEMO_MODE", "false")
    config.reset_settings_cache()
    client = TypeSafeClient(api_key="k", retry=RetryPolicy(max_retries=0),
                            transport=httpx2.MockTransport(lambda r: httpx2.Response(500, json={"e": 1})))
    monkeypatch.setattr(js, "_service", js.JevService(client=client))
    out = bs.run_benchmark("jev", limit=6, demo=False, workers=2)
    run = out["runs"]["jev"]
    assert run["n_errors"] == 6 and all(r["error_kind"] == "api" for r in run["rows"])
    assert run["summary"]["metrics"]["n_valid"] == 0


def test_claude_price_config_enables_cost(monkeypatch):
    monkeypatch.setenv("CLAUDE_PRICE_INPUT_PER_MTOK", "3")
    monkeypatch.setenv("CLAUDE_PRICE_OUTPUT_PER_MTOK", "15")
    config.reset_settings_cache()
    case = load_dataset()[0]
    from backend.schemas.decision import ClaudeDecision

    d = ClaudeDecision(model="m", usage={"input_tokens": 1_000_000, "output_tokens": 0}, **good_decision())
    assert bs.claude_row(case, d)["cost_usd"] == pytest.approx(3.0)


# ---------------------------------------------------------------- pipeline -----
def test_pipeline_demo_end_to_end_persists_run():
    res = pipeline_service.evaluate_by_id("DEMO-04")
    assert res.error is None and res.mode == "demo"
    assert res.routing.route in ("urgent_review", "escalate_to_human") and res.routing.requires_human  # a severe case is never automated
    assert res.explanation.source == "template" and any("Demo Mode" in w for w in res.explanation.warnings)
    assert res.reference["urgency"] == "URGENT"
    runs = case_repo.list_runs()
    assert len(runs) == 1 and runs[0]["patient_id"] == "DEMO-04" and runs[0]["route"] == res.routing.route


def test_pipeline_persist_false_writes_nothing():
    pipeline_service.evaluate_by_id("DEMO-01", persist=False)
    assert case_repo.list_runs() == []


def test_pipeline_live_missing_key_returns_error_not_exception(routine_case):
    res = pipeline_service.evaluate_case(routine_case, demo=False)
    assert res.decision is None and "TYPESAFE_API_KEY" in res.error
    assert case_repo.list_runs()[0]["error"]  # failures are logged to history too


def test_pipeline_uses_claude_explanation_when_live_and_key_present(monkeypatch, severe_case):
    _live_jev(monkeypatch)
    monkeypatch.setattr(cs, "_service", cs.ClaudeService(client=FakeClient(good_explanation())))
    res = pipeline_service.evaluate_case(severe_case, demo=False)
    assert res.explanation.source == "claude" and res.mode == "live"


def test_pipeline_explanation_falls_back_to_template_when_claude_errors(monkeypatch, severe_case):
    import anthropic
    import httpx

    _live_jev(monkeypatch)
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://x"))
    monkeypatch.setattr(cs, "_service", cs.ClaudeService(client=FakeClient(raises=err)))
    res = pipeline_service.evaluate_case(severe_case, demo=False)
    assert res.error is None and res.explanation.source == "template"
    assert any("unavailable" in w for w in res.explanation.warnings)


def test_case_detail_includes_reference_and_latest_run():
    pipeline_service.evaluate_by_id("DEMO-02")
    d = pipeline_service.case_detail("DEMO-02")
    assert d["reference"]["specialty"] and d["latest_run"]["patient_id"] == "DEMO-02"


# ------------------------------------------------------------------ export -----
def test_case_report_exports():
    res = pipeline_service.evaluate_by_id("DEMO-04", persist=False).model_dump()
    md = export_service.case_report_markdown(res)
    assert "Research prototype. Not for clinical diagnosis or treatment." in md and "Demo Mode" in md
    assert "not clinical truth" in md
    js_out = json.loads(export_service.case_report_json(res))
    assert "raw_response" not in js_out["decision"] and js_out["safety_notice"]
    assert "raw_response" in json.loads(export_service.case_report_json(res, include_raw=True))["decision"]


def test_benchmark_csv_export_flattens_rows():
    out = bs.run_benchmark("jev", limit=5, demo=True)
    csv_text = export_service.rows_to_csv(out["runs"]["jev"]["rows"])
    header = csv_text.splitlines()[0].split(",")
    assert "case_id" in header and "predictions.urgency" in header and "reference.specialty" in header
    assert len(csv_text.strip().splitlines()) == 6
    assert export_service.rows_to_csv([]) == ""
