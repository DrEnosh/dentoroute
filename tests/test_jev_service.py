"""Jev integration tests. The REAL official SDK is exercised against a mock HTTP transport,
so request/response handling is tested without a live API key."""
from __future__ import annotations

import copy
import json

import httpx2
import pytest
from typesafe_sdk import RetryPolicy, TypeSafeClient

from backend.schemas.labels import SPECIALTIES, URGENCY_LEVELS
from backend.services import demo_simulator, jev_service as js
from backend.utils import config
from backend.utils.confidence import choice_confidence, score_confidence
from backend.utils.errors import InvalidModelOutputError, JevServiceError, MissingApiKeyError


# ------------------------------------------------------------------ helpers --
def good_raw(case) -> dict:
    raw = demo_simulator.simulate(case)
    raw = copy.deepcopy(raw)
    raw.pop("simulated", None)
    raw["model"] = "jev-1.13.0"
    return raw


def make_service(handler, monkeypatch) -> tuple[js.JevService, list]:
    monkeypatch.setenv("DENTOROUTE_DEMO_MODE", "false")
    config.reset_settings_cache()
    calls: list = []

    def wrapped(request):
        calls.append(json.loads(request.content))
        return handler(request)

    client = TypeSafeClient(
        api_key="test-key-123", transport=httpx2.MockTransport(wrapped), retry=RetryPolicy(max_retries=0)
    )
    return js.JevService(client=client), calls


# ------------------------------------------------------- question design -----
def test_seven_atomic_questions_in_expected_primitives():
    spec = js.QUESTION_SPECS
    assert list(spec) == [
        "urgency", "specialty", "infection_concern", "same_day_review", "human_review",
        "complexity", "referral_priority",
    ]
    assert {q: s["type"] for q, s in spec.items()} == {
        "urgency": "score", "specialty": "choice", "infection_concern": "noul", "same_day_review": "noul",
        "human_review": "noul", "complexity": "score", "referral_priority": "score",
    }
    assert len(spec["urgency"]["criteria"]) == 4
    assert set(spec["specialty"]["criteria"]) == set(SPECIALTIES)  # closed label set, no invented labels


def test_question_text_from_spec_is_used_verbatim():
    s = js.QUESTION_SPECS
    assert "meaningful concern for an active odontogenic infection or spreading infection risk" in s["infection_concern"]["instructions"]
    assert s["same_day_review"]["instructions"].startswith("Does the available case information support same-day human clinical review")
    assert "escalated to human review before an automated workflow proceeds" in s["human_review"]["instructions"]


def test_sdk_question_objects_build():
    qs = js.build_sdk_questions()
    assert set(qs) == set(js.QUESTION_SPECS)
    assert qs["urgency"].type == "score" and qs["specialty"].type == "choice" and qs["human_review"].type == "noul"


def test_confidence_formulas_match_official_docs_examples():
    assert score_confidence([0.0, 0.57, 0.43]) == pytest.approx(0.35, abs=0.01)  # docs: bug_severity
    assert score_confidence([0.0, 0.0, 1.0]) == 1.0
    assert score_confidence([0.5, 0.0, 0.5]) == 0.0  # torn between opposite ends
    assert choice_confidence([0.61, 0.35, 0.04]) == pytest.approx(0.42, abs=0.01)  # docs: department
    assert choice_confidence([0.34, 0.40, 0.02, 0.24]) == pytest.approx(0.2, abs=0.01)


# ------------------------------------------------------------- live success --
def test_live_request_shape_and_normalization(severe_case, monkeypatch):
    raw = good_raw(severe_case)
    svc, calls = make_service(lambda r: httpx2.Response(200, json=raw), monkeypatch)
    d = svc.evaluate(severe_case, demo=False, use_cache=False)

    body = calls[0]
    assert set(body) == {"state", "model", "questions"}
    assert body["model"] == "jev-latest"
    assert set(body["questions"]) == set(js.QUESTION_SPECS)  # all questions in ONE request
    assert "patient_id" not in body["state"] and body["state"]["fever"] is True

    assert d.mode == "live" and d.model == "jev-1.13.0" and d.cached is False
    assert d.latency_ms is not None and d.latency_ms >= 0
    assert d.urgency.label in URGENCY_LEVELS
    assert set(d.specialty.probabilities) == set(SPECIALTIES)
    assert abs(sum(d.urgency.probabilities.values()) - 1) < 0.02
    assert d.raw_response["answers"]["urgency"]["type"] == "score"


def test_cache_prevents_repeat_calls(routine_case, monkeypatch):
    raw = good_raw(routine_case)
    svc, calls = make_service(lambda r: httpx2.Response(200, json=raw), monkeypatch)
    first = svc.evaluate(routine_case, demo=False)
    second = svc.evaluate(routine_case, demo=False)
    assert len(calls) == 1
    assert first.cached is False and second.cached is True
    assert second.urgency.label == first.urgency.label
    assert second.latency_ms == pytest.approx(first.latency_ms)  # original latency preserved as metadata


def test_cache_key_depends_on_state_and_model(routine_case, severe_case):
    svc = js.JevService()
    assert svc.cache_key(routine_case) != svc.cache_key(severe_case)
    assert svc.cache_key(routine_case) == svc.cache_key(routine_case.model_copy())
    assert svc.cache_key(routine_case, "jev-latest") != svc.cache_key(routine_case, "jev-1.13.0")
    # the synthetic ID is not part of the model input, so it must not affect the key
    assert svc.cache_key(routine_case) == svc.cache_key(routine_case.model_copy(update={"patient_id": "SYN-OTHER"}))


# ----------------------------------------------------------------- failures --
def test_missing_api_key_gives_clear_error(routine_case, monkeypatch):
    monkeypatch.setenv("DENTOROUTE_DEMO_MODE", "false")
    config.reset_settings_cache()
    with pytest.raises(MissingApiKeyError) as exc:
        js.JevService().evaluate(routine_case, demo=False)
    assert "TYPESAFE_API_KEY" in exc.value.user_message


def test_timeout_is_a_handled_error(routine_case, monkeypatch):
    def boom(request):
        raise httpx2.ReadTimeout("slow", request=request)

    svc, _ = make_service(boom, monkeypatch)
    with pytest.raises(JevServiceError) as exc:
        svc.evaluate(routine_case, demo=False, use_cache=False)
    assert "timed out" in exc.value.user_message.lower()


def test_connection_failure_is_handled(routine_case, monkeypatch):
    def boom(request):
        raise httpx2.ConnectError("no route", request=request)

    svc, _ = make_service(boom, monkeypatch)
    with pytest.raises(JevServiceError):
        svc.evaluate(routine_case, demo=False, use_cache=False)


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors_are_handled(status, routine_case, monkeypatch):
    svc, _ = make_service(lambda r: httpx2.Response(status, json={"error": "nope"}), monkeypatch)
    with pytest.raises(JevServiceError):
        svc.evaluate(routine_case, demo=False, use_cache=False)


def test_sdk_level_malformed_response_is_handled(routine_case, monkeypatch):
    svc, _ = make_service(lambda r: httpx2.Response(200, json={"answers": {"urgency": {"type": "score"}}}), monkeypatch)
    with pytest.raises(InvalidModelOutputError):
        svc.evaluate(routine_case, demo=False, use_cache=False)


# ------------------------------------------------- normalization/validation --
@pytest.fixture
def raw(routine_case):
    return good_raw(routine_case)


def test_probability_out_of_range_rejected(raw):
    raw["answers"]["infection_concern"]["noul"] = 1.4
    with pytest.raises(InvalidModelOutputError):
        js.normalize_response(raw, mode="live")


def test_nan_probability_rejected(raw):
    raw["answers"]["human_review"]["noul"] = float("nan")
    with pytest.raises(InvalidModelOutputError):
        js.normalize_response(raw, mode="live")


def test_distribution_must_sum_to_one(raw):
    raw["answers"]["urgency"]["probabilities"] = {"0": 0.5, "1": 0.1, "2": 0.1, "3": 0.1}
    with pytest.raises(InvalidModelOutputError) as exc:
        js.normalize_response(raw, mode="live")
    assert "sum" in exc.value.detail


def test_invented_specialty_label_rejected(raw):
    raw["answers"]["specialty"]["choice"] = "cosmetic_dentistry"
    with pytest.raises(InvalidModelOutputError):
        js.normalize_response(raw, mode="live")


def test_specialty_probability_keys_must_match_allowed_labels(raw):
    probs = raw["answers"]["specialty"]["probabilities"]
    probs["cosmetic_dentistry"] = probs.pop("endodontics")
    with pytest.raises(InvalidModelOutputError):
        js.normalize_response(raw, mode="live")


def test_missing_answer_and_wrong_type_rejected(raw):
    bad = copy.deepcopy(raw)
    del bad["answers"]["complexity"]
    with pytest.raises(InvalidModelOutputError):
        js.normalize_response(bad, mode="live")
    bad = copy.deepcopy(raw)
    bad["answers"]["urgency"]["type"] = "choice"
    with pytest.raises(InvalidModelOutputError):
        js.normalize_response(bad, mode="live")
    with pytest.raises(InvalidModelOutputError):
        js.normalize_response({"nope": 1}, mode="live")


def test_score_out_of_range_rejected(raw):
    raw["answers"]["urgency"]["score"] = 7.5
    with pytest.raises(InvalidModelOutputError):
        js.normalize_response(raw, mode="live")


def test_prediction_is_argmax_and_rounded_label_kept_separately(raw):
    # bimodal: score ~1.5 rounds to a level that has ~no probability mass
    raw["answers"]["urgency"].update(
        score=1.5, probabilities={"0": 0.5, "1": 0.0, "2": 0.0, "3": 0.5}, confidence=0.0
    )
    raw["answers"]["urgency"]["probabilities"]["0"] = 0.5
    d = js.normalize_response(raw, mode="live")
    assert d.urgency.label in ("ROUTINE", "EMERGENCY")
    assert d.urgency.rounded_label in ("URGENT", "PRIORITY")
    assert d.urgency.confidence == 0.0


# --------------------------------------------------------------- demo mode ---
def test_demo_mode_needs_no_key_and_is_labeled(severe_case):
    d = js.JevService().evaluate(severe_case)  # DENTOROUTE_DEMO_MODE=true via conftest
    assert d.mode == "demo"
    assert "NOT Jev" in d.model
    assert d.latency_ms is None  # no fabricated latency
    assert d.raw_response["simulated"] is True


def test_demo_mode_is_deterministic(severe_case):
    a = js.JevService().evaluate(severe_case)
    b = js.JevService().evaluate(severe_case)
    assert a.model_dump() == b.model_dump()


def test_demo_output_passes_the_same_validation_as_live(severe_case, routine_case):
    for case in (severe_case, routine_case):
        js.normalize_response(demo_simulator.simulate(case), mode="demo")


def test_demo_simulator_ranks_severe_above_routine(severe_case, routine_case):
    sev = js.JevService().evaluate(severe_case)
    rou = js.JevService().evaluate(routine_case)
    assert sev.urgency.score > rou.urgency.score
    assert sev.infection_concern.probability > rou.infection_concern.probability


def test_auto_mode_resolution(monkeypatch):
    monkeypatch.setenv("DENTOROUTE_DEMO_MODE", "auto")
    config.reset_settings_cache()
    assert config.get_settings().resolve_demo_mode() is True  # no key -> demo
    monkeypatch.setenv("TYPESAFE_API_KEY", "abc")
    config.reset_settings_cache()
    assert config.get_settings().resolve_demo_mode() is False
    assert config.get_settings().resolve_demo_mode(override=True) is True  # request override wins


def test_api_key_never_logged(routine_case, monkeypatch, caplog):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-super-secret-999")
    config.reset_settings_cache()
    import logging

    from backend.utils.logging_utils import get_logger

    logger = get_logger("leaktest")
    handler = logging.getLogger("dentoroute").handlers[0]
    records: list[str] = []
    handler.emit = lambda rec: records.append(handler.format(rec)) if handler.filter(rec) else None  # type: ignore[assignment]
    logger.info("calling with key sk-super-secret-999 now")
    assert records and "sk-super-secret-999" not in records[0] and "REDACTED" in records[0]
