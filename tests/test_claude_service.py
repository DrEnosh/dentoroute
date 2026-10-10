"""Claude layer tests with an injected fake Anthropic client (no key, no network)."""
from __future__ import annotations

from types import SimpleNamespace

import anthropic
import httpx
import pytest

from backend.services import claude_service as cs
from backend.services import jev_service as js
from backend.services.routing_service import route_decision
from backend.services.signal_service import find_unsupported_claims, positive_signals
from backend.utils.errors import ClaudeUnavailableError, InvalidModelOutputError, MissingApiKeyError

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


class FakeClient:
    def __init__(self, tool_input=None, raises=None):
        self.tool_input, self.raises, self.calls = tool_input, raises, []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            exc = self.raises.pop(0) if isinstance(self.raises, list) else self.raises
            if exc is not None:
                raise exc
        block = SimpleNamespace(type="tool_use", input=self.tool_input)
        return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=120, output_tokens=60))


def good_explanation(**over):
    base = dict(
        summary="This synthetic case was routed to an urgent review because the engine flagged fever and swelling.",
        engine_findings="The engine gave a high probability to infection concern.",
        why_route="The urgency score crossed the policy threshold.",
        urgency_signals=["fever"], human_review_note="Human review is required.", uncertainty="Confidence is high.",
    )
    base.update(over)
    return base


@pytest.fixture
def bundle(severe_case):
    d = js.JevService().evaluate(severe_case)
    return severe_case, d, route_decision(d)


# ------------------------------------------------------------ explanation ----
def test_grounded_explanation_is_returned(bundle):
    case, d, r = bundle
    client = FakeClient(good_explanation(urgency_signals=["fever", "swelling (right mandibular buccal region)"]))
    out = cs.ClaudeService(client=client).explain(case, d, r)
    assert out.source == "claude" and out.grounding_ok and out.latency_ms is not None
    assert out.disclaimer


def test_forced_tool_choice_and_constrained_signals(bundle):
    case, d, r = bundle
    client = FakeClient(good_explanation())
    cs.ClaudeService(client=client).explain(case, d, r)
    kw = client.calls[0]
    assert kw["tool_choice"] == {"type": "tool", "name": "submit_explanation"}
    enum = kw["tools"][0]["input_schema"]["properties"]["urgency_signals"]["items"]["enum"]
    assert set(enum) == set(positive_signals(case))  # Claude can only cite facts that exist
    assert "ONLY the facts" in kw["system"]


def test_invented_finding_triggers_template_fallback(routine_case):
    d = js.JevService().evaluate(routine_case)
    r = route_decision(d)
    client = FakeClient(good_explanation(summary="The patient has a high fever and facial swelling."))
    out = cs.ClaudeService(client=client).explain(routine_case, d, r)
    assert out.source == "template" and out.grounding_ok is False
    assert out.grounding_issues and out.warnings


def test_negated_findings_are_allowed(routine_case):
    assert find_unsupported_claims("There is no fever and no swelling.", routine_case) == []
    assert find_unsupported_claims("Without trismus or trauma.", routine_case) == []


def test_unsupported_claim_detection(routine_case, severe_case):
    assert find_unsupported_claims("The patient presented with fever.", routine_case)
    assert find_unsupported_claims("Possible trismus noted.", routine_case)
    assert find_unsupported_claims("The patient presented with fever.", severe_case) == []


def test_template_explanation_is_grounded_and_complete(bundle):
    case, d, r = bundle
    t = cs.template_explanation(case, d, r)
    assert t.source == "template" and t.grounding_ok
    assert find_unsupported_claims(" ".join([t.summary, t.engine_findings, t.why_route, t.uncertainty]), case) == []
    assert "synthetic" in t.summary.lower() and "not a" in t.disclaimer.lower()


def test_template_explanation_for_routine_case_does_not_invent(routine_case):
    d = js.JevService().evaluate(routine_case)
    t = cs.template_explanation(routine_case, d, route_decision(d))
    assert find_unsupported_claims(" ".join([t.summary, t.engine_findings, t.why_route]), routine_case) == []


def test_missing_explanation_fields_rejected(bundle):
    case, d, r = bundle
    bad = good_explanation()
    del bad["why_route"]
    with pytest.raises(InvalidModelOutputError):
        cs.ClaudeService(client=FakeClient(bad)).explain(case, d, r)


# --------------------------------------------------------------- failures ----
def test_missing_key_raises_clear_error(bundle):
    case, d, r = bundle
    svc = cs.ClaudeService()
    assert svc.available is False
    with pytest.raises(MissingApiKeyError) as exc:
        svc.explain(case, d, r)
    assert "ANTHROPIC_API_KEY" in exc.value.user_message


@pytest.mark.parametrize(
    "exc",
    [
        anthropic.APITimeoutError(request=REQ),
        anthropic.APIConnectionError(request=REQ),
        anthropic.AuthenticationError("bad", response=httpx.Response(401, request=REQ), body=None),
        anthropic.RateLimitError("slow down", response=httpx.Response(429, request=REQ), body=None),
        anthropic.InternalServerError("boom", response=httpx.Response(500, request=REQ), body=None),
    ],
)
def test_api_failures_become_handled_errors(bundle, exc):
    case, d, r = bundle
    with pytest.raises(ClaudeUnavailableError):
        cs.ClaudeService(client=FakeClient(raises=exc)).explain(case, d, r)


def test_temperature_rejection_retries_without_it(bundle):
    case, d, r = bundle
    bad = anthropic.BadRequestError(
        "temperature is not supported for this model", response=httpx.Response(400, request=REQ), body=None
    )
    client = FakeClient(good_explanation(), raises=[bad, None])
    out = cs.ClaudeService(client=client).explain(case, d, r)
    assert out.source == "claude" and len(client.calls) == 2
    assert "temperature" in client.calls[0] and "temperature" not in client.calls[1]


def test_no_tool_block_is_invalid_output(bundle):
    case, d, r = bundle
    client = FakeClient()
    client.messages.create = lambda **k: SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")], usage=None)
    with pytest.raises(InvalidModelOutputError):
        cs.ClaudeService(client=client).explain(case, d, r)


# ------------------------------------------------------------- benchmark -----
def good_decision(**over):
    base = dict(
        urgency="URGENT", urgency_confidence=0.8, specialty="endodontics", specialty_confidence=0.7,
        infection_concern=True, infection_probability=0.7, same_day_review=True, same_day_probability=0.8,
        human_review=False, human_review_probability=0.2, complexity="MODERATE", referral_priority="HIGH",
    )
    base.update(over)
    return base


def test_decide_parses_valid_output_and_records_usage(severe_case):
    client = FakeClient(good_decision())
    out = cs.ClaudeService(client=client).decide(severe_case)
    assert out.urgency == "URGENT" and out.usage == {"input_tokens": 120, "output_tokens": 60}
    assert client.calls[0]["tool_choice"]["name"] == "submit_decisions"
    assert "patient_id" not in client.calls[0]["messages"][0]["content"]  # same input as Jev


def test_decide_uses_same_questions_as_jev(severe_case):
    client = FakeClient(good_decision())
    cs.ClaudeService(client=client).decide(severe_case)
    system = client.calls[0]["system"]
    for qid in js.QUESTION_SPECS:
        assert f"[{qid}]" in system


@pytest.mark.parametrize(
    "over",
    [dict(urgency="SUPER_URGENT"), dict(specialty="cosmetic_dentistry"), dict(complexity="EASY"),
     dict(referral_priority="NOW"), dict(infection_probability=1.7)],
)
def test_invalid_claude_decisions_are_schema_failures(over):
    with pytest.raises(InvalidModelOutputError):
        cs.ClaudeService.parse_decision(good_decision(**over), latency_ms=1.0, usage=None, model="m")


def test_missing_decision_field_is_schema_failure():
    data = good_decision()
    del data["human_review"]
    with pytest.raises(InvalidModelOutputError):
        cs.ClaudeService.parse_decision(data, latency_ms=1.0, usage=None, model="m")


def test_cost_is_not_measured_unless_prices_configured():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert cs.estimate_cost(usage, None, None) is None
    assert cs.estimate_cost(None, 3.0, 15.0) is None
    assert cs.estimate_cost(usage, 3.0, 15.0) == pytest.approx(18.0)
