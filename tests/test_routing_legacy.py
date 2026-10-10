"""LEGACY (v1) routing policy: kept as the comparison baseline. The conservative policy is in test_safety_gate.py."""
from __future__ import annotations

import copy

import pytest

from backend.services import jev_service as js
from backend.services.routing_service import route_decision as _route_active
from backend.utils.config import load_routing_policy
from tests.helpers import make_decision

LEGACY = load_routing_policy("legacy")


def route_decision(decision, policy=None):
    return _route_active(decision, policy or LEGACY)


def test_thresholds_come_from_yaml_policy():
    policy = LEGACY
    r = route_decision(make_decision(human=0.9), policy)
    assert r.thresholds["human_review_probability"] == policy["thresholds"]["human_review_probability"]


def test_r1_human_review_escalates_and_beats_everything():
    d = make_decision(human=0.85, urgency=(0, 0, 0.1, 0.9), same_day=0.95)
    r = route_decision(d)
    assert r.route == "escalate_to_human" and r.triggered_rule == "R1_human_review_probability"
    assert r.requires_human is True


def test_r1_boundary_is_inclusive():
    t = LEGACY["thresholds"]["human_review_probability"]
    assert route_decision(make_decision(human=t)).route == "escalate_to_human"
    assert route_decision(make_decision(human=t - 0.01)).route != "escalate_to_human"


@pytest.mark.parametrize(
    "kwargs, expected_text",
    [
        (dict(urgency=(0, 0, 0.1, 0.9)), "Urgency score"),
        (dict(same_day=0.9), "Same-day"),
        (dict(infection=0.95), "Infection-concern"),
    ],
)
def test_r2_urgent_signals(kwargs, expected_text):
    r = route_decision(make_decision(**kwargs))
    assert r.route == "urgent_review" and r.triggered_rule == "R2_urgent_signals"
    assert any(expected_text in x for x in r.reasons)
    assert r.requires_human


def test_r2_lists_every_triggering_signal():
    r = route_decision(make_decision(urgency=(0, 0, 0.1, 0.9), same_day=0.9, infection=0.95))
    assert len(r.reasons) == 3


def test_r3_confident_case_is_automated_with_specialty_label():
    r = route_decision(make_decision(specialty=("endodontics", 0.92)))
    assert r.route == "automated_specialty_route" and r.triggered_rule == "R3_confident_specialty"
    assert r.destination_specialty == "endodontics"
    assert r.workflow_label.upper().startswith("ENDODONTIC")
    assert r.requires_human is False


def test_r4_low_specialty_confidence_goes_to_human_review():
    r = route_decision(make_decision(specialty=("endodontics", 0.2)))
    assert r.route == "human_review" and r.triggered_rule == "R4_low_confidence"
    assert r.confidence_band == "low"
    assert "does not force an answer" in r.reasons[0]


def test_r4_low_urgency_confidence_goes_to_human_review():
    # urgency split between ROUTINE and EMERGENCY -> score ~1.5 (not >= 2.5) but confidence ~0
    r = route_decision(make_decision(urgency=(0.5, 0, 0, 0.5), specialty=("endodontics", 0.95)))
    assert r.route == "human_review"
    assert any("urgency confidence" in x for x in r.reasons)


def test_custom_policy_is_respected():
    policy = copy.deepcopy(LEGACY)
    policy["thresholds"]["human_review_probability"] = 0.05
    assert route_decision(make_decision(human=0.1), policy).route == "escalate_to_human"


def test_policy_never_returns_unknown_route():
    from itertools import product

    for hr, sd, sp in product((0.05, 0.5, 0.9), (0.1, 0.9), (0.2, 0.9)):
        r = route_decision(make_decision(human=hr, same_day=sd, specialty=("periodontics", sp)))
        assert r.route in {"escalate_to_human", "urgent_review", "automated_specialty_route", "human_review"}
        assert r.reasons


def test_ambiguous_demo_case_is_escalated_to_a_human():
    """The LinkedIn 'ambiguous low-confidence' demo case must not be automated."""
    from backend.services.synthetic_case_service import load_demo_cases

    case = next(c for c in load_demo_cases() if c.patient_id == "DEMO-10")
    r = _route_active(js.JevService().evaluate(case.state))
    assert r.route in ("human_review", "escalate_to_human")
    assert r.requires_human


def test_routine_demo_case_is_automated_and_emergency_is_not():
    from backend.services.synthetic_case_service import load_demo_cases

    cases = {c.patient_id: c for c in load_demo_cases()}
    svc = js.JevService()
    routine = _route_active(svc.evaluate(cases["DEMO-01"].state))
    assert routine.route != "urgent_review"
    for cid, c in cases.items():
        if c.reference and c.reference.urgency == "EMERGENCY":
            assert _route_active(svc.evaluate(c.state)).requires_human
