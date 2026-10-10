"""Every rule of the conservative safety gate, in isolation, plus ordering, defaults and policy validation.

Thresholds here are fixed test values, independent of whatever tuned numbers ship in config/routing_policy.yaml.
They are research-demo thresholds, not clinical thresholds.
"""
from __future__ import annotations

import copy
import random

import pytest

from backend.schemas.labels import SPECIALTIES, URGENCY_LEVELS
from backend.services import pipeline_service, risk_model
from backend.services.policy_service import MAX_LIMITS, MIN_LIMITS, validate_policy
from backend.services.routing_service import fail_safe_routing, route_decision, route_features
from backend.utils.config import load_routing_policy
from backend.utils.errors import PolicyError
from tests.helpers import benign, make_decision

TH = {
    "human_review_probability": 0.50, "infection_probability": 0.35, "same_day_probability": 0.50, "urgency_score": 1.50,
    "urgent_tail_probability": 0.30, "risk_score": 0.40, "urgency_confidence_min": 0.60, "specialty_confidence_min": 0.70,
    "specialty_top_probability_min": 0.55, "boolean_certainty_min": 0.40,
}
# A toy risk model: score rises with the model's own P(infection). Lets R3's learned-risk branch be tested exactly.
TOY_MODEL = {
    "features": risk_model.NUMERIC_FEATURES, "mean": [0.0] * 10, "scale": [1.0] * 10,
    "coef": [0, 0, 0, 0, 0, 6.0, 0, 0, 0, 0], "intercept": -3.0,
}


@pytest.fixture
def policy():
    p = copy.deepcopy(load_routing_policy("conservative"))
    p["thresholds"] = dict(TH)
    p["risk_model"] = None
    p["provenance"] = {"data_mode": "live"}
    return p


def go(d, policy):
    return route_decision(d, policy)


# ---------------------------------------------------------------------- R6 -----
def test_r6_benign_case_is_automated_only_when_every_gate_passes(policy):
    r = go(benign(), policy)
    assert r.route == "automated_specialty_route" and r.triggered_rule == "R6_all_gates_passed"
    assert r.destination_specialty == "general_dentistry" and not r.requires_human
    assert r.workflow_label.startswith("GENERAL DENTISTRY")
    assert r.policy_name == "conservative"


# ---------------------------------------------------------------------- R0 -----
def test_r0_fail_safe_is_human_review_with_reason():
    r = fail_safe_routing("Jev returned an invalid response")
    assert r.route == "human_review" and r.triggered_rule == "R0_fail_safe" and r.requires_human
    assert "No automated action was taken" in r.reasons[0]


def test_r0_pipeline_failure_never_produces_an_automated_route(routine_case):
    res = pipeline_service.evaluate_case(routine_case, demo=False)  # no key -> provider failure
    assert res.error and res.decision is None
    assert res.routing.route == "human_review" and res.routing.triggered_rule == "R0_fail_safe"


def test_r0_invalid_model_output_is_human_review(routine_case, monkeypatch):
    from backend.services import jev_service
    from backend.utils.errors import InvalidModelOutputError

    def boom(self, case, demo=None, use_cache=True):
        raise InvalidModelOutputError("Jev returned malformed output.")

    monkeypatch.setattr(jev_service.JevService, "evaluate", boom)
    res = pipeline_service.evaluate_case(routine_case)
    assert res.routing.route == "human_review" and res.routing.triggered_rule == "R0_fail_safe"
    assert res.routing.requires_human


# ---------------------------------------------------------------------- R1 -----
def test_r1_human_review_probability_escalates(policy):
    r = go(benign(human=0.6), policy)
    assert r.route == "escalate_to_human" and r.triggered_rule == "R1_human_review_probability"


def test_r1_boundary_is_inclusive(policy):
    assert go(benign(human=TH["human_review_probability"]), policy).triggered_rule == "R1_human_review_probability"
    assert go(benign(human=TH["human_review_probability"] - 0.02), policy).triggered_rule != "R1_human_review_probability"


def test_r1_beats_every_later_rule(policy):
    d = benign(human=0.9, infection=0.9, urgency=(0, 0, 0.1, 0.9), same_day=0.9)
    assert go(d, policy).triggered_rule == "R1_human_review_probability"


# ---------------------------------------------------------------------- R2 -----
def test_r2_high_infection_concern_goes_to_a_human_even_when_everything_else_is_benign(policy):
    r = go(benign(infection=0.40), policy)
    assert r.route == "urgent_review" and r.triggered_rule == "R2_infection_concern" and r.requires_human
    assert "40%" in r.reasons[0]


def test_r2_boundary_is_inclusive_and_threshold_is_lower_than_legacy(policy):
    t = TH["infection_probability"]
    assert go(benign(infection=t), policy).triggered_rule == "R2_infection_concern"
    assert go(benign(infection=t - 0.05), policy).triggered_rule != "R2_infection_concern"
    legacy = load_routing_policy("legacy")["thresholds"]["infection_probability_urgent"]
    assert t < legacy  # the old gate waited for 85%


def test_r2_legacy_would_have_automated_the_same_infection_signal():
    legacy = load_routing_policy("legacy")
    assert route_decision(benign(infection=0.60), legacy).route == "automated_specialty_route"


def test_r2_beats_r3_and_later(policy):
    d = benign(infection=0.5, urgency=(0, 0, 0.2, 0.8), same_day=0.9)
    assert go(d, policy).triggered_rule == "R2_infection_concern"


# ---------------------------------------------------------------------- R3 -----
@pytest.mark.parametrize(
    "override, text",
    [
        # score = P(PRIORITY) + 2*P(URGENT) + 3*P(EMERGENCY) = 0.70 + 0.87 = 1.57 >= 1.5, while the tail is 0.29 < 0.30
        (dict(urgency=(0.01, 0.70, 0.0, 0.29)), "Urgency score"),
        # tail 0.40 >= 0.30 while the score (0.80) stays below 1.5
        (dict(urgency=(0.6, 0.0, 0.4, 0.0)), "URGENT or EMERGENCY"),
        (dict(same_day=0.6), "Same-day"),
    ],
    ids=["urgency_score", "urgent_tail", "same_day"],
)
def test_r3_each_urgent_signal_triggers_on_its_own(policy, override, text):
    r = go(benign(**override), policy)
    assert r.route == "urgent_review" and r.triggered_rule == "R3_urgent_risk" and r.requires_human
    assert any(text in x for x in r.reasons)
    assert len(r.reasons) == 1  # only the signal under test fired


def test_r3_clean_urgent_call_is_not_automated_anymore(policy):
    """The original flaw: a confident URGENT call (score ~2.0) used to be automated."""
    d = benign(urgency=(0.0, 0.05, 0.9, 0.05), infection=0.1, same_day=0.1, human=0.1, specialty=("endodontics", 0.95))
    assert route_decision(d, load_routing_policy("legacy")).route == "automated_specialty_route"
    r = go(d, policy)
    assert r.route == "urgent_review" and r.triggered_rule == "R3_urgent_risk"


def test_r3_emergency_call_is_never_automated(policy):
    r = go(benign(urgency=(0.0, 0.0, 0.1, 0.9)), policy)
    assert r.route == "urgent_review"


def test_r3_learned_risk_score_triggers_and_is_reported(policy):
    policy["risk_model"] = TOY_MODEL
    # infection 0.30 is below R2 (0.35) but the toy risk model scores it above 0.4? 1/(1+e^-(6*0.30-3)) = 0.23 -> no
    assert go(benign(infection=0.30), policy).route == "automated_specialty_route"
    policy["thresholds"]["infection_probability"] = 0.50  # let R2 stay quiet while risk rises
    r = go(benign(infection=0.49), policy)               # risk = 1/(1+e^(0.06)) ~ 0.485 >= 0.40
    assert r.triggered_rule == "R3_urgent_risk" and any("Learned reference-risk score" in x for x in r.reasons)


def test_r3_lists_every_signal_that_fired(policy):
    r = go(benign(same_day=0.7, urgency=(0.0, 0.0, 0.5, 0.5)), policy)
    assert len(r.reasons) >= 3  # score, tail, same-day


def test_r3_beats_r4_and_r5(policy):
    d = benign(same_day=0.9, specialty=("emergency_dental_service", 0.2))
    assert go(d, policy).triggered_rule == "R3_urgent_risk"


# ---------------------------------------------------------------------- R4 -----
def test_r4_low_urgency_confidence(policy):
    d = benign(urgency=(0.5, 0.5, 0.0, 0.0))   # score 0.5, no urgent tail, but a flat distribution -> low confidence
    r = go(d, policy)
    assert r.route == "human_review" and r.triggered_rule == "R4_low_confidence"
    assert any("urgency confidence" in x for x in r.reasons) and "does not force an answer" in r.reasons[0]


def test_r4_low_specialty_confidence(policy):
    r = go(benign(specialty=("general_dentistry", 0.2)), policy)
    assert r.route == "human_review" and r.triggered_rule == "R4_low_confidence"
    assert any("specialty confidence" in x for x in r.reasons)


def test_r4_low_top_specialty_probability(policy):
    policy["thresholds"]["specialty_confidence_min"] = 0.60
    policy["thresholds"]["specialty_top_probability_min"] = 0.90
    r = go(benign(specialty=("general_dentistry", 0.85)), policy)
    assert r.triggered_rule == "R4_low_confidence" and any("top specialty probability" in x for x in r.reasons)


@pytest.mark.parametrize("field, label", [("infection", "infection concern"), ("same_day", "same-day review"), ("human", "human review")])
def test_r4_uncertain_safety_answer_goes_to_a_human(policy, field, label):
    # trigger thresholds at their hard ceilings so p=0.42 does not fire R1/R2/R3, but is still too uncertain for R4
    policy["thresholds"].update(infection_probability=0.50, same_day_probability=0.50, human_review_probability=0.50)
    r = go(benign(**{field: 0.42}), policy)   # certainty |2*0.42-1| = 0.16 < 0.40
    assert r.route == "human_review" and r.triggered_rule == "R4_low_confidence"
    assert any(label in x and "uncertain" in x for x in r.reasons)


def test_r4_confident_negative_answers_do_not_trigger(policy):
    assert go(benign(infection=0.01, same_day=0.01, human=0.01), policy).route == "automated_specialty_route"


def test_r4_confidence_boundary_is_strict_below(policy):
    d = benign()
    floor = d.urgency.confidence
    policy["thresholds"]["urgency_confidence_min"] = floor           # exactly at the floor passes
    assert go(d, policy).route == "automated_specialty_route"
    policy["thresholds"]["urgency_confidence_min"] = min(1.0, floor + 0.001)
    assert go(d, policy).triggered_rule == "R4_low_confidence"


def test_r4_beats_r5(policy):
    r = go(benign(specialty=("oral_pathology", 0.3)), policy)
    assert r.triggered_rule == "R4_low_confidence"


# ---------------------------------------------------------------------- R5 -----
def test_r5_emergency_service_destination_is_never_automated(policy):
    r = go(benign(specialty=("emergency_dental_service", 0.97)), policy)
    assert r.route == "urgent_review" and r.triggered_rule == "R5_specialty_requires_human" and r.requires_human


def test_r5_oral_pathology_destination_requires_a_person(policy):
    r = go(benign(specialty=("oral_pathology", 0.97)), policy)
    assert r.route == "human_review" and r.triggered_rule == "R5_specialty_requires_human"


def test_r5_other_specialties_are_not_blocked(policy):
    for spec in ("general_dentistry", "periodontics", "orthodontics", "endodontics"):
        assert go(benign(specialty=(spec, 0.97)), policy).route == "automated_specialty_route"


def test_r5_list_comes_from_policy(policy):
    policy["human_only_specialties"] = ["periodontics"]
    assert go(benign(specialty=("periodontics", 0.97)), policy).triggered_rule == "R5_specialty_requires_human"
    assert go(benign(specialty=("oral_pathology", 0.97)), policy).route == "automated_specialty_route"


# ------------------------------------------------- default-to-human properties ---
def _oracle_all_gates_pass(f, th, human_only):
    return (f["p_human_review"] < th["human_review_probability"] and f["p_infection"] < th["infection_probability"]
            and f["urgency_score"] < th["urgency_score"] and f["urgent_tail"] < th["urgent_tail_probability"]
            and f["p_same_day"] < th["same_day_probability"]
            and f["urgency_confidence"] >= th["urgency_confidence_min"] and f["specialty_confidence"] >= th["specialty_confidence_min"]
            and f["specialty_top_probability"] >= th["specialty_top_probability_min"]
            and all(abs(2 * f[k] - 1) >= th["boolean_certainty_min"] for k in ("p_infection", "p_same_day", "p_human_review"))
            and f["specialty"] not in human_only)


def _random_decision(rng):
    def dist(n):
        raw = [rng.random() ** 3 for _ in range(n)]
        s = sum(raw)
        return tuple(x / s for x in raw)

    spec = rng.choice(SPECIALTIES)
    return make_decision(urgency=dist(4), specialty=(spec, rng.uniform(0.15, 0.99)), infection=rng.random() ** 2, same_day=rng.random() ** 2,
                         human=rng.random() ** 2, complexity=dist(4), referral=dist(4))


def test_automation_happens_if_and_only_if_every_gate_passes(policy):
    rng = random.Random(7)
    automated = 0
    for _ in range(1500):
        d = _random_decision(rng)
        f = risk_model.features_from_decision(d)
        got = route_features(f, policy).route == "automated_specialty_route"
        assert got == _oracle_all_gates_pass(f, policy["thresholds"], policy["human_only_specialties"])
        automated += got
    assert 0 < automated < 1500  # the property is exercised on both sides


def test_tightening_any_threshold_never_creates_new_automation(policy):
    """Monotonicity: a more cautious policy automates a subset of what a looser one automates."""
    rng = random.Random(11)
    decisions = [risk_model.features_from_decision(_random_decision(rng)) for _ in range(600)]
    tighter = copy.deepcopy(policy)
    for k in MAX_LIMITS:
        tighter["thresholds"][k] = policy["thresholds"][k] * 0.5
    for k in MIN_LIMITS:
        tighter["thresholds"][k] = min(1.0, policy["thresholds"][k] + 0.1)
    for f in decisions:
        if route_features(f, tighter).route == "automated_specialty_route":
            assert route_features(f, policy).route == "automated_specialty_route"


def test_rule_order_is_fixed(policy):
    d = benign(human=0.9, infection=0.9, same_day=0.9, specialty=("emergency_dental_service", 0.1))
    order = []
    f = risk_model.features_from_decision(d)
    order.append(route_features(f, policy).rule)
    f["p_human_review"] = 0.0
    order.append(route_features(f, policy).rule)
    f["p_infection"] = 0.0
    order.append(route_features(f, policy).rule)
    f["p_same_day"] = 0.0
    f["urgency_confidence"], f["specialty_confidence"] = 1.0, 1.0
    f["p_human_review"] = f["p_infection"] = f["p_same_day"] = 0.0
    f["specialty_top_probability"] = 1.0
    order.append(route_features(f, policy).rule)
    assert order == ["R1_human_review_probability", "R2_infection_concern", "R3_urgent_risk", "R5_specialty_requires_human"]


def test_every_route_that_is_not_automated_requires_a_human(policy):
    rng = random.Random(3)
    for _ in range(400):
        r = go(_random_decision(rng), policy)
        assert r.requires_human == (r.route != "automated_specialty_route")


# --------------------------------------------------------- risk model plumbing ---
def test_risk_score_is_none_without_a_model_and_bounded_with_one():
    f = risk_model.features_from_decision(benign())
    assert risk_model.score(None, f) is None
    assert 0.0 <= risk_model.score(TOY_MODEL, f) <= 1.0
    f["p_infection"] = 1e9
    assert risk_model.score(TOY_MODEL, f) == pytest.approx(1.0)  # no overflow


def test_features_from_decision_equal_features_from_row():
    from tests.helpers import reference, row_from_decision

    d = benign(urgency=(0.2, 0.3, 0.3, 0.2), infection=0.4)
    a = risk_model.features_from_decision(d)
    b = risk_model.features_from_row(row_from_decision(d, reference(), "c", "train"))
    for k in a:
        assert a[k] == pytest.approx(b[k]) if isinstance(a[k], float) else a[k] == b[k]


# ------------------------------------------------------------ policy validation --
def test_shipped_policies_are_valid_and_labelled_research_demo():
    for name in ("conservative", "legacy"):
        p = validate_policy(load_routing_policy(name))
        assert "not clinical thresholds" in p["disclaimer"].lower()
    text = open("config/routing_policy.yaml", encoding="utf-8").read().lower()
    assert "not clinical thresholds" in text and "research-demo" in text


def test_routing_result_carries_the_research_demo_disclaimer(policy):
    r = go(benign(), policy)
    assert "not clinical thresholds" in r.disclaimer.lower()
    policy["provenance"] = {"data_mode": "demo"}
    assert "SIMULATED" in go(benign(), policy).disclaimer


@pytest.mark.parametrize("key, bad", [
    ("infection_probability", 0.85), ("human_review_probability", 0.8), ("urgency_score", 2.5), ("urgent_tail_probability", 0.9),
    ("same_day_probability", 0.8), ("risk_score", 0.9),
    ("urgency_confidence_min", 0.1), ("specialty_confidence_min", 0.3), ("specialty_top_probability_min", 0.2), ("boolean_certainty_min", 0.0),
])
def test_policy_looser_than_the_hard_limits_is_rejected(policy, key, bad):
    policy["thresholds"][key] = bad
    with pytest.raises(PolicyError) as exc:
        validate_policy(policy)
    assert key in exc.value.detail
    with pytest.raises(PolicyError):
        route_decision(benign(), policy)  # a loose policy can never route a case


def test_hard_limits_live_in_code_not_in_the_policy_file(policy):
    policy["limits"] = {"max": {"infection_probability": 0.99}}  # a file cannot raise its own ceiling
    policy["thresholds"]["infection_probability"] = 0.9
    with pytest.raises(PolicyError):
        validate_policy(policy)


def test_policy_must_define_every_threshold_and_valid_labels(policy):
    del policy["thresholds"]["boolean_certainty_min"]
    with pytest.raises(PolicyError):
        validate_policy(policy)
    p2 = copy.deepcopy(policy)
    p2["thresholds"] = dict(TH)
    p2["human_only_specialties"] = ["cosmetic_dentistry"]
    with pytest.raises(PolicyError):
        validate_policy(p2)
    p3 = copy.deepcopy(p2)
    p3["human_only_specialties"] = []
    p3["risk_model"] = {**TOY_MODEL, "coef": [0.0]}
    with pytest.raises(PolicyError):
        validate_policy(p3)
    with pytest.raises(PolicyError):
        validate_policy({"rules": "something-else"})


def test_nan_threshold_is_rejected(policy):
    policy["thresholds"]["infection_probability"] = float("nan")
    with pytest.raises(PolicyError):
        validate_policy(policy)


def test_default_policy_setting_selects_conservative_rules():
    assert load_routing_policy()["rules"] == "conservative"
    assert load_routing_policy("legacy")["rules"] == "legacy"


def test_pipeline_uses_the_conservative_policy_by_default():
    res = pipeline_service.evaluate_by_id("DEMO-04", persist=False)
    assert res.routing.policy_name == "conservative"
    assert res.routing.thresholds["infection_probability"] <= MAX_LIMITS["infection_probability"]
    assert URGENCY_LEVELS  # sanity: labels importable
